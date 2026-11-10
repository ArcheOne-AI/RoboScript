"""The supervisor, for real: service processes of `stack_service.py`, started in order and stopped
by group, and the bounded stop of process groups that ignore SIGTERM."""

import json
import os
import signal
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import anyio
import anyio.lowlevel
import httpx

import roboscript
from roboscript.settings import repo_root
from roboscript.stack import (
    Interpreters,
    ServiceSpec,
    StackConfig,
    stop_groups,
    up,
)

SERVICE = str(Path(__file__).with_name("stack_service.py"))
CORE_SRC = Path(roboscript.__file__).parents[1]


def config(run_dir: Path, *services: ServiceSpec) -> StackConfig:
    python = Path(sys.executable)
    return StackConfig(
        interpreters=Interpreters(agent=python, sim=python),
        sim_pythonpath=[CORE_SRC],
        run_dir=run_dir,
        services=list(services),
    )


async def assert_gone(pids: Iterable[int], groups: Iterable[int]) -> None:
    """No process with these pids and no member of these groups is left (zombies included)."""
    pids, groups = list(pids), list(groups)
    with anyio.fail_after(5):
        while True:
            alive = [p for p in pids if _exists(os.kill, p)]
            alive += [g for g in groups if _exists(os.killpg, g)]
            if not alive:
                return
            await anyio.sleep(0.05)


def _exists(send: Callable[[int, int], None], pid: int) -> bool:
    try:
        send(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_services_start_in_order_and_stop_by_group(artifacts_dir: Path) -> None:
    done = artifacts_dir / "a-finally.txt"
    stack = config(
        artifacts_dir,
        ServiceSpec(
            name="b", argv=[SERVICE], interpreter="sim", depends_on=["a"], env={"RS_TEST": "b"}
        ),
        ServiceSpec(
            name="a",
            argv=[SERVICE, "--helper", "--two-planes", "--finally-file", str(done)],
            gpu=7,
            stop_grace_s=0.5,  # its helper ignores SIGTERM: SIGKILL after 0.5 s
        ),
    )
    async with up(stack) as running:
        assert list(running.endpoints) == ["a", "b"]  # a first: b depends on it
        second = running.urls["a"]["toy.b"]
        assert running.urls == {"a": {"toy.a": running.endpoints["a"], "toy.b": second}, "b": {}}
        written = json.loads(
            (artifacts_dir / ".roboscript" / "endpoints.json").read_text(encoding="utf-8")
        )
        assert {name: (e["url"], e["urls"]) for name, e in written.items()} == {
            name: (running.endpoints[name], running.urls[name]) for name in ("a", "b")
        }
        async with httpx.AsyncClient() as http:
            a = (await http.get(f"{running.endpoints['a']}/info")).json()
            b = (await http.get(f"{running.endpoints['b']}/info")).json()
            assert (await http.get(f"{second}/readyz")).status_code == 200
    assert a["pid"] == a["pgid"] == running.pids["a"] == written["a"]["pid"]  # its own group
    assert a["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert b["env"] == {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(CORE_SRC),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "RS_TEST": "b",
        "RS_ROOT": str(repo_root()),  # every service finds .env from any working directory
        "RS_BLOB_DIR": str(artifacts_dir / "blobs"),  # the stack's, by default
        "RS_BLOB_ROOTS": None,  # only a shared stack's services get one
    }
    assert a["env"]["RS_BLOB_DIR"] == b["env"]["RS_BLOB_DIR"] == str(running.blob_dir)
    assert running.blob_dir.is_dir()
    # SIGTERM ended serve() normally: the service's `finally` ran, for both of its servers.
    assert done.read_text(encoding="utf-8") == "serve returned"
    # The helper ignored SIGTERM and still went with its group.
    await assert_gone([a["helper"]], running.pids.values())
    assert f"--ready-file {artifacts_dir}/.roboscript/ready/a.json" in (
        (artifacts_dir / "logs" / "a.log").read_text(encoding="utf-8")
    )


async def test_stop_groups_kills_what_ignores_sigterm() -> None:
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    process = await anyio.open_process([sys.executable, "-c", stubborn], start_new_session=True)
    await anyio.sleep(0.3)  # let it install its handler
    started = anyio.current_time()
    await stop_groups([(process, 0.2)])
    assert process.returncode == -signal.SIGKILL and anyio.current_time() - started < 5
    await assert_gone([process.pid], [process.pid])
