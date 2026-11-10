"""The Agent host process (`python -m roboscript.learning.agent_host`): episodes of a benchmark
client against the process."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
import yaml
from pydantic import SecretStr

from roboscript import rpc
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.host import AgentHost
from roboscript.contracts.ids import TaskName
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.kernel import entrypoints
from roboscript.memory import (
    FileMemoryStore,
    MemoryDoc,
    VersionSource,
    build_manifest,
    write_manifest,
)
from roboscript.records import read_events
from roboscript.rpc import read_ready
from roboscript.settings import write_secret
from roboscript.testing.host import MockClient
from roboscript.testing.memory import sample_memory

pytestmark = pytest.mark.skipif(
    "scripted" not in entrypoints.names("roboscript.brains"),
    reason="the Agent host runs in the agent venv (roboscript installed, with entry points)",
)
REPO = Path(__file__).resolve().parents[3]
TASK = TaskName("lift_red")
RATIONALE = {"stage": "wait", "scene": "three boxes", "progress": "at rest", "plan": "stay"}


async def serving(url: str) -> bool:
    async with httpx.AsyncClient(trust_env=False) as http:
        return (await http.get(f"{url}/readyz")).status_code == 200


async def serve(
    tmp_path: Path,
    artifacts_dir: Path,
    *,
    method: str,
    toolsets: list[dict[str, Any]],
    replies: list[Any],
    memory: MemoryDoc | None = None,
    episodes: int = 2,
    environ: dict[str, str] | None = None,
    access: str = "library",
) -> tuple[int, list[Any]]:
    """The Agent host process on `tmp_path`'s frozen Memory (`memory`, else the sample), serving
    a MockEnv client `episodes` episodes, then stopped: its exit code and the verdicts. `environ`
    adds to its environment; `access`, how the Actor gets the Memory (`memory_access`)."""
    store = FileMemoryStore(tmp_path / "memory")
    doc = memory or sample_memory(TASK)
    await store.commit(doc, status="s1_draft", source=VersionSource(kind="s1"))
    await store.freeze(TASK, 0, reason="for the test")
    manifest = await build_manifest(store, [TASK])
    write_manifest(tmp_path / "manifest.json", manifest)
    config = {
        "manifest": str(tmp_path / "manifest.json"),
        "memory": {"name": "file", "options": {"root": str(tmp_path / "memory")}},
        "blob_dir": str(tmp_path / "blobs"),
        "recipe": str(REPO / "recipes" / "wicl"),
        "episodes": {
            "records": str(tmp_path / "host"),
            "embodiment": str(MOCK_BIMANUAL),
            "latch_finish": False,  # the script finishes once
        },
        "method": method,
        "memory_access": access,
        "toolsets": toolsets,
        "brain": {"name": "scripted", "config": {"replies": replies}},
    }
    (tmp_path / "host.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    token, ready_file, log = (
        "a-host-token-for-the-tests",
        tmp_path / "ready.json",
        artifacts_dir / "host.log",
    )
    write_secret(tmp_path / "host.token", SecretStr(token))  # as the launcher hands it over
    command = [sys.executable, "-m", "roboscript.learning.agent_host", "--config"]
    options = ["--token-file", str(tmp_path / "host.token"), "--ready-file", str(ready_file)]
    with log.open("wb") as output:
        process = await anyio.open_process(
            [*command, str(tmp_path / "host.yaml"), *options],
            stdout=output,
            stderr=subprocess.STDOUT,
            env={k: v for k, v in os.environ.items() if k != "RS_HOST_TOKEN"} | (environ or {}),
            start_new_session=True,
        )
    verdicts: list[Any] = []
    try:
        with anyio.fail_after(60):  # listening at once, ready once the Memory is loaded
            while (ready := read_ready(ready_file)) is None or not await serving(ready.url):
                if process.returncode is not None:
                    return process.returncode, verdicts
                await anyio.sleep(0.1)
        host = rpc.client(AgentHost, ready.url, token=SecretStr(token), timeout=10)  # type: ignore[type-abstract]
        blobs = LocalBlobStore(tmp_path / "blobs")  # shared with the host, as on one machine
        env = MockEnv(MockConfig(task=TASK, blob_dir=tmp_path / "blobs"), blobs)
        client = MockClient(env, host, "lift_red.clean.seed0", TASK, manifest.id)
        with anyio.fail_after(60):
            verdicts = [await client.episode() for _ in range(episodes)]
        await rpc.aclose(host)
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGTERM)
        with anyio.fail_after(30):
            code = await process.wait()
    return code, verdicts


async def test_the_process_serves_a_benchmark_client(tmp_path: Path, artifacts_dir: Path) -> None:
    hold = {"name": "hold", "arguments": {"rationale": RATIONALE, "steps": 2}}
    code, verdicts = await serve(
        tmp_path,
        artifacts_dir,
        method="roboscript.testing.phases:TestMethod",  # its Actor holds in p3
        toolsets=[{"name": "sensing"}, {"name": "hold"}],
        replies=[[hold], [{"name": "finish", "arguments": {"note": "done"}}]],
    )
    assert code == 0, (artifacts_dir / "host.log").read_text(encoding="utf-8")
    assert [verdict.steps_used for verdict in verdicts] == [40, 40]
    for index in (0, 1):
        episode = tmp_path / "host" / "lift_red.clean.seed0" / f"ep{index:03d}"
        summary = json.loads((episode / "summary.json").read_text(encoding="utf-8"))
        assert summary["end"]["reason"] == "finish" and summary["turns"] == 2, summary
        official = {"success": False, "steps": 40, "step_limit": 40, "failed_calls": 0}
        assert summary["official"] == official and summary["infra"] is False
        # library mode, the default: the manifest's Memory is the library, seen and not read
        assert summary["memory_access"] == "library" and summary["library"]["read"] == []
        assert summary["library"]["seen"] == [summary["library"]["own"]]
        logged = {event.episode_id for event in read_events(episode / "events.jsonl")}
        assert logged == {summary["episode_id"]}  # this episode's events, not the other's


SECRET = "h" * 40  # the run's RS_HUB_SECRET
