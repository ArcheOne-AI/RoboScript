"""RoboTwin's P1 glue (`examples/simBenchmarks/RoboTwin/p1`): the parts that run without a
simulator, a GPU or a model. The scripts import each other by name (their directory is on
`sys.path` when run), so this file puts that directory there and loads `build.py` and
`freeze.py` under names of their own. It checks the origin tolerance, the executor and tolerance
the bench binds, the success condition, and that a stop signal ends the build."""

import argparse
import asyncio
import importlib.util
import math
import os
import signal
import sys
from pathlib import Path
from types import ModuleType

import anyio
import pytest

P1 = Path(__file__).resolve().parents[4] / "examples" / "simBenchmarks" / "RoboTwin" / "p1"
if str(P1) not in sys.path:
    sys.path.insert(0, str(P1))

import entry_check  # noqa: E402

pytestmark = pytest.mark.unit


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"robotwin_p1_{name}", P1 / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = load("build")
freeze = load("freeze")


# -- entry_check ----------------------------------------------------------------------------------


def test_origin_tolerance_is_robotwins_planner_threshold() -> None:
    metres, degrees = entry_check.origin_tolerance()
    assert metres == 0.005
    # cuRobo's rotation threshold 0.05 is the norm of the error quaternion's vector part
    assert degrees == pytest.approx(math.degrees(2 * math.asin(0.05)))
    assert degrees == pytest.approx(5.732, abs=1e-3)
    assert (entry_check.CUT.home_m, entry_check.CUT.home_deg) == (metres, degrees)


def test_bench_binds_robotwins_executor_and_tolerance() -> None:
    from roboscript_robotwin.bindings import BINDINGS

    found = entry_check.bench()
    assert found.executor == BINDINGS.executor
    assert found.origin_tolerance == entry_check.origin_tolerance()
    assert found.workspace is entry_check.UNPROBED


# -- demo_export ----------------------------------------------------------------------------------


# -- build ----------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _repository_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests' directories lie in the repository (test-results/); `--runs` must not."""
    monkeypatch.setattr(build, "REPO", Path("/nonexistent/repository"))


def args_for(tmp_path: Path, *extra: str) -> argparse.Namespace:
    recordings = tmp_path / "recordings"
    recordings.mkdir(exist_ok=True)
    argv = ["--run", "r", "--runs", str(tmp_path / "runs"), "--python", str(tmp_path / "py"),
            "--robotwin-root", str(tmp_path / "RoboTwin"), "--recordings", str(recordings),
            "--gpus", "0,1", *extra]  # fmt: skip
    return build.checked(build.parser().parse_args(argv))


def test_success_condition_reads_check_success(tmp_path: Path) -> None:
    bench = build.RoboTwin(args_for(tmp_path))
    envs = tmp_path / "RoboTwin" / "envs"
    envs.mkdir(parents=True)
    (envs / "open_microwave.py").write_text(
        "class open_microwave:\n"
        "    def play_once(self):\n        pass\n\n"
        "    def check_success(self):\n        return self.angle > 0.6\n",
        encoding="utf-8",
    )
    assert bench.success_condition("open_microwave") == (
        "def check_success(self):\n    return self.angle > 0.6"
    )
    assert bench.success_condition("missing") == ""


# -- freeze ---------------------------------------------------------------------------------------


def test_a_stop_signal_ends_the_build_and_what_it_started() -> None:
    started: list[asyncio.subprocess.Process] = []

    async def body() -> None:  # a service the build's context managers own
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)", start_new_session=True
        )
        started.append(child)
        try:
            asyncio.get_running_loop().call_later(0.2, os.kill, os.getpid(), signal.SIGTERM)
            await asyncio.sleep(30)
        finally:  # shielded, as the stack supervisor's stop is
            child.terminate()
            with anyio.CancelScope(shield=True):
                await child.wait()

    with pytest.raises(SystemExit) as stopped:
        build.until_stopped(body)
    assert stopped.value.code == 128 + signal.SIGTERM
    assert started[0].returncode is not None
