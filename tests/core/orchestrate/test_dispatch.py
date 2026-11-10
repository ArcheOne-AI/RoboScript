"""Dispatch over fake workers: local code copies and run roots in `tmp_path`, reached through the
`local` transport (wrapped to cut a worker off or spoil a copy), each unit a `stub_unit.py`
standing in for the pipeline. The version and config refusals, the transport conformance and the
balanced assignment policy."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from roboscript.contracts.dispatch import (
    AgentReply,
    StartRequest,
    StatusRequest,
    StopRequest,
    UnitPaths,
)
from roboscript.contracts.ids import TaskName
from roboscript.orchestrate.dispatch import (
    DispatchConfig,
    PinnedPath,
    UnitSpec,
    WorkerSpec,
)
from roboscript.orchestrate.dispatch.agent import start_token
from roboscript.orchestrate.dispatch.assign import AssignmentPolicy, load_policy
from roboscript.orchestrate.dispatch.config import load_pipeline, local_keys, provenance
from roboscript.orchestrate.dispatch.transport import (
    open_transport,
)
from roboscript.testing.assignment import AssignmentConformance
from roboscript.testing.transport import TransportConformance, TransportTarget

STUB = Path(__file__).resolve().parent / "stub_unit.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
TASKS = [TaskName(t) for t in ("alpha", "beta", "gamma", "delta")]


def code_copy(path: Path, sha: str = SHA[:7]) -> Path:
    (path / ".roboscript").mkdir(parents=True)
    (path / ".roboscript" / "source.sha").write_text(sha + "\n", encoding="utf-8")
    (path / "configs").mkdir()
    (path / "configs" / "method.yaml").write_text("budget: 3\n", encoding="utf-8")
    return path


def worker(tmp: Path, name: str, gpus: list[int], **more: Any) -> WorkerSpec:
    code = tmp / name / "code"
    if not code.exists():
        code_copy(code)
    values = {
        "name": name,
        "transport": "local",
        "host": "localhost",
        "code_dir": str(code),
        "python": sys.executable,
        "gpus": gpus,
        "run_root": str(tmp / name / "runs"),
        "set": {"root": str(tmp / name / "runs")},
        **more,
    }
    return WorkerSpec.model_validate(values)


def dispatch(
    tmp: Path, workers: list[WorkerSpec], stub: dict[str, Any] | None = None, **more: Any
) -> DispatchConfig:
    pipeline = tmp / "pipeline.yaml"
    data = {
        "root": "runs/x",
        "tasks": ["ignored"],
        "gpus": [0],
        "learning": {"budget": 3},
        "stub": stub or {},
    }
    pipeline.write_text(yaml.safe_dump(data), encoding="utf-8")
    unit = UnitSpec(
        argv=[sys.executable, str(STUB), "--resume", "{dir}"],
        local_keys=["root"],
        forced={"gpu_cap_gb": 47.0},
        terminal=["done", "not_passed"],
        tombstones=["*/pipeline.json"],
        pinned=[PinnedPath(path="configs/method.yaml")],
        stop_grace_s=5,
    )
    values = {
        "name": "d1",
        "root": tmp / "coord" / "runs",
        "sha": SHA,
        "workers": workers,
        "tasks": TASKS,
        "costs": {"alpha": 4, "beta": 3, "gamma": 2, "delta": 1},
        "pipeline": pipeline,
        "unit": unit,
        "poll_s": 0.1,
        "relaunch_waits_s": [0.1, 0.1],
        "command_timeout_s": 60,
        **more,
    }
    return DispatchConfig.model_validate(values)


def agent(code: Path, command: str, request: Any) -> AgentReply:
    done = subprocess.run(
        [sys.executable, "-m", "roboscript.orchestrate.dispatch.agent", command],
        cwd=code,
        input=request.model_dump_json().encode(),
        capture_output=True,
        check=True,
    )
    return AgentReply.model_validate_json(done.stdout)


# -- the agent -------------------------------------------------------------------------------


def start_request(
    config: DispatchConfig, w: WorkerSpec, tasks: list[str], **more: Any
) -> StartRequest:
    pipeline = load_pipeline(config)
    values = {
        "unit": UnitPaths(run_root=w.run_root, unit_dir=f"{w.run_root}/d1/u"),
        "sha": SHA,
        "provenance": provenance(pipeline, local_keys(config.unit)),
        "config": {**pipeline, "tasks": tasks, "gpus": w.gpus},
        "config_file": "pipeline.yaml",
        "local_keys": local_keys(config.unit),
        "argv": config.unit.argv,
        "tasks": tasks,
        **more,
    }
    return StartRequest.model_validate(values)


def test_agent_refuses_another_version_or_config(tmp_path: Path) -> None:
    w = worker(tmp_path, "w1", [0])
    config = dispatch(tmp_path, [w], stub={"ticks": 200, "tick_s": 0.05})
    code = Path(w.code_dir)
    wrong = agent(code, "start", start_request(config, w, ["alpha"], sha="fedcba9876543"))
    assert not wrong.ok and "not the dispatch's" in (wrong.error or "")
    pinned = agent(
        code, "start", start_request(config, w, ["alpha"], pinned={"configs/method.yaml": "0" * 64})
    )
    assert not pinned.ok and "differs" in (pinned.error or "")
    started = agent(code, "start", start_request(config, w, ["alpha"]))
    assert started.ok and started.reply and started.reply["started"]
    again = agent(code, "start", start_request(config, w, ["alpha"]))
    assert again.ok and again.reply and not again.reply["started"]  # alive: left alone
    other = agent(code, "start", start_request(config, w, ["alpha"], provenance="1" * 64))
    assert not other.ok and "provenance" in (other.error or "")
    paths = UnitPaths(run_root=w.run_root, unit_dir=f"{w.run_root}/d1/u")
    pid = started.reply["pid"]
    stopped = agent(code, "stop", StopRequest(unit=paths, grace_s=5))
    assert stopped.ok and stopped.reply == {"pid": pid, "was_alive": True, "killed": False}
    assert start_token(pid) is None
    status = agent(code, "status", StatusRequest(unit=paths, status_files=["status.json"]))
    assert status.ok and status.reply and not status.reply["alive"] and status.reply["rows"]


# -- the coordinator ---------------------------------------------------------------------------


# -- transports --------------------------------------------------------------------------------


class TestLocalTransport(TransportConformance):
    @pytest.fixture
    def transport_target(self, tmp_path: Path) -> TransportTarget:
        (tmp_path / "remote").mkdir()
        (tmp_path / "local").mkdir()
        return TransportTarget(
            open_transport("local", "localhost"), str(tmp_path / "remote"), tmp_path / "local"
        )


class TestBalancedPolicy(AssignmentConformance):
    @pytest.fixture
    def policy_target(self) -> AssignmentPolicy:
        return load_policy("balanced")
