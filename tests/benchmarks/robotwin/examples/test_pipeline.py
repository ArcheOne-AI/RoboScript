"""The pipeline on MockEnv, without the simulator or a model (pipeline_fakes): scripted brains, the
oracle lift, a MockPlanner service per GPU, the fake gateway and RoboTwin's official scripts
stood in, each job in a worker process. A P3 block whose client fails is run again; an S1
without an official success stops and counts 0 in P3; the report counts what the records say; P3
counts per mode and a lower count closes blocks; an S2 at its cap goes on with its best Memory,
and under `p3.unconverged: stop` is not evaluated; the practice latch switch reaches P3; a lost
episode keeps its block open until a resumed run ends it, and a block lost too often fails its
job; the pipeline config example and the script's real wiring."""

import dataclasses
import importlib
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import types
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import anyio
import anyio.abc
import pytest
from pipeline_fakes import (
    HELPERS,
    TESTS,
    fake_robotwin,
    record_demo,
    wiring,
)
from pipeline_fakes import config as pipeline_config
from robotwin_pipeline import (
    PipelineConfig,
    RunEnv,
    compose_config,
    leftovers,
    load_directory,
    new_directory,
    report,
    run_pipeline,
    schedule,
)
from robotwin_pipeline import run as run_module
from robotwin_pipeline.config import MOTION
from robotwin_pipeline.run import (
    definition,
    demo_host,
    failure_of,
    gate_variant,
)

from roboscript.agent.session import Usage
from roboscript.contracts.events import Event
from roboscript.contracts.hub import EpisodeEnd
from roboscript.contracts.ids import EpisodeId, ManifestId, TaskName
from roboscript.kernel import entrypoints
from roboscript.learning import EpisodeOutcome
from roboscript.learning.episode import set_aside
from roboscript.memory import FileMemoryStore, VersionSource, object_roles
from roboscript.motion import MockPlanner
from roboscript.orchestrate.jobs import RunKey, RunState
from roboscript.settings import load_config_file, repo_root
from roboscript_methods.wicl.ledger import LedgerRow
from roboscript_methods.wicl.practice import PracticeConfig
from roboscript_robotwin.benchmark import HERE
from roboscript_robotwin.eval import install, launch, wire
from roboscript_robotwin.eval import job as p3_job
from roboscript_robotwin.eval.blocks import (
    block_done,
    p3_pending,
)
from roboscript_robotwin.eval.job import p3_config
from roboscript_robotwin.eval.launch import P3Config, P3Motion, P3Result, P3Run

EXAMPLE = HERE.parents[4] / "examples" / "simBenchmarks" / "RoboTwin"

RED, BLUE = TaskName("lift_red"), TaskName("lift_blue")
SCRIPTS = EXAMPLE / "scripts"
agent_venv = pytest.mark.skipif(
    "scripted" not in entrypoints.names("roboscript.brains"),
    reason="P3's Agent host runs in the agent venv (roboscript installed, with entry points)",
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The env servers' token, and a fake RoboTwin checkout with P3 installed (the install
    recorded in `tmp_path`, as every worker is told; `tmp_path/home` holds the launcher's
    locks)."""
    monkeypatch.setenv("RS_ENV_TOKEN", secrets.token_urlsafe(24))
    checkout, expected = fake_robotwin(tmp_path)
    record = tmp_path / "install.json"
    monkeypatch.setattr(install, "record_path", lambda: record)
    monkeypatch.setenv("RS_TEST_P3_RECORD", str(record))
    install.install(checkout, expected=expected)
    monkeypatch.setenv("ROBOTWIN_ROOT", str(checkout))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def state(directory: Path, task: str, variant: str) -> RunState:
    return RunState.load(directory / task / variant, RunKey(TaskName(task), variant))


def records(directory: Path) -> list[Path]:
    """The pid records left (none once every group is stopped)."""
    return schedule._record_files(directory)


@agent_venv
async def test_a_p3_block_whose_client_fails_is_run_again(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The official client exits 1, then exits 0 without its result: neither block counts, each
    is set aside and the job respawned; its next block runs whole. A launch again refuses a
    changed P3 config; with the P3 mode `randomized` it runs only the frozen variant's P3 there,
    a new block beside the clean one."""
    monkeypatch.setenv("RS_TEST_CLIENT_FAILS", str(root / "client-failed"))
    monkeypatch.setenv("RS_TEST_CLIENT_NO_RESULT", str(root / "client-no-result"))
    await record_demo(RED, root / "demos" / RED / "package")
    config = pipeline_config(root, [RED], {"default": {}})
    directory = new_directory(config)
    with anyio.fail_after(240):
        found = await run_pipeline(config, wiring(), directory)
    assert found["runs"] == {"done": 2}
    ended = state(directory, RED, "default")
    assert (ended.status, ended.launches, ended.respawns) == ("done", 4, 2)  # S2, P3 thrice
    blocks = directory / RED / "default" / "P3" / "clean"
    failed = P3Result.model_validate_json((blocks / "block1-aborted1" / "result.json").read_bytes())
    assert (failed.client_exit, failed.truncated, failed.config_error) == (1, False, None)
    rateless = P3Result.model_validate_json(
        (blocks / "block1-aborted2" / "result.json").read_bytes()
    )
    assert (rateless.client_exit, rateless.success_rate, rateless.truncated) == (0, None, False)
    counted = P3Result.model_validate_json((blocks / "block1" / "result.json").read_bytes())
    assert (counted.client_exit, counted.success_rate) == (0, 0.0)
    [row] = [row for row in found["rows"] if row["run"] == "lift_red/default"]
    assert list(row["p3"]["blocks"]) == ["clean/block1"]  # the set-aside blocks count nowhere
    assert leftovers(directory) == [] and records(directory) == []
    p3_yaml = (root / "p3.yaml").read_text(encoding="utf-8")
    (root / "p3.yaml").write_text(
        p3_yaml.replace("stop_grace_s: 5", "stop_grace_s: 6"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"lift_red/default \(p3\)"):
        await run_pipeline(load_directory(directory), wiring(), directory, p3_modes=["randomized"])
    (root / "p3.yaml").write_text(p3_yaml, encoding="utf-8")
    ledger = (directory / RED / "default" / RED / "S2" / "ledger.jsonl").read_bytes()
    with anyio.fail_after(120):
        found = await run_pipeline(
            load_directory(directory), wiring(), directory, p3_modes=["randomized"]
        )
    assert found["runs"] == {"done": 2}
    reopened = state(directory, RED, "default")
    assert (reopened.status, reopened.launches) == ("done", 5)  # its P3 in randomized scenes
    assert state(directory, RED, "S1").launches == 1
    assert (directory / RED / "default" / RED / "S2" / "ledger.jsonl").read_bytes() == ledger
    randomized = P3Result.model_validate_json(
        (directory / RED / "default" / "P3" / "randomized" / "block1" / "result.json").read_bytes()
    )
    assert (randomized.task_config, randomized.success_rate) == ("rs_agentic_randomized_p3", 0.0)
    [row] = [row for row in found["rows"] if row["run"] == "lift_red/default"]
    assert list(row["p3"]["modes"]) == ["clean", "randomized"]
    launched = json.loads(
        (directory / "launches.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert (launched["jobs"], launched["p3_modes"]) == (["lift_red/default"], ["randomized"])
    assert leftovers(directory) == [] and records(directory) == []


"""S1 on MockEnv that cannot exit: the hinted first attempt fails (its lift too low), the
Reviser's fix makes the unhinted second succeed, and the attempts run out before two passes."""


async def test_an_s1_without_an_official_success_stops_and_counts_0_in_p3(root: Path) -> None:
    """S1 runs out of its one attempt, which had no official success: under the `success` rule
    it has no fallback (`s1_exit: none`, why), the task stops (S1 not passed, its variant
    blocked), and the summary counts the variant 0 of P3's episodes in each mode."""
    await record_demo(RED, root / "demos" / RED / "package")
    more = {"protocols": {"imitate": {"max_attempts": 1}}, "p3": {"randomized": True}}
    config = pipeline_config(root, [RED], {"default": {}}, **more)
    directory = new_directory(config)
    with anyio.fail_after(180):
        found = await run_pipeline(config, wiring(), directory)
    assert found["runs"] == {"not_passed": 1, "blocked": 1}
    (exit_path,) = (directory / RED / "S1").glob("*/S1/*/exit.json")
    exited = json.loads(exit_path.read_bytes())
    assert (exited["passed"], exited["s1_exit"], exited["fallback"]) == (False, "none", None)
    assert exited["fallback_reason"] == "no official success"
    assert not (directory / RED / "S1" / "fallback.json").exists()
    assert not (directory / RED / "default" / RED / "S2").exists()
    summary = json.loads((directory / "summary.json").read_bytes())
    assert (summary["s1_best_memory"], summary["s1_stopped"]) == ([], [RED])
    for mode in ("clean", "randomized"):
        overall = summary["p3_overall"][mode]
        assert (overall["successes"], overall["n"], overall["rate"]) == (0, 100, 0.0)
        assert overall["zeroed"] == [f"{RED}/default"] and overall["pending"] == []
    page = (directory / "summary.md").read_text(encoding="utf-8")
    assert "P3 clean: 0/100 over 1 variants" in page
    assert "no exit, stopped (no official success)" in page
    # a launch again leaves it stopped: the fallback was found under the rule
    loaded = load_directory(directory)
    jobs = {job["job"]: job["action"] for job in schedule.plan(loaded, directory)["jobs"]}
    assert jobs[f"{RED}/S1"] == "skip (not_passed)"


def test_the_pipeline_config_example() -> None:
    config = compose_config(
        str(repo_root() / "examples/simBenchmarks/RoboTwin/configs" / "pipeline.example.yaml")
    )
    assert len(config.tasks) == 50  # none named: every task
    assert (config.gpus, config.max_per_gpu, config.max_runs) == ([0], 10, 15)
    assert (config.gpu_cap_gb, config.gpu_margin_gb, config.free_gb_per_place) == (46, 1, 6.5)
    assert config.watchdog_s == 1800
    assert config.gpu_claims == Path("/dev/shm/gpu_claims")
    assert config.shared_stacks == Path("/dev/shm/gpu_stacks")
    assert config.shared_linger_s == 60
    assert (config.service_mib("motion"), config.service_mib("vision")) == (2500, 8500)
    assert (config.p3.blocks, config.p3.episodes, config.p3.modes) == ([1], 100, ["clean"])
    assert (config.respawn_waits_s, config.min_free_gb) == ([60, 300, 900], 100)
    recipe = config.recipe().protocols["practice"]
    practice = {
        variant: PracticeConfig.model_validate(
            {**recipe, **config.protocol("practice", "t", variant)}
        )
        for variant in config.variants
    }
    assert {v: (p.diagnosis, p.budget_factor) for v, p in practice.items()} == {
        "default": (True, 1),
    }
    assert config.demo_of("lift_pot") == repo_root() / "data/demos/recorded/lift_pot/package"
    assert config.per_gpu[0] == MOTION and config.planner == "motion"
    vision = config.per_gpu[1]
    assert (vision.name, vision.module, vision.interpreter) == (
        "vision",
        "roboscript_vision",
        "vision",
    )


def test_the_scripts_real_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The toolsets `run_s2` mounts with the demo toolset (for S1), the manipulation toolset and
    the scene toolsets on the job's GPU's services; the script as the worker; P3's launcher config
    per job."""
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(
        repo_root() / "examples/simBenchmarks/RoboTwin/configs" / "manipulation.example.yaml",
        configs / "manipulation.yaml",
    )
    monkeypatch.syspath_prepend(str(SCRIPTS))
    script = importlib.import_module("run_pipeline")
    wired = script.wiring(root=tmp_path)
    assert Path(wired.worker[1]) == (SCRIPTS / "run_pipeline.py").resolve()
    assert (wired.worker[2], wired.oracle) == ("--worker", None)
    services = {"motion": "http://127.0.0.1:18301", "vision": "http://127.0.0.1:18302"}
    env = RunEnv(RED, 4, MockPlanner(), services["motion"], services, None)  # type: ignore[arg-type]
    mounted = {provider.name: provider for provider in wired.toolsets(env)}
    assert list(mounted) == [
        "sensing",
        "manipulation",
        "demo",
        "perception",
        "geometry",
        "worldstate",
        "done",
    ]
    assert mounted["manipulation"].config.planner.url == services["motion"]  # type: ignore[attr-defined]
    scene = mounted["perception"].config  # type: ignore[attr-defined]
    assert (scene.planner, scene.vision) == (services["motion"], services["vision"])
    # the run's VLM model (`PipelineConfig.vlm_model`) reaches every scene toolset and the demo
    # toolset's; without one, the scene config's stands
    own = scene.vlm.model
    env = dataclasses.replace(env, vlm_model="gpt-6.1-sol")
    mounted = {provider.name: provider for provider in wired.toolsets(env)}
    for name in ("perception", "geometry", "worldstate"):
        assert mounted[name].config.vlm.model == "gpt-6.1-sol"  # type: ignore[attr-defined]
    assert mounted["demo"].config.scene.vlm.model == "gpt-6.1-sol"  # type: ignore[attr-defined]
    assert own != "gpt-6.1-sol"
    base = load_config_file(EXAMPLE / "configs" / "p3.example.yaml", P3Config)
    launcher = p3_config(base, tmp_path / "run", 0, services["motion"], None, tmp_path / "blobs")
    assert (launcher.gpu, launcher.motion, launcher.vision) == (
        0,
        P3Motion(url=services["motion"]),
        None,
    )
    assert launcher.host["memory"] == {"name": "file", "options": {"root": str(tmp_path / "run")}}


def test_the_report_counts_what_the_records_say(tmp_path: Path) -> None:
    """Of the episodes that count: receipts, hard failures by workflow stage, failed ends, fresh
    identifications; voided and set-aside ones apart; P3's rate from a complete block's official
    result and a cut block's host counts, a running block's progress beside (and a block whose
    client wrote no rate counted nowhere); a counted block with over 5% infrastructure failures
    flagged, a success standing even with `infra` beside it."""
    key = RunKey(RED, "default")
    run = tmp_path / RED / "default"
    s2 = run / RED / "S2"
    fresh = {"status": "identified"}
    grasp = [
        call("c1", "lift", {"rationale": {"stage": "grasp"}}),
        result("c1", {"reason": "empty_grasp", "ok": False}),
        call("c2", "observe", {}),
        result("c2", {"frame": "f", "arms": {"left": {"gripper": 1.0}}}),
        fact({"roles": {"box": {"reason": "ok"}, "base": {"reason": "ambiguous"}}}),
    ]
    episode(s2 / "10000" / "try1", "s2", False, "fail_fast:empty_grasp@grasp", grasp, fresh)
    both = {"left": {"reason": "ok", "ok": True}, "right": {"reason": "lost_hold", "ok": False}}
    moved = [call("c3", "together", {"rationale": {"stage": "lift"}}), result("c3", {"arms": both})]
    reused = {"status": "reused", "source": "try1"}
    episode(s2 / "10000" / "try2", "s2", True, "success", [*moved, grasp[-1]], reused)
    lost = [result("c4", {"reason": "plan_failed", "ok": False})]
    episode(s2 / "10000" / "try1-infra1", "s2", False, "gateway_error", lost, end_class="infra")
    episode(s2 / "10001-aborted1" / "try1", "s2", False, "finish", lost)
    rows = [
        ledger_row(10000, 1, False, "merge", ["empty grasp in stage grasp"]),
        ledger_row(10001, 2, True, "pass", []),
    ]
    text = "".join(row.model_dump_json() + "\n" for row in rows)
    (s2 / "ledger.jsonl").write_text(text + '{"seed"', encoding="utf-8")  # a row half appended
    blocks = run / "P3" / "clean"
    block(  # its second episode a success with `infra` beside it
        blocks / "block1", [True, True, False], rate=0.43, exit=0, truncated=False, guard=2, infra=1
    )
    block(blocks / "block2", [True, False], rate=None, exit=-9, truncated=True, infra=1)  # cut
    block(blocks / "block3", [True], rate=None, exit=None, truncated=None)  # running
    block(blocks / "block4", [True], rate=None, exit=0, truncated=False)  # no _result.txt
    block(blocks / "block1-aborted1", [True], rate=None, exit=1, truncated=False, guard=0)
    config = pipeline_config(tmp_path, [RED], {"default": {}})
    config = config.model_copy(update={"p3": config.p3.model_copy(update={"episodes": 2})})
    found = report.report(config).job_row(tmp_path, key, 0.0, deep=True)
    assert (found["s2"]["seeds"], found["s2"]["curve"], found["s2"]["streak"]) == (2, [0, 1], 1)
    clean = found["p3"]["modes"]["clean"]
    assert (clean["successes"], clean["n"], clean["n_official"], clean["n_host"]) == (
        44,
        102,
        100,
        2,
    )
    assert (clean["host_ended"], clean["running_ended"], clean["running_successes"]) == (5, 2, 2)
    assert list(found["p3"]["blocks"]) == [f"clean/block{n}" for n in (1, 2, 3, 4)]
    block1, block2 = (found["p3"]["blocks"][f"clean/block{n}"] for n in (1, 2))
    assert (block1["infra"], block1["host_successes"], block2["infra"]) == (0, 2, 1)
    taxonomy = found["taxonomy"]
    assert taxonomy["receipts"]["s2"] == {"empty_grasp": 1, "ok": 1, "lost_hold": 1}
    assert taxonomy["hard_failures"]["s2"] == {"empty_grasp@grasp": 1, "lost_hold@lift": 1}
    assert taxonomy["identification"]["s2"] == {"ok": 1, "ambiguous": 1}  # fresh only
    assert taxonomy["openings"]["s2"] == {"identified": 1, "reused": 1}
    assert taxonomy["failed_ends"] == {
        "s1": {},
        "s2": {"fail_fast:empty_grasp@grasp": 1},
        "p3": {"wall_clock": 1},
    }
    assert taxonomy["infra_ends"]["s2"] == {"gateway_error": 1}
    assert taxonomy["infra_ends"]["p3"] == {"step_limit": 1}
    assert taxonomy["s2_labels"] == {"empty grasp in stage grasp": 1}
    assert (found["infra"], found["guard_trips"]) == ({"s1": 0, "s2": 1, "p3": 1}, 1)
    assert found["tokens"]["s2"] == 4 * (14 + 5)  # the brain's and perception's, all four
    assert found["tokens"]["p3"] == 8 * 14
    assert report.summary(tmp_path, config)["flags"] == [
        "P3 lift_red/default clean/block2: 1 of 2 episodes failed by infrastructure"
    ]
    more = config.model_copy(update={"p3": config.p3.model_copy(update={"episodes": 3})})
    again = report.report(more).job_row(tmp_path, key, 0.0, deep=False)["p3"]
    assert not again["blocks"]["clean/block2"]["done"]  # cut after 2 of the 3 asked: runs again
    assert again["modes"]["clean"]["running_ended"] == 4
    assert block_done(blocks / "block2", 2) and not block_done(blocks / "block2", 3)
    assert definition(more, key) == definition(config, key)  # not frozen: a resume may ask more
    both = {"blocks": [1, 2]}
    assert not p3_pending(config.p3.model_copy(update=both), run)
    assert p3_pending(more.p3.model_copy(update=both), run)


"""A GPU's motion service that fails its first start on GPU 4 (marker file), then serves."""
"""A GPU's motion service that runs as `fake_motion --config <how>` on its first start on GPU 4
(marker file), then serves."""


def _env() -> dict[str, str]:
    """A process's environment that finds `roboscript_robotwin`, the core and the tests'
    helpers."""
    paths = [str(HERE.parent), str(repo_root() / "packages" / "core" / "src"), str(TESTS)]
    paths += [str(EXAMPLE), str(HELPERS)]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}


@asynccontextmanager
async def launched(directory: Path, root: Path) -> AsyncIterator[anyio.abc.Process]:
    """The pipeline in a process of its own, as run_pipeline.py starts it (killed at the end if
    it still runs), its output in `root/launch.log`."""
    command = [sys.executable, str(TESTS / "pipeline_fakes.py"), "launch", str(directory)]
    with (root / "launch.log").open("wb") as log:
        process = await anyio.open_process(
            command, stdout=log, stderr=subprocess.STDOUT, env=_env()
        )
    try:
        yield process
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def episode(
    directory: Path,
    stage: str,
    success: bool,
    end: str,
    events: list[dict[str, Any]],
    identification: dict[str, Any] | None = None,
    end_class: str | None = None,
) -> None:
    """An S1 or S2 Actor episode's summary and events."""
    outcome = EpisodeOutcome(
        task=RED,
        stage=stage,  # type: ignore[arg-type]
        role="actor",
        seed=10000,
        directory=directory,
        episode_id=EpisodeId("ep-1"),
        env_episode=EpisodeId("env-1"),
        instruction="lift the red box",
        success=success,
        end_class=end_class or ("success" if success else "fail_fast"),  # type: ignore[arg-type]
        end=EpisodeEnd(reason=end, at=0.0),
        usage=Usage(input_tokens=11, output_tokens=3),
        perception_usage=Usage(input_tokens=4, output_tokens=1),
        identification=identification,
        wall_clock_s=2.0,
    )
    directory.mkdir(parents=True)
    config = {"task": RED, "stage": stage, "role": "actor"}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "summary.json").write_text(outcome.model_dump_json(), encoding="utf-8")
    lines = [
        Event(ts=0.0, seq=n, service="test", **e).model_dump_json() for n, e in enumerate(events)
    ]
    (directory / "events.jsonl").write_text(
        "".join(line + "\n" for line in lines), encoding="utf-8"
    )


def call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "tool.call", "data": {"call_id": call_id, "name": name, "arguments": arguments}}


def result(call_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "tool.result", "data": {"call_id": call_id, "result": data}}


def fact(value: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "scene.fact", "data": {"name": "identified", "value": value}}


def ledger_row(seed: int, k: int, first: bool, outcome: str, labels: list[str]) -> LedgerRow:
    return LedgerRow(
        seed=seed,
        k=k,
        solvable=True,
        instruction="lift the red box",
        tries=[],
        first_try_pass=first,
        streak_after=int(first),
        outcome=outcome,  # type: ignore[arg-type]
        memory_before=0,
        memory_after=0,
        failure_labels=labels,
    )


def block(
    out: Path,
    successes: list[bool],
    *,
    rate: float | None,
    exit: int | None,
    truncated: bool | None,
    guard: int | None = None,
    infra: int | None = None,
) -> None:
    """A P3 block: the Agent host's episode summaries (the `guard`-th tripped its runaway guard,
    the `infra`-th failed by infrastructure) and, unless it runs (`truncated` None), its
    `result.json`."""
    summaries = []
    for n, success in enumerate(successes):
        reason = "finish" if success else "wall_clock" if n == guard else "step_limit"
        summary = {
            "end": {"reason": reason, "at": 0},
            "official": {"success": success, "steps": 10, "step_limit": 10},
            "infra": n == infra,
            "guard_tripped": n == guard,
            "usage": {"input_tokens": 11, "output_tokens": 3},
            "elapsed_s": 1.0,
        }
        summaries.append(summary)
        (out / "host" / "k" / f"ep{n:03d}").mkdir(parents=True)
        (out / "host" / "k" / f"ep{n:03d}" / "config.json").write_text("{}", encoding="utf-8")
        (out / "host" / "k" / f"ep{n:03d}" / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    if truncated is None:
        return
    done = P3Result(
        task=RED,
        seed=1,
        task_config="rs_agentic_clean_p3",
        manifest=ManifestId("mf-0123456789ab"),
        official_dir=None,
        success_rate=rate,
        episodes=summaries,
        infra_episodes=0,
        truncated=truncated,
        client_exit=exit,
        pids={},
    )
    (out / "result.json").write_text(done.model_dump_json(), encoding="utf-8")


def _interrupt(block: Path) -> Path:
    """Make a finished P3 block look like a run its launch left unfinished: set aside, no
    `result.json` (as a stopped driver leaves it, `schedule.prune`)."""
    aside = set_aside(block)
    assert aside is not None
    (aside / "result.json").unlink()
    return aside


def _ended(block: Path) -> int:
    result = P3Result.model_validate_json((block / "result.json").read_bytes())
    return sum(e["official"] is not None for e in result.episodes)


def _evaluations(run: Path) -> list[str]:
    lines = (run / "worker.log").read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.startswith("$ ") and " evaluate " in line]


@agent_venv
async def test_p3_counts_per_mode_and_a_lower_count_closes_blocks(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clean blocks of 4, then that run interrupted: a launch asking 5 resumes it (4 carried) and
    stops at 5. That run interrupted too, a launch asking 3 clean and 2 randomized closes the
    clean block from it (all 5 count, nothing run: no client, no pids) and starts randomized
    from scratch for exactly 2; the dry run says so first. `--redo-p3` of randomized runs
    exactly 2 again from scratch; the closed clean block stays. The definition ignores both
    counts."""
    monkeypatch.setenv("RS_FAKE_EPISODES", "6")
    await record_demo(RED, root / "demos" / RED / "package")
    config = pipeline_config(root, [RED], {"default": {}}, p3={"episodes": 4})
    directory = new_directory(config)
    with anyio.fail_after(240):
        await run_pipeline(config, wiring(), directory)
    run = directory / RED / "default"
    clean = run / "P3" / "clean"
    assert _ended(clean / "block1") == 4
    first = _interrupt(clean / "block1")
    assert len(launch.finished(first)) == 4

    five = load_directory(directory, ["p3.episodes=5"])
    with anyio.fail_after(240):
        await run_pipeline(five, wiring(), directory, p3_modes=["clean"])
    carried = json.loads((clean / "block1" / "carried.json").read_bytes())
    assert len(carried["episodes"]) == 4 and _ended(clean / "block1") == 5
    assert len(launch.finished(clean / "block1")) == 5
    second = _interrupt(clean / "block1")
    assert second.name == "block1-aborted2"

    lower = ["p3.episodes=3", "p3.episodes_randomized=2", "p3.randomized=true"]
    loaded = load_directory(directory, lower)
    assert (loaded.p3.episodes_of("clean"), loaded.p3.episodes_of("randomized")) == (3, 2)
    key = RunKey(RED, "default")
    assert definition(loaded, key) == definition(config, key)
    planned = schedule.plan(loaded, directory, p3_modes=["clean", "randomized"])
    [row] = [j for j in planned["jobs"] if j["job"] == key.name]
    assert row["p3"] == {
        "clean/block1": "close at 3: 5 finished in block1-aborted2",
        "randomized/block1": "start (2 episodes)",
    }
    assert not (clean / "block1").exists()  # the dry run wrote nothing
    before = len(_evaluations(run))
    with anyio.fail_after(240):
        found = await run_pipeline(loaded, wiring(), directory, p3_modes=["clean", "randomized"])
    assert found["runs"] == {"done": 2}
    closed = clean / "block1"
    assert json.loads((closed / "closed.json").read_bytes())["finished"] == 5
    assert not (closed / "pids.json").exists() and _ended(closed) == 5
    assert block_done(closed, 3) and not block_done(closed, 6)
    assert [e["index"] for e in json.loads((closed / "carried.json").read_bytes())["episodes"]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    randomized = run / "P3" / "randomized" / "block1"
    assert _ended(randomized) == 2 and not (randomized / "carried.json").exists()
    assert len(launch.finished(randomized)) == 2
    assert len(_evaluations(run)) == before + 1  # one job, its clean block closed already
    launched = json.loads((directory / "launches.jsonl").read_text("utf-8").splitlines()[-1])
    assert launched["closed_p3"] == {key.name: ["clean/block1"]}
    [row] = [r for r in report.status(directory, loaded)["rows"] if r["run"] == key.name]
    assert row["p3"]["blocks"]["clean/block1"]["closed"]
    assert (row["p3"]["modes"]["clean"]["n"], row["p3"]["modes"]["randomized"]["n"]) == (5, 2)

    with anyio.fail_after(240):
        await run_pipeline(
            loaded, wiring(), directory, p3_modes=["randomized"], redo_p3=[(RED, "randomized")]
        )
    assert (run / "P3" / "randomized" / "block1-redone1").is_dir()
    assert _ended(randomized) == 2 and not (randomized / "carried.json").exists()
    assert (closed / "closed.json").is_file() and _ended(closed) == 5
    assert leftovers(directory) == [] and records(directory) == []


PARALLEL: Final = {"randomized": "spare", "spare_parallel": True}


CAPPED: Final = {"protocols": {"practice": {"max_seeds": 1}}}
"""One counted seed: S2 never converges (two first-try passes stop it early)."""


def _stopped_at_cap(run: Path, version: int, hashed: str) -> None:
    """Make a variant's S2 look stopped at its cap under `p3.unconverged: stop`: no
    `s2_exit.json`, the Memory frozen at `version` (the cap's), `stop.json` naming it."""
    s2 = run / RED / "S2"
    (s2 / "s2_exit.json").unlink()
    folder = FileMemoryStore(run).folder(RED)
    for aside in folder.glob("frozen.before*.json"):
        aside.unlink()
    FileMemoryStore(run).refreeze(RED, version, reason="not converged")
    for aside in folder.glob("frozen.before*.json"):
        aside.unlink()
    stop = json.loads((s2 / "stop.json").read_bytes())
    stop |= {"version": version, "hash": hashed}
    (s2 / "stop.json").write_text(json.dumps(stop), encoding="utf-8")


@agent_venv
async def test_an_s2_at_its_cap_goes_on_with_its_best_memory(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`p3.unconverged: best_memory` (the default): a variant whose
    S2 stops at its seed cap goes on to P3 with the version of the best first-try record, frozen
    for P3 (`s2_exit.json`), and counts in `p3_overall`. One that stopped under `stop`
    with the cap's version frozen gets the pick at the next launch: the cap's record kept aside,
    its P3 blocks (run on the cap's version) moved aside and run again. One with no first-try
    success ends `not_converged` and counts 0 in P3. P3 does not run with end targets read for
    the cap's roles: they are read again for the picked version (the old file kept aside), and
    a failed reading puts it back and fails the job."""
    await record_demo(RED, root / "demos" / RED / "package")
    config = pipeline_config(root, [RED], {"default": {}}, p3=PARALLEL, **CAPPED)
    assert config.p3.unconverged == "best_memory"
    assert "s2_unconverged" not in definition(config, RunKey(RED, "default"))
    directory = new_directory(config)
    run = directory / RED / "default"
    with anyio.fail_after(300):
        found = await run_pipeline(config, wiring(), directory)
    assert found["runs"] == {"done": 2}
    stop = json.loads((run / RED / "S2" / "stop.json").read_bytes())
    assert (stop["reason"], stop["k"]) == ("not converged", 1)
    exited = json.loads((run / RED / "S2" / "s2_exit.json").read_bytes())
    start = json.loads((run / RED / "S2" / "start.json").read_bytes())
    assert exited["s2_exit"] == "best_memory"
    assert exited["picked"]["successes"] == exited["picked"]["n"] == 1
    assert exited["version"] == start["version"] == stop["version"]  # 1 of 1 on the start
    assert "redone_p3" not in exited  # the frozen version stood
    frozen = await FileMemoryStore(run).frozen(RED)
    assert frozen is not None and frozen.version == exited["version"]
    for mode in ("clean", "randomized"):
        assert block_done(run / "P3" / mode / "block1"), mode
    summary = json.loads((directory / "summary.json").read_bytes())
    assert summary["s2_best_memory"] == [f"{RED}/default v{exited['version']:03d}"]
    assert summary["s2_stopped"] == []
    for mode in ("clean", "randomized"):
        overall = summary["p3_overall"][mode]
        assert overall["variants"] == 1 and overall["not_counted"] == overall["zeroed"] == []
        assert overall["all"]["n"] == overall["n"]
    page = (directory / "summary.md").read_text(encoding="utf-8")
    assert f"best_memory v{exited['version']:03d} (1/1" in page
    assert "S2 did not converge, P3 on its best Memory" in page

    # stopped under `stop`, with a later version (the cap's) frozen
    store = FileMemoryStore(run)
    picked = await store.load(RED, exited["version"])
    (store.folder(RED) / "frozen.json").unlink()  # as if S2 had merged once more before its cap
    later = await store.commit(
        picked.doc.replace(notes="- changed after the last seed."),
        status="edited",
        source=VersionSource(kind="edited"),
        parent=picked.meta.version,
    )
    _stopped_at_cap(run, later.meta.version, later.meta.hash)
    main = state(directory, RED, "default")
    main.end(run, "not_converged", "S2 stopped at its seed cap without converging")
    planned = schedule.plan(load_directory(directory), directory)
    row = next(j for j in planned["jobs"] if j["job"] == f"{RED}/default")
    assert row["action"] == f"S2's best Memory v{exited['version']:03d}: its P3 goes on"
    with anyio.fail_after(300):
        found = await run_pipeline(load_directory(directory), wiring(), directory)
    assert found["runs"] == {"done": 2}
    launch_ = json.loads(
        (directory / "launches.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert launch_["gated"] == {f"{RED}/default": f"picked v{exited['version']:03d}"}
    again = json.loads((run / RED / "S2" / "s2_exit.json").read_bytes())
    assert (again["version"], again["instead_of"]["version"]) == (
        exited["version"],
        later.meta.version,
    )
    assert again["redone_p3"] == {"clean": ["block1"], "randomized": ["block1"]}
    frozen = await store.frozen(RED)
    assert frozen is not None and frozen.version == exited["version"]
    assert frozen.summary["s2_exit"] == "best_memory"
    aside = json.loads((store.folder(RED) / "frozen.before1.json").read_bytes())
    assert aside["version"] == later.meta.version
    manifest = json.loads((run / "P3" / "manifest.json").read_bytes())
    assert json.dumps(manifest).count(frozen.hash) >= 1  # written afresh for the pick
    for mode in ("clean", "randomized"):
        assert block_done(run / "P3" / mode / "block1"), mode
        assert (run / "P3" / mode / "block1-redone1").is_dir(), mode

    # end targets (P3 reads them with a done goal): read again when the picked version's roles
    # are not the file's, the old file kept aside; kept when they are
    goal = config.model_copy(update={"done_goal": True})
    key = RunKey(RED, "default")
    s2 = run / RED / "S2"
    assert run_module.end_targets_due(config, directory, key) is None  # no done goal: none read
    package = run_module.end_targets_due(goal, directory, key)  # none read yet
    assert package == goal.demo_of(RED)
    (s2 / "end_targets.json").write_text(
        json.dumps({"roles": ["cap"], "targets": []}), encoding="utf-8"
    )
    roles = sorted(object_roles(picked.doc))
    assert roles != ["cap"] and run_module.end_targets_due(goal, directory, key) == package
    calls: list[tuple[Any, ...]] = []

    async def reading(driver: Any, folder: Path, pkg: Any, doc: Any, **kw: Any) -> Any:
        calls.append((folder, doc, kw))
        if kw.get("overwrite") and len(calls) > 1:
            return None, "no reading within 1 s"
        (folder / "end_targets.json").write_text(
            json.dumps({"roles": roles, "targets": []}), encoding="utf-8"
        )
        return [], "read"

    async def no_import(*_: Any) -> None:
        return None

    monkeypatch.setattr(run_module, "read_end_targets", reading)
    monkeypatch.setattr(run_module, "import_blobs", no_import)
    runner = types.SimpleNamespace(driver=None, blobs=None)
    await run_module.refresh_end_targets(goal, runner, directory, key)  # type: ignore[arg-type]
    (folder, doc, kw), *_ = calls
    assert folder == s2 and doc == picked.doc and kw["overwrite"] is True
    assert json.loads((s2 / "end_targets.json").read_bytes())["roles"] == roles
    assert json.loads((s2 / "end_targets.before1.json").read_bytes())["roles"] == ["cap"]
    assert run_module.end_targets_due(goal, directory, key) is None  # the roles match: kept
    await run_module.refresh_end_targets(goal, runner, directory, key)
    assert len(calls) == 1
    (s2 / "end_targets.json").write_text(
        json.dumps({"roles": ["cap"], "targets": []}), encoding="utf-8"
    )
    with pytest.raises(run_module.EndTargetsUnread, match="P3 does not start") as failed:
        await run_module.refresh_end_targets(goal, runner, directory, key)
    assert run_module.failure_of(failed.value) == "infra"  # an outage: respawned
    assert json.loads((s2 / "end_targets.json").read_bytes())["roles"] == ["cap"]  # put back
    assert not (s2 / "end_targets.before2.json").exists()
    # a P3-mode launch leaves it out, and says so
    with caplog.at_level(logging.WARNING, logger="robotwin_pipeline.schedule"):
        assert await schedule._pending_jobs(goal, directory, False, ["clean"]) == []
    assert "end targets of S2's best Memory are read first" in caplog.text
    # a package without frames has no end to read: P3 goes on without (as a practice does)
    real = run_module.load_package
    monkeypatch.setattr(
        run_module, "load_package", lambda p: real(p).model_copy(update={"frames": []})
    )
    assert run_module.end_targets_due(goal, directory, key) is None
    monkeypatch.setattr(run_module, "load_package", real)
    (s2 / "end_targets.json").unlink()
    for aside in s2.glob("end_targets.before*.json"):
        aside.unlink()

    # no first-try success at the cap: it stops and counts 0
    ledger = run / RED / "S2" / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    rows = [{**r, "first_try_pass": False, "outcome": "skip"} for r in rows]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (run / RED / "S2" / "s2_exit.json").unlink()
    loaded = load_directory(directory)
    assert gate_variant(loaded, directory, RunKey(RED, "default")) == "stopped before P3"
    main = state(directory, RED, "default")
    assert main.status == "not_converged"
    assert main.error is not None and "no first-try success" in main.error
    assert json.loads((run / RED / "S2" / "s2_exit.json").read_bytes())["s2_exit"] == "none"
    found = report.write_summary(directory, loaded)
    assert found["s2_stopped"] == [f"{RED}/default"] and found["s2_best_memory"] == []
    for mode in ("clean", "randomized"):
        overall = found["p3_overall"][mode]
        assert overall["zeroed"] == [f"{RED}/default"]
        assert (overall["successes"], overall["n"], overall["rate"]) == (0, 100, 0.0)
    page = (directory / "summary.md").read_text(encoding="utf-8")
    assert "S2 stopped (no first-try success; P3 counted 0)" in page
    planned = schedule.plan(loaded, directory)
    row = next(j for j in planned["jobs"] if j["job"] == f"{RED}/default")
    assert "with no first-try success, counted 0 in P3" in row["action"]


@agent_venv
async def test_an_s2_at_its_cap_under_stop_is_not_evaluated(root: Path) -> None:
    """`p3.unconverged: stop`: a variant whose S2 stops at its seed cap without converging may
    not go on to P3 (`p3_gated`): it ends `not_converged`, no P3 block runs, and P3 leaves it out
    of its rate, named as not counted."""
    await record_demo(RED, root / "demos" / RED / "package")
    config = pipeline_config(root, [RED], {"default": {}}, p3={"unconverged": "stop"}, **CAPPED)
    assert config.p3.unconverged == "stop"
    directory = new_directory(config)
    run = directory / RED / "default"
    with anyio.fail_after(300):
        found = await run_pipeline(config, wiring(), directory)
    stop = json.loads((run / RED / "S2" / "stop.json").read_bytes())
    assert stop["reason"] == "not converged"
    assert run_module.p3_gated(config, run, RED)
    assert state(directory, RED, "default").status == "not_converged"
    assert found["runs"] == {"done": 1, "not_converged": 1}  # the task's S1 job, the variant
    for mode in config.p3.modes:
        assert not block_done(run / "P3" / mode / "block1"), mode
    summary = json.loads((directory / "summary.json").read_bytes())
    for mode in config.p3.modes:
        overall = summary["p3_overall"][mode]
        assert overall["not_counted"] == [f"{RED}/default"], mode
        assert overall["n"] == 0, mode


def test_the_practice_latch_switch_reaches_p3() -> None:
    """One switch: a practice that sets `latch_finish` gives its P3 host the same."""
    assert "episodes" not in demo_host({})
    assert demo_host({"latch_finish": False})["episodes"] == {"latch_finish": False}


MANIFEST_ID = ManifestId("mf-" + "0" * 12)


def _lost_result(out: Path, task: str, lost: Sequence[int]) -> None:
    """A block's run whose client exited 0 with the official rate (the lost episodes counted as
    failures, as the official client counts them) and its outcomes, `lost` lost."""
    outcomes = [
        {"index": i, "seed": 200000 + i, **({"lost": True, "losses": 1} if i in lost else {})}
        | ({} if i in lost else {"success": True})
        for i in range(3)
    ]
    result = P3Result(
        task=TaskName(task), seed=1, task_config="demo_clean", manifest=MANIFEST_ID,
        official_dir=out / "official", success_rate=0.66, episodes=[], infra_episodes=len(lost),
        truncated=False, client_exit=0, pids={}, outcomes=outcomes,
    )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(result.model_dump_json(), encoding="utf-8")
    lines, wins = [], 0  # the official client's log, our policy's mark before a lost episode
    for i in range(3):
        if i in lost:
            lines.append(wire.INFRA_LOST)
        wins += i not in lost
        lines.append(f"Success rate: {wins}/{i + 1} => 1%, current seed: {200000 + i}")
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / "client.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _p3_job(tmp_path: Path) -> tuple[PipelineConfig, Path]:
    """A variant's job directory ready for P3 (`run._evaluate`, `eval.job.evaluate_blocks`): its
    manifest, one clean block."""
    config = pipeline_config(
        tmp_path, [RED], {"default": {}}, p3={"config": str(tmp_path / "p3.yaml"), "episodes": 3}
    )
    run_dir = tmp_path / "job"
    (run_dir / "P3").mkdir(parents=True)
    (run_dir / "P3" / "manifest.json").write_text("{}", encoding="utf-8")
    return config, run_dir


async def test_a_lost_episode_keeps_its_block_open_until_a_resumed_run_ends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block whose run ended with an episode
    infrastructure lost (the official rate counting it as a failure) is not done; the job sets
    it aside and resumes it at once, from that run, and the block is done once the resumed run
    ended the episode."""
    config, run_dir = _p3_job(tmp_path)
    calls: list[P3Run] = []

    async def evaluate(run: P3Run, launcher: P3Config) -> None:
        calls.append(run)
        _lost_result(run.out_dir, RED, [1] if len(calls) == 1 else [])

    monkeypatch.setattr(p3_job, "evaluate", evaluate)
    block = run_dir / "P3" / "clean" / "block1"
    _lost_result(tmp_path / "probe", RED, [1])
    assert not block_done(tmp_path / "probe")  # the official rate is there, the loss is not over
    await run_module._evaluate(
        config,
        RunKey(RED, "default"),
        run_dir,
        4,
        {"motion": "http://127.0.0.1:1"},
        tmp_path,
        ["clean"],
    )
    assert [run.resume for run in calls] == [None, block.parent / "block1-aborted1"]
    assert block_done(block) and not p3_pending(config.p3, run_dir)


async def test_a_block_lost_too_often_fails_its_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume that refuses an episode lost `MAX_LOSSES` runs in a row is not a
    start that failed nor infrastructure: the job fails at once with that message (no restart,
    no respawn)."""
    config, run_dir = _p3_job(tmp_path)
    calls: list[P3Run] = []

    async def evaluate(run: P3Run, launcher: P3Config) -> None:
        calls.append(run)
        raise launch.LostRepeatedly("infrastructure lost episodes [...] 3 runs in a row")

    monkeypatch.setattr(p3_job, "evaluate", evaluate)
    with pytest.raises(launch.LostRepeatedly) as caught:
        await run_module._evaluate(
            config,
            RunKey(RED, "default"),
            run_dir,
            4,
            {"motion": "http://x:1"},
            tmp_path,
            ["clean"],
        )
    assert len(calls) == 1 and failure_of(caught.value) == "error"
