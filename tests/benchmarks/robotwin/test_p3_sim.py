"""P3 through RoboTwin's own `eval_policy_client.py` and `policy_model_server.py` (marker `sim`):
the launcher runs two episodes of beat_block_hammer (seed block 0, clean, unseen instructions)
with a scripted brain that holds still for five actions, lifts the left gripper 5 cm through the
manipulation toolset (planned by the run's cuRobo motion service) and finishes, through the Agent
host, then stops the client. Checked: the step accounting (five held actions, the lift, then held
still to the limit), the lift's receipt, the official result directory and videos (moved into the
run's), every relay call far below the client's 30 s socket timeout, no process left, and nothing
left in the RoboTwin checkout: the test installs P3 only if it is not installed yet, and then
uninstalls it. One simulator process (the client) and the motion service.

    RS_ENV_FILE=<.env with ROBOTWIN_ROOT> python -m pytest -m sim \
        tests/benchmarks/robotwin/test_p3_sim.py
"""

import os
import re
import sys
from pathlib import Path

import anyio
import pytest

from roboscript.agent import read_lines
from roboscript.contracts.ids import TaskName
from roboscript.memory import FileMemoryStore, VersionSource, build_manifest, write_manifest
from roboscript.records import read_events
from roboscript.settings import repo_root
from roboscript.stack import Interpreters
from roboscript.testing.memory import sample_memory
from roboscript_robotwin.benchmark import HERE
from roboscript_robotwin.config import robotwin_root
from roboscript_robotwin.eval import install
from roboscript_robotwin.eval.launch import P3Config, P3Run, evaluate

pytestmark = pytest.mark.sim
TASK = TaskName("beat_block_hammer")
GPU = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
RATIONALE = {"stage": "wait", "scene": "a hammer", "progress": "at rest", "plan": "stay"}


def _results(root: Path) -> list[str]:
    """The checkout's `eval_result/` tree: the same before and after the test."""
    top = root / "eval_result"
    return sorted(str(p.relative_to(root)) for p in [top, *top.rglob("*")]) if top.exists() else []


def _taken(root: Path) -> list[Path]:
    """The paths of P3's links in the checkout that exist, as ours or another's."""
    return [link.path for link in install.links(root) if os.path.lexists(link.path)]


async def test_two_official_episodes(tmp_path: Path, artifacts_dir: Path) -> None:
    root = robotwin_root()
    agent = repo_root() / ".venv" / "bin" / "python"
    if root is None or not agent.is_file():
        pytest.skip("needs ROBOTWIN_ROOT (environment or RS_ENV_FILE) and the agent venv")
    store = FileMemoryStore(tmp_path / "memory")
    await store.commit(sample_memory(TASK), status="s1_draft", source=VersionSource(kind="s1"))
    await store.freeze(TASK, 0, reason="for the test")
    manifest = await build_manifest(store, [TASK])
    write_manifest(tmp_path / "manifest.json", manifest)
    hold = {"name": "hold", "arguments": {"rationale": RATIONALE, "steps": 5}}
    lift = {"name": "move_by", "arguments": {"rationale": RATIONALE, "arm": "left"}}
    lift["arguments"] |= {"z": 0.05}
    motion = tmp_path / "motion.yaml"
    motion.write_text(
        (repo_root() / "examples/simBenchmarks/RoboTwin/configs" / "motion.example.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    config = P3Config(
        interpreters=Interpreters(agent=agent, sim=Path(sys.executable)),
        sim_pythonpath=[repo_root() / "packages" / "core" / "src", HERE.parent],
        gpu=GPU,
        gpus=[GPU],
        motion={"config": str(motion)},
        host={
            "memory": {"name": "file", "options": {"root": str(tmp_path / "memory")}},
            "recipe": str(repo_root() / "recipes" / "wicl"),
            "episodes": {"embodiment": str(HERE / "embodiments" / "aloha_agilex.yaml")},
            "toolsets": [
                {"name": "sensing"},
                {"name": "hold"},
                {"name": "manipulation"},
            ],
            "brain": {
                "name": "scripted",
                "config": {
                    "replies": [[hold], [lift], [{"name": "finish", "arguments": {"note": "ok"}}]]
                },
            },
        },
    )
    root = root.resolve()
    results = _results(root)
    created = not install.installed(root)
    taken = _taken(root)
    if created and taken:
        pytest.skip(f"another checkout's P3 is installed in {root}: {taken}")
    if created:
        try:
            install.install(root)
        except FileExistsError as error:  # this checkout's install is another RoboTwin's
            pytest.skip(str(error))
    out = artifacts_dir / "p3"
    run = P3Run(task=TASK, seed=0, manifest=tmp_path / "manifest.json", out_dir=out, max_episodes=2)
    try:
        with anyio.fail_after(900):
            result = await evaluate(run, config)
    finally:
        if created:  # never the links of a run that installed them
            assert len(install.uninstall(root)) == 3

    assert result.truncated and result.success_rate is None
    for episode in result.episodes[:2]:
        official = {"success": False, "steps": 400, "step_limit": 400, "failed_calls": 0}
        assert episode["official"] == official and not episode["infra"]
        assert episode["end"]["reason"] == "finish" and episode["turns"] == 3
        assert episode["instruction"]  # RoboTwin's unseen instruction, as the agent saw it
    slot = out / "host" / "beat_block_hammer.rs_agentic_clean_p3.seed0"
    turns = read_lines(slot / "ep000" / "turns.jsonl")
    [held], [lifted] = (turn["exchanges"][0]["calls"] for turn in turns[:2])
    assert "steps_used=5" in held["content"][0]["text"], held
    receipt = lifted["content"][0]["text"]
    assert receipt.startswith("Receipt: ok (done).\nleft TCP commanded") and "394 left" in receipt
    reports = [
        event.data
        for event in read_events(slot / "ep000" / "events.jsonl")
        if event.kind == "p3.report"
    ]
    executed = [(r["executed"], r["step"], r["stopped"]) for r in reports]
    assert executed == [(5, 5, "completed"), (1, 6, "completed")]

    assert result.official_dir == out / "official" and (out / "official" / "episode0.mp4").is_file()
    assert result.infra_episodes == 0
    assert _results(root) == results  # moved out, and the directories its client made removed
    server_log = (out / "logs" / "policy_server.log").read_text(encoding="utf-8")
    slowest = [float(s) for s in re.findall(r"max (\d+\.\d+) s", server_log)]
    assert len(slowest) >= 8 and max(slowest) < 10, slowest  # the client times out at 30 s
    client_log = (out / "logs" / "client.log").read_text(encoding="utf-8")
    assert "Communication error" not in client_log and "Traceback" not in client_log
    for group in result.pids.values():
        with pytest.raises(ProcessLookupError):
            os.killpg(group, 0)
