"""The generalized contracts: arms by declared name, action kinds, joint groups, the support
surface, success by latch, terminal score or an outside judge, resets by seed, initial-state index
or a person, and reruns. Fake declarations of a single-arm Panda, a BEHAVIOR-like mobile bimanual
robot and a real arm pass the conformance suites; the startup checks refuse each mismatch with a
line that names it."""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from roboscript.contracts.capabilities import (
    EnvCapabilities,
    episode_reset,
    missing,
)
from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot_server import RobotDescription, RobotLimits
from roboscript.contracts.skills import (
    SkillNeeds,
    SkillSet,
    SkillSpec,
)
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs import Benchmark, TaskInfo, load_benchmark
from roboscript.learning.prechecks import SKILLSETS, RunFacts, run_prechecks
from roboscript.learning.runspec import RunSpec
from roboscript.testing import precheck
from roboscript.testing.benchmarks import BenchmarkConformance, BenchmarkTarget
from roboscript.testing.embodiments import EmbodimentConformance, EmbodimentTarget
from roboscript.tools.skillsets import SkillSetPlugin

MOCK = yaml.safe_load(MOCK_BIMANUAL.read_text(encoding="utf-8"))
ARM = MOCK["arms"]["left"]
WRIST = {"kind": "wrist", "type": "rgbd", "width": 320, "height": 240}


def _panda() -> dict[str, Any]:
    """One arm named `arm`, driven by ee deltas in its base frame (OSC_POSE)."""
    return MOCK | {
        "name": "panda_like",
        "action_dim": 7,
        "arms": {"arm": ARM | {"action": [0, 6], "gripper_index": 6}},
        "cameras": {
            "agentview": MOCK["cameras"]["overhead"],
            "eye_in_hand": WRIST | {"arm": "arm"},
        },
        "actions": {
            "joint_position": False,
            "ee_delta": {"arms": ["arm"], "frame": "base", "max_translation_m": 0.05},
        },
    }


def _r1() -> dict[str, Any]:
    """BEHAVIOR-like: two arms, a torso of four joints and a holonomic base."""
    return MOCK | {
        "name": "r1_like",
        "action_dim": 18,
        "joint_groups": {"torso": {"joints": ["t1", "t2", "t3", "t4"], "action": [14, 18]}},
        "actions": {"base": {"holonomic": True, "max_speed_m_s": 1.0, "max_turn_rad_s": 1.0}},
    }


def _real_arm() -> dict[str, Any]:
    """A real single arm commanded by joint positions, its one arm named `arm`."""
    return MOCK | {
        "name": "real_arm",
        "action_dim": 7,
        "arms": {"arm": ARM | {"action": [0, 6], "gripper_index": 6}},
        "cameras": {"wrist": WRIST | {"arm": "arm"}},
    }


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / f"{data['name']}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("make", [_panda, _r1, _real_arm])
async def test_each_new_embodiment_passes_the_embodiment_suite(tmp_path: Path, make: Any) -> None:
    results = await precheck(EmbodimentConformance(), EmbodimentTarget(_write(tmp_path, make())))
    assert results and all(why is None for why in results.values()), results


def test_the_core_resets_by_seed_where_offered_else_by_initial_state() -> None:
    assert episode_reset(EnvCapabilities(reset={"seed", "initial_state"})) == "seed"
    assert episode_reset(EnvCapabilities(reset={"initial_state"})) == "initial_state"
    with pytest.raises(ValueError, match=r"reset by \['manual'\] cannot open an EpisodeSpec"):
        episode_reset(EnvCapabilities(reset={"manual"}, real_robot=True))


ROBOT = EnvCapabilities(success={"external"}, reset={"manual"}, demos=True, real_robot=True)
BY_INITIAL_STATE = EnvCapabilities(success={"latch"}, reset={"initial_state"})
BEHAVIOR = EnvCapabilities(success={"terminal_score"}, reset={"initial_state"}, reruns=False)


def test_a_method_needing_a_latch_is_refused_by_a_terminal_score() -> None:
    latch = EnvCapabilities(success={"latch"}, reset={"initial_state"})
    assert missing(latch, BEHAVIOR) == [
        "success judged by latch",
        "reruns of one reset (the env opens each only once)",
    ]
    once = latch.model_copy(update={"success": frozenset({"terminal_score"}), "reruns": False})
    assert missing(once, BEHAVIOR) == []


def test_a_real_robot_is_judged_from_outside_and_reset_by_hand() -> None:
    limits = RobotLimits(
        control_hz=200,
        workspace_min=(-0.5, -0.5, 0.0),
        workspace_max=(0.5, 0.5, 0.6),
        max_tcp_speed_m_s=0.25,
        max_joint_speed_rad_s=1.0,
        heartbeat_timeout_s=0.5,
    )
    args = {"name": "fake", "tasks": ["lift_red"], "embodiment": "real_arm", "cameras": ["wrist"]}
    RobotDescription(**args, capabilities=ROBOT, limits=limits)
    with pytest.raises(ValidationError, match="a real robot cannot declare"):
        EnvCapabilities(success={"terminal_score"}, reset={"manual"}, real_robot=True)


def _skill(name: str, **needs: Any) -> SkillSpec:
    return SkillSpec(
        name=name,
        description=f"{name}.",
        parameters={"type": "object"},
        needs=SkillNeeds(**needs),
        chain=["parameters", "executor", "env_execute", "receipt"],
    )


def _skillset(*skills: SkillSpec, name: str = "fake") -> SkillSet:
    return SkillSet(name=name, version="1", layer="benchmark", skills=list(skills))


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A module `fake_bench` holding single-arm and BEHAVIOR-like benchmarks and their skillsets,
    importable by path as a third party's in-repo example would be."""
    module = ModuleType("fake_bench")
    tasks = {TaskName("open_drawer"): TaskInfo(step_limit=600)}
    module.NUDGE = SkillSetPlugin(_skillset(_skill("nudge_ee", actions=["ee_delta"]), name="nudge"))
    module.DRIVE = SkillSetPlugin(_skillset(_skill("drive_to", actions=["base"]), name="drive"))
    module.PANDA = Benchmark(
        name="panda_like",
        tasks=tasks,
        env_server="roboscript.envs.mock",  # importable here; the suite only imports it
        interpreter="agent",
        embodiment=_write(tmp_path, _panda()),
        capabilities=BY_INITIAL_STATE,
        skillsets=["fake_bench:NUDGE"],
    )
    module.BEHAVIOR = module.PANDA.model_copy(
        update={
            "name": "behavior_like",
            "embodiment": _write(tmp_path, _r1()),
            "capabilities": BEHAVIOR,
            "skillsets": ["fake_bench:DRIVE"],
        }
    )
    module.DRIVE_ON_PANDA = module.PANDA.model_copy(update={"skillsets": ["fake_bench:DRIVE"]})
    monkeypatch.setitem(sys.modules, "fake_bench", module)
    return module


@pytest.mark.parametrize("name", ["PANDA", "BEHAVIOR"])
async def test_each_new_benchmark_passes_the_benchmark_suite(fakes: ModuleType, name: str) -> None:
    results = await precheck(BenchmarkConformance(), BenchmarkTarget(f"fake_bench:{name}"))
    assert results and all(why is None for why in results.values()), results


def _facts(env: str, embodiment: str, **extra: Any) -> RunFacts:
    benchmark = load_benchmark(env)
    spec = RunSpec.model_validate(
        {
            "env": {"name": env},
            "embodiment": {"name": embodiment},
            "method": {"name": "a_method"},
            "tasks": list(benchmark.tasks),
            "seeds": {"blocks": {}},
        }
        | extra
    )
    return RunFacts(spec=spec, offered=benchmark.capabilities, required=EnvCapabilities())


async def test_the_skillsets_precheck_names_each_misfit(fakes: ModuleType) -> None:
    panda = str(fakes.PANDA.embodiment)
    assert await run_prechecks([(SKILLSETS, {})], _facts("fake_bench:PANDA", panda)) == []
    assert await run_prechecks([(SKILLSETS, {})], _facts("fake_bench:DRIVE_ON_PANDA", panda)) == [
        "skillsets: drive_to: executes ['base'] actions, panda_like takes ['ee_delta']"
    ]
    # a core skillset's entries mount where they fit: on one arm `together` is left out, said
    # in a note, and the run goes on
    with_core = _facts("fake_bench:PANDA", panda, skillsets=[{"name": "manipulation"}])
    assert await run_prechecks([(SKILLSETS, {})], with_core) == []
