"""The interface contracts: phases, env capabilities, artifacts, the
skillset manifest and the robot server's models."""

import pytest
from pydantic import TypeAdapter, ValidationError

from roboscript.contracts.capabilities import (
    EnvCapabilities,
    ManualReset,
    OpenSpec,
    ResetSpec,
    SeedReset,
    SensingCapabilities,
    missing,
)
from roboscript.contracts.env import EnvDiscovery
from roboscript.contracts.phase import eval_tier_violations
from roboscript.contracts.skills import SkillSet, SkillSpec, StepResult, VerificationReport

HASH = "0" * 64


def test_eval_tier_violations() -> None:
    assert eval_tier_violations("learn", {"actor": "P"}) == []
    assert eval_tier_violations("eval", {"actor": "D"}) == []
    assert eval_tier_violations("eval", {"actor": "D", "reviser": "P"}) == [
        "role 'reviser' has tier P"
    ]


def test_a_real_robot_declares_no_privilege() -> None:
    EnvCapabilities(success={"external"}, reset={"manual"}, demos=True, real_robot=True)
    for bad in (
        {"truth": True},
        {"expert": True},
        {"success": {"latch"}},
        {"reset": {"seed", "manual"}},
    ):
        with pytest.raises(ValidationError, match="a real robot cannot declare"):
            EnvCapabilities(**({"reset": {"manual"}, "real_robot": True} | bad))


def test_missing_capabilities_say_what_lacks() -> None:
    wicl = EnvCapabilities(success={"latch"}, reset={"seed"}, truth=True, expert=True, demos=True)
    simulated = EnvCapabilities(
        success={"latch"}, reset={"seed"}, truth=True, expert=True, demos=True
    )
    robot = EnvCapabilities(success={"external"}, reset={"manual"}, demos=True, real_robot=True)
    assert missing(wicl, simulated) == []
    assert missing(wicl, robot) == [
        "success judged by latch",
        "reset by seed",
        "truth",
        "expert",
    ]


def test_reset_spec_is_discriminated_by_kind() -> None:
    adapter: TypeAdapter[ResetSpec] = TypeAdapter(ResetSpec)
    assert adapter.validate_python({"kind": "seed", "seed": 3}) == SeedReset(seed=3)
    assert isinstance(adapter.validate_python({"kind": "manual", "prompt": "p"}), ManualReset)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "seed", "index": 3})
    spec = OpenSpec(task="lift_red", reset=ManualReset(prompt="put the cube on the mark"))
    assert OpenSpec.model_validate_json(spec.model_dump_json()) == spec


def _skill(name: str = "grasp_object", **extra: object) -> SkillSpec:
    return SkillSpec.model_validate(
        {
            "name": name,
            "description": "Grasp an object.",
            "parameters": {"type": "object"},
            "chain": ["parameters", "perception", "planning", "executor", "receipt"],
        }
        | extra
    )


def test_skill_chain_follows_the_order() -> None:
    _skill(chain=["parameters", "executor"])
    for chain in (["executor", "planning"], ["planning", "planning"]):
        with pytest.raises(ValidationError, match="must follow"):
            _skill(chain=chain)


def test_skillset_manifest() -> None:
    report = VerificationReport(
        steps={"interface": StepResult(passed=True), "ceiling": StepResult(passed=False, value=0.8)}
    )
    skillset = SkillSet(
        name="manipulation",
        version="1",
        layer="core",
        skills=[_skill(bindings=["planner_profile"])],
        report=report,
    )
    assert SkillSet.kind == "skillset"
    assert SkillSet.model_validate_json(skillset.model_dump_json()) == skillset
    assert not report.passed
    with pytest.raises(ValidationError, match="named twice"):
        SkillSet(name="m", version="1", layer="core", skills=[_skill(), _skill()])
    with pytest.raises(ValidationError):
        SkillSet(name="m", version="1", layer="somewhere")


SENSING = SensingCapabilities(
    scene_identity=True, measured_proprio=True, depth=True, cameras=["head"], step_unit="action"
)


def test_a_discovery_declares_its_sensing() -> None:
    with pytest.raises(ValidationError, match="declares its sensing"):
        EnvDiscovery(
            contracts={},
            benchmark="b",
            embodiment="e",
            upstream_commit=None,
            config_hash=HASH,
            capabilities=EnvCapabilities(),
        )


def test_missing_sensing() -> None:
    offered = EnvCapabilities(sensing=SENSING)
    assert missing(EnvCapabilities(sensing=SENSING), offered) == []
    need = SENSING.model_copy(update={"cameras": ["head", "wrist"], "step_unit": "frame"})
    assert missing(EnvCapabilities(sensing=need), offered) == [
        "camera wrist",
        "steps counted in frame (the env counts action)",
    ]
    assert missing(EnvCapabilities(sensing=SENSING), EnvCapabilities()) == [
        "sensing (the env declares none)"
    ]
