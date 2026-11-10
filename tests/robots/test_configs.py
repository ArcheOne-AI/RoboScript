"""The three configurations' fake robots: each fake profile and descriptor pass the core suites, a
configuration cannot claim what a real robot never has, and a run naming a robot as its env is
refused before it starts."""

from typing import Any

import pytest
from pydantic import ValidationError

from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.embodiments import load_embodiment
from roboscript.envs import load_benchmark
from roboscript.learning.prechecks import (
    ENV_CAPABILITIES,
    REAL_ROBOT,
    RunFacts,
    run_prechecks,
)
from roboscript.learning.runspec import RunSpec
from roboscript.testing import precheck
from roboscript.testing.benchmarks import BenchmarkConformance, BenchmarkTarget
from roboscript.testing.embodiments import EmbodimentConformance, EmbodimentTarget
from roboscript_robots import dual_arm, mobile_manipulator, single_arm
from roboscript_robots.base import (
    HardwareBackend,
    RobotConfig,
    load_robot,
)

CONFIGURATIONS = [
    (single_arm, "fake_single_arm", single_arm.SingleArmConfig, single_arm.SingleArmBackend),
    (dual_arm, "fake_dual_arm", dual_arm.DualArmConfig, dual_arm.DualArmBackend),
    (
        mobile_manipulator,
        "fake_mobile_manipulator",
        mobile_manipulator.MobileManipulatorConfig,
        mobile_manipulator.MobileManipulatorBackend,
    ),
]
IDS = [name for _, name, _, _ in CONFIGURATIONS]


def _data(name: str) -> dict[str, Any]:
    spec = load_robot(name)
    return spec.config.model_dump(mode="json")


@pytest.mark.parametrize(("package", "name", "config_model", "backend"), CONFIGURATIONS, ids=IDS)
async def test_each_fake_profile_and_descriptor_pass_the_core_suites(
    package: Any, name: str, config_model: type[RobotConfig], backend: type[HardwareBackend]
) -> None:
    for suite, target in (
        (EmbodimentConformance(), EmbodimentTarget(name)),
        (BenchmarkConformance(), BenchmarkTarget(name)),
    ):
        results = await precheck(suite, target)
        assert results and all(why is None for why in results.values()), results
    assert load_embodiment(name).name == name == load_benchmark(name).name


@pytest.mark.parametrize(
    "claim",
    [{"truth": True}, {"expert": True}, {"capabilities": {"truth": True}}, {"success": ["latch"]}],
)
def test_a_configuration_cannot_claim_what_a_real_robot_never_has(claim: dict[str, Any]) -> None:
    """Truth snapshots, expert pre-runs, a latch or a seed reset are not configuration fields:
    the capabilities of a real robot are fixed (docs/how-to/add-a-real-robot.md)."""
    data = _data("fake_dual_arm") | claim
    with pytest.raises(ValidationError, match="Extra inputs"):
        dual_arm.DualArmConfig.model_validate(data)


@pytest.mark.parametrize("name", IDS)
async def test_a_run_naming_a_robot_as_its_env_is_refused_before_it_starts(name: str) -> None:
    """Registered as a benchmark, a robot is still no env the runner can open episodes on (it
    serves `robot.control`, not `env.control`): the startup precheck says so, whatever the
    method needs."""
    spec = RunSpec.model_validate(
        {
            "env": {"name": name},
            "embodiment": {"name": name},
            "method": {"name": "wicl"},
            "tasks": [next(iter(load_benchmark(name).tasks))],
            "seeds": {"blocks": {}},
        }
    )
    needs_nothing = EnvCapabilities(reruns=False)
    facts = RunFacts(spec=spec, offered=load_benchmark(name).capabilities, required=needs_nothing)
    assert await run_prechecks([(ENV_CAPABILITIES, {})], facts) == [
        f"env_capabilities: {REAL_ROBOT}"
    ]


# Hardware skeletons --------------------------------------------------------------------------
