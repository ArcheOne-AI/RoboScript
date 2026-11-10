"""The toolset conformance suite (`roboscript.testing.toolsets`) on every toolset the core
registers in `roboscript.toolsets`, loaded through the entry point as a third-party package's
would be, and on toolsets that break it."""

from typing import Any

import pytest
from pydantic import BaseModel

from roboscript.testing import precheck, registered
from roboscript.testing.mock import Mock
from roboscript.testing.toolsets import ToolsetConformance, ToolsetTarget
from roboscript.tools import ToolContext, Toolset, tool

BUILTIN = registered("roboscript.toolsets")

pytestmark = pytest.mark.skipif(
    not BUILTIN, reason="roboscript is not installed here (the simulator env uses PYTHONPATH)"
)

# What a toolset needs to mount, beyond its defaults: `manipulation` binds its executor and
# planner through a skillset and a planner address (nothing is started that calls it).
NEEDS: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {
    "manipulation": ({"planner": {"url": "http://127.0.0.1:18300"}}, ("mock",)),
}


@pytest.fixture(params=BUILTIN)
def toolset_target(request: pytest.FixtureRequest) -> ToolsetTarget:
    config, skillsets = NEEDS.get(request.param, ({}, ()))
    return ToolsetTarget.registered(request.param, config, skillsets)


class TestBuiltinToolsets(ToolsetConformance):
    pass


def test_the_core_registers_its_toolsets() -> None:
    assert {"sensing", "perception", "geometry", "worldstate", "manipulation", "memory"} <= set(
        BUILTIN
    )


async def test_the_checks_run_without_pytest() -> None:
    target = ToolsetTarget.registered("mock")
    results = await precheck(ToolsetConformance(), target)
    assert results and all(why is None for why in results.values()), results


class Lax(BaseModel):
    """A config that takes any key."""


class Answer(BaseModel):
    ok: bool = True


class Lenient(Toolset):
    name = "lenient"
    Config = Lax

    @tool(kind="pure", tier="D")
    async def ping(self, ctx: ToolContext) -> Answer:
        """Answers."""
        return Answer()


class Orphan(Toolset):
    name = "orphan"

    @tool(kind="pure", tier="D", needs=("executor",))
    async def reach(self, ctx: ToolContext) -> Answer:
        """Needs a binding no skillset gives here."""
        return Answer()


class Truthful(Toolset):
    """A toolset with a tier-P tool next to a tier-D one."""

    name = "truthful"

    @tool(kind="pure", tier="P")
    async def truth(self, ctx: ToolContext) -> Answer:
        """Ground truth: tier P."""
        return Answer()

    @tool(kind="pure", tier="D")
    async def ping(self, ctx: ToolContext) -> Answer:
        """Answers."""
        return Answer()


async def test_a_toolset_with_a_privileged_tool_conforms() -> None:
    """The Hub lists a tier-P tool to a tier-P episode only; the suite expects just that."""
    results = await precheck(ToolsetConformance(), ToolsetTarget("truthful", Truthful))
    assert results["test_a_hub_starts_it_and_lists_its_tools"] is None, results


@pytest.mark.parametrize(
    ("check", "target"),
    [
        ("test_is_a_toolset_of_its_name", ToolsetTarget("other", Mock)),
        ("test_config_validates_and_refuses_unknown_keys", ToolsetTarget("lenient", Lenient)),
        ("test_a_provider_mounts_it", ToolsetTarget("orphan", Orphan)),
    ],
)
async def test_the_suite_catches_a_broken_toolset(check: str, target: ToolsetTarget) -> None:
    results = await precheck(ToolsetConformance(), target)
    assert results[check] is not None, results
