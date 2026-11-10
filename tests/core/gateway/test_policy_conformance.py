"""The routing policy conformance suite (`roboscript.testing.routing`) on every policy the core
registers in `roboscript.gateway_policies`, loaded through the entry point as a third-party
package's would be, and on policies that break it."""

from collections.abc import Mapping, Sequence

import pytest

from roboscript.gateway.config import Tier
from roboscript.gateway.router import POLICIES
from roboscript.kernel import entrypoints
from roboscript.testing import precheck, registered
from roboscript.testing.routing import Factory, RoutingPolicyConformance

BUILTIN = registered(POLICIES)

pytestmark = pytest.mark.skipif(
    not BUILTIN, reason="roboscript is not installed here (the simulator env uses PYTHONPATH)"
)


@pytest.fixture(params=BUILTIN)
def policy_factory(request: pytest.FixtureRequest) -> Factory:
    factory: Factory = entrypoints.load(POLICIES, request.param)
    return factory


class TestBuiltinPolicies(RoutingPolicyConformance):
    pass


def test_the_core_registers_the_default_policy() -> None:
    assert "default" in BUILTIN


async def test_the_checks_run_without_pytest() -> None:
    factory: Factory = entrypoints.load(POLICIES, "default")
    results = await precheck(RoutingPolicyConformance(), factory)
    assert results and all(why is None for why in results.values()), results


class _Base:
    def order(
        self,
        model: str,
        tier: Tier,
        ready: Sequence[str],
        *,
        pinned: str | None,
        last_good: str | None,
        load: Mapping[str, float],
    ) -> list[str]:
        return list(ready)


class Drops(_Base):
    """Leaves an endpoint out."""

    def order(self, model: str, tier: Tier, ready: Sequence[str], **rest: object) -> list[str]:  # type: ignore[override]
        return list(ready)[:1]


class Repeats(_Base):
    """Names the first endpoint twice."""

    def order(self, model: str, tier: Tier, ready: Sequence[str], **rest: object) -> list[str]:  # type: ignore[override]
        return [*ready, *ready[:1]]


class Sorts(_Base):
    """Sorts the list it was given in place."""

    def order(self, model: str, tier: Tier, ready: Sequence[str], **rest: object) -> list[str]:  # type: ignore[override]
        ready.sort(reverse=True)  # type: ignore[attr-defined]
        return list(ready)


@pytest.mark.parametrize(
    ("check", "policy"),
    [
        ("test_order_is_the_endpoints_it_was_given", Drops),
        ("test_order_is_the_endpoints_it_was_given", Repeats),
        ("test_order_is_the_endpoints_it_was_given", Sorts),
    ],
)
async def test_the_suite_catches_a_broken_policy(check: str, policy: type[_Base]) -> None:
    results = await precheck(RoutingPolicyConformance(), policy)
    assert results[check] is not None, results


async def test_the_suite_catches_a_factory_that_hands_out_one_policy() -> None:
    one = _Base()
    results = await precheck(RoutingPolicyConformance(), lambda: one)
    assert results["test_the_factory_builds_a_new_policy_each_time"] is not None
