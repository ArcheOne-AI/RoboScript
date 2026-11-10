"""Conformance checks for gateway routing policies (`roboscript.gateway_policies`,
docs/concepts/agent-runtime.md).

A policy is registered as a factory (a class) that builds one with no arguments; the router asks
a policy only to order the endpoints it may take, so a policy conforms when `order` returns
exactly the endpoints it was given, each once, in the order of its preference, whatever the load,
pin and last good endpoint, for any number of calls in a row and without changing what it was
given; the factory builds a new policy each time; and a `Router` that routes with it
hands out slots on the endpoints its config names, keeps a session on its pin and frees what is
released.

The target is the registered factory. Use the checks from pytest by subclassing
`RoutingPolicyConformance` as a `Test*` class with a `policy_factory` fixture, or without pytest
via `roboscript.testing.precheck`:

```python
@pytest.fixture(params=registered("roboscript.gateway_policies", "roboscript_mypolicy"))
def policy_factory(request):
    return entrypoints.load("roboscript.gateway_policies", request.param)

class TestMyPolicy(RoutingPolicyConformance): ...
```
"""

import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from roboscript.contracts.ids import SessionId
from roboscript.gateway.config import GatewayConfig, Tier
from roboscript.gateway.router import Lease, Router, RoutingPolicy
from roboscript.testing import unit

Factory = Callable[[], RoutingPolicy]

ENDPOINTS = ("e1", "e2", "e3")
_READY: tuple[Sequence[str], ...] = ((), ("e1",), ENDPOINTS, ("e3", "e1"))
_LOADS: tuple[Mapping[str, float], ...] = (
    {},
    dict.fromkeys(ENDPOINTS, 0.0),
    {"e1": 0.9, "e2": 0.1, "e3": 0.5},
)


def _config(directory: Path) -> GatewayConfig:
    both = ["responses", "chat"]
    return GatewayConfig.model_validate(
        {
            "schema": "rs-gateway/v1",
            "listen": "127.0.0.1:0",
            "endpoints": {
                e: {"url_env": "U", "key_env": "K", "protocols": both} for e in ENDPOINTS
            },
            "models": {
                "m": {
                    "deployments": [
                        {"endpoint": e, "upstream": "up", "max_inflight": 2} for e in ENDPOINTS
                    ]
                }
            },
            "state_file": str(directory / "state.json"),
        }
    )


class RoutingPolicyConformance:
    """The checks, as `test_*` methods taking a policy factory."""

    @unit
    async def test_order_is_the_endpoints_it_was_given(self, policy_factory: Factory) -> None:
        policy = policy_factory()
        tiers: tuple[Tier, ...] = ("primary", "fallback")
        for _ in range(3):  # a policy may keep state between calls
            for ready in _READY:
                for load in _LOADS:
                    for tier in tiers:
                        for pinned, last_good in ((None, None), ("e2", "e3"), ("gone", "e1")):
                            given, loads = list(ready), dict(load)
                            order = policy.order(
                                "m", tier, given, pinned=pinned, last_good=last_good, load=loads
                            )
                            where = (ready, load, tier, pinned, last_good, order)
                            assert isinstance(order, list), where
                            assert sorted(order) == sorted(ready), where
                            assert (given, loads) == (list(ready), dict(load)), "it changed input"

    @unit
    async def test_the_factory_builds_a_new_policy_each_time(self, policy_factory: Factory) -> None:
        first, second = policy_factory(), policy_factory()
        assert first is not second
        ready, load = list(ENDPOINTS), dict.fromkeys(ENDPOINTS, 0.0)
        for policy in (first, second):
            order = policy.order("m", "primary", ready, pinned=None, last_good=None, load=load)
            assert sorted(order) == sorted(ready)

    async def test_a_router_routes_with_it(self, policy_factory: Factory) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            router = Router(_config(Path(directory)), policy_factory())
            sessions = [SessionId(f"s{i}") for i in range(4)]
            for session in sessions * 2:
                lease = await _take(router, session)
                assert lease.deployment.endpoint in ENDPOINTS
                lease.release()
            # A session stays on the endpoint that answered it while that has a free slot.
            first = await _take(router, sessions[0])
            router.succeeded(sessions[0], "m", "m", first.deployment.endpoint)  # it answered
            first.release()
            route = router.route(sessions[0], "m", "responses")
            assert route is not None and route.pinned == first.deployment.endpoint
            again = await router.acquire(route)
            assert again is not None and again.deployment.endpoint == first.deployment.endpoint
            again.release()
            # Every slot the config gives can be held at once, and releasing frees them.
            held = [await _take(router, None) for _ in range(2 * len(ENDPOINTS))]
            by_endpoint = [lease.deployment.endpoint for lease in held]
            assert sorted(by_endpoint) == sorted(ENDPOINTS * 2), by_endpoint
            for lease in held:
                lease.release()
            (await _take(router, None)).release()


async def _take(router: Router, session: SessionId | None) -> Lease:
    route = router.route(session, "m", "responses")
    assert route is not None
    lease = await router.acquire(route)
    assert lease is not None
    return lease
