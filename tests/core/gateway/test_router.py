"""roboscript.gateway.router: sessions stick to where they succeeded, aliases resolve, failures cool
a deployment down, and fallback deployments serve only when no primary can."""

from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from roboscript.contracts.ids import SessionId
from roboscript.gateway.config import GatewayConfig
from roboscript.gateway.router import (
    DefaultPolicy,
    Router,
)
from roboscript.testing.gateway import FakeClock

S1, S2, S3 = SessionId("s1"), SessionId("s2"), SessionId("s3")


def make_config(
    tmp_path: Path,
    speaks: dict[str, list[str]],
    caps: dict[str, int] | None = None,
    *,
    floors: dict[str, int] | None = None,
    pools: dict[str, str] | None = None,
    **changes: Any,
) -> GatewayConfig:
    """Model `m` on every endpoint of `speaks` (those starting with `f` in the fallback tier, with
    `caps` as their max_inflight and `floors` as their min_inflight, on the `pools` given), model
    `n` on the first one only, alias `brain`."""
    deployments = [
        {"endpoint": e, "upstream": "up"}
        | ({"tier": "fallback"} if e[0] == "f" else {})
        | ({"max_inflight": caps[e]} if caps and e in caps else {})
        | ({"min_inflight": floors[e]} if floors and e in floors else {})
        for e in speaks
    ]
    config = {
        "schema": "rs-gateway/v1",
        "listen": "127.0.0.1:0",
        "endpoints": {
            e: {"url_env": "U", "key_env": "K", "protocols": p}
            | ({"pool": pools[e]} if pools and e in pools else {})
            for e, p in speaks.items()
        },
        "models": {"m": {"deployments": deployments}, "n": {"deployments": deployments[:1]}},
        "aliases": {"brain": "m"},
        "cooldown_s": 30,
        "retry_after_cap_s": 120,
        "state_file": str(tmp_path / "state.json"),
    }
    return GatewayConfig.model_validate(config | changes)


def flat(tmp_path: Path, **changes: Any) -> GatewayConfig:
    """All primary: e1 (responses only), e2, e3."""
    both = ["responses", "chat"]
    return make_config(tmp_path, {"e1": ["responses"], "e2": both, "e3": both}, **changes)


def tiered(tmp_path: Path) -> GatewayConfig:
    """Primary p1 (responses only) and p2, fallback f1 and f2."""
    both = ["responses", "chat"]
    return make_config(tmp_path, {"p1": ["responses"], "p2": both, "f1": both, "f2": both})


async def take(
    router: Router, session: SessionId | None, name: str = "m", wire: Any = "responses"
) -> str:
    """Where the request's first attempt goes (its slot is freed again at once)."""
    route = router.route(session, name, wire)
    assert route is not None
    lease = await router.acquire(route)
    assert lease is not None
    lease.release()
    return lease.deployment.endpoint


def cooling(router: Router, model: str = "m") -> dict[str, bool]:
    deployments = router.snapshot()["models"][model]["deployments"]
    return {endpoint: d["cooling"] for endpoint, d in deployments.items()}


async def test_sessions_stick_to_where_they_succeeded(tmp_path: Path) -> None:
    router = Router(flat(tmp_path), DefaultPolicy(), clock=FakeClock())
    first = router.route(S1, "m", "responses")
    assert first is not None and first.pinned is None and first.model == "m"
    assert await take(router, S1) == "e1"
    router.succeeded(S1, "m", "m", "e2")
    assert await take(router, None) == "e2"  # the last good one of those not taken yet
    router.succeeded(S2, "m", "m", "e3")  # the last good moves on ...
    route = router.route(S1, "m", "responses")
    assert route is not None and route.pinned == "e2"
    assert await take(router, S1) == "e2"  # ... but s1 stays where it was
    assert await take(router, S2) == "e3"
    await router.state.flush()


def test_aliases_resolve_and_unknown_names_do_not(tmp_path: Path) -> None:
    router = Router(flat(tmp_path), DefaultPolicy())
    route = router.route(S1, "brain", "responses")
    assert route is not None and route.model == "m"
    assert router.route(S1, "gpt-4o", "responses") is None
    assert router.route(None, "gpt-4o", "chat") is None


async def test_failures_cool_a_deployment_down(tmp_path: Path) -> None:
    clock = FakeClock()
    router = Router(flat(tmp_path), DefaultPolicy(), clock=clock)
    router.succeeded(S1, "m", "m", "e1")
    router.failed("m", "e1", None)  # cooldown_s: 30
    route = router.route(S1, "m", "responses")
    assert route is not None and route.pinned == "e1"
    lease = await router.acquire(route)
    assert lease is not None and lease.deployment.endpoint != "e1" and lease.move == "cooling"
    lease.release()
    assert await take(router, None, "n") == "e1"  # cooldowns are per model
    clock.advance(29.5)
    assert await take(router, S1) != "e1"
    clock.advance(0.5)
    assert await take(router, S1) == "e1"
    router.failed("m", "e2", retry_after=7)  # Retry-After wins ...
    router.failed("m", "e3", retry_after=1e6)  # ... up to retry_after_cap_s: 120
    assert router.wait_s("m", "responses") == 0  # e1 is ready
    router.failed("m", "e1", retry_after=0.5)
    # every deployment cools down: the one ready first is tried all the same ...
    assert await take(router, None) == "e1"
    # ... by an attempt after a failure too (e1 and e3 have not failed on this request) ...
    route = router.route(None, "m", "responses")
    assert route is not None
    lease = await router.acquire(route, ["e2"])
    assert lease is not None and lease.deployment.endpoint == "e1"
    lease.release()
    # ... until every deployment has failed on it
    assert await router.acquire(route, ["e1", "e2", "e3"]) is None
    assert router.wait_s("m", "responses") == pytest.approx(0.5)
    assert router.wait_s("m", "chat") == pytest.approx(7)  # e1 does not speak chat
    clock.advance(0.5)
    assert cooling(router) == {"e1": False, "e2": True, "e3": True}
    clock.advance(6.5)
    assert cooling(router) == {"e1": False, "e2": False, "e3": True}
    clock.advance(112.5)
    assert cooling(router)["e3"]
    clock.advance(0.5)
    assert not cooling(router)["e3"]
    await router.state.flush()


async def test_fallback_deployments_serve_only_when_no_primary_can(tmp_path: Path) -> None:
    clock = FakeClock()
    router = Router(tiered(tmp_path), DefaultPolicy(), clock=clock)
    assert await take(router, S1) == "p1"
    assert await take(router, None, wire="chat") == "p2"  # p1 speaks no chat
    router.failed("m", "p1", None)
    router.failed("m", "p2", None)
    route = router.route(S1, "m", "responses")
    assert route is not None
    lease = await router.acquire(route)
    assert lease is not None and lease.deployment.endpoint in ("f1", "f2")
    assert not lease.overflow  # the primaries cool down: a failover, not an overflow
    lease.release()
    router.succeeded(S1, "m", "m", "f2")
    assert await take(router, S1) == "f2"  # the pin holds in its tier
    clock.advance(30)  # the primaries are back: the session leaves its fallback pin
    route = router.route(S1, "m", "responses")
    assert route is not None and route.pinned == "f2"
    lease = await router.acquire(route)
    assert lease is not None and lease.deployment.endpoint in ("p1", "p2")
    assert lease.move == "tier"
    lease.release()
    await router.state.flush()


def _installed() -> bool:
    try:
        metadata.distribution("roboscript")
    except metadata.PackageNotFoundError:
        return False
    return True
