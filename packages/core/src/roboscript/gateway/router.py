"""Deployment choice (docs/concepts/agent-runtime.md).

Each attempt holds one of its deployment's `max_inflight` slots until its answer is through (a
`Lease`). An attempt takes a free slot in the best tier that has one, among the deployments that
speak its protocol (and, for a request whose tool outputs carry images, take them), are not
cooling down and have not failed on the request; the policy orders
that tier's deployments (by default: least loaded, then in turn, the model's last good one first
among those not taken yet). Serving from a lower tier because the better one's ready deployments
are all full is an overflow. When every ready deployment is full, the request queues, first come
first served, for up to `queue_timeout_s` in all (`QueueTimeout`). A session keeps the deployment
that answered it last (its pin), waiting up to `pin_wait_s` from the request's start for a slot
there; it moves when that wait runs out, when its pin cools down or failed on the request, or
when its pin is a fallback and a primary has a free slot. When every deployment a request has not
tried is cooling down, its attempt (the first or one after a failure) takes the one whose cooldown
ends first (on a tie a primary, then the last good one) if it has a free slot; queued requests
wait for a cooldown to end. A failure cools a deployment down; a request runs out of deployments
only once every one has failed on it. Pins live in memory only (the latest `max_pins` sessions);
each model's last good deployment survives restarts in `state_file`, written at start and, off the
requests' path, whenever it changes.

Endpoints limit concurrency, often per account and below what the config says, so the cap in
force adapts (AIMD), leaning towards keeping it large. With `aimd_cut: failure_share` (the
default) it is halved, never below `min_inflight`, only when at least `aimd_cut_failure_share` of
the cap key's attempts in the last `aimd_cut_window_s` failed (any failure: a refusal, a timeout,
a cut stream, a 5xx ...), counted once there are `aimd_cut_min_calls` of them; the window then
starts afresh, and attempts that were in flight at that cut do not count toward the next. A lone
failure only cools its deployment down and counts toward health. With `aimd_cut: signal` every
overload signal (429, 503, 529, a concurrency or overload code), timeout or broken stream
(`congestion`) halves it, once per burst. `aimd_increase_after`
answers in a row add one, and each `aimd_recover_s` without a cut or an overload refusal
multiplies the cap by `aimd_recover_factor` (at least one more; a deployment that gets few calls,
such as a fallback, recovers too), never above `max_inflight`. Deployments on endpoints of one
`pool` share one cap (between their largest `min_inflight` and the sum of their `max_inflight`) and
one count of calls in flight. Health: once
`health_min_calls` attempts in the last `stats_window_s` include a `health_failure_rate` share
of failures, a deployment leaves rotation (open) for twice `cooldown_s`, doubled again on each
open in a row, up to `health_cooldown_max_s`; then it takes a single call (probing), whose
answer re-admits it (closed, the window forgotten) and whose failure opens it again. Cap and
health changes go to `on_change` (the gateway log)."""

import asyncio
import itertools
import json
import logging
import math
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import anyio

from roboscript.contracts.events import EventKind
from roboscript.contracts.ids import SessionId
from roboscript.gateway.config import TIERS, Deployment, GatewayConfig, Tier, Wire
from roboscript.gateway.stats import Stats
from roboscript.kernel import entrypoints
from roboscript.records import write_atomic

log = logging.getLogger(__name__)

POLICIES: Final = "roboscript.gateway_policies"
_STATE: Final = "rs-gateway-state/1"

Health = Literal["closed", "open", "probing"]
_OVERLOAD_STATUS: Final = frozenset({429, 503, 529})
_OVERLOAD_WORDS: Final = ("concurrency", "overload", "rate_limit", "service_unavailable")
_SLOW_OUTCOMES: Final = frozenset({"timeout"})
_CUT_OUTCOMES: Final = frozenset(
    {
        "stream_error",  # an `error` event from the upstream
        "no_final_event",  # a stream that ended early
        "upstream_disconnected",  # a stream that broke off after it had begun
        "RemoteProtocolError",
        "ReadError",
        "WriteError",
    }
)
_RECHECK_S: Final = 0.05  # a recovery timer that fired a hair early looks again this soon

Move = Literal["full", "cooling", "failed", "tier", "protocol", "images", "hedge"]
"""Why a session left its pin: its wait for a slot ran out, it cools down, it failed on the
request, a primary had a free slot while it is a fallback, it does not speak the protocol, it
does not take the images in the request's tool outputs (`tool_output_images: false`), or a
hedge on another deployment answered first (`Router.try_acquire`)."""


class RoutingPolicy(Protocol):
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
        """The endpoints of `ready` in the order of preference; the router takes the first.
        `ready` lists, in config order, the endpoints of `model`'s deployments in `tier` that
        speak the request's protocol, are not cooling down, have a free slot and have not failed
        on the request; `load` is each one's share of its slots in use (0 <= load < 1);
        `last_good` is the model's last good endpoint, possibly not in `ready`; `pinned` is the
        session's, never in `ready` (the router takes a free pin without asking)."""
        ...


class DefaultPolicy:
    """The least loaded endpoint; among equally loaded ones, in turn: the one it took least
    recently, those it has not taken yet first, and among those the last good one, then config
    order. Taking turns before the last good one spreads light sequential traffic over the
    accounts instead of filling one account's rate window."""

    def __init__(self) -> None:
        self._taken: dict[tuple[str, Tier], dict[str, int]] = {}  # endpoint -> when, per tier
        self._clock = itertools.count()

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
        taken = self._taken.setdefault((model, tier), {})
        order = sorted(ready, key=lambda e: (load.get(e, 0.0), taken.get(e, -1), e != last_good))
        if order:
            taken[order[0]] = next(self._clock)  # the router takes the first
        return order


def load_policy(name: str) -> RoutingPolicy:
    """A new instance of the policy registered as `name` in `roboscript.gateway_policies`."""
    factory: Callable[[], RoutingPolicy] = entrypoints.load(POLICIES, name)
    return factory()


@dataclass(frozen=True)
class Route:
    model: str
    deployments: list[Deployment]  # those that serve the request, in config order
    pinned: str | None  # the session's endpoint before this request
    wire: Wire = "responses"
    tool_images: bool = False  # the request's tool outputs carry images


@dataclass(eq=False)
class Lease:
    """An attempt's slot on `deployment`; `release()` frees it (again: nothing). `queued_s`: how
    long the request waited for it; `overflow`: a better tier's ready deployments were all
    full; `move`: why the session left its pin, if it did."""

    deployment: Deployment
    queued_s: float
    overflow: bool
    move: Move | None
    _free: Callable[[], None]
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self._free()


class QueueTimeout(Exception):
    """The request's `queue_timeout_s` ran out; `waited_s`: in this last wait."""

    def __init__(self, waited_s: float) -> None:
        super().__init__(f"no free slot for {waited_s:.1f} s")
        self.waited_s = waited_s


@dataclass(frozen=True)
class _Pin:
    model: str  # an alias resolves once per session
    endpoint: str | None = None  # set by the session's first success


@dataclass(frozen=True)
class _Take:
    deployment: Deployment
    overflow: bool = False
    move: Move | None = None


@dataclass(frozen=True)
class _Wait:
    until: float  # time.monotonic() when this wait runs out
    pin: bool  # for a slot on the session's pin; else for any slot (queued)


@dataclass(eq=False)
class _Waiter:
    """A request looking for a slot for its next attempt: what it tried, since when, its wait
    budgets (`queue_left_s` of `queue_timeout_s`), whether it queued and for what, and the slot
    once granted."""

    route: Route
    tried: frozenset[str]
    started: float
    pin_until: float
    queue_left_s: float
    queue_until: float | None = None
    queued: bool = False
    on_pin: bool = False
    granted: "asyncio.Future[Lease | None]" = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


@dataclass
class _Health:
    state: Health = "closed"
    opens: int = 0  # opened in a row, without an answer in between


def overload(outcome: int | str) -> bool:
    """Whether a failed attempt's outcome (a status, or `response_failed:<code>` and the like)
    says the upstream has more calls than it takes: it restarts the cap's quiet interval (and
    with `aimd_cut: signal` halves the cap)."""
    if isinstance(outcome, int):
        return outcome in _OVERLOAD_STATUS
    return any(word in outcome for word in _OVERLOAD_WORDS)


def congestion(outcome: int | str) -> str | None:
    """Whether a failed attempt's outcome says its deployment has more calls than it carries, and
    how it says so, for the cap event's `reason`: `overload` (a refusal, see `overload`),
    `timeout` (nothing came within the first-byte timeout) or `cut` (a stream that broke off,
    stalled, ended early or ended in an error event); the last two move the cap only with
    `aimd_cut: signal`. None for any other failure (a connection that was refused, a 401, a
    5xx other than 503 and 529, ...): the cap stays."""
    if overload(outcome):
        return "overload"
    if outcome in _SLOW_OUTCOMES:
        return "timeout"
    return "cut" if outcome in _CUT_OUTCOMES else None


class Router:
    """Pins per (session, requested name); per deployment its slots in use and its cooldown;
    last good per model; the requests waiting for a slot; `stats` for `/admin/stats`.

    `clock` measures cooldowns and the caps' quiet intervals (monotonic seconds); waits for a
    slot run on real time. `on_change(kind, data)` hears of cap and health changes
    (`gateway.cap`, `gateway.health`).
    """

    def __init__(
        self,
        config: GatewayConfig,
        policy: RoutingPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_change: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self._on_change = on_change
        self.policy = policy
        self.state = StateFile(config.state_file)
        self.stats = Stats(config.stats_window_s)
        self._clock = clock
        self._pins: OrderedDict[tuple[SessionId, str], _Pin] = OrderedDict()
        self._cooling: dict[tuple[str, str], float] = {}  # (model, endpoint) -> until
        self._inflight: Counter[tuple[str, str]] = Counter()  # (model, endpoint) -> slots used
        self._waiters: dict[str, deque[_Waiter]] = {}  # per model, in order of arrival
        # Adaptive caps, per pool or else per (model, endpoint): the most, the floor, in force,
        # in flight.
        self._cap_max: Counter[str] = Counter()
        self._cap_min: Counter[str] = Counter()
        self._models_of: dict[str, set[str]] = {}  # cap key -> the models whose calls count
        for model, spec in config.models.items():
            for d in spec.deployments:
                key = self._cap_key(model, d.endpoint)
                self._cap_max[key] += d.max_inflight
                self._cap_min[key] = max(self._cap_min[key], d.min_inflight)
                self._models_of.setdefault(key, set()).add(model)
        self._cap = dict(self._cap_max)
        self._used: Counter[str] = Counter()
        self._streak: Counter[str] = Counter()  # answers since the cap last changed
        self._cut_at: dict[str, float] = {}  # time.monotonic() of the last halving
        # aimd_cut failure_share: per cap key, (clock(), failed) of its recent attempts, failures
        self._window: dict[str, deque[tuple[float, bool]]] = {}
        self._window_failed: Counter[str] = Counter()
        self._quiet_since: dict[str, float] = {}  # clock(): last overload signal or quiet raise
        self._signaled: dict[str, tuple[str, str]] = {}  # (model, endpoint) of the last signal
        self._armed: set[str] = set()  # caps whose recovery timer is set
        self._health: dict[tuple[str, str], _Health] = {}
        self._last_good = self.state.load(config)
        self.state.save(self._last_good)  # an unwritable state file fails now, not on a success

    def route(
        self, session: SessionId | None, name: str, wire: Wire, *, tool_images: bool = False
    ) -> Route | None:
        """Where a request for `name` (a model or an alias) may go; None if it is neither.
        `tool_images`: its tool outputs carry images, which leaves out the deployments on
        endpoints with `tool_output_images: false`."""
        key = None if session is None else (session, name)
        pin = None if key is None else self._pins.get(key)
        model = self.config.resolve(name) if pin is None else pin.model
        if model is None:
            return None
        if key is not None:
            self._remember(key, _Pin(model) if pin is None else pin)
        deployments = [
            d
            for d in self._speaking(model, wire)
            if not tool_images or self.config.endpoints[d.endpoint].tool_output_images
        ]
        pinned = None if pin is None else pin.endpoint
        return Route(model, deployments, pinned, wire, tool_images)

    def forced(self, name: str, wire: Wire, endpoint: str) -> Route | None:
        """A route to `name`'s deployment on `endpoint` alone, whatever its capabilities, for
        an endpoint probe (the admin-only `x-rs-deployment` header): no session, no pin, no
        failover. None if `name` is no model or alias; no deployments if the model has none on
        `endpoint` that speaks `wire`."""
        model = self.config.resolve(name)
        if model is None:
            return None
        deployments = [d for d in self._speaking(model, wire) if d.endpoint == endpoint]
        return Route(model, deployments, None, wire)

    async def acquire(
        self,
        route: Route,
        tried: Collection[str] = (),
        *,
        since: float | None = None,
        queued_s: float = 0.0,
    ) -> Lease | None:
        """A slot for the request's next attempt, avoiding the deployments it `tried`: at once,
        or once one comes free for its pin or its place in the queue; None when no deployment
        is left to try. `since`: when the request began (its pin wait counts from then, default
        now); `queued_s`: how long it has queued already. Raises `QueueTimeout`."""
        now = time.monotonic()
        pin_until = (now if since is None else since) + self.config.pin_wait_s
        queue_left = self.config.queue_timeout_s - queued_s
        waiter = _Waiter(route, frozenset(tried), now, pin_until, queue_left)
        choice = self._choose(waiter)
        if not isinstance(choice, _Wait):
            return self._grant(waiter, choice)
        waiter.queued = True
        queue = self._waiters.setdefault(route.model, deque())
        queue.append(waiter)
        try:
            while not waiter.granted.done():
                waiter.on_pin = choice.pin
                now = time.monotonic()
                if not choice.pin and now >= choice.until:
                    raise QueueTimeout(now - waiter.started)
                await asyncio.wait({waiter.granted}, timeout=max(choice.until - now, 0.0))
                if waiter.granted.done():
                    break
                self._dispatch(route.model)  # a wait ran out: slots still go in order
                if not waiter.granted.done():
                    again = self._choose(waiter)
                    if isinstance(again, _Wait):
                        choice = again
                    else:
                        waiter.granted.set_result(self._grant(waiter, again))
            return waiter.granted.result()
        except BaseException:
            granted = waiter.granted
            if granted.done() and granted.exception() is None and granted.result() is not None:
                granted.result().release()  # granted as the request gave up or was cancelled
            raise
        finally:
            queue.remove(waiter)

    def try_acquire(self, route: Route, tried: Collection[str]) -> Lease | None:
        """A slot at once for a hedge of the request (`GatewayConfig.hedge_quantile`): on a ready
        deployment it has not `tried`, with a free slot under its cap; None when there is none or
        when calls queue for the model, whose slots a hedge never takes. Never a cooling one."""
        if any(not w.granted.done() for w in self._waiters.get(route.model, ())):
            return None
        now = time.monotonic()
        # `queued` keeps `_choose` off cooling deployments and the pin wait is over at once
        waiter = _Waiter(route, frozenset(tried), now, now, 0.0, queued=True)
        choice = self._choose(waiter)
        if not isinstance(choice, _Take):
            return None
        waiter.queued = False  # not a wait for the stats
        move: Move | None = "hedge" if choice.deployment.endpoint != route.pinned else None
        return self._grant(waiter, _Take(choice.deployment, choice.overflow, move))

    def hedge_after_s(self, model: str) -> float:
        """How long an aggregated attempt of `model` runs before it is hedged: the configured
        quantile of its recent answered attempts, at least `hedge_min_s`."""
        quantile = self.config.hedge_quantile
        assert quantile is not None
        recent = self.stats.latency_quantile(model, quantile * 100)
        return max(self.config.hedge_min_s, recent or 0.0)

    def failed(
        self,
        model: str,
        endpoint: str,
        retry_after: float | None,
        *,
        overloaded: bool = False,
        reason: str = "overload",
        began: float | None = None,
    ) -> None:
        """Cool a deployment down for `cooldown_s`, or for the upstream's Retry-After, capped;
        sessions waiting for it move on, and waiting requests look again when it is over. An
        `overloaded` attempt (a refusal, a timeout, a cut stream: see `congestion`; `reason` goes
        into the cap event) is a signal to the cap (see `_signal`); the failure counts toward
        health and toward the cap's failure share (see `_tally`)."""
        seconds = (
            self.config.cooldown_s
            if retry_after is None
            else min(retry_after, self.config.retry_after_cap_s)
        )
        self._cool(model, endpoint, seconds)
        began = time.monotonic() if began is None else began
        if overloaded:
            self._signal(model, endpoint, began, reason)
        self.outcome(model, endpoint, failed=True, began=began)
        self._dispatch(model)

    def congested(self, model: str, endpoint: str, reason: str, *, began: float) -> None:
        """A signal that `endpoint` carries too many calls from an attempt `failed` does not
        see: a stream that broke off after it had begun. A signal to the cap as in `failed` (its
        failure itself goes to `outcome`)."""
        self._signal(model, endpoint, began, reason)

    def outcome(
        self, model: str, endpoint: str, *, failed: bool, began: float | None = None
    ) -> None:
        """An attempt's outcome, for health and the cap: an answer (a streamed one once it is
        through), or a failure (`failed` records its own; this for a stream that broke off).
        `began`: when the attempt began (time.monotonic()), for the cap's failure share."""
        self.stats.attempted(model, endpoint, failed=failed)
        if self.config.aimd_cut == "failure_share":
            self._tally(model, endpoint, failed, began)
        health = self._health.setdefault((model, endpoint), _Health())
        if not failed:
            if health.state != "closed":
                health.state, health.opens = "closed", 0
                self.stats.forget_outcomes(model, endpoint)
                self._changed(EventKind.GATEWAY_HEALTH, model, endpoint, health="closed")
            self._raise(model, endpoint)
            return
        calls, rate = self.stats.failure_rate(model, endpoint)
        if health.state == "probing" or (
            health.state == "closed"
            and calls >= self.config.health_min_calls
            and rate >= self.config.health_failure_rate
        ):
            health.state, health.opens = "open", health.opens + 1
            seconds = min(
                self.config.cooldown_s * 2**health.opens, self.config.health_cooldown_max_s
            )
            self._cool(model, endpoint, seconds)
            log.warning(
                "%s on %s: out of rotation for %.0f s (%d of %d attempts failed)",
                model,
                endpoint,
                seconds,
                round(rate * calls),
                calls,
            )
            data = {"cooldown_s": seconds, "attempts": calls, "failure_rate": round(rate, 3)}
            self._changed(EventKind.GATEWAY_HEALTH, model, endpoint, health="open", **data)

    def succeeded(self, session: SessionId | None, name: str, model: str, endpoint: str) -> None:
        """Pin the session to `endpoint`, which becomes the model's last good deployment."""
        if session is not None:
            self._remember((session, name), _Pin(model, endpoint))
        if self._last_good.get(model) != endpoint:
            self._last_good[model] = endpoint
            self.state.save_soon(lambda: dict(self._last_good))

    def wait_s(self, model: str, wire: Wire) -> float:
        """Seconds until the first of `model`'s deployments that speak `wire` is ready again."""
        waits = [self._wait(model, d.endpoint) for d in self._speaking(model, wire)]
        return max(0.0, min(waits, default=0.0))

    def retry_after_s(self, route: Route) -> int:
        """When to try again after a queue timeout: about when a slot comes free, i.e. the
        shortest recent median call of the route's deployments (at least 1 s)."""
        medians = [self.stats.median_s(route.model, d.endpoint) for d in route.deployments]
        return max(1, math.ceil(min((m for m in medians if m is not None), default=1.0)))

    def snapshot(self) -> dict[str, Any]:
        """Per model and deployment: slots in use and cap, cooldown, waiting requests and the
        stats (`/admin/stats`)."""
        models = {}
        for name, spec in self.config.models.items():
            queue = self._waiters.get(name, ())
            deployments = {}
            for d in spec.deployments:
                cooling = max(0.0, self._wait(name, d.endpoint))
                key = self._cap_key(name, d.endpoint)
                deployments[d.endpoint] = {
                    "tier": d.tier,
                    "inflight": self._inflight[(name, d.endpoint)],
                    "max_inflight": d.max_inflight,
                    "min_inflight": d.min_inflight,
                    "pool": self.config.endpoints[d.endpoint].pool,
                    "cap": self._cap[key],  # in force, for the pool if there is one
                    "cap_floor": self._cap_min[key],
                    "cap_inflight": self._used[key],
                    "health": self._health_state(name, d.endpoint),
                    "cooling": cooling > 0,
                    "cooling_s": round(cooling, 1),
                    "waiting_for_pin": sum(
                        w.on_pin and w.route.pinned == d.endpoint for w in queue
                    ),
                    **self.stats.deployment(name, d.endpoint),
                }
            models[name] = {
                "waiting": len(queue),
                "last_good": self._last_good.get(name),
                **self.stats.model(name),
                "deployments": deployments,
            }
        return {"window_s": self.config.stats_window_s, "models": models}

    def _choose(self, waiter: _Waiter) -> _Take | _Wait | None:
        """Where the request's next attempt goes now, or why it must wait; None: nowhere."""
        route, model, now = waiter.route, waiter.route.model, time.monotonic()
        left = [d for d in route.deployments if d.endpoint not in waiter.tried]
        if not left:
            return None
        ready = [d for d in left if self._wait(model, d.endpoint) <= 0]
        free = [d for d in ready if self._has_slot(model, d)]
        move: Move | None = None
        if route.pinned is not None:
            pin = next((d for d in route.deployments if d.endpoint == route.pinned), None)
            if pin is None:
                speaks = route.wire in self.config.endpoints[route.pinned].protocols
                move = "images" if route.tool_images and speaks else "protocol"
            elif pin.endpoint in waiter.tried:
                move = "failed"
            elif pin not in ready:
                move = "cooling"
            elif any(_rank(d) < _rank(pin) for d in free):
                move = "tier"
            elif pin in free:  # a fallback pin while the primaries are full: an overflow too
                return _Take(pin, any(_rank(d) < _rank(pin) for d in ready))
            elif now < waiter.pin_until:
                return _Wait(waiter.pin_until, pin=True)
            else:
                move = "full"
        if free:
            best = min(map(_rank, free))
            pool = {d.endpoint: d for d in free if _rank(d) == best}
            load = {e: self._load(model, d) for e, d in pool.items()}
            first = self.policy.order(
                model,
                TIERS[best],
                list(pool),
                pinned=route.pinned,
                last_good=self._last_good.get(model),
                load=load,
            )[0]
            return _Take(pool[first], any(_rank(d) < best for d in ready), move)
        if not ready and not waiter.queued:  # all left cool down: an attempt tries the one
            last_good = self._last_good.get(model)  # ready first (on a tie a primary, last good)

            def soonest(d: Deployment) -> tuple[float, int, bool]:
                return self._wait(model, d.endpoint), _rank(d), d.endpoint != last_good

            first = min(left, key=soonest)
            if self._has_slot(model, first):
                return _Take(first, move=move)
        if waiter.queue_until is None:
            waiter.queue_until = now + waiter.queue_left_s
        return _Wait(waiter.queue_until, pin=False)

    def _grant(self, waiter: _Waiter, choice: _Take | None) -> Lease | None:
        if choice is None:
            return None
        key = (waiter.route.model, choice.deployment.endpoint)
        self._inflight[key] += 1
        self._used[self._cap_key(*key)] += 1
        if choice.overflow:
            self.stats.count(key[0], "overflow")
        queued = 0.0
        if waiter.queued:
            queued = time.monotonic() - waiter.started
            self.stats.waited(*key, queued)
        free = partial(self._free, *key)
        return Lease(choice.deployment, round(queued, 3), choice.overflow, choice.move, free)

    def _free(self, model: str, endpoint: str) -> None:
        self._inflight[(model, endpoint)] -= 1
        key = self._cap_key(model, endpoint)
        self._used[key] -= 1
        self._dispatch_all(key)

    def _dispatch_all(self, key: str) -> None:
        """Dispatch every model whose calls count toward the cap `key` (a pool's: several)."""
        for model in sorted(self._models_of[key]):
            self._dispatch(model)

    def _dispatch(self, model: str) -> None:
        """Hand free slots to the requests waiting for `model`, in their order of arrival. A
        policy that fails while choosing for a request fails that request, not the caller."""
        for waiter in list(self._waiters.get(model, ())):
            if waiter.granted.done():
                continue
            try:
                choice = self._choose(waiter)
            except Exception as error:
                waiter.granted.set_exception(error)
                continue
            if not isinstance(choice, _Wait):
                waiter.granted.set_result(self._grant(waiter, choice))

    def _speaking(self, model: str, wire: Wire) -> list[Deployment]:
        endpoints = self.config.endpoints
        return [
            d
            for d in self.config.models[model].deployments
            if wire in endpoints[d.endpoint].protocols
        ]

    def _cap_key(self, model: str, endpoint: str) -> str:
        pool = self.config.endpoints[endpoint].pool
        return f"pool:{pool}" if pool is not None else f"{model}@{endpoint}"

    def _has_slot(self, model: str, deployment: Deployment) -> bool:
        """Below its `max_inflight` (a probing deployment: nothing in flight) and its cap."""
        most = 1 if self._health_state(model, deployment.endpoint) == "probing" else None
        if self._inflight[(model, deployment.endpoint)] >= (most or deployment.max_inflight):
            return False
        key = self._cap_key(model, deployment.endpoint)
        return self._used[key] < self._cap[key]

    def _load(self, model: str, deployment: Deployment) -> float:
        key = self._cap_key(model, deployment.endpoint)
        own = self._inflight[(model, deployment.endpoint)] / deployment.max_inflight
        return max(own, self._used[key] / self._cap[key])

    def _health_state(self, model: str, endpoint: str) -> Health:
        """Its health; an open deployment whose time out is over is probing from now on."""
        health = self._health.get((model, endpoint))
        if health is None:
            return "closed"
        if health.state == "open" and self._wait(model, endpoint) <= 0:
            health.state = "probing"
            self._changed(EventKind.GATEWAY_HEALTH, model, endpoint, health="probing")
        return health.state

    def _cool(self, model: str, endpoint: str, seconds: float) -> None:
        until = self._clock() + seconds
        health = self._health.get((model, endpoint))
        if health is not None and health.state == "open":  # a late failure keeps its time out
            until = max(until, self._cooling.get((model, endpoint), until))
        self._cooling[(model, endpoint)] = until
        asyncio.get_running_loop().call_later(seconds, self._dispatch, model)

    def _signal(self, model: str, endpoint: str, began: float, reason: str = "overload") -> None:
        """A signal that the cap key carries too many calls, from an attempt that `began`
        (time.monotonic()). With `aimd_cut: signal`, each one (a refusal, a timeout, a cut
        stream) halves the cap, once for all the attempts that began before the last halving
        (those in flight then signal the same overload), and restarts its quiet interval. With
        `aimd_cut: failure_share`, an overload refusal only restarts the quiet interval (the
        failure share decides cuts, see `_tally`)."""
        key = self._cap_key(model, endpoint)
        if self.config.aimd_cut == "failure_share":
            if reason == "overload":
                self._quiet_since[key], self._signaled[key] = self._clock(), (model, endpoint)
                self._arm(key)
            return
        self._quiet_since[key], self._signaled[key] = self._clock(), (model, endpoint)
        if began >= self._cut_at.get(key, -math.inf):  # else in flight at the last cut: the same
            self._halve(model, endpoint, reason)
        self._arm(key)

    def _tally(self, model: str, endpoint: str, failed: bool, began: float | None) -> None:
        """Count an attempt into its cap key's window of the last `aimd_cut_window_s`; once it
        holds `aimd_cut_min_calls` attempts of which at least `aimd_cut_failure_share` failed, a
        failure halves the cap and the window starts afresh. Attempts in flight at the last cut
        (`began` before it) do not count: they tell of the load that cut already answered."""
        key = self._cap_key(model, endpoint)
        if began is not None and began < self._cut_at.get(key, -math.inf):
            return
        config, now = self.config, self._clock()
        window = self._window.setdefault(key, deque())
        window.append((now, failed))
        self._window_failed[key] += failed
        while window[0][0] < now - config.aimd_cut_window_s:
            self._window_failed[key] -= window.popleft()[1]
        calls, failures = len(window), self._window_failed[key]
        if not failed or calls < config.aimd_cut_min_calls:
            return
        if failures >= config.aimd_cut_failure_share * calls:
            window.clear()
            self._window_failed[key] = 0
            self._quiet_since[key], self._signaled[key] = now, (model, endpoint)
            self._halve(model, endpoint, "failure_share", f"{failures} of {calls} failed")
            self._arm(key)

    def _halve(self, model: str, endpoint: str, reason: str, detail: str = "") -> None:
        key = self._cap_key(model, endpoint)
        was = self._cap[key]
        # Half of what was in flight when it overloaded (the attempt that failed included),
        # which is at or below the cap in force; rounded up so that one call in flight keeps
        # its slot; never below the floor.
        half = math.ceil(min(was, self._used[key]) / 2)
        self._cap[key] = max(self._cap_min[key], half)
        self._cut_at[key], self._streak[key] = time.monotonic(), 0
        if self._cap[key] != was:
            cap = self._cap[key]
            why = f"{reason}: {detail}" if detail else reason
            log.warning("%s: cap %d -> %d (%s on %s)", key, was, cap, why, endpoint)
            self._changed(EventKind.GATEWAY_CAP, model, endpoint, cap=cap, was=was, reason=reason)

    def _raise(self, model: str, endpoint: str) -> None:
        key = self._cap_key(model, endpoint)
        if self._cap[key] >= self._cap_max[key]:
            return
        self._streak[key] += 1
        if self._streak[key] >= self.config.aimd_increase_after:
            was, self._streak[key] = self._cap[key], 0
            self._cap[key] = was + 1
            data = {"cap": self._cap[key], "was": was, "reason": "answers"}
            self._changed(EventKind.GATEWAY_CAP, model, endpoint, **data)
            self._dispatch_all(key)

    def _arm(self, key: str) -> None:
        """While the cap is below its max, look again when its quiet interval ends."""
        if key in self._armed or self._cap[key] >= self._cap_max[key]:
            return
        self._armed.add(key)
        due = self._quiet_since[key] + self.config.aimd_recover_s - self._clock()
        asyncio.get_running_loop().call_later(max(due, _RECHECK_S), self._recover, key)

    def _recover(self, key: str) -> None:
        """Multiply the cap by `aimd_recover_factor` (raise it by one at least) for each
        `aimd_recover_s` that has passed without an overload signal (up to its max), so that a
        deployment that gets few calls recovers too, not only one that answers
        `aimd_increase_after` in a row; the requests waiting get the slots."""
        self._armed.discard(key)
        was, interval = self._cap[key], self.config.aimd_recover_s
        steps = int((self._clock() - self._quiet_since[key]) // interval)
        if steps > 0 and was < self._cap_max[key]:
            cap, most, factor = was, self._cap_max[key], self.config.aimd_recover_factor
            for _ in range(steps):
                cap = min(most, max(cap + 1, math.ceil(cap * factor)))
                if cap >= most:
                    break
            self._cap[key] = cap
            self._quiet_since[key] += steps * interval
            self._streak[key] = 0
            data = {"cap": self._cap[key], "was": was, "reason": "quiet"}
            self._changed(EventKind.GATEWAY_CAP, *self._signaled[key], **data)
            self._dispatch_all(key)
        self._arm(key)

    def _changed(self, kind: str, model: str, endpoint: str, **data: Any) -> None:
        if self._on_change is not None:
            pool = self.config.endpoints[endpoint].pool
            self._on_change(kind, {"model": model, "endpoint": endpoint, "pool": pool, **data})

    def _wait(self, model: str, endpoint: str) -> float:
        until = self._cooling.get((model, endpoint))
        return 0.0 if until is None else until - self._clock()

    def _remember(self, key: tuple[SessionId, str], pin: _Pin) -> None:
        self._pins[key] = pin
        self._pins.move_to_end(key)
        if len(self._pins) > self.config.max_pins:
            self._pins.popitem(last=False)


def _rank(deployment: Deployment) -> int:
    return TIERS.index(deployment.tier)


class StateFile:
    """Each model's last good deployment on disk. `save_soon` writes from a worker thread, one
    write at a time and each time the latest state, so that no request waits for the disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._writes: set[asyncio.Task[None]] = set()

    def load(self, config: GatewayConfig) -> dict[str, str]:
        """The saved last good deployments that are still deployments of their model."""
        if not self.path.exists():
            return {}
        fix = f"{self.path}: delete it to start without last-good state"
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError:
            raise ValueError(f"{fix} (not JSON)") from None
        valid = isinstance(state, dict) and state.get("schema") == _STATE
        if not valid or not isinstance(state.get("last_good"), dict):
            raise ValueError(f"{fix} (not an {_STATE} file)")
        deployments = {
            name: {d.endpoint for d in m.deployments} for name, m in config.models.items()
        }
        return {m: e for m, e in state["last_good"].items() if e in deployments.get(m, ())}

    def save(self, last_good: dict[str, str]) -> None:
        text = json.dumps({"schema": _STATE, "last_good": last_good}, indent=2, sort_keys=True)
        write_atomic(self.path, (text + "\n").encode())

    def save_soon(self, latest: Callable[[], dict[str, str]]) -> None:
        task = asyncio.get_running_loop().create_task(self._save(latest))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def flush(self) -> None:
        """Wait for the writes under way."""
        await asyncio.gather(*self._writes)

    async def _save(self, latest: Callable[[], dict[str, str]]) -> None:
        async with self._lock:
            try:
                await anyio.to_thread.run_sync(self.save, latest())
            except OSError as error:  # routing goes on; a restart would miss this change
                log.error("cannot save the last-good state: %s", error)
