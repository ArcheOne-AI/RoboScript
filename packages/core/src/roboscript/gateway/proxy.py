"""Forwarding (docs/concepts/agent-runtime.md). A request tries its route's deployments in turn,
each attempt holding a slot of its deployment (`Router.acquire`) until its answer is through.

Responses calls are always streamed upstream: a streaming client gets the stream passed through,
a non-streaming one the response the stream ends with, read only up to its final event. Chat calls
go upstream as the client sent them. Until the client has received something, a connection error,
timeout, 401, 403, 429, 5xx, a stream that breaks off or ends early, an `error` event or a failed
response (`response.failed`, unless its code says the request itself is at fault, `_REQUEST`:
some upstreams report overload this way, `gateway_concurrency_limit` or `server_is_overloaded`
after a 200) cools the deployment down and the whole request is tried on the next one, primaries
first (one that cools down too, when all the rest do); an overload signal, a timeout or a stream
that breaks off or ends in an error also halves the cap in force (`Router.failed`, `congestion`).
Other 4xx and a failure of the request's own are passed through. An aggregated call has received
nothing before its final event; a streaming client nothing before the first event past the
stream's preamble (`response.created`, `response.in_progress`, `response.queued`, which are held
back meanwhile). The client gets 503 only once every deployment has failed on its request
(`no_deployment_available`) or none had a free slot within `queue_timeout_s` (`queue_timeout`).
Once a stream's final event is through, the call has succeeded, whatever follows.
Two behaviours serve `stream_only` endpoints: a string `input` is sent as its one user message,
and a final event whose `output` such an endpoint leaves empty is completed from the streamed
items. A request that names no reasoning effort gets its model's `default_reasoning_effort`, if it
has one; every request's line in the log says which effort went upstream and whether it was
filled in. A
Responses request whose tool outputs carry images skips endpoints with `tool_output_images:
false` (`Router.route`). An admin's `x-rs-deployment` sends a request to one deployment alone,
for endpoint probes (`Router.forced`). For a session a Hub registered, the answer's tool calls
are permitted before the client sees them (`roboscript.gateway.permits`). With `hedge_quantile`,
an aggregated Responses call whose attempt runs long gets a second attempt on another deployment
and takes the first answer (`Proxy._race`).
"""

import email.utils
import json
import logging
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, Final

import anyio
import httpx
from pydantic import SecretStr
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response, StreamingResponse

from roboscript.contracts.events import EventKind
from roboscript.contracts.ids import EpisodeId, SessionId
from roboscript.contracts.trace import TraceContext
from roboscript.gateway import sse
from roboscript.gateway.config import Deployment, Endpoint, GatewayConfig, Tier, Upstream, Wire
from roboscript.gateway.permits import (
    FAILED,
    ChatHold,
    PermitError,
    RequestPermits,
    ResponsesHold,
    SessionKeyError,
    Sessions,
)
from roboscript.gateway.router import Lease, QueueTimeout, Route, Router, congestion
from roboscript.records import EventWriter

logger = logging.getLogger(__name__)

PATHS: Final = {"responses": "/responses", "chat": "/chat/completions"}
_FINAL: Final = frozenset({"response.completed", "response.incomplete", "response.failed"})
_MARKS: Final = (b"response.output_item.done", *(kind.encode() for kind in _FINAL))
_AUTH: Final = frozenset({401, 403})
_PREAMBLE: Final = frozenset({"response.created", "response.in_progress", "response.queued"})
_REQUEST: Final = frozenset(
    {
        "invalid_prompt",
        "invalid_request",
        "invalid_image",
        "invalid_image_url",
        "content_filter",
        "content_policy_violation",
        "context_length_exceeded",
        "string_above_max_length",
        "empty_image_file",
        "unsupported_parameter",
    }
)
"""Error codes (or types, less an `_error` suffix) of a failed response that are the request's
own fault, which another deployment would fail the same way: passed through. Any other failed
response (overload, an interrupted stream, no code at all) is tried elsewhere."""
_BROKEN: Final = frozenset({"upstream_disconnected", "no_final_event", "upstream_response_failed"})
"""How an answer fails on the deployment's side once it has begun (not a failover: a stream
broken off or unfinished, a response that failed)."""


def upstream_client() -> httpx.AsyncClient:
    """The client for all upstream calls. Streams hold a connection for minutes, so there is no
    cap: waiting for a pooled connection must never pass for a slow upstream. Idle connections
    live a minute; proxy variables are ignored."""
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None, keepalive_expiry=60)
    return httpx.AsyncClient(limits=limits, trust_env=False)


def error_response(
    status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """An error in OpenAI's shape, so that OpenAI clients report its message."""
    error = {"message": message, "type": "rs_gateway_error", "code": code, "param": None}
    return JSONResponse({"error": error}, status, headers=headers)


class _Lost(Exception):
    """A hedged attempt whose answer came after the other one's: it leaves the call alone."""


class _Failed(Exception):
    """An attempt that failed while the client had received nothing: cool down, try the next."""

    def __init__(
        self, outcome: int | str, detail: str | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(outcome)
        self.outcome, self.detail, self.retry_after = outcome, detail, retry_after


@dataclass
class _Call:
    """One client request on its way through; it ends as one line of the request log."""

    wire: Wire
    session: SessionId | None
    requested: str | None
    stream: bool
    trace: TraceContext | None
    started: float = field(default_factory=time.monotonic)
    model: str | None = None
    deployment: str | None = None
    tier: Tier | None = None
    affinity_break: bool = False
    move: str | None = None  # why the session left its pin
    degradation: str | None = None  # "overflow": a better tier was full
    queued_s: float = 0.0  # waiting for slots, all attempts together
    attempts: list[dict[str, Any]] = field(default_factory=list)
    began: float | None = None  # when the attempt that answered began
    answered: float | None = None  # when its answer was through (the final event)
    first_byte_s: float | None = None
    status: int | None = None
    error: str | None = None
    usage: Any = None
    permits: RequestPermits | None = None  # the session is registered for permits
    request_fault: bool = False  # its failed response was the request's own fault
    forced: bool = False  # an admin sent it to one deployment (`x-rs-deployment`): no pin
    effort: str | None = None  # the reasoning effort sent upstream (None: none at all)
    effort_filled: bool = False  # the caller named none: the model's default went instead
    claimed: bool = False  # an attempt's answer is the call's (`_answered`): a hedge's is not
    racing: dict[str, float] = field(default_factory=dict)  # hedged attempts running: began


class Proxy:
    """Sends requests upstream on `http` and logs each one to `log` (never with a secret); the
    answers of `sessions` registered for permits are permitted before they go out."""

    def __init__(
        self,
        config: GatewayConfig,
        upstreams: dict[str, Upstream],
        router: Router,
        http: httpx.AsyncClient,
        log: EventWriter,
        sessions: Sessions | None = None,
    ) -> None:
        self.config = config
        self.upstreams = upstreams
        self.router = router
        self.http = http
        self.log = log
        self.sessions = sessions
        self._degraded: set[tuple[str, Wire]] = set()  # served by a fallback deployment last

    async def forward(
        self,
        wire: Wire,
        body: dict[str, Any],
        session: SessionId | None,
        key: SecretStr | None,
        trace: TraceContext | None,
        *,
        deployment: str | None = None,
    ) -> Response:
        """Answer a request whose body is a JSON object with a string `model`; `key` is the
        session's key, needed when a Hub registered the session for permits. `deployment`: the
        one endpoint to send it to (an admin's probe), whatever its capabilities."""
        name: str = body["model"]
        call = _Call(wire, session, name, body.get("stream") is True, trace)
        call.forced = deployment is not None
        if self.sessions is not None:
            try:
                call.permits = self.sessions.permits(session, key, wire, body)
            except SessionKeyError:
                message = "this session belongs to a Hub episode: send its key after the session id"
                return self._end(call, 403, "session_key_required", message)
        images = wire == "responses" and tool_output_images(body)
        if deployment is None:
            route = self.router.route(session, name, wire, tool_images=images)
        else:
            route = self.router.forced(name, wire, deployment)
        if route is None:
            message = f"no model or alias {name!r}; GET /v1/models lists them"
            return self._end(call, 404, "model_not_found", message)
        call.model = route.model
        body = self._with_effort(call, route.model, body)
        if not route.deployments:
            if deployment is not None:
                message = f"{route.model} has no deployment on {deployment!r} that speaks {wire}"
                return self._end(call, 400, "unknown_deployment", message)
            speaking = self.router.route(None, route.model, wire)
            if images and speaking is not None and speaking.deployments:
                message = f"no deployment of {route.model} takes images in tool outputs"
                return self._end(call, 400, "unsupported_tool_output_images", message)
            message = f"no deployment of {route.model} speaks {wire}"
            return self._end(call, 400, "unsupported_protocol", message)
        try:
            return await self._forward(call, route, body)
        except anyio.get_cancelled_exc_class():
            if call.status is None:  # the client left before its answer (app.py): its line
                call.status, call.error = 499, "client_disconnected"
                self._log(call)
            raise

    async def _forward(self, call: _Call, route: Route, body: dict[str, Any]) -> Response:
        """Attempts, each on a slot `Router.acquire` hands out, until one answers."""
        tried: list[str] = []
        while True:
            try:
                lease = await self.router.acquire(
                    route, tried, since=call.started, queued_s=call.queued_s
                )
            except QueueTimeout as full:
                call.queued_s += full.waited_s
                self.router.stats.count(route.model, "queue_timeout")
                message = f"every deployment of {route.model} stayed full for {call.queued_s:.0f} s"
                wait = {"retry-after": str(self.router.retry_after_s(route))}
                return self._end(call, 503, "queue_timeout", message, wait)
            if lease is None:
                break
            call.queued_s += lease.queued_s
            if self._hedged(call):
                answer = await self._race(call, route, lease, body, tried)
                if answer is not None:
                    return answer
                continue
            began = time.monotonic()
            try:
                return await self._attempt(call, route, lease, body, began)
            except _Failed as failed:
                self._failed(call, route, lease, failed, began)
                tried.append(lease.deployment.endpoint)
        attempts = "; ".join(f"{a['deployment']}: {a['outcome']}" for a in call.attempts)
        message = f"no deployment of {route.model} answered ({attempts})"
        wait = {"retry-after": str(math.ceil(self.router.wait_s(route.model, call.wire)))}
        return self._end(call, 503, "no_deployment_available", message, wait)

    def _failed(
        self, call: _Call, route: Route, lease: Lease, failed: _Failed, began: float
    ) -> None:
        """An attempt that failed before the client got anything: logged, counted, its
        deployment cooled down, its slot freed."""
        deployment = lease.deployment
        call.attempts.append(_record(deployment, failed.outcome, began, failed.detail))
        self.router.stats.error(route.model, deployment.endpoint, str(failed.outcome))
        signal = congestion(failed.outcome)
        self.router.failed(
            route.model,
            deployment.endpoint,
            failed.retry_after,
            overloaded=signal is not None,
            reason=signal or "overload",
            began=began,
        )
        lease.release()  # once it cools down, so that no waiting call takes the slot

    def _hedged(self, call: _Call) -> bool:
        """An aggregated Responses call, which nothing has reached yet until its final event, so
        that a second attempt may run beside the first (`hedge_quantile`)."""
        return (
            self.config.hedge_quantile is not None
            and call.wire == "responses"
            and not call.stream
            and not call.forced
        )

    async def _race(
        self, call: _Call, route: Route, lease: Lease, body: dict[str, Any], tried: list[str]
    ) -> Response | None:
        """The attempt on `lease`, and once it has run `Router.hedge_after_s` a second one on
        another deployment with a free slot (`Router.try_acquire`; none when calls queue or no
        slot is free): the first answer is the call's and the other attempt is cancelled (logged
        as `hedge_lost`, not a failure). A failed attempt is handled as without a hedge and its
        deployment added to `tried`; one that fails past the hedge's time while the other runs
        is hedged again at once; None when every attempt of the race failed (the caller tries
        the next deployment). A hedge moves the session as any move does: encrypted
        reasoning is stripped from its request (`_rewrite`), and its answer re-pins the session."""
        send, receive = anyio.create_memory_object_stream[
            tuple[Lease, float, "Response | _Failed | _Lost"]
        ](2)
        running: dict[str, Lease] = {}
        answer: Response | None = None

        async def run(lease: Lease, began: float) -> None:
            result: Response | _Failed | _Lost
            try:
                result = await self._attempt(call, route, lease, body, began)
            except (_Failed, _Lost) as error:
                result = error
            await send.send((lease, began, result))

        async with anyio.create_task_group() as group:

            def start(lease: Lease) -> None:
                began = time.monotonic()
                running[lease.deployment.endpoint] = lease
                call.racing[lease.deployment.endpoint] = began
                group.start_soon(run, lease, began)

            start(lease)
            hedge_at: float | None = time.monotonic() + self.router.hedge_after_s(route.model)
            while running:
                got = None
                wait = math.inf if hedge_at is None else max(0.0, hedge_at - time.monotonic())
                with anyio.move_on_after(wait):
                    got = await receive.receive()
                if got is None:  # the hedge's time
                    hedge_at = None
                    extra = self.router.try_acquire(route, [*tried, *running])
                    if extra is not None:
                        self.router.stats.count(route.model, "hedge")
                        start(extra)
                    continue
                done, began, result = got
                del running[done.deployment.endpoint]
                if not call.claimed:
                    del call.racing[done.deployment.endpoint]
                if call.claimed and not isinstance(result, Response):
                    continue  # the rival's answer is the call's; this one is logged as lost
                if isinstance(result, _Failed):
                    self._failed(call, route, done, result, began)
                    tried.append(done.deployment.endpoint)
                    if hedge_at is None and running:  # past the hedge's time: hedge again
                        hedge_at = time.monotonic()
                elif isinstance(result, Response):  # (a `_Lost` comes only after a claim)
                    answer = result
                    if done is not lease:
                        self.router.stats.count(route.model, "hedge_won")
                    break
            group.cancel_scope.cancel()  # the rest: lost (`_answered`), their slots freed
        call.racing.clear()
        return answer

    def _with_effort(self, call: _Call, model: str, body: dict[str, Any]) -> dict[str, Any]:
        """`body`, with `model`'s `default_reasoning_effort` when the caller names no effort;
        the effort that goes upstream is logged either way."""
        call.effort = reasoning_effort(call.wire, body)
        default = self.config.models[model].default_reasoning_effort
        if call.effort is not None or default is None or _names_effort(call.wire, body):
            return body
        call.effort, call.effort_filled = default, True
        if call.wire == "chat":
            return {**body, "reasoning_effort": default}
        reasoning = body.get("reasoning")
        kept = reasoning if isinstance(reasoning, dict) else {}  # e.g. a summary setting
        return {**body, "reasoning": {**kept, "effort": default}}

    def refuse(
        self, wire: Wire, status: int, code: str, message: str, session: SessionId | None = None
    ) -> Response:
        """Answer (and log) a request that is not forwarded at all."""
        return self._end(_Call(wire, session, None, False, None), status, code, message)

    async def _attempt(
        self,
        call: _Call,
        route: Route,
        lease: Lease,
        body: dict[str, Any],
        began: float,
    ) -> Response:
        """One deployment's answer; `_Failed` while the client has received nothing of it. The
        lease is released once the answer is through (a relayed stream releases it itself), or
        after `_Failed` by the caller."""
        held = False
        try:
            answer = await self._answer(call, route, lease, body, began)
            held = isinstance(answer, StreamingResponse)
            return answer
        except _Failed:
            held = True
            raise
        finally:
            if not held:
                lease.release()

    async def _answer(
        self,
        call: _Call,
        route: Route,
        lease: Lease,
        body: dict[str, Any],
        began: float,
    ) -> Response:
        deployment = lease.deployment
        upstream = self.upstreams[deployment.endpoint]
        endpoint = self.config.endpoints[deployment.endpoint]
        moved = route.pinned is not None and deployment.endpoint != route.pinned
        sent = _rewrite(body, deployment.upstream, endpoint, call.wire, moved)
        response, first, rest = await self._open(upstream, call.wire, sent)
        status = response.status_code
        if status in _AUTH or status == 429 or status >= 500:
            await response.aclose()
            outcome = f"upstream_auth_{status}" if status in _AUTH else status
            raise _Failed(outcome, _detail(first, upstream.key), _retry_after(response))
        first_byte_s = time.monotonic() - call.started
        # The queue wait, which the client may leave out of its own time budgets.
        headers = {"x-rs-deployment": deployment.endpoint, "x-rs-queued-s": f"{call.queued_s:.3f}"}
        if moved:
            headers["x-rs-affinity-break"] = "1"
        if response.is_success and call.wire == "responses" and not call.stream:
            final = await self._aggregate(call, response, first, rest, upstream.key)
            call.answered = time.monotonic()
            self._answered(call, lease, status, began, first_byte_s, moved)
            if isinstance(final, str):
                call.request_fault = True
                return self._end(call, 502, "upstream_response_failed", final, headers)
            self._succeeded(call, route, lease, body["model"])
            try:
                if call.permits is not None:
                    await call.permits.response(final)
            except PermitError:
                return self._end(call, 502, "permit_failed", FAILED, headers)
            return self._send(call, 200, json.dumps(final).encode(), headers, "application/json")
        events = None
        if response.is_success and call.stream:
            events = await _prelude(call.wire, response, _events(first, rest), upstream.key)
        try:
            self._answered(call, lease, status, began, first_byte_s, moved)
        except _Lost:
            await response.aclose()
            raise
        media_type = response.headers.get("content-type", "application/json")
        if events is not None:
            self._succeeded(call, route, lease, body["model"])
            relay = self._relay(call, events, lease)
            close = BackgroundTask(self._closed, call, response, lease)
            return StreamingResponse(
                relay, headers=headers, media_type=media_type, background=close
            )
        try:
            content = await _read(first, rest, response)
        except httpx.HTTPError as error:
            message = f"{deployment.endpoint} broke off its answer ({type(error).__name__})"
            return self._end(call, 502, "upstream_disconnected", message, headers)
        call.answered = time.monotonic()
        if not response.is_success:  # the request's own fault: passed through
            content = _scrub(content, upstream.key)
            return self._send(call, status, content, headers, media_type, "upstream_status")
        self._succeeded(call, route, lease, body["model"])
        payload = _json(content)
        call.usage = payload.get("usage") if isinstance(payload, dict) else None
        try:
            if call.permits is not None and call.wire == "chat":
                await call.permits.chat(payload)
        except PermitError:
            return self._end(call, 502, "permit_failed", FAILED, headers)
        return self._send(call, 200, content, headers, media_type)

    async def _open(
        self, upstream: Upstream, wire: Wire, body: dict[str, Any]
    ) -> tuple[httpx.Response, bytes, AsyncIterator[bytes]]:
        """POST `body` and wait for the first chunk of the answer (b"" if it is empty)."""
        stream = body.get("stream") is True
        request = self.http.build_request(
            "POST",
            upstream.url + PATHS[wire],
            content=json.dumps(body).encode(),
            headers={
                "authorization": f"Bearer {upstream.key.get_secret_value()}",
                "content-type": "application/json",
                "accept": "text/event-stream" if stream else "application/json",
            },
            # The read timeout also bounds the gaps between chunks once the answer has started.
            timeout=httpx.Timeout(
                self.config.first_byte_timeout_s, connect=self.config.connect_timeout_s, pool=None
            ),
        )
        response = None
        try:
            with anyio.fail_after(self.config.first_byte_timeout_s):
                response = await self.http.send(request, stream=True)
                chunks = response.aiter_bytes()
                first = await anext(chunks, b"")
        except BaseException as error:
            if response is not None:
                with anyio.CancelScope(shield=True):
                    await response.aclose()
            if isinstance(error, (TimeoutError, httpx.TransportError)):
                kind = "no first byte" if isinstance(error, TimeoutError) else type(error).__name__
                raise _Failed(_outcome(error), kind) from None
            raise
        return response, first, chunks

    async def _aggregate(
        self,
        call: _Call,
        response: httpx.Response,
        first: bytes,
        rest: AsyncIterator[bytes],
        key: SecretStr,
    ) -> dict[str, Any] | str:
        """Read a Responses stream up to its final event and stop: the response it completed,
        or why it failed. Breaking off earlier is a failed attempt: the client has nothing yet."""
        items = _Items()
        opened, seen, content = time.monotonic(), 0, None  # events; when the last non-preamble
        try:
            async with aclosing(_events(first, rest)) as events:
                async for event in events:
                    parsed = _json(sse.data(event))
                    kind = parsed.get("type") if isinstance(parsed, dict) else None
                    seen += 1
                    if kind is not None and kind not in _PREAMBLE:
                        content = time.monotonic()
                    if kind == "response.output_item.done":
                        items.add(parsed)
                    elif kind == "error":
                        text = f"{parsed.get('message') or parsed.get('error')}"
                        raise _Failed("stream_error", _detail(text.encode(), key))
                    elif kind in _FINAL and isinstance(parsed.get("response"), dict):
                        final: dict[str, Any] = parsed["response"]
                        call.usage = final.get("usage")
                        if kind == "response.failed":
                            if (code := _request_fault(final)) is None:
                                raise _Failed(_failed_outcome(final), _detail(event, key))
                            return f"the upstream response failed ({code}): {final.get('error')}"
                        items.complete(final)
                        return final
        except httpx.HTTPError as error:
            last = "none" if content is None else f"{content - opened:.0f} s in"
            detail = f"{type(error).__name__} after {seen} events (last content: {last})"
            raise _Failed(_outcome(error), detail) from None
        finally:
            with anyio.CancelScope(shield=True):  # a lost hedge is closed as it is cancelled
                await response.aclose()
        raise _Failed("no_final_event")

    async def _relay(
        self, call: _Call, events: AsyncGenerator[bytes, None], lease: Lease
    ) -> AsyncIterator[bytes]:
        """Pass a stream through whole events at a time. Once its final event is through, the
        call has succeeded (and its slot is free); a break before that ends the client's stream
        with an error event. In a session registered for permits, the events of an answer's
        tool calls are held back until the answer is complete and its permits are taken; if
        they are not, the stream ends with an error event instead, and a stream that breaks off
        drops them."""
        items = _Items()
        hold: ResponsesHold | ChatHold | None = None
        if call.permits is not None:
            hold = call.permits.hold(call.wire)
        try:
            async with aclosing(events):
                async for event in events:
                    relayed, final = _relayed(call, event, items)
                    try:
                        out = [relayed] if hold is None else await hold.feed(relayed)
                    except PermitError:
                        call.status, call.error = 200, "permit_failed"
                        yield sse.error_event(call.wire, "permit_failed", FAILED)
                        return
                    if final:
                        call.status, call.answered = 200, time.monotonic()
                        lease.release()
                    for piece in out:
                        yield piece
        except httpx.HTTPError as error:
            if call.status is None:
                call.status, call.error = 200, "upstream_disconnected"
                message = f"{call.deployment} broke off the stream ({type(error).__name__})"
                yield sse.error_event(call.wire, "upstream_disconnected", message)
            return
        finally:
            lease.release()
        if call.status is None:
            call.status, call.error = 200, "no_final_event"

    async def _closed(self, call: _Call, response: httpx.Response, lease: Lease) -> None:
        """After a relayed stream, also when the client went away mid-stream."""
        lease.release()
        await response.aclose()
        if call.status is None:
            call.status, call.error = 200, "client_disconnected"
        self._log(call)

    def _answered(
        self,
        call: _Call,
        lease: Lease,
        status: int,
        began: float,
        first_byte_s: float,
        moved: bool,
    ) -> None:
        if call.claimed:  # a hedged attempt whose rival answered first
            raise _Lost()
        call.claimed = True
        deployment = lease.deployment
        call.attempts.append(_record(deployment, status, began))
        for rival, rival_began in call.racing.items():  # cancelled once this answer is through
            if rival != deployment.endpoint:
                call.attempts.append(
                    {"deployment": rival, "outcome": "hedge_lost", "s": _since(rival_began)}
                )
        call.deployment, call.tier = deployment.endpoint, deployment.tier
        call.first_byte_s, call.affinity_break = first_byte_s, moved
        call.began = began
        call.move = lease.move if moved else None
        call.degradation = "overflow" if lease.overflow else None

    def _succeeded(self, call: _Call, route: Route, lease: Lease, name: str) -> None:
        """Pin and last good, a move logged; a warning whenever a model moves onto or off
        fallback deployments, except by overflow, which is routine under load. A forced call
        (a probe) changes none of that."""
        if call.forced:
            return
        deployment = lease.deployment
        self.router.succeeded(call.session, name, route.model, deployment.endpoint)
        if call.move is not None:
            self.router.stats.count(route.model, f"move_{call.move}")
            logger.info(
                "session %s (%s): moved from %s to %s (%s)",
                call.session,
                route.model,
                route.pinned,
                deployment.endpoint,
                call.move,
            )
        if lease.overflow:
            return
        key, degraded = (route.model, call.wire), deployment.tier == "fallback"
        if degraded == (key in self._degraded):
            return
        if degraded:
            self._degraded.add(key)
            logger.warning("%s (%s): degraded to fallback %s", *key, deployment.endpoint)
        else:
            self._degraded.discard(key)
            logger.warning("%s (%s): back on primary %s", *key, deployment.endpoint)

    def _send(
        self,
        call: _Call,
        status: int,
        content: bytes,
        headers: dict[str, str],
        media_type: str,
        error: str | None = None,
    ) -> Response:
        call.status, call.error = status, error
        self._log(call)
        return Response(content, status, headers=headers, media_type=media_type)

    def _end(
        self,
        call: _Call,
        status: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> Response:
        call.status, call.error = status, code
        self._log(call)
        return error_response(status, code, message, headers)

    def _log(self, call: _Call) -> None:
        """The request's line in the request log; its deployment's answer or error in the stats."""
        now = time.monotonic()
        data = {
            "session": call.session,
            "requested": call.requested,
            "model": call.model,
            "deployment": call.deployment,
            "tier": call.tier,
            "degradation": call.degradation,
            "protocol": call.wire,
            "stream": call.stream,
            "status": call.status,
            "error": call.error,
            "attempts": call.attempts,
            "queued_s": round(call.queued_s, 3),
            "first_byte_s": None if call.first_byte_s is None else round(call.first_byte_s, 3),
            "total_s": round(now - call.started, 3),
            "usage": call.usage,
            "affinity_break": call.affinity_break,
            "move": call.move,
            "permits": None if call.permits is None else _permits(call.permits),
            "forced": call.forced,
            "reasoning_effort": call.effort,
            "effort_filled": call.effort_filled,
        }
        episode = None if call.session is None else EpisodeId(call.session)
        self.log.emit(EventKind.GATEWAY_REQUEST, data, episode_id=episode, trace=call.trace)
        if call.model is None or call.deployment is None or call.began is None:
            return
        if call.error in _BROKEN:
            self.router.stats.error(call.model, call.deployment, call.error)
            if not call.forced and not call.request_fault:
                self.router.outcome(call.model, call.deployment, failed=True, began=call.began)
                if (signal := congestion(call.error)) is not None:  # a stream cut mid-way
                    self.router.congested(call.model, call.deployment, signal, began=call.began)
        elif call.answered is not None:  # not a client that left first
            first_byte_s = (call.first_byte_s or 0.0) - (call.began - call.started)
            total_s = call.answered - call.began
            self.router.stats.served(call.model, call.deployment, total_s, first_byte_s)
            if not call.forced:
                self.router.outcome(call.model, call.deployment, failed=False, began=call.began)


def _rewrite(
    body: dict[str, Any], upstream: str, endpoint: Endpoint, wire: Wire, moved: bool
) -> dict[str, Any]:
    """The body to send upstream: `model` renamed, and a Responses call streamed, with a string
    `input` as the one user message it stands for where a `stream_only` endpoint needs a list, and
    after a forced move without encrypted reasoning, which is bound to the organization that
    produced it (its summary stays), and without the ids of replayed items that the API would not
    have given (`_with_api_id`). Chat bodies go as they are, but for `model`."""
    sent = {**body, "model": upstream}
    if wire == "responses":
        sent["stream"] = True
        if endpoint.stream_only and isinstance(sent.get("input"), str):
            sent["input"] = [{"role": "user", "content": sent["input"]}]
        if moved and isinstance(sent.get("input"), list):
            sent["input"] = [_with_api_id(_without_encrypted_content(i)) for i in sent["input"]]
    return sent


def _names_effort(wire: Wire, body: dict[str, Any]) -> bool:
    """The request names an effort in some form, even `null` or on the other wire's field
    (a Chat request with nested `reasoning.effort`): the caller's choice stands."""
    reasoning = body.get("reasoning")
    nested = isinstance(reasoning, dict) and "effort" in reasoning
    return (nested or "reasoning_effort" in body) if wire == "chat" else nested


def reasoning_effort(wire: Wire, body: dict[str, Any]) -> str | None:
    """The reasoning effort a request names: Responses `reasoning.effort`, Chat
    `reasoning_effort`; None if it names none."""
    if wire == "chat":
        effort = body.get("reasoning_effort")
    else:
        reasoning = body.get("reasoning")
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    return effort if isinstance(effort, str) else None


def tool_output_images(body: dict[str, Any]) -> bool:
    """Whether a Responses body's `input` holds a tool's output (`function_call_output`,
    `custom_tool_call_output`, ...) whose `output` is a list with an `input_image` part."""
    items = body.get("input")
    if not isinstance(items, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("type", "")).endswith("_call_output")
        and isinstance(item.get("output"), list)
        and any(isinstance(p, dict) and p.get("type") == "input_image" for p in item["output"])
        for item in items
    )


def _without_encrypted_content(item: Any) -> Any:
    if isinstance(item, dict) and item.get("type") == "reasoning":
        return {key: value for key, value in item.items() if key != "encrypted_content"}
    return item


_ID_PREFIXES: Final = {"message": "msg_", "function_call": "fc_"}
"""The prefix the API's ids of these output items carry, which it checks on items replayed as
input; their ids are optional there."""


def _with_api_id(item: Any) -> Any:
    """A message or function call without its `id` when the id lacks the API's prefix: some
    OpenAI-compatible upstreams answer with message ids that read `item_...` and take them back,
    but the API, where a move lands the session within the turn, refuses the request (HTTP 400
    `invalid_value` on `input[n].id`). Reasoning items keep theirs: a replayed reasoning item
    needs its id."""
    if not isinstance(item, dict):
        return item
    prefix, ident = _ID_PREFIXES.get(item.get("type", "")), item.get("id")
    if prefix is None or not isinstance(ident, str) or ident.startswith(prefix):
        return item
    return {key: value for key, value in item.items() if key != "id"}


class _Items:
    """The output items a Responses stream announces in `response.output_item.done` events. Some
    upstreams (`stream_only` ones) end the stream with a final response whose `output` is empty;
    `complete` fills it in, as the API itself sends it."""

    def __init__(self) -> None:
        self._items: dict[int, Any] = {}

    def add(self, event: dict[str, Any]) -> None:
        index = event.get("output_index")
        self._items[index if isinstance(index, int) else len(self._items)] = event.get("item")

    def complete(self, response: dict[str, Any]) -> bool:
        """Fill an empty `output` with the items, in `output_index` order; whether it did."""
        if response.get("output") or not self._items:
            return False
        response["output"] = [self._items[index] for index in sorted(self._items)]
        return True


def _relayed(call: _Call, event: bytes, items: _Items) -> tuple[bytes, bool]:
    """An event as relayed (a final Responses event with its empty `output` completed) and
    whether it is the stream's final event; notes the usage."""
    if call.wire == "chat":
        if b"[DONE]" in event and sse.data(event) == "[DONE]":
            return event, True
        if b'"usage"' in event:
            chunk = _json(sse.data(event))
            if isinstance(chunk, dict) and chunk.get("usage"):
                call.usage = chunk["usage"]
        return event, False
    if not any(mark in event for mark in _MARKS):  # most events are deltas: no parsing
        return event, False
    parsed = _json(sse.data(event))
    if not isinstance(parsed, dict):
        return event, False
    kind, response = parsed.get("type"), parsed.get("response")
    if kind == "response.output_item.done":
        items.add(parsed)
    elif kind in _FINAL and isinstance(response, dict):
        call.usage = response.get("usage")
        if kind == "response.failed":  # relayed as it is; the deployment's error all the same
            call.error = "upstream_response_failed"
            call.request_fault = _request_fault(response) is not None
        return (sse.encode(kind, parsed) if items.complete(response) else event), True
    return event, False


async def _prelude(
    wire: Wire, response: httpx.Response, events: AsyncGenerator[bytes, None], key: SecretStr
) -> AsyncGenerator[bytes, None]:
    """A stream for a streaming client, read on to its first event past the preamble, which is
    held back: `_Failed` (the response closed) if it fails before then, as nothing has gone out
    yet. A chat stream's first event is past its preamble."""
    held: list[bytes] = []
    try:
        async for event in events:
            held.append(event)
            parsed = _json(sse.data(event))
            if not isinstance(parsed, dict):
                if wire == "chat" or sse.data(event).strip():
                    break
                continue  # a comment or keep-alive
            kind = parsed.get("type")
            if kind == "error" or (wire == "chat" and "error" in parsed):
                raise _Failed("stream_error", _detail(event, key))
            failed = parsed.get("response") if kind == "response.failed" else None
            if isinstance(failed, dict) and _request_fault(failed) is None:
                raise _Failed(_failed_outcome(failed), _detail(event, key))
            if wire == "chat" or kind not in _PREAMBLE:
                break
        else:
            raise _Failed("no_final_event")
    except BaseException as error:
        with anyio.CancelScope(shield=True):
            await events.aclose()
            await response.aclose()
        if isinstance(error, httpx.HTTPError):
            raise _Failed(_outcome(error)) from None
        raise
    return _chain(held, events)


def _request_fault(response: dict[str, Any]) -> str | None:
    """The code of a failed response's error if it is the request's own fault (`_REQUEST`)."""
    error = response.get("error")
    if not isinstance(error, dict):
        return None
    for field_ in ("code", "type"):
        value = error.get(field_)
        if isinstance(value, str) and _REQUEST & {value, value.removesuffix("_error")}:
            return value
    return None


def _failed_outcome(response: dict[str, Any]) -> str:
    """`response_failed:<code>` for the log and the overload check (`unknown` without one)."""
    error = response.get("error")
    code = error.get("code") or error.get("type") if isinstance(error, dict) else None
    return f"response_failed:{code if isinstance(code, str) and code else 'unknown'}"


async def _chain(
    held: list[bytes], events: AsyncGenerator[bytes, None]
) -> AsyncGenerator[bytes, None]:
    """`held`, then the rest of `events` (closed with this one)."""
    try:
        for event in held:
            yield event
        async for event in events:
            yield event
    finally:
        await events.aclose()


async def _events(first: bytes, rest: AsyncIterator[bytes]) -> AsyncGenerator[bytes, None]:
    """The whole events of a stream, then whatever follows the last one."""
    splitter = sse.Splitter()
    async for chunk in _chunks(first, rest):
        for event in splitter.feed(chunk):
            yield event
    if tail := splitter.rest():
        yield tail


async def _chunks(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    if first:
        yield first
    async for chunk in rest:
        yield chunk


async def _read(first: bytes, rest: AsyncIterator[bytes], response: httpx.Response) -> bytes:
    """The whole answer; closes the response."""
    try:
        return first + b"".join([chunk async for chunk in rest])
    finally:
        await response.aclose()


def _since(began: float) -> float:
    return round(time.monotonic() - began, 3)


def _record(
    deployment: Deployment, outcome: int | str, began: float, detail: str | None = None
) -> dict[str, Any]:
    record = {"deployment": deployment.endpoint, "outcome": outcome}
    record["s"] = _since(began)
    if detail:
        record["detail"] = detail
    return record


def _outcome(error: BaseException) -> str:
    timed_out = isinstance(error, (TimeoutError, httpx.TimeoutException))
    return "timeout" if timed_out else type(error).__name__


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds from a Retry-After header: delta seconds or an HTTP date."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _scrub(content: bytes, key: SecretStr) -> bytes:
    """Mask an upstream key that an upstream echoes in an error."""
    return content.replace(key.get_secret_value().encode(), b"**********")


def _detail(content: bytes, key: SecretStr) -> str:
    """The start of an upstream's error text for the log, masked before it is cut."""
    return _scrub(content, key)[:300].decode(errors="replace")


def _permits(permits: RequestPermits) -> dict[str, Any]:
    """A request's permits for the log: made, skipped, and why the Hub did not take them."""
    logged: dict[str, Any] = {"issued": permits.issued, "skipped": permits.skipped}
    if permits.error is not None:
        logged["error"] = permits.error
    return logged


def _json(text: str | bytes) -> Any:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None
