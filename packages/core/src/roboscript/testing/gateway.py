"""Test support for the model gateway: `FakeUpstream`, OpenAI-compatible endpoints with scripted
replies in the shapes of the Responses and Chat Completions APIs (streamed or not), and
`GatewayRig`, a real gateway on a local socket in front of one, its admin plane open for Hubs that
register sessions for permits."""

import asyncio
import contextlib
import json
import random
import secrets
import socket
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import httpx
import openai
import yaml
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp
from typing_extensions import Self

from roboscript import rpc
from roboscript.contracts.events import EventKind
from roboscript.contracts.gateway import GatewayAdmin
from roboscript.gateway.app import Gateway
from roboscript.gateway.client import openai_client
from roboscript.gateway.proxy import upstream_client
from roboscript.gateway.router import DefaultPolicy
from roboscript.records import read_events
from roboscript.rpc import Ready, read_ready, serve

USAGE = {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14}
CHAT_USAGE = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}


@dataclass(frozen=True)
class Text:
    """Answer with `text`, streamed if the request asks for it: `gap_s` apart per event, and held
    open `linger_s` after the last one (or until the upstream closes)."""

    text: str = "ok"
    gap_s: float = 0.0
    linger_s: float = 0.0


@dataclass(frozen=True)
class ToolCall:
    """Answer with a call of `name`; `arguments` is JSON text, sent as it is."""

    name: str
    arguments: str
    call_id: str = "call_1"


@dataclass(frozen=True)
class Status:
    """Answer with an error status, an OpenAI-style error body (or `body`) and `headers`."""

    status: int
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Hang:
    """Answer like `Text()`, but only after `seconds` (or at once when the upstream closes)."""

    seconds: float


@dataclass(frozen=True)
class Drop:
    """Stream the first `after` events of a `Text()` answer (maybe all of them), then break the
    connection."""

    after: int


@dataclass(frozen=True)
class Events:
    """Stream these events (dicts, or "[DONE]") as they are, `gap_s` apart."""

    events: list[Any]
    gap_s: float = 0.0


Reply = Text | ToolCall | Status | Hang | Drop | Events


@dataclass(frozen=True)
class Seen:
    """A request as an endpoint received it."""

    wire: str
    body: dict[str, Any]
    headers: dict[str, str]


class FakeEndpoint:
    """Answers each request with the next queued reply, else `Text()`, and records it. A
    `stream_only` endpoint behaves like a stream-only upstream: it refuses chat (500),
    unstreamed calls and a string `input` (400), and its final event has an empty `output`.
    While `hold` is an unset event, its requests wait for it (after the upstream's own).

    Overload, as upstreams show it (at once, not held): with more than `limit` requests in flight
    (this one counted), a request gets `limit_reply` (by default a 200 whose stream fails with
    `gateway_concurrency_limit`); else, with the probability `overload(inflight)`, a 200 whose
    stream fails with `server_is_overloaded`. `inflight` counts requests from their arrival
    until their answer is through; `peak` is its highest value."""

    def __init__(self, name: str, *, stream_only: bool = False) -> None:
        self.name = name
        self.stream_only = stream_only
        self.seen: list[Seen] = []
        self.replies: deque[Reply] = deque()
        self.cut = 0  # streams the gateway left before they ended
        self.hold: asyncio.Event | None = None
        self.limit: int | None = None
        self.limit_reply: Reply = failed_stream("gateway_concurrency_limit")
        self.overload: Callable[[int], float] | None = None
        self.default: Reply = Text()  # the reply when none is queued
        self.inflight = 0
        self.peak = 0
        self.refused = 0  # requests answered with an overload
        self.random = random.Random(0)

    def reply(self, *replies: Reply) -> None:
        self.replies.extend(replies)


class FakeUpstream:
    """Endpoints at `<url>/<name>/v1/responses` and `<url>/<name>/v1/chat/completions`;
    `endpoints` maps names to `stream_only`. While `hold` is an unset event, every request waits
    for it."""

    def __init__(self, endpoints: dict[str, bool]) -> None:
        self.endpoints = {name: FakeEndpoint(name, stream_only=s) for name, s in endpoints.items()}
        self.released = asyncio.Event()  # set: hanging and lingering replies end at once
        self.hold: asyncio.Event | None = None
        self.app = Starlette(
            routes=[
                Route("/{name}/v1/responses", self._responses, methods=["POST"]),
                Route("/{name}/v1/chat/completions", self._chat, methods=["POST"]),
            ]
        )

    def __getitem__(self, name: str) -> FakeEndpoint:
        return self.endpoints[name]

    async def _responses(self, request: Request) -> Response:
        return await self._answer(request, "responses")

    async def _chat(self, request: Request) -> Response:
        return await self._answer(request, "chat")

    async def _answer(self, request: Request, wire: str) -> Response:
        endpoint = self.endpoints[request.path_params["name"]]
        endpoint.inflight += 1
        endpoint.peak = max(endpoint.peak, endpoint.inflight)
        streaming = False
        try:
            response = await self._reply(endpoint, request, wire)
            if isinstance(response, StreamingResponse):
                response.body_iterator = _counted(response.body_iterator, endpoint)
                streaming = True
            return response
        finally:
            if not streaming:
                endpoint.inflight -= 1

    async def _reply(self, endpoint: FakeEndpoint, request: Request, wire: str) -> Response:
        body = await request.json()
        endpoint.seen.append(Seen(wire, body, dict(request.headers)))
        inflight = endpoint.inflight  # on arrival; a refusal is not held
        if endpoint.limit is not None and inflight > endpoint.limit:
            endpoint.refused += 1
            endpoint.replies.appendleft(endpoint.limit_reply)
        elif endpoint.overload is not None and endpoint.random.random() < endpoint.overload(
            inflight
        ):
            endpoint.refused += 1
            endpoint.replies.appendleft(failed_stream("server_is_overloaded"))
        else:
            if self.hold is not None:
                await self.hold.wait()
            if endpoint.hold is not None:
                await endpoint.hold.wait()
        if endpoint.stream_only:
            if wire == "chat":
                return _error(500, "chat/completions is not supported")
            if body.get("stream") is not True:
                return _error(400, "Stream must be set to true")
            if not isinstance(body.get("input"), list):
                return _error(400, "input must be a list")
        reply = endpoint.replies.popleft() if endpoint.replies else endpoint.default
        if isinstance(reply, Status):
            content = reply.body or {"error": {"message": f"fake status {reply.status}"}}
            return JSONResponse(content, reply.status, headers=reply.headers)
        if isinstance(reply, Events):
            return self._stream(endpoint, reply.events, wire, gap_s=reply.gap_s)
        if isinstance(reply, Hang):
            with anyio.move_on_after(reply.seconds):
                await self.released.wait()
        answer = reply if isinstance(reply, (Text, ToolCall)) else Text()
        model = str(body.get("model"))
        payload = (
            responses_payload(model, answer) if wire == "responses" else chat_payload(model, answer)
        )
        if body.get("stream") is not True:
            return JSONResponse(payload)
        events = responses_events(payload) if wire == "responses" else chat_chunks(payload)
        if endpoint.stream_only:  # like stream-only upstreams: the final event has no output
            events[-1] = {**events[-1], "response": {**payload, "output": []}}
        drop = reply.after if isinstance(reply, Drop) else None
        if isinstance(answer, Text):
            return self._stream(endpoint, events, wire, drop, answer.gap_s, answer.linger_s)
        return self._stream(endpoint, events, wire, drop)

    def _stream(
        self,
        endpoint: FakeEndpoint,
        events: list[Any],
        wire: str,
        drop_after: int | None = None,
        gap_s: float = 0.0,
        linger_s: float = 0.0,
    ) -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            try:
                for index, event in enumerate(events[:drop_after]):
                    if index and gap_s:
                        await anyio.sleep(gap_s)
                    yield encode_event(event, wire)
                if linger_s:
                    with anyio.move_on_after(linger_s):
                        await self.released.wait()
            except BaseException:  # the gateway went away first
                endpoint.cut += 1
                raise
            if drop_after is not None:
                raise ConnectionAbortedError("the fake upstream breaks the connection")

        return StreamingResponse(body(), media_type="text/event-stream")


async def _counted(body: Any, endpoint: FakeEndpoint) -> AsyncIterator[Any]:
    try:
        async for chunk in body:
            yield chunk
    finally:
        endpoint.inflight -= 1


def failed_stream(code: str) -> Events:
    """A 200 whose Responses stream fails at once with `code`, as overloaded upstreams answer."""
    failed = {"status": "failed", "error": {"code": code, "message": f"fake {code}"}}
    return Events(
        [
            {"type": "response.created", "response": {"status": "in_progress"}},
            {"type": "response.failed", "response": failed},
        ]
    )


def responses_payload(model: str, answer: Text | ToolCall) -> dict[str, Any]:
    """A completed Responses API response."""
    if isinstance(answer, ToolCall):
        item: dict[str, Any] = {
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "call_id": answer.call_id,
            "name": answer.name,
            "arguments": answer.arguments,
        }
    else:
        text = {"type": "output_text", "text": answer.text, "annotations": []}
        item = {"id": "msg_1", "type": "message", "status": "completed", "role": "assistant"}
        item["content"] = [text]
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [item],
        "usage": USAGE,
    }


def responses_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The events of a streamed Responses API response, ending with `response.completed`."""
    started = {**payload, "status": "in_progress", "output": [], "usage": None}
    events: list[dict[str, Any]] = [{"type": "response.created", "response": started}]
    for index, item in enumerate(payload["output"]):
        at = {"item_id": item["id"], "output_index": index}
        if item["type"] == "function_call":
            added = {**item, "status": "in_progress", "arguments": ""}
            events += [
                {"type": "response.output_item.added", "output_index": index, "item": added},
                {
                    "type": "response.function_call_arguments.delta",
                    **at,
                    "delta": item["arguments"],
                },
                {
                    "type": "response.function_call_arguments.done",
                    **at,
                    "arguments": item["arguments"],
                },
            ]
        else:
            added = {**item, "status": "in_progress", "content": []}
            text = item["content"][0]["text"]
            events += [
                {"type": "response.output_item.added", "output_index": index, "item": added},
                {"type": "response.output_text.delta", **at, "content_index": 0, "delta": text},
                {"type": "response.output_text.done", **at, "content_index": 0, "text": text},
            ]
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    events.append({"type": "response.completed", "response": payload})
    return [{**event, "sequence_number": number} for number, event in enumerate(events)]


def chat_payload(model: str, answer: Text | ToolCall) -> dict[str, Any]:
    """A Chat Completions response."""
    if isinstance(answer, ToolCall):
        function = {"name": answer.name, "arguments": answer.arguments}
        call = {"id": answer.call_id, "type": "function", "function": function}
        message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": [call]}
        finish = "tool_calls"
    else:
        message, finish = {"role": "assistant", "content": answer.text}, "stop"
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": CHAT_USAGE,
    }


def chat_chunks(payload: dict[str, Any]) -> list[Any]:
    """The chunks of a streamed chat completion: tool-call arguments in two pieces, usage last
    (as with `stream_options.include_usage`), then "[DONE]"."""

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        choice = {"index": 0, "delta": delta, "finish_reason": finish}
        head = {"id": payload["id"], "object": "chat.completion.chunk", "created": 0}
        return {**head, "model": payload["model"], "choices": [choice]}

    choice = payload["choices"][0]
    message = choice["message"]
    chunks: list[Any] = [chunk({"role": "assistant", "content": ""})]
    for index, call in enumerate(message.get("tool_calls") or []):
        arguments = call["function"]["arguments"]
        half = len(arguments) // 2
        opened = {"name": call["function"]["name"], "arguments": ""}
        first = {"index": index, "id": call["id"], "type": "function", "function": opened}
        chunks.append(chunk({"tool_calls": [first]}))
        for part in (arguments[:half], arguments[half:]):
            chunks.append(
                chunk({"tool_calls": [{"index": index, "function": {"arguments": part}}]})
            )
    if message.get("content"):
        chunks.append(chunk({"content": message["content"]}))
    chunks.append(chunk({}, choice["finish_reason"]))
    chunks.append({**chunk({}), "choices": [], "usage": payload["usage"]})
    chunks.append("[DONE]")
    return chunks


def encode_event(event: Any, wire: str) -> bytes:
    """One event on the wire; a Responses stream names each event's type."""
    if event == "[DONE]":
        return b"data: [DONE]\n\n"
    name = f"event: {event['type']}\n" if wire == "responses" else ""
    return f"{name}data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status)


@asynccontextmanager
async def serving(app: ASGIApp, directory: Path) -> AsyncIterator[str]:
    """Serve `app` on a free local port for the block; yields its base URL."""
    path = directory / f".ready-{secrets.token_hex(4)}.json"
    task = asyncio.get_running_loop().create_task(serve(app, ready_file=path))
    try:
        with anyio.fail_after(10):
            while (ready := read_ready(path)) is None:
                if task.done():
                    task.result()  # raises what stopped the server
                await anyio.sleep(0.01)
        yield ready.url
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@asynccontextmanager
async def running(service: Coroutine[Any, Any, None], ready_file: Path) -> AsyncIterator[Ready]:
    """Run a service that writes `ready_file` as its own task until the block ends (pytest
    fixtures set up and tear down in different tasks); yields the ready file once written."""
    task = asyncio.get_running_loop().create_task(service)
    try:
        with anyio.fail_after(30):
            while (ready := read_ready(ready_file)) is None:
                if task.done():
                    task.result()  # raises what stopped the service
                await anyio.sleep(0.01)
        yield ready
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class FakeClock:
    """A monotonic clock that moves only when told to."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class GatewayRig:
    """A real gateway on a local socket in front of a `FakeUpstream`:
    `async with GatewayRig(tmp_path, ["ep1", "ep2"]) as rig:`.

    Endpoints speak both protocols, except the `stream_only` ones (responses only); those in
    `unreachable` point at a closed port. `models` maps logical models to endpoints (default: `m`
    on all of them, in order), each deployed as `<model>-up`, in the fallback tier on the
    `fallback` endpoints, with `caps[endpoint]` as its `max_inflight` and `floors[endpoint]` as
    its `min_inflight` (default: the config's); `no_tool_images` endpoints have
    `tool_output_images: false`; `pools` maps endpoints to their `pool`; `aliases` and further
    keywords (`GatewayConfig` fields) go into the config. Cooldowns and the caps' quiet intervals
    run on `clock`; files go to `root`.
    `admin_token` opens the admin plane.
    """

    def __init__(
        self,
        root: Path,
        endpoints: Sequence[str],
        *,
        stream_only: Sequence[str] = (),
        unreachable: Sequence[str] = (),
        fallback: Sequence[str] = (),
        no_tool_images: Sequence[str] = (),
        caps: dict[str, int] | None = None,
        floors: dict[str, int] | None = None,
        pools: dict[str, str] | None = None,
        models: dict[str, list[str]] | None = None,
        aliases: dict[str, str] | None = None,
        **settings: Any,
    ) -> None:
        self.root = root
        self.upstream = FakeUpstream({name: name in stream_only for name in endpoints})
        self.unreachable = unreachable
        self.clock = FakeClock()
        self.token = SecretStr(f"tok-{secrets.token_hex(8)}")
        self.admin_token = SecretStr(f"adm-{secrets.token_hex(12)}")
        self.keys = {name: f"sk-{name}-{secrets.token_hex(8)}" for name in endpoints}
        self.config: dict[str, Any] = {
            "schema": "rs-gateway/v1",
            "listen": "127.0.0.1:0",
            "endpoints": {
                name: {
                    "url_env": f"EP{index}_URL",
                    "key_env": f"EP{index}_KEY",
                    "protocols": ["responses"] if name in stream_only else ["responses", "chat"],
                    "stream_only": name in stream_only,
                    **({"pool": pools[name]} if pools and name in pools else {}),
                    "tool_output_images": name not in no_tool_images,
                }
                for index, name in enumerate(endpoints)
            },
            "models": {
                model: {
                    "deployments": [
                        _deployment(
                            model, e, e in fallback, (caps or {}).get(e), (floors or {}).get(e)
                        )
                        for e in names
                    ]
                }
                for model, names in (models or {"m": list(endpoints)}).items()
            },
            "aliases": aliases or {},
            "state_file": str(root / "gateway-state.json"),
            "log_file": str(root / "gateway.jsonl"),
            **settings,
        }
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> Self:
        stack = self._stack
        self.root.mkdir(parents=True, exist_ok=True)
        upstream = await stack.enter_async_context(serving(self.upstream.app, self.root))
        dead = f"http://127.0.0.1:{_closed_port()}"
        lines = [
            f"{spec['url_env']}={dead if name in self.unreachable else upstream}/{name}/v1\n"
            f"{spec['key_env']}={self.keys[name]}\n"
            for name, spec in self.config["endpoints"].items()
        ]
        self.env_api = self.root / ".env.api"
        self.env_api.write_text("".join(lines), encoding="utf-8")
        self.config_path = self.root / "gateway.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        http = await stack.enter_async_context(upstream_client())
        self.gateway = Gateway.from_files(
            self.config_path,
            http=http,
            env_api=self.env_api,
            token=self.token,
            admin_token=self.admin_token,
            policy=DefaultPolicy(),
            clock=self.clock,
        )
        stack.push_async_callback(self.gateway.aclose)
        self.url = await stack.enter_async_context(serving(self.gateway.app, self.root))
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        client = httpx.AsyncClient(base_url=self.url, timeout=30, trust_env=False, limits=limits)
        self.http = await stack.enter_async_context(client)
        stack.callback(self.upstream.released.set)  # first on exit: nothing waits on a Hang
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._stack.aclose()

    def auth(self, session: str | None = None) -> dict[str, str]:
        key = self.token.get_secret_value() + ("" if session is None else f":{session}")
        return {"authorization": f"Bearer {key}"}

    async def post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        session: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return await self.http.post(
            path, json=body, headers={**self.auth(session), **(headers or {})}
        )

    def openai(self, session: str, key: SecretStr | None = None) -> openai.AsyncOpenAI:
        """An OpenAI client of the gateway for `session` (with its `key`, if registered)."""
        return openai_client(session, key=key, base_url=f"{self.url}/v1", token=self.token)

    def admin(self) -> GatewayAdmin:
        """A client of the admin plane, as a Hub holds it."""
        return rpc.client(
            GatewayAdmin,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
            self.url,
            token=self.admin_token,
            http=self.http,
            timeout=30,
        )

    async def stats(self) -> dict[str, Any]:
        """`GET /admin/stats`, as a monitor with the admin token sees it."""
        headers = {"authorization": f"Bearer {self.admin_token.get_secret_value()}"}
        response = await self.http.get("/admin/stats", headers=headers)
        response.raise_for_status()
        stats: dict[str, Any] = response.json()
        return stats

    def log(self) -> list[dict[str, Any]]:
        """The data of the request log's request lines."""
        return self.events(EventKind.GATEWAY_REQUEST)

    def events(self, *kinds: str) -> list[dict[str, Any]]:
        """The data of the log's lines of `kinds` (`gateway.cap`, `gateway.health`, ...)."""
        self.gateway.log.flush()  # the gateway appends on a thread
        events = read_events(Path(self.config["log_file"]))
        return [event.data for event in events if event.kind in kinds]

    def log_text(self) -> str:
        self.gateway.log.flush()
        return Path(self.config["log_file"]).read_text(encoding="utf-8")


def _deployment(
    model: str, endpoint: str, fallback: bool, cap: int | None, floor: int | None
) -> dict[str, Any]:
    deployment: dict[str, Any] = {"endpoint": endpoint, "upstream": f"{model}-up"}
    if fallback:
        deployment["tier"] = "fallback"
    if cap is not None:
        deployment["max_inflight"] = cap
    if floor is not None:
        deployment["min_inflight"] = floor
    return deployment


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port
