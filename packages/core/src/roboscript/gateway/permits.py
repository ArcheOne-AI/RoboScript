"""Permits (docs/concepts/agent-runtime.md): the gateway's side of "only model-proposed calls
run".

A served Hub registers each episode's session on the admin plane (`Sessions`, which implements
`GatewayAdmin`), with the hash of a key that the session's requests must then carry. Once a model
answer in a registered session is complete (Responses `response.completed` or `incomplete`; Chat
the last `finish_reason` or `[DONE]`), each tool call in it (Responses `function_call` items, Chat
`tool_calls`, by `call_id`) becomes a single-use `Permit`, all posted in one batch to the Hub's
intake before the client sees any of the calls: an aggregated answer waits for the Hub's
acknowledgement, and a stream holds back its events from the first one of a tool call until
then. A failed or broken-off answer gets no permits. A called name maps to a Hub tool through the
description the request's `tools` gave it (`description_hash`), so a client may rename or prefix
tools; calls of anything else, and calls whose arguments are not plain JSON, get none. When the
Hub does not acknowledge, the answer's calls never reach the client: it fails with
`permit_failed`.
"""

import hmac
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Final

import httpx
from pydantic import SecretStr

from roboscript import rpc
from roboscript.contracts.errors import ProtocolError, RSError
from roboscript.contracts.gateway import (
    UNHASHABLE,
    Permit,
    PermitIntake,
    SessionRegistration,
    args_hash,
    description_hash,
    session_key_hash,
)
from roboscript.contracts.ids import EpisodeId, SessionId, ToolName
from roboscript.gateway import sse
from roboscript.gateway.config import Wire

FAILED: Final = "The gateway could not register this answer's tool calls with the Hub; try again."
"""What the client of a failed permit post is told (the log has the details)."""

_COMPLETE: Final = frozenset({"response.completed", "response.incomplete"})
_FAILED: Final = "response.failed"
_MARKS: Final = (b"function_call", *(kind.encode() for kind in (*_COMPLETE, _FAILED)))
_CALL_EVENTS: Final = frozenset(
    {
        "response.output_item.added",
        "response.output_item.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    }
)

Call = tuple[str, Any, Any]
"""(call id, the name the model called, the arguments as the model wrote them), the last two
as they came: anything but an offered name and a JSON object's text gets no permit."""


class PermitError(Exception):
    """The Hub did not acknowledge an answer's permits."""


class SessionKeyError(Exception):
    """A request of a registered session without its key."""


@dataclass(frozen=True)
class _Session:
    registration: SessionRegistration
    intake: PermitIntake


class Sessions:
    """The registered sessions: `GatewayAdmin` fills them, requests read them. Registrations
    lapse at their `expires_at` (Unix time on `clock`); permits are posted within `timeout_s`."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        timeout_s: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.http = http
        self.timeout_s = timeout_s
        self.clock = clock
        self._sessions: dict[SessionId, _Session] = {}

    async def register(self, registration: SessionRegistration) -> None:
        if registration.expires_at <= self.clock():
            raise ProtocolError("invalid_request", "the registration has already expired")
        intake = rpc.client(
            PermitIntake,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
            registration.permit_url,
            token=registration.permit_token,
            http=self.http,
            timeout=self.timeout_s,
        )
        self._sessions[registration.session_id] = _Session(registration, intake)

    async def unregister(self, session_id: SessionId) -> None:
        self._sessions.pop(session_id, None)

    def permits(
        self, session: SessionId | None, key: SecretStr | None, wire: Wire, body: dict[str, Any]
    ) -> "RequestPermits | None":
        """The permits of a request of `session`, None unless the session is registered;
        `SessionKeyError` if it is and `key` is not its key."""
        found = None if session is None else self._sessions.get(session)
        if found is None:
            return None
        if found.registration.expires_at <= self.clock():
            del self._sessions[found.registration.session_id]
            return None
        given = "" if key is None else session_key_hash(key)
        if not hmac.compare_digest(given, found.registration.session_key_hash):
            raise SessionKeyError(session)
        offered = _offered(wire, body.get("tools"), found.registration.tools)
        return RequestPermits(found, offered)


class RequestPermits:
    """One request's permits: the Hub tools its `tools` offer, by the names the model sees them
    under, and how many permits it made (`issued`) and could not make (`skipped`: calls of other
    tools, or with arguments that are not plain JSON)."""

    def __init__(self, session: _Session, offered: dict[str, ToolName]) -> None:
        self.session = session
        self.offered = offered
        self.issued = 0
        self.skipped = 0
        self.error: str | None = None  # why the Hub did not take them, for the log
        self._done: set[str] = set()

    async def grant(self, calls: Iterable[Call]) -> None:
        """Post one permit per call not permitted yet, in one batch; `PermitError` unless the
        Hub acknowledges. A call counts as permitted once the Hub has acknowledged it."""
        episode = EpisodeId(self.session.registration.session_id)
        permits: dict[str, Permit] = {}
        for call_id, name, arguments in calls:
            if call_id in self._done or call_id in permits:
                continue
            tool, digest = self.offered.get(name), _args_hash(arguments)
            if tool is None or digest is None:
                self.skipped += 1
                continue
            permits[call_id] = Permit(
                episode_id=episode, tool=tool, args_hash=digest, call_id=call_id
            )
        if not permits:
            return
        try:
            await self.session.intake.issue(list(permits.values()))
        except RSError as error:
            self.error = f"{type(error).__name__}: {error}"
            raise PermitError(self.error) from error
        self._done |= permits.keys()
        self.issued += len(permits)

    async def response(self, response: dict[str, Any]) -> None:
        """Permit the calls of a complete Responses answer."""
        output = response.get("output")
        items = output if isinstance(output, list) else []
        await self.grant(
            (i["call_id"], i.get("name"), i.get("arguments")) for i in items if _is_call(i)
        )

    async def chat(self, payload: Any) -> None:
        """Permit the calls of a complete Chat Completions answer."""
        choices = payload.get("choices") if isinstance(payload, dict) else None
        calls: list[Call] = []
        for choice in choices if isinstance(choices, list) else []:
            message = choice.get("message") if isinstance(choice, dict) else None
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            for tool_call in tool_calls if isinstance(tool_calls, list) else []:
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                if isinstance(function, dict) and isinstance(tool_call.get("id"), str):
                    calls.append((tool_call["id"], function.get("name"), function.get("arguments")))
        await self.grant(calls)

    def hold(self, wire: Wire) -> "ResponsesHold | ChatHold":
        return ResponsesHold(self) if wire == "responses" else ChatHold(self)


class _Hold:
    """Events held back (None: not holding): `_pass` sends an event on, or keeps it."""

    def __init__(self, permits: RequestPermits) -> None:
        self.permits = permits
        self._held: list[bytes] | None = None

    def _pass(self, event: bytes) -> list[bytes]:
        if self._held is None:
            return [event]
        self._held.append(event)
        return []

    def _release(self, event: bytes) -> list[bytes]:
        released, self._held = [*(self._held or []), event], None
        return released


class ResponsesHold(_Hold):
    """Holds a Responses stream back from its first tool-call event until its answer is
    complete; `feed(event)` returns the events to send on now, in order."""

    async def feed(self, event: bytes) -> list[bytes]:
        if not any(mark in event for mark in _MARKS):  # most events: no parsing
            return self._pass(event)
        parsed = _json(sse.data(event))
        kind = parsed.get("type") if isinstance(parsed, dict) else None
        if kind in _COMPLETE or kind == _FAILED:
            response = parsed.get("response")
            if kind in _COMPLETE and isinstance(response, dict):
                await self.permits.response(response)
            return self._release(event)
        if self._held is None and _part_of_call(kind, parsed):
            self._held = [event]
            return []
        return self._pass(event)


class ChatHold(_Hold):
    """Holds a Chat Completions stream back from its first `tool_calls` chunk until every choice
    with calls has finished (or `[DONE]`); `feed(chunk)` returns the chunks to send on now."""

    def __init__(self, permits: RequestPermits) -> None:
        super().__init__(permits)
        self._calls: dict[tuple[int, int], dict[str, str]] = {}  # (choice, index) -> call
        self._finished: set[int] = set()

    async def feed(self, chunk: bytes) -> list[bytes]:
        data = sse.data(chunk)
        if data == "[DONE]":
            return await self._complete(chunk)
        if b"tool_calls" not in chunk and b"finish_reason" not in chunk:
            return self._pass(chunk)
        parsed = _json(data)
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        for choice in choices if isinstance(choices, list) else []:
            if isinstance(choice, dict):
                self._follow(choice)
        if not self._calls:
            return self._pass(chunk)
        if {number for number, _ in self._calls} <= self._finished:
            return await self._complete(chunk)
        if self._held is None:
            self._held = []
        return self._pass(chunk)

    def _follow(self, choice: dict[str, Any]) -> None:
        number, delta = choice.get("index", 0), choice.get("delta")
        pieces = delta.get("tool_calls") if isinstance(delta, dict) else None
        for piece in pieces if isinstance(pieces, list) else []:
            if not isinstance(piece, dict):
                continue
            key = (number, piece.get("index", 0))
            call = self._calls.setdefault(key, {"id": "", "name": "", "arguments": ""})
            function = piece.get("function")
            if not isinstance(function, dict):
                function = {}
            call["id"] = piece.get("id") or call["id"]
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""
        if choice.get("finish_reason"):
            self._finished.add(number)

    async def _complete(self, chunk: bytes) -> list[bytes]:
        """Permit the calls (once: `grant` skips permitted ones), then send everything on."""
        calls = [(c["id"], c["name"], c["arguments"]) for c in self._calls.values() if c["id"]]
        await self.permits.grant(calls)
        return self._release(chunk)


def _offered(wire: Wire, tools: Any, registered: dict[str, ToolName]) -> dict[str, ToolName]:
    """Model-visible name -> Hub tool for the function tools a request offers whose
    descriptions the Hub registered."""
    offered: dict[str, ToolName] = {}
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        spec = tool.get("function") if wire == "chat" else tool
        if not isinstance(spec, dict):
            continue
        name, description = spec.get("name"), spec.get("description")
        if isinstance(name, str) and isinstance(description, str):
            found = registered.get(description_hash(description))
            if found is not None:
                offered[name] = found
    return offered


def _part_of_call(kind: Any, parsed: Any) -> bool:
    """Whether a Responses event is part of a tool call."""
    if kind not in _CALL_EVENTS:
        return False
    return kind.startswith("response.function_call") or _is_call(parsed.get("item"))


def _is_call(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "function_call"
        and isinstance(item.get("call_id"), str)
    )


def _args_hash(arguments: Any) -> str | None:
    """The hash of arguments as the model wrote them (JSON text), None unless they are a plain
    JSON object."""
    value = _json(arguments) if isinstance(arguments, str) else None
    if not isinstance(value, dict):
        return None
    try:
        return args_hash(value)
    except UNHASHABLE:
        return None


def _json(text: str | bytes) -> Any:
    """Parsed JSON, None if it is not JSON (or too deep to parse)."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None
