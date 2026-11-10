"""What the Hub's MCP endpoint and remote toolset servers share: bearer authentication, the MCP
app (stateless: both client generations, no protocol sessions to keep), the `_meta` keys that
carry episode context, results as structured content, and errors as RFC 9457 problems inside
tool results."""

import json
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Final, TypeVar

import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, SecretStr, ValidationError
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from roboscript.contracts.errors import Problem, ProtocolError, RSError, error_from_problem
from roboscript.contracts.trace import TraceContext, use_trace
from roboscript.hub.tokens import TokenClaims
from roboscript.rpc import make_app

CONTEXT: Final = "roboscript/context"
"""Request `_meta`: the `CallContext` of a Hub-to-toolset call (tokens revealed)."""
FRAMES: Final = "roboscript/frames"
"""Request `_meta`: the episode's frames and scene facts now (an `EpisodeSeed`)."""
PARENT: Final = "roboscript/parent"
"""Request `_meta`: a Hub-to-toolset call's handle, which the toolset's own calls back through
the Hub carry to say whose they are."""
HUB: Final = "roboscript/hub"
"""Request `_meta`: the Hub's MCP endpoint, where a remote toolset calls other toolsets."""
TRACEPARENT: Final = "traceparent"
"""Request `_meta`: W3C trace context (SEP-414)."""
ENDED: Final = "roboscript/episode_ended"
"""Result `_meta`: whether the episode is over after this call."""
CODE: Final = "roboscript/code"
"""Result `_meta`: an error result's code (the Hub's `CallOutcome.error_code`)."""
PROBLEM: Final = "roboscript/problem"
"""Result `_meta`: the problem a toolset raised, so the caller can raise it again."""
WRITES: Final = "roboscript/writes"
"""Result `_meta`: what a remote tool put into the episode's frame store (events to apply)."""

_LOOPBACK: Final = ("127.0.0.1", "localhost", "::1")

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class BearerAuth:
    """ASGI middleware: a request needs `Authorization: Bearer <token>` that `verify` accepts
    (returning its claims, raising `ProtocolError` otherwise), else it gets 401 with the Bearer
    challenge, before its body is read. Handlers find the claims and the token in
    `request.state`."""

    def __init__(self, app: ASGIApp, verify: Callable[[str], TokenClaims]) -> None:
        self.app = app
        self.verify = verify

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scheme, _, token = Headers(scope=scope).get("authorization", "").partition(" ")
        try:
            if scheme.lower() != "bearer" or not token:
                raise ProtocolError("unauthorized", "send Authorization: Bearer <token>")
            claims = self.verify(token)
        except ProtocolError as error:
            body = {"error": "invalid_token", "error_description": error.detail}
            challenge = {"WWW-Authenticate": 'Bearer error="invalid_token"'}
            await JSONResponse(body, 401, headers=challenge)(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["claims"], state["bearer"] = claims, SecretStr(token)
        await self.app(scope, receive, send)


def mcp_app(
    server: Server[Any],
    verify: Callable[[str], TokenClaims],
    host: str,
    routes: list[BaseRoute] | None = None,
) -> Starlette:
    """`server` at `/mcp` behind `BearerAuth`, stateless (no `Mcp-Session-Id`: handshake-era
    clients initialize on every connection, and nothing outlives a request), with DNS-rebinding
    protection on loopback, `/healthz`, `/readyz` and extra `routes`."""
    manager = StreamableHTTPSessionManager(
        app=server, stateless=True, security_settings=_security(host)
    )
    endpoint = BearerAuth(StreamableHTTPASGIApp(manager), verify)
    return make_app([Route("/mcp", endpoint), *(routes or [])], lifespan=lambda app: manager.run())


def claims_of(ctx: ServerRequestContext[Any, Any]) -> TokenClaims:
    """The verified claims of the request a handler serves."""
    claims: TokenClaims = ctx.request.state.claims  # type: ignore[union-attr]
    return claims


def bearer_of(ctx: ServerRequestContext[Any, Any]) -> SecretStr:
    """The verified bearer token of the request a handler serves."""
    bearer: SecretStr = ctx.request.state.bearer  # type: ignore[union-attr]
    return bearer


def _security(host: str) -> TransportSecuritySettings | None:
    """DNS-rebinding protection (Host and Origin checks) for a server bound to loopback."""
    if host not in _LOOPBACK:
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
    )


@contextmanager
def traced(meta: Mapping[str, Any] | None) -> Iterator[None]:
    """Run under a child of the caller's span from `_meta.traceparent`, if it sent a valid one."""
    try:
        trace = TraceContext.from_traceparent(str((meta or {})[TRACEPARENT])).child()
    except (KeyError, ValueError):
        yield
        return
    with use_trace(trace):
        yield


def structured(result: BaseModel, meta: dict[str, Any] | None = None) -> types.CallToolResult:
    """A tool's result model as structured content, with its JSON as the text. Computed fields
    stay in: the output schema (serialization mode) requires them."""
    data = result.model_dump(mode="json")
    text = types.TextContent(text=json.dumps(data, ensure_ascii=False))
    return types.CallToolResult(content=[text], structured_content=data, meta=meta)  # type: ignore[call-arg]


def from_structured(model: type[M], result: types.CallToolResult) -> M:
    """The result model back from a `structured` answer, raising the error an error result
    stands for; the computed fields are recomputed."""
    raise_for(result)
    data = result.structured_content if isinstance(result.structured_content, dict) else {}
    computed = model.model_computed_fields
    return model.model_validate({key: value for key, value in data.items() if key not in computed})


def error_result(error: Exception) -> types.CallToolResult:
    """A toolset's exception as an error result the caller turns back into it (`raise_for`);
    anything but an `RSError` travels as `internal`, its traceback going to this log."""
    if not isinstance(error, RSError):
        log.error("a toolset failed unexpectedly", exc_info=error)
        error = RSError("internal", f"{type(error).__name__}: {error}")
    problem = error.to_problem()
    text = types.TextContent(text=problem.detail or problem.title)
    meta = {PROBLEM: problem.model_dump(mode="json")}
    return types.CallToolResult(content=[text], is_error=True, meta=meta)  # type: ignore[call-arg]


def raise_for(result: types.CallToolResult) -> None:
    """Raise the error an error result stands for (see `error_result`)."""
    if not result.is_error:
        return
    try:
        problem = Problem.model_validate((result.meta or {})[PROBLEM])
    except (KeyError, ValidationError):
        text = " ".join(part.text for part in result.content if isinstance(part, types.TextContent))
        raise RSError("internal", text or "the remote tool failed") from None
    raise error_from_problem(problem)
