"""The gateway service (docs/concepts/agent-runtime.md): `POST /v1/responses`,
`POST /v1/chat/completions` and `GET /v1/models` for holders of the gateway token, plus `/healthz`
and `/readyz`, and, with an admin token, the admin plane: `gateway.admin`, where a Hub registers
sessions for permits, `GET /admin/stats`, the load per deployment (`Router.snapshot`), and
`POST /admin/scan-logs`, whether log files hold one of the gateway's secrets (a self-check that no
other process needs the secrets for).

A request carries its session id after the token (`Bearer <token>:<session id>`, which works in
any client that takes an API key) or in an `x-rs-session-id` or Codex `session-id` header, in
that order of precedence. Sessions stick to their deployment and alias resolution. A session a
Hub registered for permits also needs its key, after the session id (`Bearer
<token>:<session id>:<key>`) or in an `x-rs-session-key` header.

An endpoint probe sends a request to one deployment with `x-rs-deployment: <endpoint>`, allowed
only with the admin token in `x-rs-admin-token`: no failover, no pin, the endpoint's capability
flags not applied (`gateway.probe` measures them this way).
"""

import hmac
import json
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from typing_extensions import Self

from roboscript.contracts.errors import ProtocolError, RSError
from roboscript.contracts.gateway import GatewayAdmin
from roboscript.contracts.ids import SessionId
from roboscript.contracts.trace import TraceContext
from roboscript.gateway.config import (
    GatewayConfig,
    Upstream,
    Wire,
    env_api_file,
    load,
    read_upstreams,
)
from roboscript.gateway.permits import Sessions
from roboscript.gateway.proxy import Proxy, error_response
from roboscript.gateway.router import Router, RoutingPolicy, load_policy
from roboscript.records import EventWriter
from roboscript.rpc import authorize, make_app, problem_response, routes
from roboscript.settings import EnvSettings

SESSION_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CHUNK: Final = 1 << 20  # bytes of a log read at a time by /admin/scan-logs


class _Refused(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class Gateway:
    """The ASGI app (`app`) and its routing state. `policy` defaults to the one the config names;
    `clock` measures cooldowns. Without `admin_token` there is no admin plane, so no session is
    ever registered for permits. `aclose()` finishes state writes and closes the request log."""

    def __init__(
        self,
        config: GatewayConfig,
        upstreams: dict[str, Upstream],
        *,
        token: SecretStr,
        http: httpx.AsyncClient,
        admin_token: SecretStr | None = None,
        policy: RoutingPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not token.get_secret_value() or ":" in token.get_secret_value():
            raise ValueError("RS_GATEWAY_TOKEN must be set and must not contain ':'")
        if admin_token is not None and len(admin_token.get_secret_value()) < 16:
            raise ValueError("RS_GATEWAY_ADMIN_TOKEN must have 16 or more characters")
        if admin_token is not None and admin_token.get_secret_value() == token.get_secret_value():
            raise ValueError("RS_GATEWAY_ADMIN_TOKEN must differ from RS_GATEWAY_TOKEN")
        self.config = config
        self._token = token
        self._admin_token = admin_token
        policy = load_policy(config.policy) if policy is None else policy
        # Appends on a thread: a request must never wait for the disk, and a saturated file
        # system would hold the event loop.
        self.log = EventWriter(
            config.log_file, "gateway", background=True, max_queued=config.log_max_queued
        )
        self.router = Router(config, policy, clock=clock, on_change=self._changed)
        self.sessions = Sessions(http, timeout_s=config.permit_timeout_s)
        self.proxy = Proxy(config, upstreams, self.router, http, self.log, self.sessions)
        admin = (
            []
            if admin_token is None
            else [
                *routes(GatewayAdmin, self.sessions, token=admin_token),
                Route("/admin/stats", self._stats, methods=["GET"]),
                Route("/admin/scan-logs", self._scan_logs, methods=["POST"]),
            ]
        )
        self.app = make_app(
            [
                Route("/v1/responses", self._responses, methods=["POST"]),
                Route("/v1/chat/completions", self._chat, methods=["POST"]),
                Route("/v1/models", self._models, methods=["GET"]),
            ],
            admin,
        )

    def _changed(self, kind: str, data: dict[str, Any]) -> None:
        """A deployment's cap or health changed: a line in the request log."""
        self.log.emit(kind, data)

    @classmethod
    def from_files(
        cls,
        source: str | Path,
        *,
        http: httpx.AsyncClient,
        env_api: Path | None = None,
        token: SecretStr | None = None,
        admin_token: SecretStr | None = None,
        policy: RoutingPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> Self:
        """Load `source` (see `config.load`) and `env_api` (default: `env_api_file()`); the
        tokens default to `RS_GATEWAY_TOKEN` and `RS_GATEWAY_ADMIN_TOKEN` (optional) from `.env`."""
        config = load(source)
        upstreams = read_upstreams(config, env_api_file() if env_api is None else env_api)
        settings = EnvSettings()
        if token is None:
            token = settings.gateway_token
            if token is None:
                raise ValueError("RS_GATEWAY_TOKEN is not set: add it to .env (see .env.example)")
        if admin_token is None:
            admin_token = settings.gateway_admin_token
        return cls(
            config,
            upstreams,
            token=token,
            http=http,
            admin_token=admin_token,
            policy=policy,
            clock=clock,
        )

    async def aclose(self) -> None:
        await self.router.state.flush()
        await anyio.to_thread.run_sync(self.log.close)  # flushes, bounded; never blocks the loop

    async def _responses(self, request: Request) -> Response:
        return await self._forward(request, "responses")

    async def _chat(self, request: Request) -> Response:
        return await self._forward(request, "chat")

    async def _forward(self, request: Request, wire: Wire) -> Response:
        session, key = None, None
        try:
            session, key = self._session(request)
            deployment = self._forced(request)
            body = _body(await request.body())
        except _Refused as refused:
            return self.proxy.refuse(wire, refused.status, refused.code, refused.message, session)
        trace = _trace(request)
        forward = self.proxy.forward(wire, body, session, key, trace, deployment=deployment)
        return await _unless_disconnected(request, forward)

    async def _stats(self, request: Request) -> Response:
        try:
            authorize(request, self._admin_token)
        except RSError as error:
            return problem_response(error)
        return JSONResponse(self.router.snapshot())

    async def _scan_logs(self, request: Request) -> Response:
        """`{"paths": [absolute file paths on this machine]}` -> `{"leaked": whether any holds
        one of the gateway's secrets (an upstream key, its client or admin token), "files": the
        paths that do}`; never a secret."""
        try:
            authorize(request, self._admin_token)
            paths = _ScanLogs.model_validate_json(await request.body()).paths
            held = [self._token, *[upstream.key for upstream in self.proxy.upstreams.values()]]
            held += [] if self._admin_token is None else [self._admin_token]
            secrets = [secret.get_secret_value().encode() for secret in held]
            await anyio.to_thread.run_sync(self.log.flush)  # the request log is scanned whole
            found = await anyio.to_thread.run_sync(_holding, paths, secrets)
        except RSError as error:
            return problem_response(error)
        except ValidationError:
            fix = 'send {"paths": [absolute paths of files]}'
            return problem_response(ProtocolError("invalid_request", fix))
        return JSONResponse({"leaked": bool(found), "files": [str(path) for path in found]})

    async def _models(self, request: Request) -> Response:
        try:
            self._session(request)
        except _Refused as refused:
            return error_response(refused.status, refused.code, refused.message)
        names = [*self.config.models, *self.config.aliases]
        data = [{"id": n, "object": "model", "created": 0, "owned_by": "rs-gateway"} for n in names]
        return JSONResponse({"object": "list", "data": data})

    def _forced(self, request: Request) -> str | None:
        """The endpoint an admin's `x-rs-deployment` names, if the request has that header."""
        endpoint = request.headers.get("x-rs-deployment")
        if endpoint is None:
            return None
        given = request.headers.get("x-rs-admin-token", "").encode()
        admin = self._admin_token
        if admin is None or not hmac.compare_digest(given, admin.get_secret_value().encode()):
            message = "x-rs-deployment needs the admin token in x-rs-admin-token"
            raise _Refused(403, "admin_token_required", message)
        return endpoint

    def _session(self, request: Request) -> tuple[SessionId | None, SecretStr | None]:
        """Check the token; return the session id and key, if the request carries them."""
        scheme, _, credential = request.headers.get("authorization", "").partition(" ")
        token, _, rest = credential.strip().partition(":")
        session, _, key = rest.partition(":")
        expected = self._token.get_secret_value().encode()
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), expected):
            message = "send Authorization: Bearer <RS_GATEWAY_TOKEN>[:<session id>[:<key>]]"
            raise _Refused(401, "invalid_api_key", message)
        headers = request.headers
        session = session or headers.get("x-rs-session-id") or headers.get("session-id") or ""
        key = key or headers.get("x-rs-session-key") or ""
        if not session:
            return None, None
        if not SESSION_ID.fullmatch(session):
            message = "a session id is 1-128 letters, digits, '.', '_' or '-'"
            raise _Refused(400, "invalid_session_id", message)
        return SessionId(session), SecretStr(key) if key else None


async def _unless_disconnected(request: Request, answer: Awaitable[Response]) -> Response:
    """`answer`, cancelled when the client disconnects before it is ready, so that a call that
    waits for a slot, or runs upstream, for a client that has left gives its place up. A
    streamed answer is ready at its first byte; its stream watches for itself."""
    response = Response(status_code=499)  # nobody receives it
    async with anyio.create_task_group() as group:

        async def watch() -> None:
            while (await request.receive())["type"] != "http.disconnect":
                pass  # the body was read already: only a disconnect is left to come
            group.cancel_scope.cancel()

        group.start_soon(watch)
        response = await answer
        group.cancel_scope.cancel()
    return response


class _ScanLogs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paths: list[Path] = Field(min_length=1)


def _holding(paths: list[Path], secrets: list[bytes]) -> list[Path]:
    """The files among `paths` that hold any of `secrets`, each read a chunk at a time (a secret
    may straddle two chunks), so that a long log costs little memory."""
    keep = max(map(len, secrets)) - 1
    found = []
    for path in paths:
        if not (path.is_absolute() and path.is_file()):
            raise ProtocolError("invalid_request", f"{path} is not the absolute path of a file")
        tail = b""
        with path.open("rb") as file:
            while chunk := file.read(_CHUNK):
                window = tail + chunk
                if any(secret in window for secret in secrets):
                    found.append(path)
                    break
                tail = window[-keep:] if keep else b""
    return found


def _body(raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw)
    except ValueError:
        raise _Refused(400, "invalid_json", "the request body is not JSON") from None
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        raise _Refused(400, "invalid_request", "send a JSON object with a string 'model'")
    return body


def _trace(request: Request) -> TraceContext | None:
    """The caller's trace, continued; None without a valid `traceparent`."""
    try:
        return TraceContext.from_traceparent(request.headers["traceparent"]).child()
    except (KeyError, ValueError):
        return None
