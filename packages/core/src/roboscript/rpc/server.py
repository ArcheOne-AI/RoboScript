"""Server side of the typed RPC: routes for an implementation, an app with probes, `serve`,
and `serve_while` for services that warm up after they listen."""

import contextlib
import hmac
import inspect
import logging
import os
import signal
import socket
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import anyio
import uvicorn
from anyio.abc import TaskStatus
from pydantic import BaseModel, SecretStr, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import BaseRoute, Route
from starlette.types import ASGIApp, Lifespan

from roboscript.contracts import versions
from roboscript.contracts.errors import MEDIA_TYPE, ProtocolError, RSError
from roboscript.contracts.ids import REVEAL
from roboscript.contracts.trace import TraceContext, new_trace, use_trace
from roboscript.kernel.signals import on_signals
from roboscript.records import write_atomic
from roboscript.rpc.interface import Method, describe

log = logging.getLogger(__name__)

Allow = Callable[[SecretStr | None, str, dict[str, Any]], bool]
"""Decides a call from its bearer token, method name and validated arguments."""

MAX_BODY: Final = 16 * 1024 * 1024
"""Request bodies are small JSON; bulk data goes through the blob store."""
FAULT_POLL_S: Final = 0.5
"""How often `serve_while` looks whether a ready service failed for good (its `fault`)."""


class ServiceFault(RuntimeError):
    """A ready service failed in a way only a restart mends (`serve_while`'s `fault`): it stopped
    serving, so that its process exits and its supervisor restarts it."""


def routes(
    interface: type,
    impl: object,
    *,
    token: SecretStr | None,
    allow: Allow | None = None,
    max_body: int = MAX_BODY,
) -> list[Route]:
    """`POST /v1/<plane>/<method>` for each method of `interface`, served by `impl`.

    With a token, every call needs `Authorization: Bearer <token>`. With `allow` instead (e.g.
    per-episode tokens), a call needs a bearer token and runs only if `allow(bearer, method,
    arguments)` is true, else it is `unauthorized`; `allow` may also raise an `RSError` (e.g.
    `EpisodeEnded`). A call without the bearer it needs is refused before its body is read, and a
    body over `max_body` bytes is refused. Every call needs a compatible `RS-Contract` header and
    runs under a child of the caller's `traceparent`.
    """
    if token is not None and allow is not None:
        raise TypeError("pass either token or allow, not both")
    spec = describe(interface)
    missing = [n for n in spec.methods if not inspect.iscoroutinefunction(getattr(impl, n, None))]
    if missing:
        raise TypeError(f"{type(impl).__name__} lacks async {spec.name} methods: {missing}")
    plane = spec.contract.plane
    return [
        Route(
            f"/v1/{plane}/{name}",
            _endpoint(spec.contract, method, getattr(impl, name), token, allow, max_body),
            methods=["POST"],
            name=f"{plane}.{name}",
        )
        for name, method in spec.methods.items()
    ]


def _endpoint(
    served: versions.Contract,
    method: Method,
    call: Callable[..., Awaitable[Any]],
    token: SecretStr | None,
    allow: Allow | None,
    max_body: int,
) -> Callable[[Request], Awaitable[Response]]:
    where = f"{served.plane}.{method.name}"

    async def endpoint(request: Request) -> Response:
        try:
            authorize(request, token)
            if allow is not None and _bearer(request) is None:
                raise ProtocolError("unauthorized", "missing bearer token")
            require_contract(request, served)
            args = method.params.validate_json(await _body(request, max_body))
        except RSError as error:
            return problem_response(error)
        except ValidationError as error:
            return problem_response(ProtocolError("invalid_request", _summary(error)))
        trace = _trace(request)
        with use_trace(trace):
            try:
                if allow is not None and not allow(_bearer(request), method.name, args):
                    raise ProtocolError("unauthorized", "missing or wrong bearer token")
                result = await call(**args)
                body = method.result.dump_json(result, by_alias=True, context=REVEAL)
            except RSError as error:
                return problem_response(error)
            except Exception:
                log.exception("%s failed (trace %s)", where, trace.trace_id)
                detail = f"unexpected error in {where} (trace {trace.trace_id}); see the server log"
                return problem_response(RSError("internal", detail))
        return Response(body, media_type="application/json")

    return endpoint


def authorize(request: Request, token: SecretStr | None) -> None:
    """Require `Authorization: Bearer <token>` (constant-time compare) unless `token` is None."""
    if token is None:
        return
    scheme, _, presented = request.headers.get("authorization", "").partition(" ")
    expected = token.get_secret_value().encode()
    if scheme.lower() != "bearer" or not hmac.compare_digest(presented.encode(), expected):
        raise ProtocolError("unauthorized", "missing or wrong bearer token")


async def _body(request: Request, limit: int) -> bytes:
    """The request body, refused past `limit` bytes (by its length header, or as it streams)."""
    too_big = ProtocolError("invalid_request", f"the request body is over {limit} bytes")
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        raise too_big
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise too_big
    return bytes(body)


def _bearer(request: Request) -> SecretStr | None:
    scheme, _, presented = request.headers.get("authorization", "").partition(" ")
    return SecretStr(presented) if scheme.lower() == "bearer" and presented else None


def require_contract(request: Request, served: versions.Contract) -> None:
    """Require an `RS-Contract` header that `served` is compatible with."""
    header = request.headers.get(versions.HEADER)
    if header is None:
        raise ProtocolError("invalid_request", f"send {versions.HEADER}: {served}")
    try:
        sent = versions.parse(header)
    except ValueError as error:
        raise ProtocolError("invalid_request", str(error)) from None
    if sent.plane != served.plane:
        raise ProtocolError("invalid_request", f"{versions.HEADER} {sent} sent to {served}")
    if not versions.compatible(sent, served):
        raise ProtocolError("unsupported_version", f"client speaks {sent}, server {served}")


def problem_response(error: RSError) -> Response:
    problem = error.to_problem()
    headers = {"WWW-Authenticate": "Bearer"} if error.code == "unauthorized" else None
    return Response(
        problem.model_dump_json(), problem.status, headers=headers, media_type=MEDIA_TYPE
    )


def _trace(request: Request) -> TraceContext:
    try:
        parent = TraceContext.from_traceparent(request.headers["traceparent"])
    except (KeyError, ValueError):  # none or invalid: W3C says to start a new trace
        return new_trace()
    return parent.child()


def _summary(error: ValidationError) -> str:
    # Locations and messages only: echoing inputs could leak a token sent in the body.
    parts = error.errors(include_url=False, include_context=False, include_input=False)
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'body'}: {e['msg']}" for e in parts)


def make_app(
    *route_groups: Sequence[BaseRoute],
    ready: Callable[[], bool | Awaitable[bool]] | None = None,
    lifespan: Lifespan[Starlette] | None = None,
) -> Starlette:
    """An app with the given routes, `GET /healthz` (200 while up) and `GET /readyz`.

    `/readyz` answers 200 when `ready()` is true (always without `ready`), else 503. `lifespan`
    runs around serving (e.g. a mounted MCP server's session manager).
    """

    async def healthz(request: Request) -> Response:
        return JSONResponse({"ok": True})

    async def readyz(request: Request) -> Response:
        ok = True if ready is None else ready()
        if inspect.isawaitable(ok):
            ok = await ok
        return JSONResponse({"ready": ok}, 200 if ok else 503)

    probes = [Route("/healthz", healthz), Route("/readyz", readyz)]
    routes = [*probes, *(route for group in route_groups for route in group)]
    return Starlette(routes=routes, lifespan=lifespan)


class Ready(BaseModel):
    """A service's ready file: its primary URL, the URL of each plane served apart, its pid."""

    url: str
    urls: dict[str, str] = {}
    pid: int

    def url_for(self, plane: str) -> str:
        return self.urls.get(plane, self.url)


def write_ready(path: Path, url: str, urls: Mapping[str, str] | None = None) -> None:
    """Write this process's ready file once every app it serves listens."""
    ready = Ready(url=url, urls=dict(urls or {}), pid=os.getpid())
    write_atomic(path, ready.model_dump_json().encode())


def read_ready(path: Path) -> Ready | None:
    """The ready file at `path`, or None while there is none."""
    try:
        return Ready.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None


async def serve(
    app: ASGIApp,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    ready_file: Path | None = None,
    shutdown_timeout_s: float = 5.0,
    task_status: TaskStatus[str] = anyio.TASK_STATUS_IGNORED,
) -> None:
    """Serve `app` until SIGINT, SIGTERM or cancellation, then shut down gracefully and return.

    The socket is bound before uvicorn starts, so port 0 needs no probing. Once listening, the
    URL goes to `task_status` (`url = await tg.start(serve, app)`, e.g. to serve several apps on
    several ports and `write_ready` them together) and, given `ready_file`, into a ready file.
    Shutdown waits `shutdown_timeout_s` for requests in flight, then cancels them. A signal ends
    `serve` normally, so callers' `finally` blocks run; uvicorn's logging config is not installed.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    # IPPROTO_TCP explicitly: asyncio disables Nagle only on sockets that say so, and without
    # that every response written in two parts waits ~40 ms for a delayed ACK.
    with socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        bound = sock.getsockname()[1]
        url = f"http://[{host}]:{bound}" if family == socket.AF_INET6 else f"http://{host}:{bound}"

        def started() -> None:
            if ready_file is not None:
                write_ready(ready_file, url)
            task_status.started(url)

        config = uvicorn.Config(
            app,
            log_config=None,
            timeout_graceful_shutdown=shutdown_timeout_s,  # type: ignore[arg-type]  # a float works
        )
        server = _Server(config, started)
        try:
            await server.serve(sockets=[sock])
        except anyio.get_cancelled_exc_class():
            if server.started:
                with anyio.CancelScope(shield=True):
                    await server.shutdown(sockets=[sock])
            raise


async def serve_while(
    app: ASGIApp,
    start: Callable[[], Awaitable[None]],
    *,
    name: str,
    port: int = 0,
    ready_file: Path | None = None,
    fault: Callable[[], str | None] | None = None,
) -> None:
    """Serve `app` until SIGINT or SIGTERM and, once it listens, `await start()`: a service's
    warm-up, during which its `/readyz` should answer 503.

    The ready file is written as soon as the app listens, so a supervisor finds the service while
    it warms up. A signal also cancels a `start` still running; an error in `start` stops serving
    and then propagates as it is. Given `fault` (what failed for good, None while the service is
    sound), a service that warmed up stops serving once `fault()` names something (looked at every
    `FAULT_POLL_S`; the requests in flight are answered first) and raises `ServiceFault`: a process
    that would only refuse calls exits instead, and its supervisor restarts it. `name` names the
    service in the log.
    """
    failure: Exception | None = None
    async with anyio.create_task_group() as tg:

        async def serve_then_stop(*, task_status: TaskStatus[str]) -> None:
            await serve(app, port=port, ready_file=ready_file, task_status=task_status)
            tg.cancel_scope.cancel()  # a signal: stop a warm-up still running too

        url = await tg.start(serve_then_stop)
        log.info("serving %s at %s; warming up", name, url)
        started = anyio.current_time()
        try:
            await start()
        except Exception as error:  # raised below, outside the group, which would wrap it
            failure = error
            tg.cancel_scope.cancel()
        else:
            log.info("%s ready after %.1f s", name, anyio.current_time() - started)
            if fault is not None:
                while (what := fault()) is None:  # noqa: ASYNC110 - a plain attribute: no event
                    await anyio.sleep(FAULT_POLL_S)
                log.error("%s failed and stops serving, to be restarted: %s", name, what)
                failure = ServiceFault(f"{name} failed: {what}")
                tg.cancel_scope.cancel()
    if failure is not None:
        raise failure


class _Server(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, started: Callable[[], None]) -> None:
        super().__init__(config)
        self._started = started

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            self._started()

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        # uvicorn's own version re-raises the signal after shutting down, killing the process
        # before callers' `finally` blocks run, and with several servers only the last one hears
        # it. Here a signal asks every server of the process to exit, and nothing is re-raised.
        with on_signals(self.handle_exit, (signal.SIGINT, signal.SIGTERM)):
            yield
