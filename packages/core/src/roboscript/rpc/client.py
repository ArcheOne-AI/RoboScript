"""Client side of the typed RPC: a proxy object implementing an interface over HTTP."""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar, cast

import anyio
import httpx
from pydantic import SecretStr

from roboscript.contracts import versions
from roboscript.contracts.errors import (
    MEDIA_TYPE,
    InfraError,
    Problem,
    ProtocolError,
    RSError,
    error_from_problem,
)
from roboscript.contracts.ids import REVEAL
from roboscript.contracts.trace import get_trace
from roboscript.rpc.interface import Method, describe
from roboscript.rpc.taped import taped, untaped

T = TypeVar("T")


def client(
    interface: type[T],
    base_url: str,
    *,
    token: SecretStr | None,
    http: httpx.AsyncClient | None = None,
    timeout: float,
    timeouts: Mapping[str, float] | None = None,
) -> T:
    """A proxy whose methods call `interface` at `base_url`.

    Each call must finish within `timeouts[method]` seconds, else `timeout`. Errors come back as
    the exceptions the server raised (see `Transport` for overruns and transport failures).
    Without `http` the proxy opens its own connections; release them with `aclose(proxy)`.
    """
    spec = describe(interface)
    if unknown := sorted(set(timeouts or {}) - set(spec.methods)):
        raise ValueError(f"timeouts name methods that {spec.name} does not have: {unknown}")
    base = base_url.rstrip("/")
    headers = {versions.HEADER: str(spec.contract)}
    if token is not None:
        headers["Authorization"] = f"Bearer {token.get_secret_value()}"
    transport = Transport(http, timeout)
    calls = {
        name: _call(
            transport,
            f"{base}/v1/{spec.contract.plane}/{name}",
            method,
            headers,
            (timeouts or {}).get(name, timeout),
        )
        for name, method in spec.methods.items()
    }
    proxy = cast(T, _Proxy(f"<{spec.name} client for {base}>", transport, calls))
    return taped(interface, proxy)  # the proxy itself unless a replay tape is on


async def aclose(proxy: object) -> None:
    """Close the connections a `client` proxy opened itself; an in-process object has none."""
    proxy = untaped(proxy)
    if isinstance(proxy, _Proxy):
        await proxy._transport.aclose()


class Transport:
    """Sends requests between services, each within a deadline.

    It uses `http` if given, else one `httpx.AsyncClient` of its own, created on first use (for
    one event loop), ignoring proxy settings in the environment, and closed by `aclose()`. An
    overrun raises `InfraError("timeout")`, a transport failure `InfraError("not_ready")`;
    httpx's own timeouts are off since the deadline covers the whole request.
    """

    def __init__(self, http: httpx.AsyncClient | None, timeout: float) -> None:
        self.timeout = timeout
        self._http = http
        self._own: httpx.AsyncClient | None = None

    async def send(
        self,
        method: str,
        url: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - the overrun becomes InfraError("timeout")
        **kwargs: Any,
    ) -> httpx.Response:
        seconds = self.timeout if timeout is None else timeout
        try:
            with anyio.fail_after(seconds):
                return await self._client().request(method, url, timeout=None, **kwargs)
        except TimeoutError:
            raise InfraError("timeout", f"{method} {url}: no answer in {seconds} s") from None
        except httpx.TransportError as error:
            detail = f"{method} {url}: {type(error).__name__}: {error}"
            raise InfraError("not_ready", detail) from error

    async def aclose(self) -> None:
        if self._own is not None:
            await self._own.aclose()
            self._own = None

    def _client(self) -> httpx.AsyncClient:
        if self._http is not None:
            return self._http
        if self._own is None:
            self._own = httpx.AsyncClient(trust_env=False)
        return self._own


class _Proxy:
    def __init__(
        self, text: str, transport: Transport, calls: dict[str, Callable[..., Awaitable[Any]]]
    ) -> None:
        self.__dict__.update(calls)
        self._text = text
        self._transport = transport

    def __repr__(self) -> str:  # never shows the headers, which hold the token
        return self._text


def _call(
    transport: Transport, url: str, method: Method, headers: dict[str, str], timeout: float
) -> Callable[..., Awaitable[Any]]:
    async def call(*args: Any, **kwargs: Any) -> Any:
        arguments = method.signature.bind(*args, **kwargs).arguments
        body = method.params.dump_json(
            method.params.validate_python(arguments), by_alias=True, context=REVEAL
        )
        trace = get_trace()
        sent = headers if trace is None else {**headers, "traceparent": trace.to_traceparent()}
        response = await transport.send("POST", url, content=body, headers=sent, timeout=timeout)
        if response.is_success:
            return method.result.validate_json(response.content)
        raise error_from_response(response)

    call.__name__ = call.__qualname__ = method.name
    return call


def error_from_response(response: httpx.Response) -> RSError:
    """The exception for an error response: its problem, or one made from the status."""
    if response.content and response.headers.get("content-type", "").startswith(MEDIA_TYPE):
        return error_from_problem(Problem.model_validate_json(response.content))
    # Not ours (e.g. no such route or a proxy in between), or the empty body of a HEAD response.
    status = response.status_code
    detail = f"HTTP {status} from {response.request.url}"
    if response.content:
        detail += f": {response.text:.200}"
    if status in (502, 503, 504):
        return InfraError("not_ready", detail)
    if status in (401, 404):
        return ProtocolError("unauthorized" if status == 401 else "not_found", detail)
    if status < 500:
        return ProtocolError("invalid_request", detail)
    return RSError("internal", detail)
