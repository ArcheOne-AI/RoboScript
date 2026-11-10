"""Typed RPC with a toy interface, in-process over ASGI."""

import logging
from collections.abc import AsyncIterator
from typing import ClassVar, Protocol

import anyio
import httpx
import pytest
from pydantic import BaseModel, SecretStr
from starlette.applications import Starlette

from roboscript.contracts.errors import (
    EpisodeEnded,
    InfraError,
    ProtocolError,
    RSError,
    ToolError,
)
from roboscript.contracts.ids import WireSecret
from roboscript.contracts.trace import TraceContext, get_trace
from roboscript.rpc import (
    client,
    make_app,
    routes,
)

TOKEN = SecretStr("t0ken-for-tests")
SECRET = "agent-plane-secret"


class Point(BaseModel):
    x: float
    y: float


class Grant(BaseModel):
    user: str
    token: WireSecret


class Toy(Protocol):
    PLANE: ClassVar[str] = "toy"
    VERSION: ClassVar[str] = "1.2"

    async def add(self, a: int, b: int = 1) -> int: ...
    async def shift(self, point: Point, *, by: float) -> Point: ...
    async def grant(self, user: str) -> Grant: ...
    async def check(self, grant: Grant) -> bool: ...
    async def fail(self, kind: str) -> None: ...
    async def sleep(self, seconds: float) -> None: ...
    async def trace(self) -> TraceContext | None: ...


class ToyImpl:
    def __init__(self) -> None:
        self.sleeping = anyio.Event()

    async def add(self, a: int, b: int = 1) -> int:
        return a + b

    async def shift(self, point: Point, *, by: float) -> Point:
        return Point(x=point.x + by, y=point.y + by)

    async def grant(self, user: str) -> Grant:
        return Grant(user=user, token=SecretStr(SECRET))

    async def check(self, grant: Grant) -> bool:
        return grant.token.get_secret_value() == SECRET

    async def fail(self, kind: str) -> None:
        if kind == "tool":
            raise ToolError("obj_3 is not registered")
        if kind == "protocol":
            raise ProtocolError("not_found", "no episode ep-1")
        if kind == "ended":
            raise EpisodeEnded("closed by the runner")
        if kind == "infra":
            raise InfraError("busy", "an action is running")
        raise RuntimeError("boom at /home/someone/private/path")

    async def sleep(self, seconds: float) -> None:
        self.sleeping.set()
        await anyio.sleep(seconds)

    async def trace(self) -> TraceContext | None:
        return get_trace()


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    app = make_app(routes(Toy, ToyImpl(), token=TOKEN))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://toy") as c:
        yield c


def toy_client(http: httpx.AsyncClient, interface: type[Toy] = Toy, **kw: object) -> Toy:
    options: dict[str, object] = {"token": TOKEN, "timeout": 5.0} | kw
    return client(interface, "http://toy", http=http, **options)


async def test_round_trip(http: httpx.AsyncClient) -> None:
    toy = toy_client(http)
    assert await toy.add(2, 3) == 5
    assert await toy.add(a=2) == 3  # the default is left out and the server's applies
    assert await toy.shift(Point(x=1, y=2), by=0.5) == Point(x=1.5, y=2.5)
    assert await toy.trace() is not None
    with pytest.raises(TypeError):
        await toy.add(1, 2, 3)
    assert "t0ken" not in repr(toy)


async def test_errors_map_back_to_their_exceptions(
    http: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    toy = toy_client(http)
    expected = {
        "tool": (ToolError, "tool_failed", "obj_3 is not registered"),
        "protocol": (ProtocolError, "not_found", "no episode ep-1"),
        "ended": (EpisodeEnded, "episode_ended", "closed by the runner"),
        "infra": (InfraError, "busy", "an action is running"),
    }
    for kind, (cls, code, detail) in expected.items():
        with pytest.raises(cls) as caught:
            await toy.fail(kind)
        assert (type(caught.value), caught.value.code, caught.value.detail) == (cls, code, detail)
    assert InfraError("busy").retryable and not ToolError().retryable

    with caplog.at_level(logging.ERROR, "roboscript.rpc"), pytest.raises(RSError) as caught:
        await toy.fail("unexpected")
    assert type(caught.value) is RSError and caught.value.code == "internal"
    assert "toy.fail" in str(caught.value.detail)
    assert "/home/someone" not in str(caught.value.detail)  # sanitized for the caller ...
    assert "boom at /home/someone" in caplog.text  # ... the traceback stays in the server log
    assert "RuntimeError" in caplog.text


async def test_bearer_token_is_required(http: httpx.AsyncClient) -> None:
    for token in (SecretStr("wrong"), None):
        with pytest.raises(ProtocolError) as caught:
            await toy_client(http, token=token).add(1)
        assert caught.value.code == "unauthorized"
    response = await http.post("/v1/toy/add", json={"a": 1}, headers={"RS-Contract": "toy/1.2"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_secrets_are_revealed_only_on_the_wire() -> None:
    wire: list[bytes] = []

    async def record(response: httpx.Response) -> None:
        wire.append(await response.aread())

    app = make_app(routes(Toy, ToyImpl(), token=TOKEN))
    transport = httpx.ASGITransport(app)
    async with httpx.AsyncClient(transport=transport, event_hooks={"response": [record]}) as http:
        toy = toy_client(http)
        grant = await toy.grant("ann")
        assert grant.token.get_secret_value() == SECRET
        assert SECRET not in grant.model_dump_json()  # default dumps (logs, events) mask it
        assert SECRET in wire[-1].decode()
        assert await toy.check(grant)  # the request carried it too
        assert not await toy.check(Grant(user="ann", token=SecretStr("forged")))


async def test_timeout_is_an_infra_error(http: httpx.AsyncClient) -> None:
    with anyio.fail_after(2), pytest.raises(InfraError) as caught:
        await toy_client(http, timeout=0.1).sleep(10)
    assert caught.value.code == "timeout" and caught.value.retryable


async def status(app: Starlette, path: str) -> int:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://x") as h:
        return (await h.get(path)).status_code
