"""The `allow` callback of `rpc.routes`: per-call authorization such as per-episode tokens."""

import hmac
from collections.abc import AsyncIterator
from typing import Any, ClassVar, Protocol

import httpx
import pytest
from pydantic import SecretStr

from roboscript.contracts.errors import EpisodeEnded, ProtocolError
from roboscript.rpc import client, make_app, routes

TOKENS = {"ep-a": "token-a", "ep-b": "token-b"}
ENDED = {"ep-old": "token-old"}


class Episodes(Protocol):
    PLANE: ClassVar[str] = "episodes"
    VERSION: ClassVar[str] = "1.0"

    async def step(self, episode_id: str, n: int = 1) -> str: ...


class EpisodesImpl:
    async def step(self, episode_id: str, n: int = 1) -> str:
        return f"{episode_id} +{n}"


calls: list[tuple[str, dict[str, Any]]] = []


def matches(bearer: SecretStr | None, expected: str | None) -> bool:
    return (
        bearer is not None
        and expected is not None
        and hmac.compare_digest(bearer.get_secret_value(), expected)
    )


def allow(bearer: SecretStr | None, method: str, args: dict[str, Any]) -> bool:
    calls.append((method, args))
    episode = args["episode_id"]
    if matches(bearer, ENDED.get(episode)):
        raise EpisodeEnded(f"{episode} is closed")
    return matches(bearer, TOKENS.get(episode))


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    calls.clear()
    app = make_app(routes(Episodes, EpisodesImpl(), token=None, allow=allow))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://x") as c:
        yield c


def episodes(http: httpx.AsyncClient, token: str | None) -> Episodes:
    secret = None if token is None else SecretStr(token)
    return client(Episodes, "http://x", token=secret, http=http, timeout=5)


async def test_each_token_opens_only_its_episode(http: httpx.AsyncClient) -> None:
    assert await episodes(http, "token-a").step("ep-a", n=2) == "ep-a +2"
    assert await episodes(http, "token-b").step("ep-b") == "ep-b +1"
    assert calls[0] == ("step", {"episode_id": "ep-a", "n": 2})
    for token, episode in [("token-a", "ep-b"), ("wrong", "ep-a"), (None, "ep-a")]:
        with pytest.raises(ProtocolError) as caught:
            await episodes(http, token).step(episode)
        assert caught.value.code == "unauthorized"


async def test_bodies_over_the_cap_are_refused() -> None:
    app = make_app(routes(Episodes, EpisodesImpl(), token=None, allow=allow, max_body=64))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://x") as http:
        headers = {"RS-Contract": "episodes/1.0", "Authorization": "Bearer token-a"}
        small = await http.post("/v1/episodes/step", json={"episode_id": "ep-a"}, headers=headers)
        big = await http.post(
            "/v1/episodes/step", json={"episode_id": "ep-a", "pad": "x" * 100}, headers=headers
        )
    assert small.status_code == 200
    assert big.status_code == 400 and "over 64 bytes" in big.json()["detail"]
