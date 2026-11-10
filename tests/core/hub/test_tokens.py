"""The Hub's tokens (docs/concepts/agent-runtime.md): what they say, a key per type, and every way
one is refused."""

import base64
import json

import jwt
import pytest
from pydantic import SecretStr

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.ids import EpisodeId, ToolName
from roboscript.hub.tokens import (
    TokenClaims,
    Tokens,
    TokenType,
    control_token,
    service_key,
)

SECRET = SecretStr("s" * 40)


class Clock:
    def __init__(self) -> None:
        self.now = 1_790_000_000.0

    def __call__(self) -> float:
        return self.now


def issue(
    tokens: Tokens, typ: TokenType = "episode", episode: str = "ep-a"
) -> tuple[str, TokenClaims]:
    token, claims = tokens.issue(
        typ,
        episode=EpisodeId(episode),
        stage="s2",
        tier="D",
        role="actor",
        tools=[ToolName("observe"), ToolName("finish")],
        catalog="c" * 64,
        ttl_s=600,
    )
    return token.get_secret_value(), claims


def refusal(tokens: Tokens, token: str, *types: TokenType) -> str:
    with pytest.raises(ProtocolError) as caught:
        tokens.verify(token, types or ("episode",))
    assert caught.value.code == "unauthorized"
    assert token not in str(caught.value)
    return str(caught.value.detail)


def b64(data: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def test_a_token_says_what_its_episode_sees() -> None:
    clock = Clock()
    tokens = Tokens.from_secret(SECRET, clock=clock)
    token, claims = issue(tokens)
    assert tokens.verify(token, ("episode",)) == claims
    assert (claims.sub, claims.typ, claims.stage, claims.tier, claims.role) == (
        "ep-a",
        "episode",
        "s2",
        "D",
        "actor",
    )
    assert claims.tools == ["observe", "finish"] and claims.catalog == "c" * 64
    assert (claims.iat, claims.exp) == (1_790_000_000, 1_790_000_600)
    assert issue(tokens)[1].jti != claims.jti
    assert json.loads(base64.urlsafe_b64decode(token.split(".")[0] + "=="))["alg"] == "HS256"


def test_each_type_has_its_own_key() -> None:
    clock = Clock()
    hub = Tokens.from_secret(SECRET, clock=clock)
    toolset = Tokens({"service": service_key(SECRET)}, clock=clock)  # what a server holds
    episode_token, _ = issue(hub, "episode")
    service_token, _ = issue(hub, "service")
    assert toolset.verify(service_token, ("service",)).typ == "service"
    assert hub.verify(service_token, ("episode", "service")).typ == "service"
    assert "not a valid token" in refusal(toolset, episode_token, "service")
    assert "not a valid token" in refusal(hub, service_token, "episode")
    with pytest.raises(KeyError):  # the server cannot make episode tokens
        issue(toolset, "episode")
    # A toolset server holds the service key: whatever it signs with it is no episode token.
    claims = json.loads(base64.urlsafe_b64decode(service_token.split(".")[1] + "=="))
    minted = jwt.encode({**claims, "typ": "episode"}, service_key(SECRET).get_secret_value())
    assert "not a valid token" in refusal(hub, minted, "episode", "service")


def test_expired_tampered_and_foreign_tokens_are_refused() -> None:
    clock = Clock()
    tokens = Tokens.from_secret(SECRET, clock=clock)
    token, _ = issue(tokens)
    head, payload, signature = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=="))
    for forged in (
        {**claims, "sub": "ep-b"},  # another episode
        {**claims, "exp": claims["exp"] + 3600},  # a longer life
        {**claims, "role": "reviser"},  # another role
    ):
        assert refusal(tokens, f"{head}.{b64(forged)}.{signature}", "episode", "service")
    other = Tokens.from_secret(SecretStr("o" * 40), clock=clock)
    assert "not a valid token" in refusal(other, token)
    assert "not a valid token" in refusal(tokens, "not-a-token")
    unsigned = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{payload}."
    assert "not a valid token" in refusal(tokens, unsigned)
    clock.now += 600
    assert "expired" in refusal(tokens, token)


def test_revoked_tokens_stay_refused_until_they_expire() -> None:
    clock = Clock()
    tokens = Tokens.from_secret(SECRET, clock=clock)
    token, claims = issue(tokens)
    kept, _ = issue(tokens)
    tokens.revoke(claims)
    assert "revoked" in refusal(tokens, token)
    tokens.verify(kept, ("episode",))  # other tokens of the episode are separate
    clock.now += 601
    tokens.revoke(issue(tokens)[1])  # revoking forgets tokens that expired meanwhile
    assert claims.jti not in tokens._revoked
    assert "expired" in refusal(tokens, token)


def test_keys_are_long_and_derived_apart() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        Tokens.from_secret(SecretStr("short"))
    with pytest.raises(ValueError, match="at least 32"):
        Tokens({"service": SecretStr("short")})
    derived = [control_token(SECRET), service_key(SECRET)]
    values = {key.get_secret_value() for key in derived}
    assert len(values) == 2 and SECRET.get_secret_value() not in "".join(values)
    assert all(len(value) == 64 for value in values)
    assert control_token(SECRET).get_secret_value() == derived[0].get_secret_value()
    other = control_token(SecretStr("o" * 40)).get_secret_value()
    assert other != derived[0].get_secret_value()
