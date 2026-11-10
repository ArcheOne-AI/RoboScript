"""Tokens of a served Hub (docs/concepts/agent-runtime.md).

MCP 2026-07-28 has no protocol sessions, so what a request may see rides on its token: one
short-lived HS256 JWT per episode and purpose, naming the episode (`sub`), its stage, tier, role,
visible tools in order and the hash of the catalog they come from. The types:

- `episode`: a brain's bearer at the MCP endpoint; its calls are the model's (permits apply);
- `service`: the Hub's bearer at a remote toolset server, which calls other toolsets of the
  episode back through the Hub with it, outside the pipeline, while the Hub's call is running.

Each type is signed with its own key derived from `RS_HUB_SECRET`, so neither can pass for the
other, and toolset servers hold only the service key (`RS_HUB_SERVICE_KEY`, `service_key`), which
opens nothing else. Arguments never carry an episode id: the token decides which episode a
request is for. A token is refused once expired or revoked; the Hub revokes an episode's tokens
when it closes the episode. The control plane's bearer is derived from the secret too.
"""

import hashlib
import hmac
import math
import time
from collections.abc import Callable, Collection, Mapping
from typing import Final, Literal

import jwt
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError
from typing_extensions import Self

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.ids import EpisodeId, PhaseName, RoleName, ToolName, new_id
from roboscript.contracts.tools import Tier
from roboscript.settings import EnvSettings

TokenType = Literal["episode", "service"]
ISSUER: Final = "roboscript.hub"
MIN_SECRET: Final = 32  # characters; HS256 wants a key of at least 256 bits
_ALGORITHM: Final = "HS256"


class TokenClaims(BaseModel):
    """What a token says; `iat` and `exp` are Unix seconds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    iss: Literal["roboscript.hub"] = ISSUER
    sub: EpisodeId
    typ: TokenType
    stage: PhaseName
    tier: Tier
    role: RoleName
    tools: list[ToolName]
    catalog: str
    iat: int
    exp: int
    jti: str


class Tokens:
    """Issues and verifies tokens with a key per type (`from_secret` derives both; a toolset
    server holds only the service key); `clock` (Unix seconds) decides expiry. A revoked token
    stays refused until it expires, then it is forgotten."""

    def __init__(
        self, keys: Mapping[TokenType, SecretStr], *, clock: Callable[[], float] = time.time
    ) -> None:
        if short := sorted(
            t for t, key in keys.items() if len(key.get_secret_value()) < MIN_SECRET
        ):
            raise ValueError(f"the {short} token key must have at least {MIN_SECRET} characters")
        self._keys = {typ: key.get_secret_value() for typ, key in keys.items()}
        self.clock = clock
        self._revoked: dict[str, int] = {}  # jti -> exp

    @classmethod
    def from_secret(cls, secret: SecretStr, *, clock: Callable[[], float] = time.time) -> Self:
        if len(secret.get_secret_value()) < MIN_SECRET:
            raise ValueError(f"RS_HUB_SECRET must have at least {MIN_SECRET} characters")
        keys: dict[TokenType, SecretStr] = {
            "episode": _derive(secret, "episode/1"),
            "service": service_key(secret),
        }
        return cls(keys, clock=clock)

    def issue(
        self,
        typ: TokenType,
        *,
        episode: EpisodeId,
        stage: PhaseName,
        tier: Tier,
        role: RoleName,
        tools: list[ToolName],
        catalog: str,
        ttl_s: float,
    ) -> tuple[SecretStr, TokenClaims]:
        now = int(self.clock())
        claims = TokenClaims(
            sub=episode,
            typ=typ,
            stage=stage,
            tier=tier,
            role=role,
            tools=tools,
            catalog=catalog,
            iat=now,
            exp=now + math.ceil(ttl_s),
            jti=new_id("tok"),
        )
        token = jwt.encode(claims.model_dump(), self._keys[typ], algorithm=_ALGORITHM)
        return SecretStr(token), claims

    def verify(self, token: str, types: Collection[TokenType]) -> TokenClaims:
        """The claims of a valid token of one of `types`, checked with that type's key; else
        `ProtocolError("unauthorized")` saying why (never echoing the token)."""
        for typ in types:
            try:
                payload = jwt.decode(
                    token,
                    self._keys[typ],
                    algorithms=[_ALGORITHM],
                    issuer=ISSUER,
                    # PyJWT checks times on the wall clock; expiry is checked on `clock` below.
                    options={"verify_exp": False, "verify_iat": False, "verify_nbf": False},
                )
                claims = TokenClaims.model_validate(payload)
            except (jwt.InvalidTokenError, ValidationError):
                continue
            if claims.typ != typ:
                break  # signed with this type's key, claiming another: forged
            if claims.exp <= self.clock():
                raise ProtocolError("unauthorized", "the token has expired")
            if claims.jti in self._revoked:
                raise ProtocolError("unauthorized", "the token has been revoked")
            return claims
        raise ProtocolError("unauthorized", "not a valid token of this Hub for this endpoint")

    def revoke(self, claims: TokenClaims) -> None:
        now = self.clock()
        self._revoked = {jti: exp for jti, exp in self._revoked.items() if exp > now}
        self._revoked[claims.jti] = claims.exp


def hub_secret() -> SecretStr:
    """`RS_HUB_SECRET` from the environment or `.env`."""
    secret = EnvSettings().hub_secret
    if secret is None:
        raise ValueError("RS_HUB_SECRET is not set: add it to .env (see .env.example)")
    return secret


def hub_service_key() -> SecretStr:
    """`RS_HUB_SERVICE_KEY` from the environment or `.env`: what a toolset server verifies the
    Hub's service tokens with."""
    key = EnvSettings().hub_service_key
    if key is None:
        raise ValueError(
            "RS_HUB_SERVICE_KEY is not set: add service_key(RS_HUB_SECRET) from "
            "roboscript.hub.tokens to the toolset server's .env (see .env.example)"
        )
    return key


def control_token(secret: SecretStr) -> SecretStr:
    """The bearer of the Hub's control plane: derived from `RS_HUB_SECRET`, so the runner, which
    reads the same `.env`, needs no second secret and the secret itself never travels."""
    return _derive(secret, "control/1")


def service_key(secret: SecretStr) -> SecretStr:
    """The key service tokens are signed with, the only one toolset servers get."""
    return _derive(secret, "service/1")


def _derive(secret: SecretStr, label: str) -> SecretStr:
    key = secret.get_secret_value().encode("utf-8")
    message = f"roboscript.hub.{label}".encode()
    return SecretStr(hmac.new(key, message, hashlib.sha256).hexdigest())
