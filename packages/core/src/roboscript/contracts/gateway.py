"""Gateway-Hub contract (docs/concepts/agent-runtime.md): session registration, permits and the
one canonical argument hash.

A served Hub registers each episode's session (= the episode id) on the gateway's admin plane
(`GatewayAdmin`): where to post permits, the token to post them with, the hash of the key that
the session's model requests must carry, and the episode's tools by the hash of their
descriptions. For the tool calls of each model answer in that session the gateway posts
single-use `Permit`s to the Hub's intake (`PermitIntake`), once the answer is complete and before
the client sees its calls; the Hub runs a model's call only by consuming a matching permit.
"""

import hashlib
from typing import Any, ClassVar, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, SecretStr

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.ids import EpisodeId, SessionId, ToolName, WireSecret
from roboscript.contracts.versions import VERSIONS

PermitMode = Literal["strict", "tool_only"]
"""How the Hub matches a call to a permit: tool and `args_hash` (`strict`), or the tool alone."""

MAX_DEPTH: Final = 32
"""Arguments nest at most this deep (strict tool schemas allow 10 levels)."""

UNHASHABLE: Final = (ValueError, UnicodeError, RecursionError)
"""What `args_hash` raises for arguments that are not plain JSON."""


class Permit(BaseModel):
    """A single-use permission for one model-proposed tool call (`call_id`: the model's)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: EpisodeId
    tool: ToolName
    args_hash: str
    call_id: str | None


class SessionRegistration(BaseModel):
    """The Hub tells the gateway where to post permits for a session (= Hub episode id), with
    which token, until when (Unix time), the `session_key_hash` of the key its requests must
    carry, and the episode's tools: Hub tool names by the `description_hash` of their
    descriptions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: SessionId
    permit_url: str
    permit_token: WireSecret
    expires_at: float
    session_key_hash: str
    tools: dict[str, ToolName]


def description_hash(description: str) -> str:
    """sha256 hex of a tool description with runs of whitespace collapsed: clients may rename
    or prefix a tool and re-wrap its description, but pass on its words."""
    return hashlib.sha256(" ".join(description.split()).encode("utf-8")).hexdigest()


def session_key_hash(key: SecretStr) -> str:
    return hashlib.sha256(key.get_secret_value().encode("utf-8")).hexdigest()


def args_hash(args: dict[str, Any]) -> str:
    """sha256 hex of the UTF-8 canonical JSON of tool-call arguments (gateway and Hub alike).

    Integral floats count as integers (`1.0 == 1`), as JCS (RFC 8785) writes numbers: a client
    that parses the model's arguments and serializes them again may turn one into the other.
    Arguments that are not plain JSON (non-finite numbers, lone surrogates, nesting deeper than
    `MAX_DEPTH`) raise one of `UNHASHABLE`: no permit can name them.
    """
    text = canonical_json(_jcs_numbers(args, 0))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jcs_numbers(value: Any, depth: int) -> Any:
    if depth > MAX_DEPTH:
        raise ValueError(f"arguments nest deeper than {MAX_DEPTH} levels")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _jcs_numbers(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_jcs_numbers(item, depth + 1) for item in value]
    return value


class GatewayAdmin(Protocol):
    """The gateway's admin plane (bearer `RS_GATEWAY_ADMIN_TOKEN`), which only the Hub calls."""

    PLANE: ClassVar[str] = "gateway.admin"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def register(self, registration: SessionRegistration) -> None:
        """Post permits for the session's model answers from now on, and refuse its requests
        without the session key (replacing a registration of the same session)."""
        ...

    async def unregister(self, session_id: SessionId) -> None:
        """Stop posting permits for the session (a no-op if it is not registered)."""
        ...


class PermitIntake(Protocol):
    """The Hub's permit intake; each call carries the token its session was registered with."""

    PLANE: ClassVar[str] = "hub.permits"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def issue(self, permits: list[Permit]) -> None:
        """Hold these permits (all of one episode, all of one model answer) until calls consume
        them or they expire; returning is the acknowledgement the gateway waits for."""
        ...
