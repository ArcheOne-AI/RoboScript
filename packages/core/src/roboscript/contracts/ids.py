"""Typed opaque ids and secrets for what crosses a process boundary."""

import secrets
from types import MappingProxyType
from typing import Annotated, Final, NewType

from pydantic import PlainSerializer, SecretStr, SerializationInfo, StringConstraints

EpisodeId = NewType("EpisodeId", str)
SessionId = NewType("SessionId", str)
BlobId = NewType("BlobId", str)
FrameId = NewType("FrameId", str)
ToolName = NewType("ToolName", str)
TaskName = NewType("TaskName", str)
BatchId = NewType("BatchId", str)
MemoryEntryId = NewType("MemoryEntryId", str)
"""A Memory library entry's id (`memory.library.entry_id`): `mem-` and 12 hex digits of its
version's content hash."""
ManifestId = Annotated[str, StringConstraints(pattern=r"^mf-[0-9a-f]{12}$")]
"""A Memory manifest's id: `mf-` and 12 hex digits of its task table's hash."""
PhaseName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
"""A method's own phase id, e.g. `s1`, `p4_reflect` (`contracts.phase.PhaseSpec`): the core has no
fixed list of phases."""
RoleName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
"""A method's own role id, e.g. `actor`, `reviser` (`contracts.phase.RoleSpec`)."""


def new_id(prefix: str) -> str:
    """Return `<prefix>-<12 hex chars>` from `secrets`, e.g. `ep-3f9a0c1b2d4e`."""
    return f"{prefix}-{secrets.token_hex(6)}"


REVEAL: Final = MappingProxyType({"reveal_secrets": True})
"""The only dump context that writes `WireSecret` values into JSON. Only the RPC layer (requests
and responses on the wire) and the MCP integration may dump with it; logs, events and
config hashes never do."""


def _dump_secret(value: SecretStr, info: SerializationInfo) -> str:
    if info.context and info.context.get("reveal_secrets"):
        return value.get_secret_value()
    return str(value)


WireSecret = Annotated[SecretStr, PlainSerializer(_dump_secret, when_used="json")]
"""A token field. JSON dumps write `**********` unless the context is `REVEAL`; python-mode dumps
keep the `SecretStr`, so a model -> dict -> model round trip keeps the token."""
