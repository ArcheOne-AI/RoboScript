"""Where an episode result came from: the code that ran it and the behaviour config it
ran with. One definition for every result a phase writes (each `summary.json` and `result.json`), so
that a block or a ledger mixing code
versions or configs shows it episode by episode."""

from typing import Final

from pydantic import BaseModel, ConfigDict

SOURCE_ENV: Final = "RS_SOURCE_SHA"
"""The code's revision name, as a launcher sets it (`roboscript.settings.source_sha`)."""
SOURCE_FILE: Final = ".roboscript/source.sha"
"""The file under the repository root holding the revision, read when `SOURCE_ENV` is unset."""


class Provenance(BaseModel):
    """`source`: the code's revision (None: unknown); `config`: the behaviour config's name (a
    pipeline's variant, say) and `config_hash` the sha256 of its content (None: no launcher named
    one)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str | None = None
    config: str | None = None
    config_hash: str | None = None
