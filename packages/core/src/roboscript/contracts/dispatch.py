"""What the dispatch coordinator and a worker's agent exchange (`roboscript.orchestrate.dispatch`,
docs/concepts/operations.md): one JSON request on the agent's stdin, one
`AgentReply` on its stdout.

A unit is one launch of a pipeline (a run directory) on one worker with a subset of the tasks;
the agent starts, watches, stops and drains it, and moves a finished task's files out by
manifest, so that the coordinator's run root holds the one copy.
"""

from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.ids import BlobId

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _relative(path: str) -> str:
    if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
        raise ValueError(f"{path!r}: a relative path inside its directory")
    return path


RelPath = Annotated[str, AfterValidator(_relative)]
"""A path relative to a unit or code directory that cannot climb out of it."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UnitPaths(_Model):
    """A unit's directory on its worker, which must lie inside the worker's run root: the agent
    refuses to write or delete anywhere else."""

    run_root: str
    unit_dir: str

    @model_validator(mode="after")
    def _inside(self) -> Self:
        root, unit = PurePosixPath(self.run_root), PurePosixPath(self.unit_dir)
        if not root.is_absolute() or not unit.is_absolute():
            raise ValueError("run_root and unit_dir must be absolute")
        if ".." in unit.parts or unit == root or root not in unit.parents:
            raise ValueError(f"{self.unit_dir} is not inside the run root {self.run_root}")
        return self


class FileEntry(_Model):
    size: NonNegativeInt
    sha256: Sha256


class Manifest(_Model):
    """A tree's files (relative path -> size and sha256), the blobs its text records name that
    the unit's blob store holds, and `digest`: the sha256 of the canonical file table, which
    names this exact content."""

    files: dict[str, FileEntry]
    blobs: list[BlobId] = []
    digest: Sha256


class StartRequest(_Model):
    """Start (or, when its process is gone, resume) a unit: the agent checks the code version
    (`sha`, a prefix of the frozen revision), the config (`provenance`: the hash of `config`
    without `local_keys`) and the pinned files (path -> digest), writes
    `config_file` once and launches `argv` (`{dir}` is the unit directory) detached, in a
    session of its own, with `RS_GATEWAY_URL` set to `gateway_url` when given. A unit whose
    recorded provenance differs is refused."""

    unit: UnitPaths
    sha: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    provenance: Sha256
    config: dict[str, Any]
    config_file: RelPath
    local_keys: list[str]
    argv: list[str] = Field(min_length=1)
    env: dict[str, str] = {}
    pinned: dict[RelPath, Sha256] = {}
    gateway_url: str | None = None
    tasks: list[str] = Field(min_length=1)


class UnitRecord(_Model):
    """`.dispatch/unit.json` in a unit directory: who runs it (the exact pid and its start
    token, so that a reused pid is never taken for it), the version and config it runs."""

    pid: int
    start_token: str
    sha: str
    provenance: Sha256
    tasks: list[str]
    launches: int = 1
    launched_at: str


class StartReply(_Model):
    pid: int
    started: bool
    """False: the unit's process was alive already (nothing started)."""
    launches: int


class StatusRequest(_Model):
    unit: UnitPaths
    status_files: list[RelPath] = Field(min_length=1)


class StatusReply(_Model):
    """Whether the unit's recorded process is alive, and the newest of its status files: the
    pipeline's `rows` (each with `run` = `<task>/<variant>...` and `status`) and `runs`."""

    exists: bool
    alive: bool
    pid: int | None = None
    launches: int = 0
    rows: list[dict[str, Any]] = []
    updated: str | None = None


class StopRequest(_Model):
    unit: UnitPaths
    grace_s: PositiveFloat = 120.0


class StopReply(_Model):
    pid: int | None
    was_alive: bool
    killed: bool
    """True when the grace ran out and the recorded process tree was killed."""


class ManifestRequest(_Model):
    """The manifest of `path` in the unit directory (`""`: the unit itself), leaving out the
    top-level names in `exclude`; `blobs`: the unit's blob store, whose blobs the tree's text
    records name are listed."""

    unit: UnitPaths
    path: str = ""
    exclude: list[str] = []
    blobs: RelPath | None = None


class RemoveRequest(_Model):
    """Delete a moved task's tree if it still has `digest`, keeping the files that match `keep`
    (globs relative to the task directory: the tombstones that keep a resumed launch from
    running the task again); then the blobs it named that no other tree of the unit names and
    that hold at least `blob_min_bytes` (None: none; the rest go when the unit drains)."""

    unit: UnitPaths
    task: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    digest: Sha256
    keep: list[str] = []
    blobs: RelPath | None = None
    blob_min_bytes: NonNegativeInt | None = None


class RemoveReply(_Model):
    files: int
    blobs: int


class PurgeRequest(_Model):
    """Delete a drained unit's directory (refused while its process is alive)."""

    unit: UnitPaths


class DigestRequest(_Model):
    """The digests of paths relative to the code directory (None: missing)."""

    paths: list[RelPath]


class AgentReply(_Model):
    ok: bool
    error: str | None = None
    kind: Literal["refused", "failed"] | None = None
    reply: dict[str, Any] | None = None
