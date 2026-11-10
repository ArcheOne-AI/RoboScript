"""Memory versions on disk (docs/concepts/learning.md, docs/concepts/operations.md) and
the `roboscript.memory_stores` extension point.

A store keeps each task's main line of versions, numbered from v000 and content-hashed, each with
its parent, status (`s1_draft`, `merged`, `edited`; a frozen version has a freeze record), source
(the first learning phase, a merge's seed and try, an edit committed outside a phase; a reference
to what produced it) and, for a merge, the learning rate, the budget and the merge record, whose
base must be the parent's hash; and at most one freeze record per task: the version the
evaluation reads, after which the task takes no more versions. Caches are not versions; the runner
keeps them under their seed (`write_doc`). Callers lint what they commit (every `Editor` keeps its
document lint-clean).

`FileMemoryStore` (registered as `file`) lays a task out as

    <root>/<task>/memory/v000/{memory.md, params.json, meta.json}
    <root>/<task>/memory/frozen.json

writing a version into a staging directory renamed into place, and checking its content hash on
every load (`meta.hash` stays its identity: a version committed before a parameter field was
added is checked against its files as written). A store plugin is a class whose `Config` model
validates its keyword arguments (`open_memory_store`).
"""

import re
import secrets
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Literal, Protocol

import anyio
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    StringConstraints,
    TypeAdapter,
    field_validator,
)

from roboscript.kernel import entrypoints
from roboscript.memory.cache import MergeRecord
from roboscript.memory.doc import (
    PARAMS_FILE,
    MemoryDoc,
    TaskLabel,
    content_hash,
    read_doc,
    write_doc,
    written_hash,
)
from roboscript.records import replay, write_atomic
from roboscript.settings import repo_root

GROUP: Final = "roboscript.memory_stores"
META_FILE: Final = "meta.json"
FROZEN_FILE: Final = "frozen.json"
FROZEN_ASIDE: Final = "frozen.before"
"""`FileMemoryStore.refreeze` keeps a replaced freeze record as `frozen.before<n>.json`."""
_VERSION_DIR: Final = re.compile(r"v(\d{3,})")
_TIME: Final = "%Y-%m-%dT%H:%M:%SZ"
_TASK: Final = TypeAdapter(TaskLabel)

Status = Literal["s1_draft", "merged", "edited"]
Hash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class VersionSource(BaseModel):
    """Where a version came from: S1, a merge (the seed and try whose cache it merged) or an edit
    committed outside a phase (`edited`); `ref` points at what produced it, e.g. a reviewing
    role's output in the run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["s1", "merge", "edited"]
    seed: int | None = None
    attempt: int | None = None
    ref: str | None = None
    note: str = ""


class VersionMeta(BaseModel):
    """`meta.json`: a version's number, content hash, parent, status and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.memory.version/1"] = Field(default="rs.memory.version/1", alias="schema")
    task: TaskLabel
    version: NonNegativeInt
    hash: Hash
    parent: NonNegativeInt | None = None
    status: Status
    source: VersionSource
    alpha: float | None = None
    budget: NonNegativeInt | None = None
    created: str
    merge: MergeRecord | None = None


class FreezeRecord(BaseModel):
    """`frozen.json`: the version P3 reads, why S2 stopped, a summary of the ledger, and whether
    S2's identification ran GroundingDINO's cross-check (`vision`; P3 must run as S2 did; None:
    unknown, no perception toolset in the practice's process)."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.memory.frozen/1"] = Field(default="rs.memory.frozen/1", alias="schema")
    task: TaskLabel
    version: NonNegativeInt
    hash: Hash
    reason: str
    frozen_at: str
    summary: dict[str, JsonValue] = {}
    vision: bool | None = None


@dataclass(frozen=True)
class Version:
    doc: MemoryDoc
    meta: VersionMeta


VersionRef = int | str | None
"""A version number, a label (`v003`), a content hash, or None for the latest version."""


class MemoryStore(Protocol):
    async def commit(
        self,
        doc: MemoryDoc,
        *,
        status: Status,
        source: VersionSource,
        parent: int | None = None,
        alpha: float | None = None,
        budget: int | None = None,
        merge: MergeRecord | None = None,
    ) -> Version:
        """Store `doc` (not a cache) as its task's next version, stamped with its number and
        creation time; `parent` must be a version of the task."""
        ...

    async def load(self, task: str, ref: VersionRef = None) -> Version:
        """A version, checked against its hash; LookupError if there is none such."""
        ...

    async def versions(self, task: str) -> list[VersionMeta]:
        """The task's versions, oldest first."""
        ...

    async def freeze(
        self,
        task: str,
        ref: VersionRef,
        *,
        reason: str,
        summary: Mapping[str, Any] | None = None,
        vision: bool | None = None,
    ) -> FreezeRecord:
        """Freeze a version for P3 (`vision`: see `FreezeRecord`). Freezing the frozen version
        again returns its record; another version is refused (ValueError)."""
        ...

    async def frozen(self, task: str) -> FreezeRecord | None: ...


class FileStoreConfig(BaseModel):
    """`root`: the store's directory, relative to the repository root unless absolute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path

    @field_validator("root")
    @classmethod
    def _absolute(cls, root: Path) -> Path:
        return root if root.is_absolute() else repo_root() / root


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FileMemoryStore:
    """Versions as files under `root`; `now` stamps them (UTC)."""

    Config: ClassVar[type[BaseModel]] = FileStoreConfig

    def __init__(self, root: Path, *, now: Callable[[], datetime] = _utc_now) -> None:
        self.root = root
        tape = replay.active()  # a replay pins the stamps to the recorded ones
        self.now = now if tape is None else tape.clock("memory.now", now)

    async def commit(
        self,
        doc: MemoryDoc,
        *,
        status: Status,
        source: VersionSource,
        parent: int | None = None,
        alpha: float | None = None,
        budget: int | None = None,
        merge: MergeRecord | None = None,
    ) -> Version:
        commit = partial(
            self._commit,
            doc,
            status=status,
            source=source,
            parent=parent,
            alpha=alpha,
            budget=budget,
            merge=merge,
        )
        return await anyio.to_thread.run_sync(commit)

    async def load(self, task: str, ref: VersionRef = None) -> Version:
        return await anyio.to_thread.run_sync(self._load, task, ref)

    async def versions(self, task: str) -> list[VersionMeta]:
        return await anyio.to_thread.run_sync(self._metas, task)

    async def freeze(
        self,
        task: str,
        ref: VersionRef,
        *,
        reason: str,
        summary: Mapping[str, Any] | None = None,
        vision: bool | None = None,
    ) -> FreezeRecord:
        freeze = partial(self._freeze, task, ref, reason=reason, summary=summary, vision=vision)
        return await anyio.to_thread.run_sync(freeze)

    async def frozen(self, task: str) -> FreezeRecord | None:
        return await anyio.to_thread.run_sync(self._frozen, task)

    def refreeze(
        self,
        task: str,
        ref: VersionRef,
        *,
        reason: str,
        summary: Mapping[str, Any] | None = None,
        vision: bool | None = None,
    ) -> FreezeRecord:
        """Freeze `ref` in place of the version frozen now (a freeze of `ref` stands as it is):
        the earlier record is kept aside as `frozen.before<n>.json`. For a decision taken after
        the practice stopped (a pick among its versions); synchronous, file store only."""
        meta = self._load(task, ref).meta
        existing = self._frozen(task)
        if existing is not None and existing.hash == meta.hash:
            return existing
        if existing is not None:
            path = self.folder(task) / FROZEN_FILE
            n = 1
            while (aside := path.with_name(f"{FROZEN_ASIDE}{n}.json")).exists():
                n += 1
            path.rename(aside)
        return self._freeze(task, ref, reason=reason, summary=summary, vision=vision)

    def folder(self, task: str) -> Path:
        """Where the task's versions live."""
        return self.root / _TASK.validate_python(task) / "memory"

    def _commit(
        self,
        doc: MemoryDoc,
        *,
        status: Status,
        source: VersionSource,
        parent: int | None,
        alpha: float | None,
        budget: int | None,
        merge: MergeRecord | None,
    ) -> Version:
        if doc.scratch is not None:
            raise ValueError("a cache is not a version: merge it into its Memory first")
        if (frozen := self._frozen(doc.task)) is not None:
            raise ValueError(f"{doc.task} is frozen at v{frozen.version:03d}: no more versions")
        metas = {meta.version: meta for meta in self._metas(doc.task)}
        if parent is not None and parent not in metas:
            raise LookupError(f"{doc.task} has no version v{parent:03d} to be the parent")
        if status == "merged":
            _check_merge(parent, merge, None if parent is None else metas[parent])
        number = max(metas, default=-1) + 1
        stamped = doc.replace(version=number, created=self.now().strftime(_TIME))
        meta = VersionMeta(
            task=doc.task,
            version=number,
            hash=content_hash(stamped),
            parent=parent,
            status=status,
            source=source,
            alpha=alpha,
            budget=budget,
            created=stamped.created or "",
            merge=merge,
        )
        folder = self.folder(doc.task)
        staging = folder / f".staging-{secrets.token_hex(6)}"
        staging.mkdir(parents=True)
        try:
            write_doc(staging, stamped)
            write_atomic(staging / META_FILE, (meta.model_dump_json(indent=2) + "\n").encode())
            staging.rename(folder / f"v{number:03d}")  # fails if a concurrent commit took it
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        if (tape := replay.active()) is not None:
            unstamped = content_hash(stamped.replace(created=None))
            tape.observe("memory.commit", {**meta.model_dump(mode="json"), "unstamped": unstamped})
        return Version(stamped, meta)

    def _metas(self, task: str) -> list[VersionMeta]:
        folder = self.folder(task)
        if not folder.is_dir():
            return []
        metas = []
        for entry in folder.iterdir():
            if match := _VERSION_DIR.fullmatch(entry.name):
                meta = VersionMeta.model_validate_json((entry / META_FILE).read_bytes())
                if (meta.task, meta.version) != (task, int(match[1])):
                    raise ValueError(f"{entry}: meta.json says {meta.task} v{meta.version:03d}")
                metas.append(meta)
        return sorted(metas, key=lambda meta: meta.version)

    def _load(self, task: str, ref: VersionRef) -> Version:
        metas = self._metas(task)
        if not metas:
            raise LookupError(f"no Memory versions of {task} under {self.root}")
        meta = _pick(task, metas, ref)
        directory = self.folder(task) / f"v{meta.version:03d}"
        doc = read_doc(directory)
        # schema evolution: a version written before a field was added reads with that field's
        # default, and its files as written still prove it unchanged
        if (found := content_hash(doc)) != meta.hash and meta.hash != written_hash(
            doc, (directory / PARAMS_FILE).read_bytes()
        ):
            raise ValueError(
                f"{task} v{meta.version:03d} changed after its commit (hash {found[:12]}, "
                f"committed {meta.hash[:12]}): commit an edited version instead"
            )
        return Version(doc, meta)

    def _freeze(
        self,
        task: str,
        ref: VersionRef,
        *,
        reason: str,
        summary: Mapping[str, Any] | None,
        vision: bool | None,
    ) -> FreezeRecord:
        meta = self._load(task, ref).meta
        if (existing := self._frozen(task)) is not None:
            if existing.hash == meta.hash:
                return existing
            raise ValueError(f"{task} is frozen at v{existing.version:03d} already")
        record = FreezeRecord(
            task=task,
            version=meta.version,
            hash=meta.hash,
            reason=reason,
            frozen_at=self.now().strftime(_TIME),
            summary=dict(summary or {}),
            vision=vision,
        )
        path = self.folder(task) / FROZEN_FILE
        write_atomic(path, (record.model_dump_json(indent=2) + "\n").encode())
        return record

    def _frozen(self, task: str) -> FreezeRecord | None:
        path = self.folder(task) / FROZEN_FILE
        return FreezeRecord.model_validate_json(path.read_bytes()) if path.is_file() else None


def open_memory_store(name: str = "file", options: Mapping[str, Any] | None = None) -> MemoryStore:
    """The store registered as `name` in `roboscript.memory_stores`, with its options."""
    factory = entrypoints.load(GROUP, name)
    config = factory.Config.model_validate(dict(options or {}))
    store: MemoryStore = factory(**dict(config))
    return store


def _check_merge(parent: int | None, merge: MergeRecord | None, meta: VersionMeta | None) -> None:
    """A merged version names its parent and the merge record of a cache copied from it."""
    if parent is None or merge is None or meta is None:
        raise ValueError("a merged version needs its parent and its merge record")
    if merge.jitter:
        raise ValueError("a jitter merges nothing: there is no version to commit")
    if merge.base != meta.hash:
        raise ValueError(
            f"the merge record's base {merge.base[:12]} is not v{parent:03d} ({meta.hash[:12]}): "
            "merge the cache into the version it was copied from"
        )


def _pick(task: str, metas: list[VersionMeta], ref: VersionRef) -> VersionMeta:
    if ref is None:
        return metas[-1]
    if isinstance(ref, str) and (label := re.fullmatch(r"v(\d+)", ref)):
        ref = int(label[1])
    if isinstance(ref, int):
        found = [meta for meta in metas if meta.version == ref]
    elif re.fullmatch(r"[0-9a-f]{64}", ref):
        found = [meta for meta in metas if meta.hash == ref]
    else:
        raise ValueError(f"{ref!r} is neither a version (3, 'v003') nor a sha256 hex hash")
    if not found:
        shown = ", ".join(f"v{meta.version:03d}" for meta in metas)
        raise LookupError(f"{task} has no Memory version {ref!r}; versions: {shown}")
    return found[-1]
