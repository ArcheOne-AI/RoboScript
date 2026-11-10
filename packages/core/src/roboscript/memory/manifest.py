"""The Memory manifest (docs/concepts/learning.md): the frozen Memory version of each task that
a benchmark's evaluation runs. Its id derives from that table, so the same frozen versions always
get the same id, and a benchmark's evaluation client can carry it and name its results after it.

```python
manifest = await build_manifest(store, tasks)  # every task must be frozen
write_manifest(path, manifest)
version = await load_frozen(store, read_manifest(path), "lift_red")
```
"""

import hashlib
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator
from typing_extensions import Self

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.ids import ManifestId
from roboscript.memory.doc import TaskLabel
from roboscript.memory.store import Hash, MemoryStore, Version
from roboscript.records import write_atomic


class ManifestEntry(BaseModel):
    """A task's frozen version, and whether S2 identified objects with the vision service's
    cross-check (`FreezeRecord.vision`), which P3 must then have too."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: NonNegativeInt
    hash: Hash
    vision: bool | None = None


class MemoryManifest(BaseModel):
    """`id` is `manifest_id(tasks)`; a table edited by hand does not match it."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.memory.manifest/1"] = Field(default="rs.memory.manifest/1", alias="schema")
    id: ManifestId
    created: str
    tasks: dict[TaskLabel, ManifestEntry]

    @model_validator(mode="after")
    def _id_names_the_table(self) -> Self:
        if self.id != manifest_id(self.tasks):
            raise ValueError(
                f"id {self.id} does not match its task table ({manifest_id(self.tasks)})"
            )
        return self


def manifest_id(tasks: Mapping[str, ManifestEntry]) -> ManifestId:
    """`mf-` and the first 12 hex digits of the sha256 of the canonical task table (fields that
    are None left out, so that a table without them keeps its id)."""
    table = {
        task: entry.model_dump(mode="json", exclude_none=True) for task, entry in tasks.items()
    }
    return "mf-" + hashlib.sha256(canonical_json(table).encode()).hexdigest()[:12]


async def build_manifest(store: MemoryStore, tasks: Iterable[str]) -> MemoryManifest:
    """The frozen version of each task; LookupError names a task that is not frozen."""
    entries = {}
    for task in tasks:
        record = await store.frozen(task)
        if record is None:
            raise LookupError(f"{task} has no frozen Memory: freeze it before an evaluation")
        entries[task] = ManifestEntry(
            version=record.version, hash=record.hash, vision=record.vision
        )
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return MemoryManifest(id=manifest_id(entries), created=created, tasks=entries)


def write_manifest(path: Path, manifest: MemoryManifest) -> None:
    write_atomic(path, (manifest.model_dump_json(indent=2) + "\n").encode())


def read_manifest(path: Path) -> MemoryManifest:
    return MemoryManifest.model_validate_json(path.read_bytes())


async def load_frozen(store: MemoryStore, manifest: MemoryManifest, task: str) -> Version:
    """The version `manifest` names for `task`; it must still be the task's frozen version."""
    entry = manifest.tasks.get(task)
    if entry is None:
        raise LookupError(f"manifest {manifest.id} has no Memory for {task}")
    record = await store.frozen(task)
    if record is None or record.hash != entry.hash:
        frozen = "nothing" if record is None else f"v{record.version:03d}"
        raise ValueError(
            f"manifest {manifest.id} names {task} v{entry.version:03d}, but the store has "
            f"{frozen} frozen: build the manifest again"
        )
    return await store.load(task, entry.hash)
