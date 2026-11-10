"""Task Memory as an artifact (docs/concepts/agent-runtime.md, docs/how-to/add-a-method.md):
`TaskMemory` (kind `task_memory`), its refs, the artifact store over a Memory store
(`MemoryArtifacts`, registered in `roboscript.artifact_stores` as `task_memory`), and an evaluation
manifest as an artifact stack.

A Task Memory ref is `(task_memory, <task>, v<NNN>, <content hash>)`, read off a version's meta,
a freeze record or a manifest entry; the stores, files and hashes are the Memory store's
(`memory.store`). Versions are committed through the Memory store, which records their
provenance (`contracts.artifact` says why commit is not part of the artifact protocol).

```python
artifacts = open_artifact_store("task_memory", {"name": "file", "options": {"root": "runs/x"}})
(layer,) = manifest_stack(read_manifest(path)).for_task("lift_red")
memory = await load_frozen_ref(artifacts, layer.ref)  # a TaskMemory; .version for the host
```
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Final

from roboscript.contracts.artifact import Artifact, ArtifactLayer, ArtifactRef, ArtifactStack
from roboscript.contracts.ids import TaskName
from roboscript.memory.config import StoreSpec
from roboscript.memory.doc import MemoryDoc
from roboscript.memory.manifest import ManifestEntry, MemoryManifest
from roboscript.memory.store import (
    FreezeRecord,
    MemoryStore,
    Version,
    VersionMeta,
    open_memory_store,
)

KIND: Final = "task_memory"


class TaskMemory(Artifact):
    """One Task Memory version: the document and its meta (number, hash, provenance)."""

    kind: ClassVar[str] = KIND

    doc: MemoryDoc
    meta: VersionMeta

    @property
    def version(self) -> Version:
        return Version(self.doc, self.meta)


def memory_ref(task: str, entry: VersionMeta | FreezeRecord | ManifestEntry) -> ArtifactRef:
    """The ref of `task`'s version that `entry` names."""
    return ArtifactRef(kind=KIND, name=task, version=f"v{entry.version:03d}", hash=entry.hash)


def manifest_stack(manifest: MemoryManifest) -> ArtifactStack:
    """An evaluation manifest as a stack: each task's frozen Memory, for that task only."""
    return ArtifactStack(
        layers=tuple(
            ArtifactLayer(ref=memory_ref(task, entry), task=TaskName(task))
            for task, entry in manifest.tasks.items()
        )
    )


class MemoryArtifacts:
    """`ArtifactStore` of kind `task_memory` over the Memory store `store` names (with its
    options), or over `memory` itself."""

    kind: ClassVar[str] = KIND
    Config: ClassVar[type[StoreSpec]] = StoreSpec

    def __init__(
        self,
        name: str = "file",
        options: Mapping[str, Any] | None = None,
        *,
        memory: MemoryStore | None = None,
    ) -> None:
        self.memory = memory if memory is not None else open_memory_store(name, options)

    async def load(self, ref: ArtifactRef) -> TaskMemory:
        _check_kind(ref)
        version = await self.memory.load(ref.name, ref.hash)
        return TaskMemory(doc=version.doc, meta=version.meta)

    async def versions(self, name: str) -> list[ArtifactRef]:
        return [memory_ref(name, meta) for meta in await self.memory.versions(name)]

    async def freeze(self, ref: ArtifactRef, *, reason: str) -> ArtifactRef:
        _check_kind(ref)
        return memory_ref(ref.name, await self.memory.freeze(ref.name, ref.hash, reason=reason))

    async def frozen(self, name: str) -> ArtifactRef | None:
        record = await self.memory.frozen(name)
        return None if record is None else memory_ref(name, record)


def _check_kind(ref: ArtifactRef) -> None:
    if ref.kind != KIND:
        raise ValueError(f"a {KIND} store holds no {ref.kind} {ref.name}")
