"""Task Memory as an artifact: the store conformance suite over the file store, opening it by
name, and a frozen version loading off its manifest with its hash."""

from pathlib import Path

import pytest

from roboscript.contracts.artifact import ArtifactRef
from roboscript.learning.artifact import load_frozen_ref, open_artifact_store
from roboscript.memory import (
    FileMemoryStore,
    VersionSource,
    build_manifest,
    content_hash,
    read_manifest,
    view,
    write_manifest,
)
from roboscript.memory.artifact import MemoryArtifacts, TaskMemory, manifest_stack, memory_ref
from roboscript.testing.artifacts import ArtifactStoreConformance, ArtifactStoreTarget
from roboscript.testing.memory import sample_memory

pytestmark = pytest.mark.unit


def _target(root: Path) -> ArtifactStoreTarget:
    memory = FileMemoryStore(root)
    artifacts = MemoryArtifacts(memory=memory)

    async def commit(name: str) -> ArtifactRef:
        count = len(await memory.versions(name))
        doc = sample_memory(name).replace(notes=f"Version {count}.")
        version = await memory.commit(doc, status="edited", source=VersionSource(kind="edited"))
        return memory_ref(name, version.meta)

    return ArtifactStoreTarget(store=artifacts, commit=commit)


class TestMemoryArtifacts(ArtifactStoreConformance):
    @pytest.fixture
    def artifact_store_target(self, tmp_path: Path) -> ArtifactStoreTarget:
        return _target(tmp_path)


async def test_opened_by_name_and_loaded_from_a_manifest(tmp_path: Path) -> None:
    """A frozen version, opened by name, loads off its manifest with its hash and reads as
    committed."""
    root = tmp_path / "store"
    memory = FileMemoryStore(root)
    task = "stored_task"
    await memory.commit(sample_memory(task), status="edited", source=VersionSource(kind="edited"))
    second = await memory.commit(
        sample_memory(task).replace(notes="Version 1."),
        status="edited",
        source=VersionSource(kind="edited"),
    )
    await memory.freeze(task, 1, reason="early stop")
    write_manifest(tmp_path / "manifest.json", await build_manifest(memory, [task]))
    store = open_artifact_store("task_memory", {"name": "file", "options": {"root": str(root)}})
    assert isinstance(store, MemoryArtifacts)
    stack = manifest_stack(read_manifest(tmp_path / "manifest.json"))
    (layer,) = stack.for_task(task)
    assert layer.task == task and layer.ref.version == "v001"
    loaded = await load_frozen_ref(store, layer.ref)
    assert isinstance(loaded, TaskMemory)
    assert loaded.meta.hash == layer.ref.hash == content_hash(loaded.doc)
    assert view(loaded.doc, detail=False) == view(second.doc, detail=False)
    with pytest.raises(ValueError, match=r"extra|options"):
        open_artifact_store("task_memory", {"root": str(root)})
