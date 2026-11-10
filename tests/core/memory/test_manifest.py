"""The Memory manifest: the frozen version of each task an official evaluation runs, named by an
id derived from that table."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from roboscript.memory import (
    FileMemoryStore,
    ManifestEntry,
    MemoryDoc,
    MemoryManifest,
    VersionSource,
    build_manifest,
    load_frozen,
    manifest_id,
    read_manifest,
    write_manifest,
)

S1 = VersionSource(kind="s1")


async def test_a_manifest_names_the_frozen_versions(tmp_path: Path, memory: MemoryDoc) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    for task in ("lift_red", "stack_red_on_blue"):
        await store.commit(memory.replace(task=task), status="s1_draft", source=S1)
    await store.commit(memory, status="edited", source=VersionSource(kind="edited"), parent=0)
    await store.freeze("lift_red", 1, reason="three first-try passes")
    with pytest.raises(LookupError, match="stack_red_on_blue has no frozen Memory"):
        await build_manifest(store, ["lift_red", "stack_red_on_blue"])
    await store.freeze("stack_red_on_blue", 0, reason="three first-try passes")

    manifest = await build_manifest(store, ["lift_red", "stack_red_on_blue"])
    frozen = await store.frozen("lift_red")
    assert frozen is not None and manifest.tasks["lift_red"].hash == frozen.hash
    assert manifest.tasks["lift_red"].version == 1
    assert manifest.id == manifest_id(manifest.tasks) and manifest.id.startswith("mf-")
    assert (await build_manifest(store, ["stack_red_on_blue", "lift_red"])).id == manifest.id

    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    assert read_manifest(path) == manifest
    version = await load_frozen(store, manifest, "lift_red")
    assert version.meta.version == 1 and version.meta.hash == frozen.hash
    with pytest.raises(LookupError, match="has no Memory for lift_blue"):
        await load_frozen(store, manifest, "lift_blue")


async def test_a_manifest_must_match_the_store(tmp_path: Path, memory: MemoryDoc) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    await store.commit(memory, status="s1_draft", source=S1)
    await store.commit(memory, status="edited", source=VersionSource(kind="edited"), parent=0)
    first = (await store.versions(memory.task))[0]
    stale = {memory.task: ManifestEntry(version=0, hash=first.hash)}
    manifest = MemoryManifest(id=manifest_id(stale), created="2026-09-28T00:00:00Z", tasks=stale)
    with pytest.raises(ValueError, match=r"has nothing frozen: build the manifest again"):
        await load_frozen(store, manifest, memory.task)
    await store.freeze(memory.task, 1, reason="manual")
    with pytest.raises(ValueError, match=r"v001 frozen: build the manifest again"):
        await load_frozen(store, manifest, memory.task)
    edited = manifest.model_dump(mode="json", by_alias=True)
    edited["tasks"][memory.task]["version"] = 1
    with pytest.raises(ValidationError, match="does not match its task table"):
        MemoryManifest.model_validate(edited)
