"""The file store: the conformance suite, the layout on disk, opening stores by name, a failed
commit that leaves nothing behind, task names that cannot leave the root, and refreezing."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from roboscript.kernel import entrypoints
from roboscript.memory import (
    FileMemoryStore,
    MemoryDoc,
    VersionSource,
    open_memory_store,
)
from roboscript.memory import store as memory_store
from roboscript.settings import repo_root
from roboscript.testing.memory import MemoryStoreConformance, MemoryStoreTarget


def _tamper(store: FileMemoryStore, task: str, version: int) -> None:
    path = store.folder(task) / f"v{version:03d}" / "memory.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("put it on the base", "put it on the table"),
        encoding="utf-8",
    )


class TestFileMemoryStore(MemoryStoreConformance):
    @pytest.fixture
    def memory_store_target(self, tmp_path: Path) -> MemoryStoreTarget:
        store = FileMemoryStore(tmp_path)
        return MemoryStoreTarget(store=store, tamper=lambda t, v: _tamper(store, t, v))


async def test_the_layout_on_disk(tmp_path: Path, memory: MemoryDoc) -> None:
    store = FileMemoryStore(tmp_path, now=lambda: datetime(2026, 9, 28, 12, tzinfo=timezone.utc))
    await store.commit(memory, status="s1_draft", source=VersionSource(kind="s1"))
    await store.freeze("lift_red", 0, reason="early stop")
    folder = tmp_path / "lift_red" / "memory"
    assert store.folder("lift_red") == folder
    assert sorted(path.name for path in folder.iterdir()) == ["frozen.json", "v000"]
    version = folder / "v000"
    assert sorted(path.name for path in version.iterdir()) == [
        "memory.md",
        "meta.json",
        "params.json",
    ]
    assert (
        (version / "memory.md")
        .read_text(encoding="utf-8")
        .startswith("# TASK MEMORY: lift_red   (v000, S1, 2026-09-28T12:00:00Z)\n")
    )
    meta = json.loads((version / "meta.json").read_text(encoding="utf-8"))
    assert (meta["schema"], meta["created"], meta["status"]) == (
        "rs.memory.version/1",
        "2026-09-28T12:00:00Z",
        "s1_draft",
    )
    frozen = json.loads((folder / "frozen.json").read_text(encoding="utf-8"))
    assert (frozen["schema"], frozen["version"], frozen["hash"]) == (
        "rs.memory.frozen/1",
        0,
        meta["hash"],
    )


async def test_a_failed_commit_leaves_nothing_behind(
    tmp_path: Path, memory: MemoryDoc, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileMemoryStore(tmp_path)

    def full_disk(directory: Path, doc: MemoryDoc) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(memory_store, "write_doc", full_disk)
    with pytest.raises(OSError, match="no space left"):
        await store.commit(memory, status="s1_draft", source=VersionSource(kind="s1"))
    assert list(store.folder(memory.task).iterdir()) == []
    monkeypatch.undo()
    committed = await store.commit(memory, status="s1_draft", source=VersionSource(kind="s1"))
    assert committed.meta.version == 0


async def test_task_names_cannot_leave_the_root(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path)
    for task in ("../elsewhere", "a/b", ""):
        with pytest.raises(ValueError):
            await store.load(task)


@pytest.mark.skipif(
    "file" not in entrypoints.names("roboscript.memory_stores"),
    reason="the simulator env runs roboscript from source: no entry points",
)
async def test_stores_open_by_name(tmp_path: Path) -> None:
    store = open_memory_store("file", {"root": str(tmp_path)})
    assert isinstance(store, FileMemoryStore) and store.root == tmp_path
    relative = open_memory_store("file", {"root": "runs/memory"})
    assert isinstance(relative, FileMemoryStore) and relative.root == repo_root() / "runs/memory"
    with pytest.raises(LookupError, match="nothing named 'db'"):
        open_memory_store("db")
    with pytest.raises(ValueError, match="root"):
        open_memory_store("file", {})


async def test_refreeze_keeps_the_record_it_replaces(tmp_path: Path, memory: MemoryDoc) -> None:
    """`refreeze` (a pick after the practice stopped) freezes another version in place of the
    frozen one, the earlier record kept aside as `frozen.before<n>.json`; the frozen version
    again stands as it is, nothing moved."""
    store = FileMemoryStore(tmp_path)
    await store.commit(memory, status="s1_draft", source=VersionSource(kind="s1"))
    edited = memory.replace(notes="- a merged note.")
    await store.commit(edited, status="edited", source=VersionSource(kind="edited"), parent=0)
    await store.freeze(memory.task, 1, reason="not converged", summary={"seeds": 3}, vision=True)
    record = store.refreeze(memory.task, 0, reason="not converged", summary={"picked": 0})
    assert (record.version, record.summary) == (0, {"picked": 0})
    assert (await store.frozen(memory.task)) == record
    folder = store.folder(memory.task)
    aside = json.loads((folder / "frozen.before1.json").read_bytes())
    assert (aside["version"], aside["summary"]) == (1, {"seeds": 3})
    assert store.refreeze(memory.task, 0, reason="again") == record  # it stands
    assert sorted(p.name for p in folder.glob("frozen*")) == ["frozen.before1.json", "frozen.json"]
    with pytest.raises(ValueError, match="frozen at v000 already"):
        await store.freeze(memory.task, 1, reason="not converged")
