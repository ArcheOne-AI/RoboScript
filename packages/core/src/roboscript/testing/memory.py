"""Conformance checks for Memory stores (docs/concepts/learning.md,
docs/concepts/agent-runtime.md), and a sample Memory.

A store conforms when committed versions come back as committed, numbered from v000 with their
provenance and loadable by number, label or hash; when a cache, an unknown parent, or a merged
version without its parent's merge record is refused; when a version whose files changed after
the commit is refused; and when a task freezes once, at one version, and then takes no more.

Use the checks from pytest by subclassing `MemoryStoreConformance` as a `Test*` class and
providing a `memory_store_target` fixture, or without pytest via `roboscript.testing.precheck`.
Every check is `unit`: none needs an episode.
"""

from collections.abc import Callable
from dataclasses import dataclass

from roboscript.memory import (
    CategoryParam,
    GraspSpec,
    GraspSpecParam,
    Lesson,
    MemoryDoc,
    MemoryStore,
    MergeRecord,
    NumberParam,
    PlaceSpec,
    PlaceSpecParam,
    PlaceTarget,
    VersionSource,
    WorkflowStage,
    content_hash,
)
from roboscript.testing import unit


def sample_memory(task: str = "lift_red") -> MemoryDoc:
    """A lint-clean Memory with every section and parameter kind, as S1 might leave it."""
    return MemoryDoc(
        task=task,
        instruction="Pick up the tool and put it on the base.",
        statement="Put the tool on the base, its head down.",
        objects=(
            "- tool: the item the robot grasps and puts down; a long handle and a head, its x "
            "points to the head.\n"
            "- base: what the tool goes on; a small cube, symmetric every 90 degrees."
        ),
        workflow=[
            WorkflowStage(
                name="grasp_tool",
                goal="hold the tool by its handle, lifted",
                arm="{arm_rule}: the arm on the tool's side of the robot's midline",
                call="grasp_object(arm=A, obj=tool, grasp={grasp_handle}, approach_m={pre_grasp})",
                done="holding(A) and lifted(tool, {lift_min})",
            ),
            WorkflowStage(
                name="set_down",
                goal="bring the tool's head down onto the base",
                arm="the arm holding the tool",
                call="place_object(arm=A, obj=tool, target={place_on_base})",
                done="near(A, base, {place_near}) and inside(tool, base)",
            ),
        ],
        lessons=[
            Lesson(
                id="L1",
                when="the handle points away from the robot",
                do="approach from the top and keep the yaw small",
                seed=10004,
                merges=1,
            )
        ],
        notes="Copy strategies, not numbers: measure in the current scene; pick the arm by where "
        "the tool lies. Across scenes the positions, the tool's yaw and the side it lies on "
        "vary.",
        params={
            "pre_grasp": NumberParam(
                value=0.1, unit="m", bounds=(0.05, 0.15), trust=0.03, source="S1 demo reader"
            ),
            "lift_min": NumberParam(value=0.05, unit="m", bounds=(0.02, 0.15), trust=0.02),
            "place_near": NumberParam(value=0.04, unit="m", bounds=(0.01, 0.1), trust=0.02),
            "arm_rule": CategoryParam(
                value="object_side", choices=["object_side", "holding", "instruction"]
            ),
            "grasp_handle": GraspSpecParam(
                value=GraspSpec(
                    part="handle",
                    approach="top",
                    anchor="top_center",
                    x_toward="head",
                    offset_m=(-0.03, 0.0, -0.01),
                ),
                source="demo left K2",
            ),
            "place_on_base": PlaceSpecParam(
                value=PlaceSpec(
                    target=PlaceTarget(kind="object", obj="base", anchor="top_center"),
                    frame="world_axes",
                    offset_m=(0.0, 0.0, 0.05),
                    placement_point="part:head",
                ),
                source="demo left K4",
            ),
        },
    )


@dataclass(frozen=True)
class MemoryStoreTarget:
    """The store under test; its tasks start with `prefix`, one per check, and must be new.
    `tamper(task, version)`, if given, changes a committed version's files behind the store."""

    store: MemoryStore
    prefix: str = "conformance"
    tamper: Callable[[str, int], None] | None = None


class MemoryStoreConformance:
    """The checks, as `test_*` methods taking a `MemoryStoreTarget`."""

    @unit
    async def test_commit_and_load(self, memory_store_target: MemoryStoreTarget) -> None:
        store, task = memory_store_target.store, f"{memory_store_target.prefix}_commit"
        source = VersionSource(kind="s1", note="demo reader")
        committed = await store.commit(sample_memory(task), status="s1_draft", source=source)
        meta = committed.meta
        assert (meta.task, meta.version, meta.parent, meta.status) == (task, 0, None, "s1_draft")
        assert meta.source == source and meta.hash == content_hash(committed.doc)
        assert committed.doc.version == 0 and committed.doc.created == meta.created
        assert committed.doc.replace(version=None, created=None) == sample_memory(task)
        loaded = await store.load(task)
        assert loaded.doc == committed.doc and loaded.meta == meta

    @unit
    async def test_versions_and_refs(self, memory_store_target: MemoryStoreTarget) -> None:
        store, task = memory_store_target.store, f"{memory_store_target.prefix}_versions"
        first = await store.commit(
            sample_memory(task), status="s1_draft", source=VersionSource(kind="s1")
        )
        merged = first.doc.replace(notes="Measure the hammer again after the lift.")
        record = MergeRecord(
            seed=10001, attempt=2, alpha=0.75, base=first.meta.hash, cache="0" * 64, jitter=False
        )
        source = VersionSource(kind="merge", seed=10001, attempt=2, ref="10001/cache/reviser")
        for parent, bad in (
            (0, None),
            (None, record),
            (0, record.model_copy(update={"base": "b"})),
        ):
            await _raises(ValueError, _merged(store, merged, source, parent, bad))
        jitter = record.model_copy(update={"jitter": True})
        await _raises(ValueError, _merged(store, merged, source, 0, jitter))
        second = await store.commit(
            merged, status="merged", source=source, parent=0, alpha=0.75, budget=3, merge=record
        )
        assert (second.meta.version, second.meta.parent, second.meta.merge) == (1, 0, record)
        assert (second.meta.alpha, second.meta.budget) == (0.75, 3)
        assert [meta.version for meta in await store.versions(task)] == [0, 1]
        assert (await store.load(task)).meta.version == 1
        for ref in (0, "v000", first.meta.hash):
            assert (await store.load(task, ref)).doc == first.doc
        for missing in (7, "v007", "0" * 64):
            await _raises(LookupError, store.load(task, missing))
        await _raises(ValueError, store.load(task, "latest"))
        await _raises(LookupError, store.load(f"{memory_store_target.prefix}_none"))
        assert await store.versions(f"{memory_store_target.prefix}_none") == []
        await _raises(LookupError, _with_parent(store, first.doc, 9))
        third = await store.commit(first.doc, status="edited", source=VersionSource(kind="edited"))
        assert (third.meta.version, third.meta.parent) == (2, None)

    @unit
    async def test_a_cache_is_not_a_version(self, memory_store_target: MemoryStoreTarget) -> None:
        store, task = memory_store_target.store, f"{memory_store_target.prefix}_cache"
        cache = sample_memory(task).replace(scratch=["suspect the handle is short"])
        source = VersionSource(kind="merge")
        await _raises(ValueError, store.commit(cache, status="merged", source=source))
        assert await store.versions(task) == []

    @unit
    async def test_freeze_once(self, memory_store_target: MemoryStoreTarget) -> None:
        store, task = memory_store_target.store, f"{memory_store_target.prefix}_freeze"
        source = VersionSource(kind="s1")
        first = await store.commit(sample_memory(task), status="s1_draft", source=source)
        await store.commit(
            first.doc.replace(notes="Measure twice."), status="edited", source=source
        )
        assert await store.frozen(task) is None
        record = await store.freeze(task, 0, reason="early stop", summary={"seeds": 7})
        assert (record.version, record.hash, record.reason) == (0, first.meta.hash, "early stop")
        assert record.summary == {"seeds": 7}
        assert await store.frozen(task) == record
        assert await store.freeze(task, first.meta.hash, reason="again") == record
        await _raises(ValueError, store.freeze(task, 1, reason="later"))
        assert (await store.load(task, record.hash)).doc == first.doc
        later = store.commit(
            first.doc.replace(notes="After the freeze."), status="edited", source=source
        )
        await _raises(ValueError, later)
        assert [meta.version for meta in await store.versions(task)] == [0, 1]

    @unit
    async def test_changed_files_are_refused(self, memory_store_target: MemoryStoreTarget) -> None:
        if memory_store_target.tamper is None:
            return
        store, task = memory_store_target.store, f"{memory_store_target.prefix}_tamper"
        await store.commit(sample_memory(task), status="s1_draft", source=VersionSource(kind="s1"))
        memory_store_target.tamper(task, 0)
        await _raises(ValueError, store.load(task))


async def _with_parent(store: MemoryStore, doc: MemoryDoc, parent: int) -> None:
    await store.commit(doc, status="edited", source=VersionSource(kind="edited"), parent=parent)


async def _merged(
    store: MemoryStore,
    doc: MemoryDoc,
    source: VersionSource,
    parent: int | None,
    merge: MergeRecord | None,
) -> None:
    await store.commit(doc, status="merged", source=source, parent=parent, merge=merge)


async def _raises(kind: type[Exception], call: object) -> None:
    try:
        await call  # type: ignore[misc]
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")
