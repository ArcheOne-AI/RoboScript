"""Conformance checks for artifact stores (`roboscript.artifact_stores`,
docs/concepts/agent-runtime.md).

A store conforms when a version comes back from its ref as an artifact of the store's kind; when
its versions list oldest first and an unknown name has none; when a ref it does not hold is a
LookupError and a ref of another kind a ValueError; and when a name freezes once, at one version
(again at that version returns it, another is refused), after which `load_frozen_ref` loads the
frozen version and refuses any other. Committing is the kind's own call, so the target brings it.

Use the checks from pytest by subclassing `ArtifactStoreConformance` as a `Test*` class and
providing an `artifact_store_target` fixture, or without pytest via `roboscript.testing.precheck`.
Every check is `unit`.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from roboscript.contracts.artifact import ArtifactRef, ArtifactStore
from roboscript.learning.artifact import load_frozen_ref
from roboscript.testing import unit


@dataclass(frozen=True)
class ArtifactStoreTarget:
    """The store under test and `commit(name)`, which stores a new version of `name` there (a
    different content each call) and returns its ref; names start with `prefix`, one per check,
    and must be new."""

    store: ArtifactStore
    commit: Callable[[str], Awaitable[ArtifactRef]]
    prefix: str = "conformance"


class ArtifactStoreConformance:
    """The checks, as `test_*` methods taking an `ArtifactStoreTarget`."""

    @unit
    async def test_load_by_ref(self, artifact_store_target: ArtifactStoreTarget) -> None:
        store, name = artifact_store_target.store, f"{artifact_store_target.prefix}_load"
        ref = await artifact_store_target.commit(name)
        assert (ref.kind, ref.name) == (store.kind, name)
        artifact = await store.load(ref)
        assert artifact.kind == store.kind
        assert await store.versions(name) == [ref]

    @unit
    async def test_versions_oldest_first(self, artifact_store_target: ArtifactStoreTarget) -> None:
        store, name = artifact_store_target.store, f"{artifact_store_target.prefix}_versions"
        first = await artifact_store_target.commit(name)
        second = await artifact_store_target.commit(name)
        assert first != second and await store.versions(name) == [first, second]
        assert await store.versions(f"{artifact_store_target.prefix}_none") == []

    @unit
    async def test_unknown_and_foreign_refs(
        self, artifact_store_target: ArtifactStoreTarget
    ) -> None:
        store, name = artifact_store_target.store, f"{artifact_store_target.prefix}_refs"
        ref = await artifact_store_target.commit(name)
        await _raises(LookupError, store.load(ref.model_copy(update={"hash": "0" * 64})))
        other = "other_kind" if store.kind != "other_kind" else "another_kind"
        await _raises(ValueError, store.load(ref.model_copy(update={"kind": other})))

    @unit
    async def test_freeze_once(self, artifact_store_target: ArtifactStoreTarget) -> None:
        store, name = artifact_store_target.store, f"{artifact_store_target.prefix}_freeze"
        first = await artifact_store_target.commit(name)
        second = await artifact_store_target.commit(name)
        assert await store.frozen(name) is None
        await _raises(ValueError, load_frozen_ref(store, first))
        assert await store.freeze(first, reason="conformance") == first
        assert await store.frozen(name) == first
        assert await store.freeze(first, reason="again") == first
        await _raises(ValueError, store.freeze(second, reason="later"))
        assert (await load_frozen_ref(store, first)).kind == store.kind
        await _raises(ValueError, load_frozen_ref(store, second))


async def _raises(kind: type[Exception], call: object) -> None:
    try:
        await call  # type: ignore[misc]
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")
