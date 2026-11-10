"""Frozen skillsets: the artifact store conformance suite over `SkillSetStore`, opening it by
name, and one content per version."""

from pathlib import Path

import pytest

from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.skills import Bindings, SkillSet
from roboscript.learning.artifact import load_frozen_ref, open_artifact_store
from roboscript.skills import SkillSetStore
from roboscript.testing.artifacts import ArtifactStoreConformance, ArtifactStoreTarget
from roboscript.testing.skills import MOCK_SKILLS

pytestmark = pytest.mark.unit


def _target(root: Path) -> ArtifactStoreTarget:
    store = SkillSetStore(root)

    async def commit(name: str) -> ArtifactRef:
        version = str(len(await store.versions(name)) + 1)
        return await store.commit(SkillSet(name=name, version=version, layer="built"))

    return ArtifactStoreTarget(store=store, commit=commit)


class TestSkillSetStore(ArtifactStoreConformance):
    @pytest.fixture
    def artifact_store_target(self, tmp_path: Path) -> ArtifactStoreTarget:
        return _target(tmp_path)


async def test_a_version_holds_one_content(tmp_path: Path) -> None:
    store = open_artifact_store("skillset", {"root": str(tmp_path)})
    assert isinstance(store, SkillSetStore)
    manifest = MOCK_SKILLS.manifest
    ref = await store.commit(manifest)
    assert await store.commit(manifest) == ref  # the same content again: the same ref
    changed = manifest.model_copy(update={"bindings": Bindings()})
    with pytest.raises(ValueError, match="stored with other content"):
        await store.commit(changed)
    assert await store.versions(manifest.name) == [ref]
    await store.freeze(ref, reason="validated")
    assert await load_frozen_ref(store, ref) == manifest


async def test_versions_are_file_names(tmp_path: Path) -> None:
    store = SkillSetStore(tmp_path)
    with pytest.raises(ValueError, match="not a file name"):
        await store.commit(SkillSet(name="sneaky", version="../up", layer="built"))
    bad = ArtifactRef(kind="skillset", name="sneaky", version="../up", hash="0" * 64)
    with pytest.raises(LookupError):
        await store.load(bad)
