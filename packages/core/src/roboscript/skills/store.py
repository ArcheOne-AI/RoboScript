"""Frozen skillsets (docs/how-to/add-a-skillset.md, docs/concepts/agent-runtime.md): the
`ArtifactStore` of kind `skillset`, registered in `roboscript.artifact_stores` as `skillset`.

A version is a manifest (`contracts.skills.SkillSet`) under its own `name` and `version`; its
hash is the sha256 of its canonical JSON with default-valued fields left out,
so a field added later never changes a stored hash. One version of a name holds one content: the
same manifest committed again returns its ref, another content under the same version is refused.
A name freezes once, at one version, which is what runs mount. Files: `<root>/<name>/<v>.json`
for version `v`, `versions.json` beside them (the versions in commit order) and `frozen.json` (the
ref and the reason). A version label is a file name, so it is checked like one.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import ClassVar

import anyio
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.skills import SkillSet
from roboscript.records import write_atomic

KIND = SkillSet.kind
FROZEN = "frozen.json"
INDEX = "versions.json"
VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")


class SkillSetStoreConfig(BaseModel):
    """`root`: the store's directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path


def skillset_hash(manifest: SkillSet) -> str:
    """The sha256 of `manifest`'s canonical JSON, default-valued fields left out."""
    data = manifest.model_dump(mode="json", exclude_defaults=True)
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def skillset_ref(manifest: SkillSet) -> ArtifactRef:
    return ArtifactRef(
        kind=KIND, name=manifest.name, version=manifest.version, hash=skillset_hash(manifest)
    )


class SkillSetStore:
    """`ArtifactStore` of kind `skillset` under `root` (see the module)."""

    kind: ClassVar[str] = KIND
    Config: ClassVar[type[SkillSetStoreConfig]] = SkillSetStoreConfig

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    async def commit(self, manifest: SkillSet) -> ArtifactRef:
        """Store `manifest` as version `manifest.version` of `manifest.name`; its ref."""
        ref = skillset_ref(manifest)
        if not VERSION.match(ref.version):
            raise ValueError(f"skillset {ref.name}: version {ref.version!r} is not a file name")
        path = self._path(ref.name, ref.version)
        if path.is_file():
            stored = await self._read(path)
            if skillset_hash(stored) != ref.hash:
                raise ValueError(
                    f"skillset {ref.name} {ref.version} is stored with other content: "
                    "give the new content a new version"
                )
        else:
            data = (manifest.model_dump_json(indent=2, exclude_defaults=True) + "\n").encode()
            await anyio.to_thread.run_sync(write_atomic, path, data)
        order = await self._order(ref.name)
        if ref.version not in order:  # also a version written before a crash could index it
            index = (json.dumps([*order, ref.version], indent=2) + "\n").encode()
            await anyio.to_thread.run_sync(write_atomic, self.root / ref.name / INDEX, index)
        return ref

    async def load(self, ref: ArtifactRef) -> SkillSet:
        _check_kind(ref)
        path = self._path(ref.name, ref.version)
        if not VERSION.match(ref.version) or not path.is_file():
            raise LookupError(f"no skillset {ref.name} {ref.version}")
        manifest = await self._read(path)
        if skillset_hash(manifest) != ref.hash:
            raise LookupError(f"skillset {ref.name} {ref.version} is not {ref.hash[:12]}")
        return manifest

    async def versions(self, name: str) -> list[ArtifactRef]:
        return [
            skillset_ref(await self._read(self._path(name, version)))
            for version in await self._order(name)
        ]

    async def freeze(self, ref: ArtifactRef, *, reason: str) -> ArtifactRef:
        await self.load(ref)  # a version this store holds, as `ref` says
        frozen = await self.frozen(ref.name)
        if frozen is not None:
            if frozen != ref:
                raise ValueError(f"skillset {ref.name} is frozen at {frozen.version} already")
            return frozen
        record = {"ref": ref.model_dump(mode="json"), "reason": reason}
        data = (json.dumps(record, indent=2) + "\n").encode()
        await anyio.to_thread.run_sync(write_atomic, self.root / ref.name / FROZEN, data)
        return ref

    async def frozen(self, name: str) -> ArtifactRef | None:
        path = self.root / name / FROZEN
        if not path.is_file():
            return None
        text = await anyio.to_thread.run_sync(path.read_text, "utf-8")
        return ArtifactRef.model_validate(json.loads(text)["ref"])

    async def _order(self, name: str) -> list[str]:
        path = self.root / name / INDEX
        if not path.is_file():
            return []
        text = await anyio.to_thread.run_sync(path.read_text, "utf-8")
        order: list[str] = json.loads(text)
        return order

    def _path(self, name: str, version: str) -> Path:
        return self.root / name / f"{version}.json"

    async def _read(self, path: Path) -> SkillSet:
        text = await anyio.to_thread.run_sync(path.read_text, "utf-8")
        return SkillSet.model_validate_json(text)


def _check_kind(ref: ArtifactRef) -> None:
    if ref.kind != KIND:
        raise ValueError(f"a {KIND} store holds no {ref.kind} {ref.name}")
