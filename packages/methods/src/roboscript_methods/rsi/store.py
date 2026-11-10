"""Where RSI documents are kept: a file store of kind `rsi` (`roboscript.artifact_stores`, as
`rsi`), so that an evaluation loads one by reference, frozen, as it loads a Task Memory.

Under `root`, each document name has its versions as `<name>/v<NNN>.json` (the structured form;
its ref's hash is `RsiDoc.content_hash`) and `<name>/v<NNN>.md` (`doc.render`, for people), and
the one frozen as `<name>/frozen.json`. Nothing is overwritten: a version is a new file, a name
is frozen once, and a frozen name takes no new version.

A P4 run keeps one name per round: `G<r>` (`round_name`), committed and frozen at once by
`commit_round` (a round frozen already is refused) and loaded by `load_round`. Where `root` lies
in a run's directory is the launcher's to say.
"""

import json
import os
import re
import threading
from pathlib import Path
from typing import ClassVar, Final

import anyio
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.artifact import ArtifactRef
from roboscript_methods.rsi.doc import KIND, RsiDoc, render

FROZEN: Final = "frozen.json"
ROUND: Final = re.compile(r"^G([1-9][0-9]*)$")


def round_name(round: int) -> str:
    """The document name of round `round` (from 1): `G<round>`."""
    if round < 1:
        raise ValueError(f"rounds count from 1, not {round}")
    return f"G{round}"


class RsiStoreConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path


class FileRsiStore:
    """`ArtifactStore` of kind `rsi` under `root`."""

    kind: ClassVar[str] = KIND
    Config: ClassVar[type[RsiStoreConfig]] = RsiStoreConfig

    def __init__(self, root: Path) -> None:
        self.root = root

    async def commit(self, name: str, doc: RsiDoc) -> ArtifactRef:
        """Store `doc` as `name`'s next version (an unchanged document as a new version too: a
        round that changed nothing is recorded); a frozen name is refused."""
        if (frozen := await self.frozen(name)) is not None:
            raise ValueError(f"rsi {name} is frozen at {frozen.version}: it takes no new version")
        taken = [int(path.stem[1:]) for path in self._dir(name).glob("v[0-9][0-9][0-9].json")]
        version = max(taken, default=-1) + 1
        stem = self._dir(name) / f"v{version:03d}"
        data = (json.dumps(doc.model_dump(mode="json"), indent=2) + "\n").encode()
        # the .md first: a version exists once its .json does
        await anyio.to_thread.run_sync(_create, stem.with_suffix(".md"), render(doc).encode())
        await anyio.to_thread.run_sync(_create, stem.with_suffix(".json"), data)
        return _ref(name, version, doc)

    async def load(self, ref: ArtifactRef) -> RsiDoc:
        _check_kind(ref)
        path = self._dir(ref.name) / f"{ref.version}.json"
        if not path.is_file():
            raise LookupError(f"no rsi {ref.name} {ref.version}")
        doc = RsiDoc.model_validate_json(await anyio.Path(path).read_text(encoding="utf-8"))
        if doc.content_hash() != ref.hash:
            raise LookupError(f"rsi {ref.name} {ref.version} does not have hash {ref.hash}")
        return doc

    async def versions(self, name: str) -> list[ArtifactRef]:
        found = sorted(self._dir(name).glob("v[0-9][0-9][0-9].json"))
        refs = []
        for path in found:
            doc = RsiDoc.model_validate_json(await anyio.Path(path).read_text(encoding="utf-8"))
            refs.append(_ref(name, int(path.stem[1:]), doc))
        return refs

    async def freeze(self, ref: ArtifactRef, *, reason: str) -> ArtifactRef:
        _check_kind(ref)
        if ref not in await self.versions(ref.name):
            raise LookupError(f"no rsi {ref.name} {ref.version} to freeze")
        frozen = await self.frozen(ref.name)
        if frozen is not None and frozen != ref:
            raise ValueError(f"rsi {ref.name} is frozen at {frozen.version} already")
        if frozen is None:
            data = {"ref": ref.model_dump(mode="json"), "reason": reason}
            text = (json.dumps(data, indent=2) + "\n").encode()
            await anyio.to_thread.run_sync(_create, self._dir(ref.name) / FROZEN, text)
        return ref

    async def frozen(self, name: str) -> ArtifactRef | None:
        path = self._dir(name) / FROZEN
        if not path.is_file():
            return None
        data = json.loads(await anyio.Path(path).read_text(encoding="utf-8"))
        return ArtifactRef.model_validate(data["ref"])

    async def commit_round(self, round: int, doc: RsiDoc, *, reason: str) -> ArtifactRef:
        """Commit `doc` as round `round`'s document and freeze it."""
        name = round_name(round)
        if (frozen := await self.frozen(name)) is not None:
            raise ValueError(f"round {round} is frozen already ({name} {frozen.version})")
        return await self.freeze(await self.commit(name, doc), reason=reason)

    async def load_round(self, round: int) -> RsiDoc:
        """Round `round`'s frozen document."""
        frozen = await self.frozen(round_name(round))
        if frozen is None:
            raise LookupError(f"round {round} has no frozen rsi document")
        return await self.load(frozen)

    async def rounds(self) -> list[int]:
        """The rounds with a frozen document, in order."""
        found = [ROUND.match(path.parent.name) for path in self.root.glob(f"G*/{FROZEN}")]
        return sorted(int(match.group(1)) for match in found if match)

    def _dir(self, name: str) -> Path:
        return self.root / name


def _create(path: Path, data: bytes) -> None:
    """Write `path` whole, at once, and only if it does not exist (another writer's version is
    never replaced)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise ValueError(f"{path} exists already: an rsi version is never overwritten") from None
    finally:
        tmp.unlink()


def _ref(name: str, version: int, doc: RsiDoc) -> ArtifactRef:
    return ArtifactRef(kind=KIND, name=name, version=f"v{version:03d}", hash=doc.content_hash())


def _check_kind(ref: ArtifactRef) -> None:
    if ref.kind != KIND:
        raise ValueError(f"an {KIND} store holds no {ref.kind} {ref.name}")
