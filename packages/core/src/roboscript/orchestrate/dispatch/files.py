"""Manifests of file trees (relative path -> size and sha256), the blobs a tree's records name, and
the checks a move relies on: a copy is accepted only when every file hashes as its manifest says,
and a blob only when its bytes hash to its name (the store is content-addressed)."""

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.dispatch import FileEntry, Manifest
from roboscript.contracts.ids import BlobId

TEXT: Final = frozenset({".json", ".jsonl", ".yaml", ".yml", ".md", ".txt"})
"""The records whose text may name a blob (`BlobRef.id`)."""
_HEX: Final = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_CHUNK: Final = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def files_digest(files: dict[str, FileEntry]) -> str:
    table = {name: [entry.size, entry.sha256] for name, entry in files.items()}
    return hashlib.sha256(canonical_json(table).encode("utf-8")).hexdigest()


def tree_files(root: Path, exclude: Iterable[str] = ()) -> list[str]:
    """Every regular file under `root`, relative and sorted, leaving out the top-level names in
    `exclude`; symlinks are refused (a move copies bytes, never links)."""
    if not root.exists():
        return []
    if root.is_file():
        return [""]
    skip = set(exclude)
    found = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in skip:
            continue
        if path.is_symlink():
            raise ValueError(f"{path} is a symlink: a moved tree holds files only")
        if path.is_file():
            found.append(relative.as_posix())
    return found


def blob_path(blobs: Path, blob: str) -> Path:
    """Where `LocalBlobStore` keeps a blob: `<root>/<hex[:2]>/<hex>`."""
    return blobs / blob[:2] / blob


def named_blobs(root: Path, files: Iterable[str], blobs: Path) -> set[str]:
    """The blobs that `files`' text records name and `blobs` holds."""
    found: set[str] = set()
    for name in files:
        path = root / name if name else root
        if path.suffix not in TEXT:
            continue
        for match in _HEX.findall(path.read_bytes()):
            blob = match.decode()
            if blob not in found and blob_path(blobs, blob).is_file():
                found.add(blob)
    return found


def manifest(root: Path, exclude: Iterable[str] = (), blobs: Path | None = None) -> Manifest:
    """`root`'s manifest (a file: one entry named `""`), with the blobs it names in `blobs`."""
    names = tree_files(root, exclude)
    files = {}
    for name in names:
        path = root / name if name else root
        files[name] = FileEntry(size=path.stat().st_size, sha256=sha256_file(path))
    named = sorted(named_blobs(root, names, blobs)) if blobs is not None else []
    return Manifest(files=files, blobs=[BlobId(b) for b in named], digest=files_digest(files))


def path_digest(path: Path) -> str | None:
    """A file's sha256 or a directory's manifest digest; None when it is missing."""
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return manifest(path).digest
    return None


def verify(root: Path, expected: Manifest) -> list[str]:
    """What under `root` does not match `expected` (missing, other size or hash); [] if it
    matches. Extra files are not looked at: the copy names its files."""
    wrong = []
    for name, entry in expected.files.items():
        path = root / name if name else root
        if not path.is_file():
            wrong.append(f"{name}: missing")
        elif path.stat().st_size != entry.size or sha256_file(path) != entry.sha256:
            wrong.append(f"{name}: differs")
    return wrong


def verify_blobs(blobs: Path, names: Iterable[str]) -> list[str]:
    """The blobs that are missing from `blobs` or whose bytes do not hash to their name."""
    wrong = []
    for blob in names:
        path = blob_path(blobs, blob)
        if not path.is_file() or sha256_file(path) != blob:
            wrong.append(blob)
    return wrong
