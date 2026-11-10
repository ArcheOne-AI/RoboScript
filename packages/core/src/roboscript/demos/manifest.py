"""`manifest.json`: the demonstration chosen for each task, why, and the files that hold it.

A choice comes from the ranking (`source` `auto`) or from a review (`reviewed`), whose pick
wins: `status` says whether it confirmed the automatic pick or overrode it. Tasks can be
chosen in separate runs, each entry recording the configuration it was chosen with.
"""

import hashlib
from pathlib import Path
from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import Arm

MANIFEST_SCHEMA: Final = "rs.demos/1"


class FileRef(BaseModel):
    """A file by path (relative to the manifest's directory unless absolute) and its
    `content_sha256`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    episode: NonNegativeInt
    seed: int
    rank: PositiveInt
    score: float
    reasons: list[str]


class TaskDemo(BaseModel):
    """One task's demonstration: the chosen `episode` (the automatic pick is `auto_episode`), its
    score, reasons and review `notes`; `recording` is the episode's file; `candidates` are the
    episodes the review showed, best first; `margin` is how much worse the automatic runner-up
    scored than the automatic pick."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["auto", "reviewed"]
    status: Literal["auto", "confirmed", "overridden"]
    episode: NonNegativeInt
    auto_episode: NonNegativeInt
    seed: int
    category: str | None
    instruction: str
    score: float
    reasons: list[str]
    margin: float | None
    notes: list[str]
    keypoint_counts: dict[Arm, NonNegativeInt]
    config_hash: str
    recording: FileRef
    demo: list[FileRef]
    keypoints: FileRef
    candidates: list[Candidate]


class DemoManifest(BaseModel):
    """`categories` names where the task categories came from; `created` is the last update."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.demos/1"] = Field(default=MANIFEST_SCHEMA, alias="schema")
    created: str
    source_root: str
    categories: str
    tasks: dict[TaskName, TaskDemo]


def content_sha256(path: Path) -> str:
    """sha256 of a file's bytes; of an .npz, of its arrays (name, dtype, shape and bytes, in name
    order), which the zip's own timestamps do not change."""
    digest = hashlib.sha256()
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            for name in sorted(npz.files):
                arr = np.ascontiguousarray(npz[name])
                digest.update(f"{name}|{arr.dtype.str}|{arr.shape}|".encode())
                digest.update(arr.tobytes())
        return digest.hexdigest()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path, root: Path) -> FileRef:
    """`path` relative to `root` when inside it, with its `content_sha256`."""
    shown = path.relative_to(root) if path.is_relative_to(root) else path
    return FileRef(path=shown.as_posix(), sha256=content_sha256(path))
