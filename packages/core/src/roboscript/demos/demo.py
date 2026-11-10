"""The demonstration format: one recorded episode's per-frame arm tracks and camera calibration.

A demo is a directory holding `demo.json` (`DemoMeta`) and `arrays.npz` (never pickled). Frames
are the recording's samples, as it saved them: every `frame_steps` physics steps, plus whatever
else the recorder saves (it may also save one at each action's start, repeating the frame
before, and one after the action's first physics step; `keypoints.timed_frames`). Poses are
7-vectors [x, y, z, qw, qx, qy, qz] in the world frame; grippers are commands, normalized (1
open, 0 closed). The TCP (fingertip midpoint) is the ee pose moved by the embodiment's ee->TCP
offset in the ee frame, with the ee's orientation. Images stay where they were recorded
(`FrameSource`).

A re-recording replays a recorded trajectory, one action per frame it sends, and saves a frame
after each action (`replay` names the choice of frames sent; `source_frames` maps each frame to
the trajectory's frame whose command it sent, frame 0 being the start). Its frames count at the
trajectory's cadence (`frame_steps`), so that speeds read as in the original; `physics_steps` has
the physics steps each frame actually came after (from the start). It adds the measured joints
and finger opening and every frame's observation: `frames.jsonl`, one frame-store entry per frame
(`FrameEntry`, aliased `frame_alias(frame)`), with the blobs in the blob store `blobs/`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from roboscript.contracts.geometry import Vec3
from roboscript.contracts.ids import TaskName
from roboscript.contracts.observation import FrameEntry
from roboscript.contracts.robot import Arm
from roboscript.records import write_atomic
from roboscript.spatial.pose import pose_compose

Array = NDArray[np.float64]

SCHEMA: Final = "rs.demo/1"
META_FILE: Final = "demo.json"
ARRAYS_FILE: Final = "arrays.npz"
FRAMES_FILE: Final = "frames.jsonl"
BLOBS_DIR: Final = "blobs"
_OPTIONAL: Final = ("joints", "joints_measured", "opening")  # ArmTrack's optional arrays
_FRAME_ARRAYS: Final = ("source_frames", "physics_steps")  # Demo's optional ones


class FrameSource(BaseModel):
    """One encoded image per frame, in a dataset of an HDF5 file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["hdf5"] = "hdf5"
    path: Path
    dataset: str
    codec: Literal["jpeg", "png"]


class CameraTrack(BaseModel):
    """A camera's image size and images; its per-frame calibration is in the arrays."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    width: PositiveInt
    height: PositiveInt
    images: FrameSource | None = None


class Provenance(BaseModel):
    """What the demonstration was read from: `source` names the recorder or dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode: NonNegativeInt | None = None


class DemoMeta(BaseModel):
    """`demo.json`. `instructions` are the candidates for this episode, the one to use first;
    `success_frame` is the first frame with the task's success latched, when the recording
    checked it every physics step (None: unknown); `replay` is how a re-recording chose the
    frames it sent (e.g. `s1`, every frame); `observations`: `frames.jsonl` and `blobs/` hold
    every frame's observation."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.demo/1"] = Field(default=SCHEMA, alias="schema")
    task: TaskName
    seed: int
    instructions: list[str]
    frames: PositiveInt
    frame_steps: PositiveInt
    ee_to_tcp: dict[Arm, Vec3]
    cameras: dict[str, CameraTrack] = Field(default_factory=dict)
    success_frame: NonNegativeInt | None = None
    replay: str | None = None
    observations: bool = False
    provenance: Provenance


@dataclass(frozen=True, eq=False)
class ArmTrack:
    """One arm per frame: ee and TCP poses (N, 7), gripper command (N,) and, when recorded,
    joint commands and measured joints (N, J) and the measured finger opening (N,)."""

    ee: Array
    tcp: Array
    gripper: Array
    joints: Array | None = None
    joints_measured: Array | None = None
    opening: Array | None = None


@dataclass(frozen=True, eq=False)
class CameraCalib:
    """Per-frame intrinsics (N, 3, 3) and camera-to-world transforms (N, 4, 4), OpenCV axes."""

    K: Array
    cam2world: Array


@dataclass(frozen=True, eq=False)
class Demo:
    """A demonstration: `arms` and `calib` hold the arms and cameras `meta` names;
    `source_frames` (N,) the replayed trajectory's frame each frame of a re-recording sent, and
    `physics_steps` (N,) the physics steps simulated from the start to each frame (0 first)."""

    meta: DemoMeta
    arms: dict[Arm, ArmTrack]
    calib: dict[str, CameraCalib] = field(default_factory=dict)
    source_frames: NDArray[np.int64] | None = None
    physics_steps: NDArray[np.int64] | None = None

    def steps(self) -> NDArray[np.int64]:
        """The physics steps from the start to each frame: as recorded, or at `frame_steps` a
        frame when they were not."""
        if self.physics_steps is not None:
            return self.physics_steps
        return np.arange(self.meta.frames, dtype=np.int64) * self.meta.frame_steps

    def __post_init__(self) -> None:
        n = self.meta.frames
        if set(self.arms) != set(self.meta.ee_to_tcp):
            raise ValueError(f"arms {sorted(self.arms)} != ee_to_tcp {sorted(self.meta.ee_to_tcp)}")
        if set(self.calib) != set(self.meta.cameras):
            raise ValueError(
                f"calibrated {sorted(self.calib)} != cameras {sorted(self.meta.cameras)}"
            )
        shapes: dict[str, tuple[NDArray[Any], tuple[int, ...]]] = {}
        for arm, t in self.arms.items():
            shapes |= {f"{arm}.ee": (t.ee, (n, 7)), f"{arm}.tcp": (t.tcp, (n, 7))}
            shapes[f"{arm}.gripper"] = (t.gripper, (n,))
            for key in ("joints", "joints_measured"):
                if (joints := getattr(t, key)) is not None:
                    shapes[f"{arm}.{key}"] = (joints, (n, np.shape(joints)[-1]))
            if t.opening is not None:
                shapes[f"{arm}.opening"] = (t.opening, (n,))
        for key in _FRAME_ARRAYS:
            if (values := getattr(self, key)) is not None:
                shapes[key] = (values, (n,))
        for name, c in self.calib.items():
            shapes |= {f"{name}.K": (c.K, (n, 3, 3)), f"{name}.cam2world": (c.cam2world, (n, 4, 4))}
        for key, (arr, shape) in shapes.items():
            if np.shape(arr) != shape:
                raise ValueError(f"{key} has shape {np.shape(arr)}, expected {shape}")


def tcp_from_ee(ee: ArrayLike, ee_to_tcp: Vec3) -> Array:
    """TCP poses (..., 7) of ee poses (..., 7): moved by the offset in the ee frame."""
    return pose_compose(ee, [*ee_to_tcp, 1.0, 0.0, 0.0, 0.0])


def arm_track(
    ee: ArrayLike,
    gripper: ArrayLike,
    ee_to_tcp: Vec3,
    joints: ArrayLike | None = None,
    joints_measured: ArrayLike | None = None,
    opening: ArrayLike | None = None,
) -> ArmTrack:
    ee = np.asarray(ee, dtype=float)
    return ArmTrack(
        ee=ee,
        tcp=tcp_from_ee(ee, ee_to_tcp),
        gripper=np.asarray(gripper, dtype=float),
        joints=_floats(joints),
        joints_measured=_floats(joints_measured),
        opening=_floats(opening),
    )


def frame_alias(frame: int) -> str:
    """How models know a recording's frame (0-based): `frame_1` is its first."""
    return f"frame_{frame + 1}"


def save_demo(demo: Demo, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[Any]] = {}
    for arm, t in demo.arms.items():
        arrays |= {f"{arm}.ee": t.ee, f"{arm}.tcp": t.tcp, f"{arm}.gripper": t.gripper}
        for key in _OPTIONAL:
            if (values := getattr(t, key)) is not None:
                arrays[f"{arm}.{key}"] = values
    for key in _FRAME_ARRAYS:
        if (values := getattr(demo, key)) is not None:
            arrays[key] = values
    for name, c in demo.calib.items():
        arrays |= {f"{name}.K": c.K, f"{name}.cam2world": c.cam2world}
    np.savez_compressed(directory / ARRAYS_FILE, **arrays)  # type: ignore[arg-type]
    (directory / META_FILE).write_text(demo.meta.model_dump_json(indent=1) + "\n", "utf-8")


def load_demo(directory: Path) -> Demo:
    meta = DemoMeta.model_validate_json((directory / META_FILE).read_text("utf-8"))
    with np.load(directory / ARRAYS_FILE, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    arms = {
        arm: ArmTrack(
            ee=arrays[f"{arm}.ee"],
            tcp=arrays[f"{arm}.tcp"],
            gripper=arrays[f"{arm}.gripper"],
            **{key: arrays.get(f"{arm}.{key}") for key in _OPTIONAL},
        )
        for arm in meta.ee_to_tcp
    }
    calib = {n: CameraCalib(arrays[f"{n}.K"], arrays[f"{n}.cam2world"]) for n in meta.cameras}
    try:
        return Demo(meta, arms, calib, arrays.get("source_frames"), arrays.get("physics_steps"))
    except ValueError as error:
        raise ValueError(f"{directory}: {error}") from None


def save_frames(directory: Path, entries: list[FrameEntry]) -> None:
    """`frames.jsonl`: a recording's observations, one frame-store entry per frame."""
    lines = "".join(entry.model_dump_json() + "\n" for entry in entries)
    write_atomic(directory / FRAMES_FILE, lines.encode())


def load_frames(directory: Path) -> list[FrameEntry]:
    text = (directory / FRAMES_FILE).read_text("utf-8")
    return [FrameEntry.model_validate_json(line) for line in text.splitlines()]


def _floats(values: ArrayLike | None) -> Array | None:
    return None if values is None else np.asarray(values, dtype=float)
