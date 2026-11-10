"""What S1 consumes of a task's demonstration (docs/concepts/learning.md): the package, built from
a re-recording, and how the runner loads it.

`package.json` (`DemoPackage`): the demo seed and its instruction verbatim; the keypoints with
the success frame (the referee's input: agents see them without coordinates); keyframes, images
with captions; the recording's frames as a frame source (observations as JSON blobs, seeded into
the Demo Reader's and the hinted Actor's frame stores under their aliases, `frame_<n>` as
`roboscript.demos.frame_alias` names a recording's frame). The package directory carries the
blobs it references (`blobs/`, a blob store directory), which `import_blobs` checks and puts into
the runner's store. None of this is ground truth, an expert API call or an asset point.

The ground-truth layer, `truth.json` (`DemoTruth`), is apart from it: the task objects by the
simulator's names with their categories, and their poses (and articulated objects' joint
positions) at the start, at each keypoint's frame and at the success frame, from the recording's
referee. Only the protocols
read it, for the roles the information boundary allows (docs/architecture.md):
S1's Demo Reader, hinted Actor and Reviser; S2's Reviser behind `PracticeConfig.demo_truth`, off by
default; never an unhinted S1 attempt, an S2 Actor or P3.

The package also holds what S2's Actor and P3's Agent may be shown of the demonstration
(`demo_context`, docs/concepts/learning.md): its operations (`ops`: each grasp, lift, place,
release and closed-gripper move, from the keypoints' gripper events and phases, with a keyframe
before it and one after it settled) and its goal (`goal`: the success frame; `goal_frames`).

`build_package` makes one from a recording with observations: every `every`-th frame and each
keypoint's and the success frame; per keypoint (at most `keyframes`, gripper events first) a
keyframe of the head camera and the moving arm's wrist camera, drawn like the sensing toolset's
views (`SensingConfig`: TCP dots, the head view's TCP axes, the table grid, found on the first
frame's head depth) with the keypoint's TCP marked; the operations' keyframes, from any frame of
the recording, and the goal's (head and the last operation's wrist); given the recording's
ground truth, the layer too.

`segments.json` (`DemoSegments`), beside a package whose keypoints lack where the
arm's motion toward them began (`Keypoint.since`), carries those starts, re-extracted from the
recording; S1's referee reads them (`with_segments`) to find the moments both arms move toward
at once. It stays apart so that the package, its hash and what the agents see of it do not change.
"""

import functools
import hashlib
import re
import shutil
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal

import anyio
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    ValidationError,
    model_validator,
)
from typing_extensions import Self

from roboscript.blobs import BlobStore, LocalBlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraFrame
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import TaskName
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.contracts.robot import Arm, RobotState
from roboscript.demos import (
    BLOBS_DIR,
    ArmTrack,
    Demo,
    DemoMeta,
    Keypoint,
    KeypointSet,
    RecordingConfig,
    SceneTruth,
    frame_alias,
    load_demo,
    load_frames,
)
from roboscript.demos.keypoints import GRIP_CUES
from roboscript.embodiments import EmbodimentProfile
from roboscript.perception.config import TableConfig
from roboscript.perception.frames import load_image
from roboscript.perception.table import fit_table
from roboscript.records import write_atomic
from roboscript.spatial.camera import backproject_world
from roboscript.toolsets.sensing import SensingConfig
from roboscript.toolsets.sensing.overlay import Mark, TableGrid, draw

PACKAGE_FILE: Final = "package.json"
TRUTH_FILE: Final = "truth.json"
SEGMENTS_FILE: Final = "segments.json"
SETTLE_FRAMES: Final = 10
"""Frames after a gripper event until the gripper has settled, the moment an
operation's after keyframe shows unless the arm's next keypoint comes first."""
OPERATIONS: Final = {
    "preshape": "close the gripper",
    "grasp": "grasp",
    "lift": "lift",
    "place": "place",
    "release": "release",
    "carry": "move with the gripper closed",
}
"""The keypoint phases that are operations and what each is called; a carry is how a press or a
turn shows (the arm comes to rest with the gripper closed). Approaches and retreats are not."""


class Keyframe(BaseModel):
    """Images of one moment of the demonstration (head and the moving arm's wrist, overlays
    drawn), and what they show, e.g. `left K2 (grasp)`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    caption: str
    images: list[BlobRef] = Field(min_length=1)


class Operation(BaseModel):
    """One operation of the demonstration by one arm at a keypoint (`K2`), what it is (`grasp`),
    and keyframes just before it and after it (the gripper settled, the arm at rest)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: Arm
    keypoint: str
    label: str
    before: Keyframe
    after: Keyframe


class ArmTracks(BaseModel):
    """One arm's recorded proprioception per frame: ee and TCP poses (N, 7), gripper command
    (N,) and, when recorded, the measured finger opening (N,)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ee: list[tuple[float, float, float, float, float, float, float]]
    tcp: list[tuple[float, float, float, float, float, float, float]]
    gripper: list[float]
    opening: list[float] | None = None


class DemoTracks(BaseModel):
    """The demonstration's recorded proprioception, every frame (a real robot's own record of
    its motion; no object, camera or simulator state): what a written skill's fitter reads of
    the demonstration besides perception (`tools.skill_fit`). `meta` is the recording's, its
    cameras left out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    meta: DemoMeta
    arms: dict[Arm, ArmTracks]

    def demo(self) -> Demo:
        arms = {
            arm: ArmTrack(
                ee=np.asarray(track.ee, dtype=float),
                tcp=np.asarray(track.tcp, dtype=float),
                gripper=np.asarray(track.gripper, dtype=float),
                opening=None if track.opening is None else np.asarray(track.opening, dtype=float),
            )
            for arm, track in self.arms.items()
        }
        return Demo(meta=self.meta, arms=arms)


def demo_tracks(demo: Demo) -> DemoTracks:
    """`demo`'s proprioception as a package carries it."""
    arms = {
        arm: ArmTracks(
            ee=[tuple(row) for row in np.asarray(track.ee, dtype=float).tolist()],
            tcp=[tuple(row) for row in np.asarray(track.tcp, dtype=float).tolist()],
            gripper=np.asarray(track.gripper, dtype=float).tolist(),
            opening=None if track.opening is None else np.asarray(track.opening).tolist(),
        )
        for arm, track in demo.arms.items()
    }
    return DemoTracks(meta=demo.meta.model_copy(update={"cameras": {}}), arms=arms)


class DemoPackage(BaseModel):
    """A demonstration as S1 studies it: its keypoints, drawn keyframes and operations, a subset
    of its frames (observations) and its goal; `tracks`, a blob of its recorded proprioception at
    every frame (`DemoTracks`; None: the package has none), which fitting a written
    skill reads (`tools.skill_fit`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.demo_package/1"] = Field(default="rs.demo_package/1", alias="schema")
    task: TaskName
    seed: NonNegativeInt
    instruction: str
    keypoints: KeypointSet
    keyframes: list[Keyframe] = Field(default_factory=list, max_length=24)
    frames: list[FrameEntry] = []
    ops: list[Operation] = Field(default_factory=list, max_length=24)
    goal: Keyframe | None = None
    tracks: BlobRef | None = None

    @model_validator(mode="after")
    def _one_demonstration(self) -> Self:
        kps = self.keypoints
        if (kps.task, kps.seed) != (self.task, self.seed):
            raise ValueError(
                f"the keypoints are of {kps.task} seed {kps.seed}, not {self.task} seed {self.seed}"
            )
        if kps.success_frame is None:
            raise ValueError("the keypoints have no success frame: the referee cannot judge")
        aliases = [entry.alias for entry in self.frames]
        if any(not re.fullmatch(r"frame_[1-9][0-9]*", alias) for alias in aliases):
            raise ValueError("frames are aliased frame_1, frame_2, ... (the frame store's form)")
        if len(set(aliases)) != len(aliases):
            raise ValueError("a frame alias is given twice")
        return self


class TruthFrame(BaseModel):
    """The objects at one frame of the demonstration: `what` it is (`start`, `left K2 (grasp)`,
    `success`), world poses by object and joint positions by articulated object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame: NonNegativeInt
    what: list[str]
    poses: dict[str, Pose]
    joints: dict[str, list[float]] = {}


class DemoTruth(BaseModel):
    """`truth.json`: a package's ground-truth layer; `objects` maps the simulator's name of
    each object to its category word."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.demo_truth/1"] = Field(default="rs.demo_truth/1", alias="schema")
    task: TaskName
    seed: NonNegativeInt
    objects: dict[str, str]
    frames: list[TruthFrame]


def demo_truth(keypoints: KeypointSet, truth: SceneTruth) -> DemoTruth:
    """The ground-truth layer of a demonstration from its recording's referee ground truth."""
    moments: dict[int, list[str]] = {0: ["start"]}
    for arm, kps in keypoints.arms.items():
        for kp in kps:
            moments.setdefault(kp.frame, []).append(f"{arm} {kp.name} ({kp.phase})")
    if keypoints.success_frame is not None:
        moments.setdefault(keypoints.success_frame, []).append("success")
    frames = [
        TruthFrame(
            frame=t,
            what=what,
            poses={
                name: Pose.from_list(truth.poses[t, j].tolist())
                for j, name in enumerate(truth.objects)
            },
            joints={name: q[t].tolist() for name, q in truth.joints.items()},
        )
        for t, what in sorted(moments.items())
    ]
    objects = {name: truth.categories.get(name, "") for name in truth.objects}
    return DemoTruth(task=keypoints.task, seed=keypoints.seed, objects=objects, frames=frames)


def load_demo_truth(directory: Path) -> DemoTruth | None:
    """A package directory's ground-truth layer, if it has one."""
    path = directory / TRUTH_FILE
    return DemoTruth.model_validate_json(path.read_bytes()) if path.is_file() else None


class DemoSegments(BaseModel):
    """`segments.json`: per arm and keypoint (in order), the frame its motion began
    (`Keypoint.since`), with the frames they belong to, which must match the package's."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.demo_segments/1"] = Field(default="rs.demo_segments/1", alias="schema")
    task: TaskName
    seed: NonNegativeInt
    arms: dict[Arm, list[tuple[NonNegativeInt | None, NonNegativeInt]]]  # (since, frame)


def load_segments(directory: Path) -> DemoSegments | None:
    """A package directory's motion starts, if it has them."""
    path = directory / SEGMENTS_FILE
    return DemoSegments.model_validate_json(path.read_bytes()) if path.is_file() else None


def with_segments(keypoints: KeypointSet, segments: DemoSegments) -> KeypointSet:
    """`keypoints` with the starts of `segments`, which must be of the same demonstration's
    keypoints (`ValueError` names what differs)."""
    if (segments.task, segments.seed) != (keypoints.task, keypoints.seed):
        raise ValueError(
            f"the segments are of {segments.task} seed {segments.seed}, the keypoints of "
            f"{keypoints.task} seed {keypoints.seed}"
        )
    arms = {}
    for arm, kps in keypoints.arms.items():
        starts = segments.arms.get(arm, [])
        if [frame for _, frame in starts] != [kp.frame for kp in kps]:
            raise ValueError(f"the segments' {arm} frames are not the keypoints': re-extract them")
        arms[arm] = [
            kp.model_copy(update={"since": since})
            for kp, (since, _) in zip(kps, starts, strict=True)
        ]
    return keypoints.model_copy(update={"arms": arms})


def load_package(directory: Path) -> DemoPackage:
    try:
        return DemoPackage.model_validate_json((directory / PACKAGE_FILE).read_bytes())
    except ValueError as error:
        raise ValueError(f"{directory / PACKAGE_FILE}: {error}") from None


def save_package(directory: Path, package: DemoPackage) -> None:
    write_atomic(directory / PACKAGE_FILE, (package.model_dump_json(indent=1) + "\n").encode())


async def import_blobs(directory: Path, blobs: BlobStore) -> int:
    """Put the blobs a package directory's `package.json` references (its drawn images, its
    frames' observations and their camera images) into `blobs`, the runner's store, so that they
    resolve there; how many. Each must be a regular file of the package's `blobs/`, not a link,
    of the reference's size and sha256, else `ValueError` names it. Other files stay behind."""
    package = await anyio.to_thread.run_sync(load_package, directory)
    store = LocalBlobStore(directory / BLOBS_DIR)
    refs = package_images(package)
    refs += [entry.observation for entry in package.frames]
    refs += [] if package.tracks is None else [package.tracks]
    done: set[str] = set()
    observations = {entry.observation.id for entry in package.frames}
    while refs:
        ref = refs.pop()
        if ref.id in done:
            continue
        data = await anyio.to_thread.run_sync(_blob, store, ref)
        await blobs.aput_bytes(data, ref.media_type, ref.meta)
        done.add(ref.id)
        if ref.id in observations:
            refs += _images(_parsed(store, ref, data))
    return len(done)


async def demo_robots(
    package: DemoPackage, blobs: BlobStore, frames: Iterable[int] | None = None
) -> dict[int, RobotState]:
    """The demonstration's robot (its recorded proprioception) at `frames` (None: every
    keypoint's), from a store the package's blobs were imported into; a frame the package does
    not hold is left out."""
    if frames is None:
        frames = {kp.frame for kps in package.keypoints.arms.values() for kp in kps}
    entries = {entry.alias: entry for entry in package.frames}
    robots = {}
    for t in sorted(set(frames)):
        entry = entries.get(frame_alias(t))
        if entry is not None:
            seen = Observation.model_validate_json(await blobs.aget_bytes(entry.observation))
            robots[t] = seen.robot
    return robots


def package_frames(frames: int, keypoints: KeypointSet, every: int) -> list[int]:
    """The recording's frames a package holds: every `every`-th, each keypoint's, the success
    frame."""
    marked = {kp.frame for kps in keypoints.arms.values() for kp in kps}
    if keypoints.success_frame is not None:
        marked.add(keypoints.success_frame)
    return sorted({*range(0, frames, every), *(t for t in marked if t < frames)})


def operation_plan(keypoints: KeypointSet, frames: int) -> list[tuple[Arm, Keypoint, int, int]]:
    """The operations, in frame order: each arm's keypoints of an operation phase, with the
    frame before (the arm's previous keypoint's, else the first) and after it (a gripper event's
    once settled, `SETTLE_FRAMES` on, unless the arm's next keypoint comes first; a motion
    keypoint's own: the arm at rest), within the recording's `frames`."""
    plan = []
    for arm, kps in keypoints.arms.items():
        for i, kp in enumerate(kps):
            if kp.phase not in OPERATIONS:
                continue
            before = kps[i - 1].frame if i > 0 else 0
            after = kp.frame
            if kp.grip:
                after = kp.frame + SETTLE_FRAMES
                if i + 1 < len(kps):
                    after = min(after, kps[i + 1].frame)
            plan.append((arm, kp, min(before, frames - 1), min(after, frames - 1)))
    return sorted(plan, key=lambda item: (item[1].frame, item[0]))


def goal_frames(package: DemoPackage) -> Keyframe | None:
    """The demonstration's goal as the Actor may be shown it: the success frame's head view and
    the wrist view of the arm that operated last (None: the package has no goal)."""
    return package.goal


def keyframe_plan(keypoints: KeypointSet, limit: int) -> list[tuple[Arm, Keypoint]]:
    """The keypoints that get a keyframe, in frame order: all of them, or when there are more
    than `limit`, the gripper events first and then the earliest motion keypoints."""
    order = sorted(
        ((arm, kp) for arm, kps in keypoints.arms.items() for kp in kps),
        key=lambda item: (item[1].frame, item[0]),
    )
    ranked = sorted(range(len(order)), key=lambda i: (order[i][1].cue not in GRIP_CUES, i))
    keep = set(ranked[:limit])
    return [item for i, item in enumerate(order) if i in keep]


async def build_package(
    recording: Path,
    keypoints: KeypointSet,
    instruction: str,
    directory: Path,
    config: RecordingConfig,
    profile: EmbodimentProfile,
    truth: SceneTruth | None = None,
    overlays: SensingConfig | None = None,
    *,
    table: TableConfig,
) -> DemoPackage:
    """The package of a recording with observations (a demo directory) and its keypoints,
    written to `directory` (replacing what was there; never the recording or a directory
    holding it) with its blobs, its keyframes drawn as the sensing toolset draws its views
    (`overlays`, its config; None: its defaults) on the table perception finds where `table`
    (the benchmark's binding) says it is, and with the recording's ground `truth` its
    ground-truth layer."""
    overlays = overlays or SensingConfig()
    demo = await anyio.to_thread.run_sync(load_demo, recording)
    if not demo.meta.observations:
        raise ValueError(f"{recording}: the recording has no observations to package")
    await anyio.to_thread.run_sync(_fresh, directory, recording)
    entries = await anyio.to_thread.run_sync(load_frames, recording)
    source = LocalBlobStore(recording / BLOBS_DIR)
    target = LocalBlobStore(directory / BLOBS_DIR)
    frames = [entries[t] for t in package_frames(len(entries), keypoints, config.every)]
    for entry in frames:
        await _copy(entry.observation, source, target)
        for camera in (await _observation(source, entry.observation)).cameras.values():
            for blob in (camera.rgb, camera.depth):
                if blob is not None:
                    await _copy(blob, source, target)
    head = next(name for name, spec in profile.cameras.items() if spec.kind == "head")
    first = await _observation(source, entries[0].observation)
    grid = await _grid(source, first.cameras[head], table)
    views = functools.partial(_views, source, target, profile, overlays, grid, head)
    keyframes = []
    for arm, kp in keyframe_plan(keypoints, config.keyframes):
        mark = Mark(label=f"{arm[0].upper()}{kp.name}", world=kp.tcp.p)
        images = await views(entries[kp.frame], arm, [mark])
        keyframes.append(Keyframe(caption=f"{arm} {kp.name} ({kp.phase})", images=images))
    ops = []
    for arm, kp, before, after in operation_plan(keypoints, len(entries)):
        label = OPERATIONS[kp.phase]
        mark = [Mark(label=f"{arm[0].upper()}{kp.name}", world=kp.tcp.p)]
        what = f"{arm} {kp.name} {label}"
        shown = [await views(entries[frame], arm, mark) for frame in (before, after)]
        ops.append(
            Operation(
                arm=arm,
                keypoint=kp.name,
                label=label,
                before=Keyframe(caption=f"{what}: before", images=shown[0]),
                after=Keyframe(caption=f"{what}: after", images=shown[1]),
            )
        )
    goal = None
    if keypoints.success_frame is not None and keypoints.success_frame < len(entries):
        last = ops[-1].arm if ops else next(iter(keypoints.arms))
        shown = await views(entries[keypoints.success_frame], last, [])
        caption = f"the goal: the success frame (the {last} arm moved last)"
        goal = Keyframe(caption=caption, images=shown)
    tracks = demo_tracks(demo).model_dump_json().encode()
    tracks_ref = await target.aput_bytes(tracks, "application/json", {"kind": "demo_tracks"})
    package = DemoPackage(
        task=demo.meta.task,
        seed=demo.meta.seed,
        instruction=instruction,
        keypoints=keypoints,
        keyframes=keyframes,
        frames=frames,
        ops=ops,
        goal=goal,
        tracks=tracks_ref,
    )
    await anyio.to_thread.run_sync(save_package, directory, package)
    if truth is not None:
        layer = demo_truth(keypoints, truth)
        data = (layer.model_dump_json(indent=1) + "\n").encode()
        await anyio.to_thread.run_sync(write_atomic, directory / TRUTH_FILE, data)
    return package


def validate_package(directory: Path) -> list[str]:
    """What is wrong with a package directory (nothing: an empty list): its frames must include
    each keypoint's and the success frame, every blob it references must be in its `blobs/` as
    `import_blobs` takes it (a regular file of the reference's size and sha256), every
    observation exactly an `Observation`, a ground-truth layer must be of its demonstration and
    cover its keypoints, motion starts must be of its keypoints, and nothing else may be in the
    directory."""
    package = load_package(directory)
    store = LocalBlobStore(directory / BLOBS_DIR)
    problems = []
    kps = package.keypoints
    needed = {kp.frame for arm in kps.arms.values() for kp in arm} | {kps.success_frame}
    aliases = {entry.alias for entry in package.frames}
    if missing := sorted(t for t in needed if t is not None and frame_alias(t) not in aliases):
        problems.append(f"frames {missing} (keypoints or the success frame) are not in it")
    refs = package_images(package)
    for entry in package.frames:
        try:
            data = _blob(store, entry.observation)
            seen = _parsed(store, entry.observation, data)
        except ValueError as error:
            problems.append(f"{entry.alias}: {error}")
            continue
        if seen.model_dump_json().encode() != data:
            problems.append(f"{entry.alias}: its observation holds more than an Observation")
        if (seen.frame_id, seen.step) != (entry.frame_id, entry.step):
            problems.append(f"{entry.alias}: its observation is of another frame")
        refs += _images(seen)
    refs += [] if package.tracks is None else [package.tracks]
    for ref in {ref.id: ref for ref in refs}.values():
        try:
            _blob(store, ref)
        except ValueError as error:
            problems.append(str(error))
    refs += [entry.observation for entry in package.frames]
    truth = load_demo_truth(directory)
    if truth is not None:
        covered = {frame.frame for frame in truth.frames}
        if (truth.task, truth.seed) != (package.task, package.seed):
            problems.append(f"the ground-truth layer is of {truth.task} seed {truth.seed}")
        elif missing := sorted(t for t in needed if t is not None and t not in covered):
            problems.append(f"the ground-truth layer lacks frames {missing}")
    if (segments := load_segments(directory)) is not None:
        try:
            with_segments(kps, segments)
        except ValueError as error:
            problems.append(f"{SEGMENTS_FILE}: {error}")
    stray = {p.name for p in store.root.glob("*/*")} - {ref.id for ref in refs}
    stray |= {p.name for p in directory.iterdir()} - {
        PACKAGE_FILE,
        TRUTH_FILE,
        SEGMENTS_FILE,
        BLOBS_DIR,
    }
    if stray:
        problems.append(f"{len(stray)} unreferenced files, e.g. {sorted(stray)[0]}")
    return problems


async def _views(
    source: LocalBlobStore,
    target: LocalBlobStore,
    profile: EmbodimentProfile,
    overlays: SensingConfig,
    grid: TableGrid,
    head: str,
    entry: FrameEntry,
    arm: Arm,
    marks: list[Mark],
) -> list[BlobRef]:
    """A recording frame's head view and `arm`'s wrist view, drawn as the sensing toolset draws
    them, with `marks`, put into the package's store."""
    seen = await _observation(source, entry.observation)
    wrist = next(n for n, s in profile.cameras.items() if s.kind == "wrist" and s.arm == arm)
    images = []
    for name in (head, wrist):
        image = await load_image(source, seen.cameras[name])
        drawing = functools.partial(
            draw,
            image.rgb,
            seen.cameras[name].camera,
            robot=seen.robot,
            axes_m=overlays.axes_m if name == head else None,
            grid=grid,
            grid_m=overlays.grid_m,
            marks=marks,
        )
        images.append(await target.aput_image(await anyio.to_thread.run_sync(drawing)))
    return images


def package_images(package: DemoPackage) -> list[BlobRef]:
    """The drawn images a package references: its keyframes', its operations' and its goal's."""
    shown = [*package.keyframes, *(k for op in package.ops for k in (op.before, op.after))]
    if package.goal is not None:
        shown.append(package.goal)
    return [blob for keyframe in shown for blob in keyframe.images]


def _fresh(directory: Path, recording: Path) -> None:
    """A rebuild replaces the package, never the recording: no blob or layer of the last build
    may stay."""
    if recording.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"{directory} holds the recording {recording}: package it elsewhere")
    if directory.exists():
        shutil.rmtree(directory)


def _blob(store: LocalBlobStore, ref: BlobRef) -> bytes:
    """The bytes of a package's blob: a regular file (not a link) of the reference's size and
    sha256."""
    path = store.path(ref)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise ValueError(f"{path}: missing ({ref.media_type})") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"{path}: not a regular file")
    data = path.read_bytes()
    if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.id:
        raise ValueError(f"{path}: not the blob referenced ({ref.size} bytes, sha256 {ref.id})")
    return data


def _parsed(store: LocalBlobStore, ref: BlobRef, data: bytes) -> Observation:
    try:
        return Observation.model_validate_json(data)
    except ValidationError:
        raise ValueError(f"{store.path(ref)}: not an observation") from None


def _images(seen: Observation) -> list[BlobRef]:
    return [b for c in seen.cameras.values() for b in (c.rgb, c.depth) if b is not None]


async def _observation(store: LocalBlobStore, ref: BlobRef) -> Observation:
    return Observation.model_validate_json(await store.aget_bytes(ref))


async def _copy(ref: BlobRef, source: LocalBlobStore, target: LocalBlobStore) -> None:
    if not await target.aexists(ref):
        await target.aput_bytes(await source.aget_bytes(ref), ref.media_type)


async def _grid(store: LocalBlobStore, frame: CameraFrame, config: TableConfig) -> TableGrid:
    """The table to draw the grid on, from a camera's depth, as perception finds it."""
    image = await load_image(store, frame)
    if image.depth is None:
        raise ValueError(f"camera {frame.camera.name} has no depth to find the table in")
    points, _ = await anyio.to_thread.run_sync(
        backproject_world, image.depth, image.K, image.cam2world
    )
    table = await anyio.to_thread.run_sync(fit_table, points, config)
    return TableGrid(z=table.z, extent=table.extent)
