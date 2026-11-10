"""The re-recorded demonstrations, per task under `<demos>/recorded/<task>/` (written by
`examples/simBenchmarks/RoboTwin/scripts/record_demos.py`; the recording itself by
`roboscript_robotwin.sim.record`):

    record.json           how it was recorded (`RecordedTask`): every candidate tried, the ones
                          rejected, the one recorded now
    recording/            the demo with every frame's observation (demo.json, arrays.npz,
                          frames.jsonl, blobs/)
    expert/               a glob task's re-run of the expert: the trajectory replayed
    referee/              ground truth: truth.npz, disturbance.json, validation.json and the
                          validation's expert traces
    keypoints.json        extracted from the recording; keypoints_diff.json against the
                          replayed trajectory's (`KeypointReview`): its measured poses flag,
                          its commands inform
    package/              what S1 reads: package.json, blobs/, and the ground-truth layer
                          truth.json (object poses at the keypoints' frames)
    review.html, review/  the review page, its head-camera video and keyframes

A candidate is accepted when its recording latched the official success and its replay
validation passed; otherwise the next one is recorded. Keypoint and disturbance flags only mark
a demonstration for review (`summary.json` and the index page list them first). Everything past
the recording comes from the files (`review_task`), the keypoint diff from FK through the motion
service (`compare_task`).
"""

import dataclasses
import functools
import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

import anyio
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.ids import TaskName
from roboscript.contracts.motion import Planner
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import Arm
from roboscript.demos import (
    BLOBS_DIR,
    Demo,
    DemosConfig,
    Disturbance,
    FileRef,
    KeypointDiff,
    KeypointPair,
    KeypointSet,
    SceneTruth,
    compare_keypoints,
    demo_keypoints,
    disturbance,
    load_demo,
    load_frames,
    load_truth,
)
from roboscript.demos.manifest import content_sha256, file_ref
from roboscript.demos.paths import fk
from roboscript.embodiments import EmbodimentProfile
from roboscript.learning import (
    DemoPackage,
    DemoSegments,
    Keyframe,
    build_package,
    load_package,
    validate_package,
    with_segments,
)
from roboscript.learning.demo import PACKAGE_FILE, SEGMENTS_FILE
from roboscript.perception.config import TableConfig
from roboscript.perception.frames import CameraImage, load_image
from roboscript.toolsets.sensing.overlay import Mark, draw
from roboscript_robotwin.config import PERCEPTION_TABLE
from roboscript_robotwin.demos.expert import load_episode
from roboscript_robotwin.demos.records import Attempt, RecordedTask, read, write
from roboscript_robotwin.demos.validate import VALIDATION, Validation, validation_text

REVIEW: Final = "review.json"
DIFF: Final = "keypoints_diff.json"


class Reviewed(BaseModel):
    """`review.json`: what the recording gave. Sizes in bytes; `observation_bytes` is the mean
    per recorded frame (its observation and images)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode: NonNegativeInt
    keypoints: dict[Arm, NonNegativeInt]
    disturbance: Disturbance
    package: FileRef
    package_bytes: NonNegativeInt
    package_frames: NonNegativeInt
    keyframes: NonNegativeInt
    observation_bytes: NonNegativeInt
    problems: list[str]


class KeypointReview(BaseModel):
    """`keypoints_diff.json`: our keypoints against the replayed trajectory's, extracted from its
    measured poses (like ours: its flags are the review's) and, for information, from the TCP of
    its commanded joints (FK through the motion service), which grasps never reach where the
    fingers stop on the object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode: NonNegativeInt
    commanded: KeypointDiff
    measured: KeypointDiff


def reference(
    out: Path, recorded: RecordedTask, expert_root: Path, profile: EmbodimentProfile
) -> Demo:
    """The trajectory the recording replayed: the expert data's episode, or a glob task's
    re-run of the expert."""
    if recorded.source == "expert_rerun":
        return load_demo(out / "expert")
    return load_episode(expert_root, recorded.task, recorded.episode, profile)


async def review_task(out: Path, config: DemosConfig, profile: EmbodimentProfile) -> Reviewed:
    """Keypoints, the disturbance check, the package and the review of a task whose
    `record.json` says a candidate was recorded (`record_demos.py` runs it in a process of its
    own)."""
    recorded, demo, ours, truth = await anyio.to_thread.run_sync(_keypoints, out, config)
    package = await build_package(
        out / "recording",
        ours,
        recorded.instruction,
        out / "package",
        config.recording,
        profile,
        truth,
        table=TableConfig.model_validate(PERCEPTION_TABLE),
    )
    await _render(out, demo, ours, package.keyframes, config.recording.video_fps, profile)
    holding = {arm: spec.gripper.holding_threshold for arm, spec in profile.arms.items()}
    review = functools.partial(_review, out, config, recorded, demo, ours, truth, package, holding)
    return await anyio.to_thread.run_sync(review)


def _keypoints(
    out: Path, config: DemosConfig
) -> tuple[RecordedTask, Demo, KeypointSet, SceneTruth]:
    """The recorded candidate, its demo, keypoints (written to `keypoints.json`) and truth."""
    recorded = read(out)
    assert recorded is not None and recorded.recorded
    demo = load_demo(out / "recording")
    tolerance = config.keypoints.tolerance_for(recorded.category)
    ours = demo_keypoints(demo, config.keypoints, tolerance)
    _write(out / "keypoints.json", ours)
    return recorded, demo, ours, load_truth(out / "referee")


def segments(out: Path, config: DemosConfig, package: Path) -> DemoSegments:
    """The motion starts of the recording's keypoints, written to `package`'s `segments.json`
    (a package without them keeps its bytes). `package` must be of this
    recording's keypoints; `ValueError` if they were extracted with other frames."""
    recorded = read(out)
    assert recorded is not None and recorded.recorded
    ours = demo_keypoints(
        load_demo(out / "recording"),
        config.keypoints,
        config.keypoints.tolerance_for(recorded.category),
    )
    found = DemoSegments(
        task=ours.task,
        seed=ours.seed,
        arms={arm: [(kp.since, kp.frame) for kp in kps] for arm, kps in ours.arms.items()},
    )
    with_segments(load_package(package).keypoints, found)
    _write(package / SEGMENTS_FILE, found)
    return found


def _review(
    out: Path,
    config: DemosConfig,
    recorded: RecordedTask,
    demo: Demo,
    ours: KeypointSet,
    truth: SceneTruth,
    package: DemoPackage,
    holding: Mapping[Arm, float],
) -> Reviewed:
    """The disturbance check and `review.json` with the package's sizes and problems; `holding`
    per arm the embodiment's `Gripper.holding_threshold`."""
    moved = disturbance(demo, truth, config.disturbance, holding, config.keypoints.grip_threshold)
    _write(out / "referee" / "disturbance.json", moved)
    stored = sum(p.stat().st_size for p in (out / "recording" / BLOBS_DIR).glob("*/*"))
    reviewed = Reviewed(
        episode=recorded.episode,
        keypoints={arm: len(kps) for arm, kps in ours.arms.items()},
        disturbance=moved,
        package=file_ref(out / "package" / PACKAGE_FILE, out.parent),
        package_bytes=sum(p.stat().st_size for p in (out / "package").rglob("*") if p.is_file()),
        package_frames=len(package.frames),
        keyframes=len(package.keyframes),
        observation_bytes=stored // demo.meta.frames,
        problems=validate_package(out / "package"),
    )
    _write(out / REVIEW, reviewed)
    render_page(out)
    return reviewed


async def compare_task(
    out: Path, replayed: Demo, config: DemosConfig, planner: Planner, profile: EmbodimentProfile
) -> KeypointReview:
    """`keypoints_diff.json` of the recorded candidate against the trajectory it `replayed`."""
    commanded = await commanded_poses(replayed, planner, profile)
    compare = functools.partial(_compare, out, config, replayed, commanded)
    return await anyio.to_thread.run_sync(compare)


def _compare(out: Path, config: DemosConfig, replayed: Demo, commanded: Demo) -> KeypointReview:
    recorded = read(out)
    assert recorded is not None
    ours = KeypointSet.model_validate_json((out / "keypoints.json").read_text(encoding="utf-8"))
    sent = load_demo(out / "recording").source_frames
    tolerance = config.keypoints.tolerance_for(recorded.category)
    review = KeypointReview(
        episode=recorded.episode,
        commanded=compare_keypoints(
            ours, demo_keypoints(commanded, config.keypoints, tolerance), sent, config.compare
        ),
        measured=compare_keypoints(
            ours, demo_keypoints(replayed, config.keypoints, tolerance), sent, config.compare
        ),
    )
    _write(out / DIFF, review)
    render_page(out)
    return review


async def commanded_poses(demo: Demo, planner: Planner, profile: EmbodimentProfile) -> Demo:
    """`demo` with each arm's ee and TCP from FK of its commanded joints."""
    arms = {}
    for arm, track in demo.arms.items():
        if track.joints is None:
            raise ValueError(f"{demo.meta.task}: the {arm} arm has no joint commands")
        ee, tcp = await fk(planner, profile, arm, track.joints)
        arms[arm] = dataclasses.replace(track, ee=ee, tcp=tcp)
    return dataclasses.replace(demo, arms=arms)


def reject(out: Path, validation: Validation) -> RecordedTask:
    """The recorded candidate failed its validation: `record.json` says so."""
    recorded = read(out)
    assert recorded is not None
    attempt = Attempt(
        episode=validation.episode,
        seed=validation.seed,
        strategy=recorded.strategy,
        outcome="validation failed",
        seconds=sum(a.seconds for a in validation.attempts),
    )
    note = f"episode {validation.episode} {validation_text(validation)}"
    changed = {
        "attempts": [*recorded.attempts, attempt],
        "notes": [*recorded.notes, note],
        "rejected": [*recorded.rejected, validation.episode],
    }
    return write(out, recorded.model_copy(update=changed))


def reviewed(out: Path) -> bool:
    """Whether the recorded candidate's package is built, reviewed and valid."""
    recorded = read(out)
    if recorded is None or not recorded.recorded or not (out / REVIEW).is_file():
        return False
    review = Reviewed.model_validate_json((out / REVIEW).read_text("utf-8"))
    load_package(out / "package")
    return review.episode == recorded.episode and not review.problems


def compared(out: Path) -> KeypointReview | None:
    path = out / DIFF
    return (
        KeypointReview.model_validate_json(path.read_text(encoding="utf-8"))
        if path.is_file()
        else None
    )


def validation(out: Path) -> Validation | None:
    """The replay validation of the task's recorded candidate, once one ran on its current
    package (not after another candidate was recorded or the package was rebuilt)."""
    path, recorded, package = out / "referee" / VALIDATION, read(out), out / "package"
    if recorded is None or not path.is_file() or not (package / PACKAGE_FILE).is_file():
        return None
    checked = Validation.model_validate_json(path.read_text("utf-8"))
    if checked.episode != recorded.episode:
        return None
    return checked if content_sha256(package / PACKAGE_FILE) == checked.package_sha256 else None


def status(out: Path, candidates: int) -> Literal["accepted", "rejected", "pending"]:
    """`accepted`: the recorded candidate passed validation; `rejected`: all `candidates` were."""
    recorded, checked = read(out), validation(out)
    if recorded is None:
        return "pending"
    if recorded.recorded and checked is not None and checked.passed:
        return "accepted"
    return "rejected" if len(recorded.rejected) >= candidates else "pending"


async def _render(
    out: Path,
    demo: Demo,
    keypoints: KeypointSet,
    keyframes: list[Keyframe],
    fps: int,
    profile: EmbodimentProfile,
) -> None:
    """`review/video.mp4`, the head camera with the TCP dots and every keypoint's TCP drawn,
    and the keyframes as JPEGs."""
    review = out / "review"
    review.mkdir(exist_ok=True)
    head = next(name for name, spec in profile.cameras.items() if spec.kind == "head")
    store = LocalBlobStore(out / "recording" / BLOBS_DIR)
    marks = [
        Mark(label=f"{arm[0].upper()}{kp.index}", world=kp.tcp.p)
        for arm, kps in keypoints.arms.items()
        for kp in kps
    ]
    spec = profile.cameras[head]
    command = [
        *("ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"),
        *("-s", f"{spec.width}x{spec.height}", "-r", str(fps), "-i", "-"),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", str(review / "video.mp4")),
    ]
    async with await anyio.open_process(command, stdout=None, stderr=None) as encoder:
        assert encoder.stdin is not None
        for t, entry in enumerate(load_frames(out / "recording")):
            seen = Observation.model_validate_json(await store.aget_bytes(entry.observation))
            image = await load_image(store, seen.cameras[head])
            latched = demo.meta.success_frame is not None and t >= demo.meta.success_frame
            caption = f"frame {t} ({entry.alias})" + ("  success" if latched else "")
            frame = functools.partial(_video_frame, image, seen, head, marks, caption)
            await encoder.stdin.send(await anyio.to_thread.run_sync(frame))
        await encoder.stdin.aclose()
        if await encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to write {review / 'video.mp4'}")
    package_store = LocalBlobStore(out / "package" / BLOBS_DIR)
    for n, keyframe in enumerate(keyframes, start=1):
        for view, blob in zip(("head", "wrist"), keyframe.images, strict=True):
            Image.fromarray(package_store.get_image(blob)).save(
                review / f"keyframe{n}_{view}.jpg", quality=88
            )


def _video_frame(
    image: CameraImage, seen: Observation, camera: str, marks: list[Mark], caption: str
) -> bytes:
    drawn = draw(image.rgb, seen.cameras[camera].camera, robot=seen.robot, marks=marks)
    canvas = Image.fromarray(drawn)
    ImageDraw.Draw(canvas).text((8, 8), caption, fill=(255, 255, 255))
    return np.asarray(canvas).tobytes()


def _write(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=1) + "\n", "utf-8")


class SummaryRow(BaseModel):
    """A task's row of `summary.json`: how it went, the review flags, and the raw value of every
    flag's metric: against the replayed trajectory's measured
    poses, `keypoint_ratio` is the largest keypoint distance as a share of its tolerance and
    `keypoint_offset` the largest frame offset; `commanded_ratio` is that distance share against
    its commands (information)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    category: str | None
    status: Literal["accepted", "rejected", "pending"]
    episode: NonNegativeInt
    seed: NonNegativeInt
    source: Literal["expert_data", "expert_rerun"]
    strategy: str | None
    frames: NonNegativeInt | None
    success_frame: NonNegativeInt | None
    rejected: list[NonNegativeInt]
    notes: list[str]
    flags: list[str] = []
    keypoints: dict[Arm, NonNegativeInt] | None = None
    keypoint_ratio: NonNegativeFloat | None = None
    keypoint_offset: NonNegativeInt | None = None
    keypoint_counts_differ: bool | None = None
    commanded_ratio: NonNegativeFloat | None = None
    disturbance_m: NonNegativeFloat | None = None
    strike_m: NonNegativeFloat | None = None
    side_joint_rad: NonNegativeFloat | None = None
    package: FileRef | None = None
    package_bytes: NonNegativeInt | None = None
    package_frames: NonNegativeInt | None = None
    observation_bytes: NonNegativeInt | None = None
    validation: str = "pending"
    validation_attempts: NonNegativeInt = 0


def summary_row(out: Path, candidates: int) -> SummaryRow:
    """A task's row, from its files; `candidates` is how many the manifest offers it."""
    recorded = read(out)
    assert recorded is not None
    row = {
        **recorded.model_dump(include={"task", "category", "episode", "seed", "source"}),
        **recorded.model_dump(include={"strategy", "frames", "success_frame", "rejected"}),
        "notes": recorded.notes,
        "status": status(out, candidates),
    }
    flags: list[str] = []
    diff = compared(out)
    if diff is not None and diff.episode == recorded.episode:
        flags += diff.measured.flags
        offsets = [abs(p.offset) for p in diff.measured.pairs if p.offset is not None]
        row |= {
            "keypoint_ratio": diff.measured.worst,
            "keypoint_offset": max(offsets, default=0),
            "keypoint_counts_differ": any(a != b for a, b in diff.measured.counts.values()),
            "commanded_ratio": diff.commanded.worst,
        }
    if reviewed(out):
        review = Reviewed.model_validate_json((out / REVIEW).read_text("utf-8"))
        moved = review.disturbance
        flags += [*moved.flags, *review.problems]
        row |= {
            "keypoints": review.keypoints,
            "disturbance_m": moved.disturbance_m,
            "strike_m": moved.strike_m,
            "side_joint_rad": max(moved.side_joints_rad.values(), default=0.0),
            "package": review.package,
            "package_bytes": review.package_bytes,
            "package_frames": review.package_frames,
            "observation_bytes": review.observation_bytes,
        }
    checked = validation(out)
    if checked is not None:
        row |= {
            "validation": validation_text(checked),
            "validation_attempts": len(checked.attempts),
        }
    return SummaryRow.model_validate({**row, "flags": flags})


def summary(rows: list[SummaryRow]) -> dict[str, object]:
    """`summary.json`: the tasks rejected and the ones flagged for review first, then every row."""
    return {
        "tasks": len(rows),
        "accepted": [r.task for r in rows if r.status == "accepted"],
        "rejected": [r.task for r in rows if r.status == "rejected"],
        "pending": [r.task for r in rows if r.status == "pending"],
        "flagged": {r.task: r.flags for r in rows if r.flags},
        "rows": [r.model_dump(mode="json") for r in rows],
    }


def render_page(out: Path) -> None:
    """`review.html` from the task's files, as far as they go."""
    recorded = read(out)
    assert recorded is not None
    review = (
        Reviewed.model_validate_json((out / REVIEW).read_text(encoding="utf-8"))
        if reviewed(out)
        else None
    )
    keypoints = (
        KeypointSet.model_validate_json((out / "keypoints.json").read_text(encoding="utf-8"))
        if review
        else None
    )
    keyframes = load_package(out / "package").keyframes if review else []
    diff = compared(out)
    checked = validation(out)
    current = recorded.episode
    page = task_page(
        recorded,
        review,
        keypoints,
        keyframes,
        diff if diff is not None and diff.episode == current else None,
        checked if checked is not None and checked.episode == current else None,
    )
    (out / "review.html").write_text(page, "utf-8")


_STYLE: Final = """
body { font: 14px/1.4 system-ui, sans-serif; margin: 16px; color: #1d2330; background: #f7f7f9; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin: 14px 0 4px; }
table { border-collapse: collapse; font-size: 12px; background: #fff; }
td, th { border: 1px solid #d6d9e0; padding: 2px 6px; text-align: left; }
.flag { color: #a33; font-weight: 600; } .ok { color: #2f7d32; } .info { color: #667; }
.thumbs { display: flex; flex-wrap: wrap; gap: 8px; }
figure { margin: 0; } figure img { height: 200px; } figcaption { font-size: 12px; }
"""


def task_page(
    recorded: RecordedTask,
    review: Reviewed | None,
    keypoints: KeypointSet | None,
    keyframes: list[Keyframe],
    diff: KeypointReview | None,
    checked: Validation | None,
) -> str:
    """One task's review page: the candidate and its attempts, then (as far as they exist) the
    validation, video, keypoints against the replayed trajectory, disturbance and keyframes."""
    esc = html.escape
    attempts = "".join(
        f"<li>episode {a.episode} seed {a.seed}, {a.strategy or 'expert re-run'}: {a.outcome}"
        f"{'' if a.frames is None else f', {a.frames} frames'} ({a.seconds:.0f} s)</li>"
        for a in recorded.attempts
    )
    notes = "".join(f"<li>{esc(n)}</li>" for n in recorded.notes)
    source = {"expert_data": "the expert data", "expert_rerun": "a re-run of the expert"}
    head = f"""<!doctype html><html><head><meta charset=utf-8><title>{recorded.task}</title>
<style>{_STYLE}</style></head><body><p><a href="../index.html">all tasks</a></p>
<h1>{recorded.task}: episode {recorded.episode}, seed {recorded.seed}
({recorded.category or "no category"})</h1>
<p>Instruction: <b>{esc(recorded.instruction)}</b></p>
<p>Replayed {source[recorded.source]}; strategy <b>{recorded.strategy}</b>, {recorded.frames}
frames, success latched at frame {recorded.success_frame}.</p><ul>{attempts}{notes}</ul>
<p>Replay validation (the expert's API calls through our executor, judged by the package's
keypoints): <b>{esc(validation_text(checked))}</b></p>"""
    if review is None or keypoints is None:
        return head + "</body></html>\n"
    pairs = {} if diff is None else {(p.arm, p.ours): p for p in diff.measured.pairs}
    info = {} if diff is None else {(p.arm, p.ours): p for p in diff.commanded.pairs}
    rows = []
    for arm, kps in keypoints.arms.items():
        for kp in kps:
            p, m = pairs.get((arm, kp.index)), info.get((arm, kp.index))
            later = recorded.success_frame is not None and kp.frame >= recorded.success_frame
            rows.append(
                f"<tr{' class=flag' if p is not None and p.flagged else ''}><td>{arm}</td>"
                f"<td>K{kp.index}</td><td>{kp.phase}</td><td>{kp.cue}</td><td>{kp.gripper}</td>"
                f"<td>{kp.frame}{' (not required)' if later else ''}</td>"
                f"<td>{', '.join(f'{v:+.3f}' for v in kp.tcp.p)}</td>{_cells(p)}"
                f"<td class=info>{_ratio(m)}</td></tr>"
            )
    flagged = [] if diff is None else diff.measured.flags
    diffs = "".join(f"<li class=flag>{esc(f)}</li>" for f in flagged) or (
        "<li class=ok>no flag</li>" if diff else "<li>pending</li>"
    )
    moved = review.disturbance
    objects = "".join(
        f"<tr><td>{esc(o.name)}</td><td>{'yes' if o.held else ''}</td>"
        f"<td>{'yes' if o.payload else ''}</td><td>{100 * o.free_m:.1f}</td>"
        f"<td>{100 * o.strike_m:.1f}</td><td>{100 * o.net_m:.1f}</td></tr>"
        for o in moved.objects
    )
    joints = ", ".join(f"{esc(k)} {v:.2f} rad" for k, v in moved.side_joints_rad.items())
    flags = "".join(f"<li class=flag>{esc(f)}</li>" for f in moved.flags) or (
        "<li class=ok>no flag</li>"
    )
    problems = "".join(f"<li class=flag>{esc(p)}</li>" for p in review.problems)
    figures = "".join(
        f'<figure><img src="review/keyframe{n}_{view}.jpg"><figcaption>'
        f"{esc(k.caption)} ({view})</figcaption></figure>"
        for n, k in enumerate(keyframes, start=1)
        for view in ("head", "wrist")
    )
    return f"""{head}
<p>Package: {review.package_frames} frames, {review.keyframes} keyframes,
{review.package_bytes / 1e6:.1f} MB; observations {review.observation_bytes / 1e6:.2f} MB per
frame.</p><ul>{problems}</ul>
<video src="review/video.mp4" width=640 controls loop muted preload=metadata></video>
<h2>Keypoints against the replayed trajectory's</h2>
<p>From its measured poses, as ours are; flagged beyond the keypoint's tolerance, more than a
few frames off, or when the counts differ. The last column, the distance share against its
commanded poses (FK), is for information: grasps stop above their commanded target.</p>
<ul>{diffs}</ul>
<table><tr><th>arm</th><th>kp</th><th>phase</th><th>cue</th><th>gripper</th><th>frame</th>
<th>TCP xyz (m)</th><th>reference</th><th>frames off</th><th>cm off (xy, z)</th>
<th>x tolerance</th><th>x tolerance, commanded</th></tr>{"".join(rows)}</table>
<h2>Disturbance (ground truth)</h2><ul>{flags}</ul>
<table><tr><th>object</th><th>held</th><th>payload</th><th>moved free (cm)</th>
<th>arm strike (cm)</th><th>net (cm)</th></tr>{objects}</table>
<p>Side-joint motion: {joints or "no articulated object"}.</p>
<h2>Keyframes</h2><div class=thumbs>{figures}</div></body></html>
"""


def _cells(p: KeypointPair | None) -> str:
    """A keypoint's comparison with the measured poses' keypoint (none: not compared)."""
    if p is None:
        return "<td></td>" * 4
    if p.xy_mm is None or p.dz_mm is None or p.ratio is None:
        return "<td>none</td>" + "<td></td>" * 3
    return (
        f"<td>K{p.reference} ({p.reference_phase})</td><td>{p.offset:+d}</td>"
        f"<td>{p.xy_mm / 10:.1f}, {p.dz_mm / 10:+.1f}</td><td>{p.ratio:.2f}</td>"
    )


def _ratio(p: KeypointPair | None) -> str:
    return "" if p is None or p.ratio is None else f"{p.ratio:.2f}"


def index_page(rows: list[SummaryRow]) -> str:
    esc = html.escape
    checks = "".join(
        f"<li><a href='{r.task}/review.html'>{r.task}</a>: "
        + ("<b>rejected</b> " if r.status == "rejected" else "")
        + esc(" / ".join([*r.flags, *(r.notes if r.status == "rejected" else [])]))
        + "</li>"
        for r in rows
        if r.flags or r.status == "rejected"
    )
    cells = "".join(
        f"<tr><td><a href='{r.task}/review.html'>{r.task}</a></td><td>{r.category}</td>"
        f"<td>{r.status}</td><td>{r.episode}</td><td>{r.seed}</td><td>{r.strategy}</td>"
        f"<td>{r.success_frame} / {r.frames}</td><td>{json.dumps(r.keypoints or {})}</td>"
        f"<td>{_num(r.keypoint_ratio, '.2f')}</td><td>{_num(r.keypoint_offset, 'd')}</td>"
        f"<td>{_num(r.commanded_ratio, '.2f')}</td>"
        f"<td>{_num(r.disturbance_m and 100 * r.disturbance_m, '.1f')}</td>"
        f"<td>{_num(r.strike_m and 100 * r.strike_m, '.1f')}</td>"
        f"<td>{_num(r.side_joint_rad, '.2f')}</td>"
        f"<td>{_num(r.package_bytes and r.package_bytes / 1e6, '.0f')}</td>"
        f"<td>{esc(r.validation)}</td></tr>"
        for r in rows
    )
    head = "<tr><th>task</th><th>category</th><th>status</th><th>episode</th><th>seed</th>"
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Recorded demos</title>
<style>{_STYLE}</style></head><body><h1>Re-recorded demonstrations ({len(rows)} tasks)</h1>
<h2>To check</h2><ul>{checks or "<li class=ok>nothing flagged</li>"}</ul>
<table>{head}<th>strategy</th><th>success / frames</th><th>keypoints</th>
<th>keypoint x tolerance</th><th>frames off</th><th>x tolerance, commanded</th>
<th>disturbance cm</th><th>strike cm</th>
<th>side joint rad</th><th>package MB</th><th>validation</th></tr>{cells}</table></body></html>
"""


def _num(value: float | None, spec: str) -> str:
    return "" if value is None else format(value, spec)
