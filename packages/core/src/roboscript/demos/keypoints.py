"""Keypoints of a demonstration: per arm, the ordered poses the task passes through.

Found from an arm's TCP track and gripper command alone, so the same code runs on any recording:
gripper events where the command crosses `grip_threshold` (closed below it), the ends of the
arm's legs (it came to rest: a stop, however long it then dwells), speed dips inside motion
(valleys), and shallower dips where the path turns sharply (corners). Planned legs curve smoothly
between rests, so a turn without a slowdown is the planner's arc, not a step of the task. From a
gripper change until the arm next rests, its motion is the fingers pushing it, not a leg. A motion
keypoint that moved less than `min_move_m` and turned less than `min_turn_deg` from the previous
keypoint is dropped, and one at the pose where the gripper then changes gives way to the gripper
event. A close at the start pose, before the arm has moved, is a `preshape`: only the change
counts, not where it happens. Each arm is numbered on its own (K1, K2, ...); the arms need not
arrive together. A keypoint's `since` is where the arm's motion toward it
began: the first frame after the arm's previous keypoint at which it moves or its gripper
changes, so the referee can tell the arms moving at once from one after the other.

Speeds are per physics step, so the thresholds hold at any frame rate. Frames count as
`frame_steps` apart, except that a recorder saving a frame at every action boundary repeats the
previous frame there and saves one more after the action's first physics step:
speed estimates leave out such a lone copy and the frame after a repeat
(`timed_frames`). Repeats in a row are a rest, not copies: poses from commands (the FK of joint
drive targets) repeat exactly while the robot stands, and a rest between two actions is where a
leg ends. A planner's legs typically peak at 1-2.5 mm and up to about 0.6 deg per step, and read
zero at rest.
"""

from collections.abc import Iterable
from itertools import pairwise
from typing import Annotated, Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import Arm
from roboscript.demos.demo import ArmTrack, Array, Demo
from roboscript.settings import config_hash
from roboscript.spatial.rotation import quat_angle_deg, quat_multiply

Phase = Literal["preshape", "approach", "grasp", "lift", "carry", "place", "release", "retreat"]
Cue = Literal["close", "open", "stop", "valley", "corner"]
GRIP_CUES: Final = ("close", "open")
KEYPOINTS_SCHEMA: Final = "rs.keypoints/1"
Share = Annotated[float, Field(gt=0, lt=1)]


class Tolerance(BaseModel):
    """Where a keypoint counts as reached: within `xy_m` horizontally and `z_m` vertically (a
    vertical cylinder) and, when `angle_deg` is set, with the gripper turned at most that far."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    xy_m: PositiveFloat = 0.02
    z_m: PositiveFloat = 0.01
    angle_deg: PositiveFloat | None = None


class KeypointConfig(BaseModel):
    """Extraction parameters: durations in physics steps, speeds per physics step. The combined
    speed is translation / `speed_ref_m` + rotation / `speed_ref_deg`: about 1 for a planned leg
    at full speed; the arm moves while it exceeds `moving`. After an open, the arm's motion until
    it rests is the fingers' push only within `release_push_m` of where the open began: an arm
    that goes farther (a withdraw commanded with the release) makes a leg, whose end is a
    keypoint (a retreat). After a close, the same holds beyond `close_push_m` (None: never): a
    recording whose frames lie far apart shows no rest between a close and the lift that follows
    it, so without it the lift and everything after read as the fingers' push and no keypoint
    follows the close."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grip_threshold: Share = 0.5
    speed_ref_m: PositiveFloat = 1e-3
    speed_ref_deg: PositiveFloat = 0.3
    moving: PositiveFloat = 0.05
    valley_ratio: Share = 0.2
    corner_ratio: Share = 0.6
    corner_deg: Annotated[float, Field(gt=0, le=180)] = 60.0
    corner_window_steps: PositiveInt = 45
    min_leg_steps: PositiveInt = 90
    min_move_m: PositiveFloat = 0.01
    min_turn_deg: PositiveFloat = 8.0
    lift_m: PositiveFloat = 0.02
    release_push_m: PositiveFloat = 0.03
    close_push_m: PositiveFloat | None = 0.04
    tolerance: Tolerance = Tolerance()
    angle_deg_by_category: dict[str, PositiveFloat] = Field(default_factory=dict)

    def tolerance_for(self, category: str | None) -> Tolerance:
        """The default tolerance with the orientation tolerance of the task's category."""
        angle = self.angle_deg_by_category.get(category or "", self.tolerance.angle_deg)
        return self.tolerance.model_copy(update={"angle_deg": angle})


class Keypoint(BaseModel):
    """`frame` indexes the demonstration; `gripper` is the state from this keypoint on; `since`
    is the frame the arm began its motion toward it (None: not recorded; the referee then takes
    `frame`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: PositiveInt
    frame: NonNegativeInt
    since: NonNegativeInt | None = None
    tcp: Pose
    gripper: Literal["open", "closed"]
    cue: Cue
    phase: Phase
    tolerance: Tolerance

    @model_validator(mode="after")
    def _since(self) -> Self:
        if self.since is not None and self.since > self.frame:
            raise ValueError(f"K{self.index} begins at frame {self.since}, after its frame")
        return self

    @property
    def name(self) -> str:
        return f"K{self.index}"

    @property
    def segment(self) -> tuple[int, int]:
        """The frames of the arm's motion toward this keypoint, [since, frame]."""
        return (self.frame if self.since is None else self.since), self.frame

    @property
    def grip(self) -> bool:
        """A gripper event, reached only where the gripper makes the same change."""
        return self.cue in GRIP_CUES


class KeypointSet(BaseModel):
    """`keypoints.json`: one demonstration's keypoints per arm, numbered 1, 2, ... in frame order
    (a hand-edited file must keep that); arms without any are left out. `pointing` is each arm's
    gripper axis in the TCP frame (along its ee->TCP offset), about which half a turn holds the
    same way. `success_frame` is the first frame with the task's success latched; only keypoints
    up to it can be required of an attempt, so judging needs it."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.keypoints/1"] = Field(default=KEYPOINTS_SCHEMA, alias="schema")
    task: TaskName
    seed: int
    episode: NonNegativeInt | None = None
    frame_steps: PositiveInt
    success_frame: NonNegativeInt | None = None
    config_hash: str
    pointing: dict[Arm, Vec3]
    arms: dict[Arm, list[Keypoint]]

    @model_validator(mode="after")
    def _in_order(self) -> Self:
        for arm, kps in self.arms.items():
            if arm not in self.pointing:
                raise ValueError(f"no pointing axis for the {arm} arm")
            if [k.index for k in kps] != list(range(1, len(kps) + 1)):
                raise ValueError(f"{arm} keypoints must be numbered 1, 2, ... in order")
            if any(b.frame < a.frame for a, b in pairwise(kps)):
                raise ValueError(f"{arm} keypoints must be in frame order")
        return self


def pointing_axis(ee_to_tcp: Vec3) -> Vec3:
    """The gripper's pointing axis in its own frame: the direction of the ee->TCP offset."""
    x, y, z = (np.asarray(ee_to_tcp) / np.linalg.norm(ee_to_tcp)).tolist()
    return x, y, z


def gripper_angle_deg(a: ArrayLike, b: ArrayLike, pointing: Vec3) -> Array:
    """Degrees between gripper orientations (w, x, y, z), broadcasting; a parallel gripper turned
    half a turn about its `pointing` axis holds the same way, so that counts as no turn."""
    half_turn = np.array([0.0, *pointing])
    return np.minimum(quat_angle_deg(a, b), quat_angle_deg(a, quat_multiply(b, half_turn)))


def timed_frames(tracks: Iterable[ArmTrack]) -> NDArray[np.bool_]:
    """Frames usable for speed estimates: not the recorder's copy of the frame before (a repeat,
    every track's TCP and gripper equal, with no repeat before or after it: the recorder copies
    one frame per action start) nor the frame right after a repeat. Repeats in a row are a rest:
    the robot stood still for a frame's worth of physics steps, which poses taken from commands
    (the forward kinematics of joint drive targets) show exactly, so they count, at speed 0."""
    tracks = list(tracks)
    same = np.ones(len(tracks[0].tcp) - 1, dtype=bool)
    for t in tracks:
        same &= np.all(t.tcp[1:] == t.tcp[:-1], axis=1) & (t.gripper[1:] == t.gripper[:-1])
    repeat = np.r_[False, same]  # frame t repeats frame t - 1
    before, then = np.r_[False, repeat[:-1]], np.r_[repeat[1:], False]
    lone = repeat & ~before & ~then
    after = before & ~repeat
    timed: NDArray[np.bool_] = ~(lone | after)
    return timed


def arm_keypoints(
    track: ArmTrack,
    frame_steps: int,
    config: KeypointConfig,
    tolerance: Tolerance,
    timed: NDArray[np.bool_],
) -> list[Keypoint]:
    """Keypoints of one arm; `timed` marks the frames speeds are estimated at (`timed_frames`)."""
    tcp = track.tcp
    closed = track.gripper < config.grip_threshold
    events: list[tuple[int, Cue]] = [
        (t, "close" if closed[t] else "open")
        for t in (np.flatnonzero(closed[1:] != closed[:-1]) + 1).tolist()
    ]
    held = _held(track.gripper, _speed(tcp, frame_steps, config), config.moving, tcp,
                 config.release_push_m, config.close_push_m)  # fmt: skip
    at = np.flatnonzero(timed)
    speed = _speed(tcp[at], frame_steps, config)
    speed[held[at]] = 0.0
    motion = [(int(at[j]), cue) for j, cue in _motion_cues(tcp[at], speed, frame_steps, config)]
    # At one frame a motion cue comes first, so that it can give way to the gripper event.
    points: list[tuple[int, Cue]] = []
    ref = tcp[0]
    for t, cue in sorted(events + motion, key=lambda c: (c[0], c[1] in GRIP_CUES)):
        grip = cue in GRIP_CUES
        last_moved = points and points[-1][1] not in GRIP_CUES
        if grip and last_moved and _near(tcp[points[-1][0]], tcp[t], config):
            points.pop()
        if grip or not _near(ref, tcp[t], config):
            points.append((t, cue))
            ref = tcp[t]
    phases = _phases(points, closed, tcp[:, 2], config.lift_m)
    # Where each keypoint's segment begins: the arm's first move or gripper change after the
    # previous keypoint (every frame counts, repeats included: they read as standing still).
    active = _speed(tcp, frame_steps, config) > config.moving
    active[1:] |= np.diff(track.gripper) != 0
    starts, before = [], 0
    for t, _ in points:
        began = np.flatnonzero(active[before + 1 : t + 1])
        starts.append(before + 1 + int(began[0]) if len(began) else t)
        before = t
    for i, (t, cue) in enumerate(points):
        moved = any(c not in GRIP_CUES for _, c in points[:i])
        if cue == "close" and not moved and _near(tcp[0], tcp[t], config):
            phases[i] = "preshape"
    return [
        Keypoint(
            index=k,
            frame=t,
            since=since,
            tcp=Pose.from_list(tcp[t]),
            gripper="closed" if closed[t] else "open",
            cue=cue,
            phase=phase,
            tolerance=tolerance,
        )
        for k, ((t, cue), phase, since) in enumerate(
            zip(points, phases, starts, strict=True), start=1
        )
    ]


def demo_keypoints(demo: Demo, config: KeypointConfig, tolerance: Tolerance) -> KeypointSet:
    meta = demo.meta
    timed = timed_frames(demo.arms.values())
    arms = {
        arm: arm_keypoints(track, meta.frame_steps, config, tolerance, timed)
        for arm, track in demo.arms.items()
    }
    return KeypointSet(
        task=meta.task,
        seed=meta.seed,
        episode=meta.provenance.episode,
        frame_steps=meta.frame_steps,
        success_frame=meta.success_frame,
        config_hash=config_hash(config),
        pointing={arm: pointing_axis(offset) for arm, offset in meta.ee_to_tcp.items()},
        arms={arm: kps for arm, kps in arms.items() if kps},
    )


def _speed(tcp: Array, frame_steps: int, config: KeypointConfig) -> Array:
    """Combined speed arriving at each frame (0 at the first)."""
    move = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1)
    turn = quat_angle_deg(tcp[:-1, 3:], tcp[1:, 3:])
    speed = np.zeros(len(tcp))
    speed[1:] = (move / config.speed_ref_m + turn / config.speed_ref_deg) / frame_steps
    return speed


def _held(
    gripper: Array,
    speed: Array,
    moving: float,
    tcp: Array,
    release_push_m: float,
    close_push_m: float | None = None,
) -> NDArray[np.bool_]:
    """Frames from each gripper change until the arm next rests (every frame, so an action
    boundary's repeated frame counts as rest). An arm stands while its own gripper moves; what
    the TCP does until it settles is the fingers pushing on something (closing on a handle can
    shove the arm centimetres), not a leg. Open fingers push on nothing: an arm farther than
    `release_push_m` from where an open began is making a commanded move (a withdraw), a leg
    from there on, until it rests. With `close_push_m`, so is an arm that far from where a close
    began (closing fingers shove it by less: a lift that started before a frame showed it at
    rest)."""
    diff = np.diff(gripper)
    changing, opening = np.r_[False, diff != 0], np.r_[False, diff > 0]
    held = np.zeros(len(gripper), dtype=bool)
    hold = escaped = False
    origin: Array | None = None
    push = release_push_m
    for j in range(len(gripper)):
        if changing[j] and not hold and not escaped:
            limit = release_push_m if opening[j] else close_push_m
            hold, origin = True, None if limit is None else tcp[j, :3]
            push = limit if limit is not None else push
        elif not changing[j] and hold and speed[j] <= moving:
            hold = False
        if hold and origin is not None and np.linalg.norm(tcp[j, :3] - origin) > push:
            hold = False
            escaped = True  # the rest of this open's ramp is part of the leg
        if escaped and not changing[j] and speed[j] <= moving:
            escaped = False
        held[j] = hold
    return held


def runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _motion_cues(
    tcp: Array, speed: Array, frame_steps: int, config: KeypointConfig
) -> list[tuple[int, Cue]]:
    """Leg ends ("stop", the first frame at rest), then deep dips ("valley") and shallower ones
    on a sharp turn ("corner") inside each run of motion, deepest first, `min_leg_steps` apart."""
    spacing = max(1, round(config.min_leg_steps / frame_steps))
    window = max(1, round(config.corner_window_steps / frame_steps))
    cues: list[tuple[int, Cue]] = []
    for a, b in runs(speed > config.moving):  # arriving at frames a..b-1: at rest from b
        cuts: list[tuple[int, Cue]] = []
        minima = [t for t in range(a + 1, b - 1) if speed[t] <= min(speed[t - 1], speed[t + 1])]
        for t in sorted(minima, key=lambda t: (speed[t], t)):
            left = max((c for c, _ in cuts if c < t), default=a)
            right = min((c for c, _ in cuts if c > t), default=b)
            if t - left < spacing or right - t < spacing:
                continue
            peak = min(speed[left:t].max(), speed[t:right].max())
            if speed[t] < config.valley_ratio * peak:
                cuts.append((t, "valley"))
            elif (
                speed[t] < config.corner_ratio * peak
                and _turn_deg(tcp, t, window) >= config.corner_deg
            ):
                cuts.append((t, "corner"))
        cues += [*cuts, (min(b, len(speed) - 1), "stop")]
    return cues


def _turn_deg(tcp: Array, t: int, window: int) -> float:
    """Degrees between the path's direction over the `window` frames before t and after it."""
    p = tcp[:, :3]
    before, after = p[t] - p[max(t - window, 0)], p[min(t + window, len(p) - 1)] - p[t]
    norms = np.linalg.norm(before) * np.linalg.norm(after)
    if norms < 1e-12:  # turning in place: no direction to change
        return 0.0
    return float(np.degrees(np.arccos(np.clip(before @ after / norms, -1.0, 1.0))))


def _near(a: Array, b: Array, config: KeypointConfig) -> bool:
    moved = float(np.linalg.norm(a[:3] - b[:3]))
    return moved < config.min_move_m and float(quat_angle_deg(a[3:], b[3:])) < config.min_turn_deg


def _phases(
    points: list[tuple[int, Cue]], closed: NDArray[np.bool_], z: Array, lift_m: float
) -> list[Phase]:
    """Grasp and release are the gripper events. Holding: the first move after a grasp that
    rises `lift_m` is a lift, the last move before a release a place (so is a last move down
    when nothing is released), the rest carry. Open: moves heading for a grasp approach it, the
    first move after a release and every move after the last grasp retreat."""
    grips = [i for i, (_, cue) in enumerate(points) if cue in GRIP_CUES]
    last = len(points) - 1
    out: list[Phase] = []
    for i, (t, cue) in enumerate(points):
        prev = max((g for g in grips if g < i), default=-1)
        nxt = min((g for g in grips if g > i), default=last + 1)
        prev_cue = points[prev][1] if prev >= 0 else None
        next_cue = points[nxt][1] if nxt <= last else None
        phase: Phase
        if cue in GRIP_CUES:
            phase = "grasp" if cue == "close" else "release"
        elif closed[t]:
            lifted = prev_cue == "close" and i == prev + 1 and z[t] - z[points[prev][0]] >= lift_m
            before_release = next_cue == "open" and i == nxt - 1
            lowered_last = (
                next_cue is None and 0 < i == last and z[t] <= z[points[i - 1][0]] - lift_m
            )
            phase = "lift" if lifted else "place" if before_release or lowered_last else "carry"
        elif next_cue == "close":
            phase = "retreat" if prev_cue == "open" and i == prev + 1 else "approach"
        else:
            phase = "approach" if prev_cue is None else "retreat"
        out.append(phase)
    return out
