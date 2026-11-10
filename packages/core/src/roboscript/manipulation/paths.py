"""Held paths (docs/concepts/embodied-core.md): what a gripper did with what it held, as a
`PathSpec`, and the moves that do it again.

A path is fitted (`fit_path`) from the TCP poses of the demonstration's keypoints from a close on,
in the body frame of what the gripper holds: the whole object's box as measured where it was
grasped, in the first of the frames a pose grasp takes on it (`body_frames`). Where the gripper
holds an articulated part, the body is the object the part turns on, which stays put.

What a path reproduces is the rigid motion of the held thing relative to that frame: with P_i
the path's poses, the world motion to waypoint i is M_i = F P_i P_0^-1 F^-1 (F the frame in the
world), the same for every point of what is held. Following it (`ActionApi.follow_path`), each
holding gripper goes from where it holds, G, to M_i G: a grasp a few millimeters off the
demonstration's is carried along rather than corrected, an arc about a hinge stays an arc about
the same line (no hinge is guessed), and two grippers holding one object move by the same motion,
neither retreating along its own grasp. F is chosen at run time among the body frames the box
allows (`choose_frame`): the one in which the path's first pose lies nearest a holding gripper's
TCP, position and orientation, so a box measured turned is read the way the grasp was made
(`hold_frames`: also the whole's frame in each turn its shape allows, whatever decided its x). Each
waypoint's leg is cut into straight moves along the screw motion between consecutive waypoints
(`screw`, `legs`): at most `PATH_STEP_M` of the TCP and `PATH_STEP_DEG` of turn each.

Where the path ends in a motion on one joint (`joint_tail`: from some waypoint on, every leg a
turn about one fixed axis with the TCP on one circle about it, or a slide along one line, going
on the same way), that end is a held part's pull: replayed as rigid motions from the start, a
frame a few millimeters or a degree off puts its arc beside the part's own, and the arm fights
the hinge (the fingers slip or the whole object is dragged). `ActionApi.follow_path` follows that
end on the joint instead, stepping from where the gripper is.

Pure numpy over the proxy and the spec.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm
from roboscript.manipulation.articulation import NOISE_M, Joint, circle_residual, fit_line
from roboscript.manipulation.proxy import ObjectProxy
from roboscript.manipulation.specs import HIGH_END, LOW_END, MAX_PATH_POSES, GraspSpec, PathSpec
from roboscript.manipulation.synthesis import variants
from roboscript.spatial.pose import pose_to_matrix, transform
from roboscript.spatial.rotation import (
    quat_from_axis_angle,
    quat_from_matrix,
    quat_to_matrix,
    yaw_matrix,
)

Array = NDArray[np.float64]

PATH_STEP_M: Final = 0.02
"""A path's straight moves carry the TCP at most this far each."""
PATH_STEP_DEG: Final = 10.0
"""And turn it at most this much each."""
FRAME_M_PER_DEG: Final = 0.002
"""Choosing a path's frame, a degree of the start pose's orientation counts as this many meters
of its position (a frame turned 90 deg is 18 cm away)."""
JOINT_MIN_TURN_DEG: Final = 10.0
"""A path's end turns at least this much about its axis to be followed as a hinge's turn."""
JOINT_STILL_DEG: Final = 2.0
"""A slide's legs turn at most this much each."""
JOINT_AXIS_DEG: Final = 20.0
"""A turning leg's axis lies within this of the end's (a leg turning under `JOINT_STILL_DEG`
says little about its axis and is not checked)."""
JOINT_MIN_SLIDE_M: Final = 0.02
"""A path's end slides at least this far to be followed as a slide."""
DIGITS: Final = 4
"""A fitted path's numbers are rounded to this many decimals (0.1 mm)."""


@dataclass(frozen=True)
class PathFit:
    """A fitted held path: the spec, the TCP's travel along it (m) and the turn of its motion
    from the first pose to the last (deg), and what rounding leaves over (the largest)."""

    spec: PathSpec
    travel_m: float
    turn_deg: float
    position_error_m: float
    angle_error_deg: float


def body_frames(proxy: ObjectProxy, x_toward: str | None = None) -> list[Array]:
    """The frames (world, 4x4) a held path may be written in on `proxy`, the whole object as
    measured: a pose grasp's (`fit_grasp_pose`), its box with x toward `x_toward` (a path's own
    `PathSpec.x_toward`, a front told by height), else toward its front where its part or its
    footprint decides one (`ObjectProxy.front` without height: what a path with no `x_toward` is
    written in), in each turn the box allows; the first is the one a fit writes in."""
    toward = x_toward if x_toward is not None else proxy.front(by_height=False)
    shape = GraspSpec(approach="top", x_toward=toward)
    return [proxy.pose @ variant.frame for variant in variants(shape, proxy)]


def path_front(proxy: ObjectProxy) -> str | None:
    """The `PathSpec.x_toward` a fit on `proxy` writes: its front where only height tells it
    (`ObjectProxy.front`: `low_end`), so the frame it is written in is named in the spec and
    found again at run time; None where its part or footprint decides it (or nothing does),
    the frame every path without one is followed in."""
    front = proxy.front()
    return front if front in (LOW_END, HIGH_END) else None


def hold_frames(proxy: ObjectProxy, x_toward: str | None = None) -> list[Array]:
    """The frames a held path (written toward `x_toward`, `PathSpec.x_toward`) is chosen among
    at run time (`choose_frame`): `body_frames`, and the whole's own frame in every turn its
    measured shape looks the same under (`ObjectProxy.shape_symmetry`, a half or quarter turn),
    however its x was decided. A frame decided between the fit and the run (a model's answer, a
    `far_end` cue) may lie a turn from the one the path was written in; among these the hold
    picks it."""
    frames = body_frames(proxy, x_toward)
    turns = {"pi": 180, "half_pi": 90}.get(proxy.shape_symmetry or proxy.symmetry)
    if turns is None:
        return frames
    whole = proxy.pose @ proxy.whole.frame
    for turn in range(0, 360, turns):
        frame = whole @ transform(yaw_matrix(math.radians(turn)))
        if not any(_same_frame(frame, f) for f in frames):
            frames.append(frame)
    return frames


def _same_frame(a: Array, b: Array) -> bool:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) < 1e-6 and _angle(a, b) < 1e-3


def fit_path(tcps: Sequence[Pose], proxy: ObjectProxy, arm: Arm | None = None) -> PathFit:
    """The held path of the TCP poses `tcps` (the first at the close) on `proxy`, the body as it
    was when the gripper closed, written for `arm` (the arm that held; the other follows its
    mirror image); a ValueError for fewer than two poses or more than `MAX_PATH_POSES`."""
    if not 2 <= len(tcps) <= MAX_PATH_POSES:
        raise ValueError(
            f"a held path takes 2 to {MAX_PATH_POSES} poses from the close on, got {len(tcps)}"
        )
    toward = path_front(proxy)
    frame = body_frames(proxy, toward)[0]
    inverse = np.linalg.inv(frame)
    local = [inverse @ tcp.matrix() for tcp in tcps]
    spec = PathSpec(poses=[_rounded(m) for m in local], x_toward=toward, arm=arm)
    rebuilt = [frame @ path_matrix(pose) for pose in spec.poses]
    position = max(
        float(np.linalg.norm(r[:3, 3] - np.asarray(t.p)))
        for r, t in zip(rebuilt, tcps, strict=True)
    )
    angle = max(_angle(r, t.matrix()) for r, t in zip(rebuilt, tcps, strict=True))
    travel = sum(float(np.linalg.norm(np.subtract(b.p, a.p))) for a, b in pairwise(tcps))
    return PathFit(spec, travel, _angle(tcps[0].matrix(), tcps[-1].matrix()), position, angle)


def path_matrix(pose: Sequence[float]) -> Array:
    """A path pose (x, y, z, qw, qx, qy, qz) as a 4x4 matrix."""
    return pose_to_matrix(np.asarray(pose, dtype=float))


def choose_frame(
    frames: Sequence[Array], spec: PathSpec, tcps: Sequence[Pose]
) -> tuple[Array, float, float]:
    """Of `frames`, the one in which the path's first pose lies nearest one of the holding
    grippers' TCPs `tcps` (position plus `FRAME_M_PER_DEG` per degree), with that distance (m)
    and turn (deg)."""
    if not frames or not tcps:
        raise ValueError("choosing a path's frame needs a frame and a holding gripper")
    start = path_matrix(spec.poses[0])
    best: tuple[float, Array, float, float] | None = None
    for frame in frames:
        at = frame @ start
        for tcp in tcps:
            off = float(np.linalg.norm(at[:3, 3] - np.asarray(tcp.p)))
            turn = _angle(at, tcp.matrix())
            cost = off + FRAME_M_PER_DEG * turn
            if best is None or cost < best[0]:
                best = (cost, frame, off, turn)
    assert best is not None
    return best[1], best[2], best[3]


_TCP_FLIPS: Final = (np.diag([1.0, -1.0, 1.0, 1.0]), np.diag([1.0, 1.0, -1.0, 1.0]))
"""The reflections of a TCP's own axes that make a reflected gripper pose proper again: its
closing axis reversed (y or z: the gripper as a mirrored grasp holds, in either roll about its
pointing axis), the pointing axis, x, kept."""


def mirror_path(spec: PathSpec, frame: Array, lateral: ArrayLike, flip: Array) -> PathSpec:
    """The path as the other arm makes it in `frame` (world): every pose reflected through the
    frame across the vertical plane of its axes whose normal, its x or its y, is nearer the
    robot's lateral axis `lateral` (world; as a mirrored grasp, `synthesis.mirror`), its
    orientation made proper again by the TCP-axis reflection `flip` (one of `_TCP_FLIPS`; the
    same for every pose, so the motion it makes does not depend on it)."""
    axes = np.asarray(frame, dtype=float)[:3, :3]
    along = np.abs(axes[:, :2].T @ np.asarray(lateral, dtype=float))
    plane = np.eye(4)
    plane[int(np.argmax(along)), int(np.argmax(along))] = -1.0
    poses = [_rounded(plane @ path_matrix(pose) @ flip) for pose in spec.poses]
    return spec.model_copy(update={"poses": poses})


def choose_mirrored(
    frames: Sequence[Array], spec: PathSpec, tcps: Sequence[Pose], lateral: ArrayLike
) -> tuple[Array, PathSpec, float, float]:
    """`choose_frame` for the path's mirror image (`mirror_path`), reflected in each frame (which
    of its axes is lateral depends on its turn) and made proper by whichever TCP-axis
    reflection best meets a holding gripper: the frame, the mirrored path in it, the distance
    (m) and turn (deg)."""
    best: tuple[float, Array, PathSpec, float, float] | None = None
    for frame in frames:
        for flip in _TCP_FLIPS:
            mirrored = mirror_path(spec, frame, lateral, flip)
            _, off, turn = choose_frame([frame], mirrored, tcps)
            cost = off + FRAME_M_PER_DEG * turn
            if best is None or cost < best[0]:
                best = (cost, frame, mirrored, off, turn)
    if best is None:
        raise ValueError("choosing a path's frame needs a frame and a holding gripper")
    return best[1], best[2], best[3], best[4]


def motions(frame: Array, spec: PathSpec) -> list[Array]:
    """The world motions (4x4) of what is held from the path's first pose to each later one, in
    `frame` (world): M_i = F P_i P_0^-1 F^-1."""
    poses = [path_matrix(pose) for pose in spec.poses]
    first = np.linalg.inv(poses[0])
    inverse = np.linalg.inv(frame)
    return [frame @ pose @ first @ inverse for pose in poses[1:]]


@dataclass(frozen=True)
class JointTail:
    """The end of a held path that moves on one joint: from waypoint `start` on (0: the path's
    start; waypoints counted from 1) to the last, about or along `joint` (world, its `point` on
    the axis or the line, `direction` the way the path goes, `source` `guess`); `travels` the
    travel of each of those later waypoints from waypoint `start` along it (deg about a revolute
    joint, m along a prismatic one), `target` the last."""

    start: int
    joint: Joint
    travels: tuple[float, ...]

    @property
    def target(self) -> float:
        return self.travels[-1]


def joint_tail(worlds: Sequence[Array], tcp: Array) -> JointTail | None:
    """The longest end of a held path (`worlds`, its world motions from the start, as `motions`
    gives them; `tcp` the holding TCP at the start, 4x4) of at least two legs that moves on one
    joint: a turn of at least `JOINT_MIN_TURN_DEG` about one axis (no slide along it beyond
    `NOISE_M`, every turning leg's axis within `JOINT_AXIS_DEG` of it, the TCP within `NOISE_M`
    of one circle about it) or a slide of at least `JOINT_MIN_SLIDE_M` (no leg turning over
    `JOINT_STILL_DEG`, the TCP within `NOISE_M` of one line), either going on the same way at
    every waypoint; None when no end does (a lift, a carry, a shake)."""
    full = [np.eye(4), *worlds]
    points = np.array([(m @ tcp)[:3, 3] for m in full])
    for start in range(len(full) - 2):
        tail = _tail(full[start:], points[start:])
        if tail is not None:
            return JointTail(start, tail[0], tail[1])
    return None


def _tail(full: Sequence[Array], points: Array) -> tuple[Joint, tuple[float, ...]] | None:
    """The joint the motions `full` (the first the end's start) move the TCP `points` on."""
    first = np.linalg.inv(full[0])
    angle, axis, center, slide = screw_axis(full[-1] @ first)
    steps = [screw_axis(b @ np.linalg.inv(a)) for a, b in pairwise(full)]
    if angle >= JOINT_MIN_TURN_DEG:
        if abs(slide) > NOISE_M or circle_residual(points, center, axis) > NOISE_M:
            return None
        lean = math.cos(math.radians(JOINT_AXIS_DEG))
        if any(a > JOINT_STILL_DEG and u @ axis < lean for a, u, _, _ in steps):
            return None
        joint = Joint("revolute", center, axis, 0.0, "guess")
    else:
        if any(a > JOINT_STILL_DEG for a, _, _, _ in steps):
            return None
        _, direction, residual = fit_line(points)
        if residual > NOISE_M:
            return None
        joint = Joint("prismatic", points[0], direction, 0.0, "guess")
    travels = [joint.travel(points[0], p) for p in points[1:]]
    tolerance = 1.0 if joint.kind == "revolute" else NOISE_M  # deg, m: a step back is noise
    going = all(b >= a - tolerance for a, b in pairwise([0.0, *travels]))
    least = JOINT_MIN_TURN_DEG if joint.kind == "revolute" else JOINT_MIN_SLIDE_M
    if not going or travels[-1] < least:
        return None
    return joint, tuple(travels)


def screw_axis(motion: Array) -> tuple[float, Array, Array, float]:
    """A rigid motion's screw: its turn (deg, 0 to 180), the unit axis it turns about
    (right-handed; any unit vector for no turn), a point on that axis (the origin for no turn)
    and its slide along the axis (m; for no turn, the whole shift's length)."""
    rotation, shift = motion[:3, :3], motion[:3, 3]
    q = quat_from_matrix(rotation)
    q = q if q[0] >= 0 else -q
    sin_half = float(np.linalg.norm(q[1:]))
    angle = 2.0 * math.atan2(sin_half, float(q[0]))
    if angle < 1e-9:
        return 0.0, np.array([0.0, 0.0, 1.0]), np.zeros(3), float(np.linalg.norm(shift))
    axis = q[1:] / sin_half
    along = float(shift @ axis)
    center = np.linalg.lstsq(np.eye(3) - rotation, shift - along * axis, rcond=None)[0]
    return math.degrees(angle), axis, center, along


def screw(motion: Array, s: float) -> Array:
    """The fraction `s` of a rigid motion along its screw: the turn about its axis by s of its
    angle, with s of its slide along the axis (a straight line for a pure translation, an arc
    about the axis for a pure turn)."""
    angle, axis, center, along = screw_axis(motion)
    out = np.eye(4)
    if angle == 0.0:
        out[:3, 3] = s * motion[:3, 3]
        return out
    turn = quat_to_matrix(quat_from_axis_angle(axis, math.radians(s * angle)))
    out[:3, :3] = turn
    out[:3, 3] = s * along * axis + (np.eye(3) - turn) @ center
    return out


def legs(
    worlds: Sequence[Array],
    tcps: Sequence[Array],
    step_m: float = PATH_STEP_M,
    step_deg: float = PATH_STEP_DEG,
) -> list[list[Array]]:
    """Per waypoint, the world motions (from the path's start) of its straight moves: the screw
    from the waypoint before to it, cut so that no TCP of `tcps` (4x4, where each gripper holds
    at the start) moves more than `step_m` or turns more than `step_deg` in one."""
    out: list[list[Array]] = []
    before = np.eye(4)
    for world in worlds:
        step = world @ np.linalg.inv(before)
        n = 1
        for tcp in tcps:
            a, b = before @ tcp, world @ tcp
            far = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
            n = max(n, math.ceil(far / step_m - 1e-9), math.ceil(_angle(a, b) / step_deg - 1e-9))
        out.append([screw(step, k / n) @ before for k in range(1, n + 1)])
        before = world
    return out


def _rounded(m: Array) -> tuple[float, float, float, float, float, float, float]:
    q = quat_from_matrix(m[:3, :3])
    q = q if q[0] >= 0 else -q
    values = [*m[:3, 3], *q]
    x, y, z, w, qx, qy, qz = (round(float(v), DIGITS) + 0.0 for v in values)
    return (x, y, z, w, qx, qy, qz)


def _angle(a: ArrayLike, b: ArrayLike) -> float:
    """The angle (deg) between two 4x4 poses' orientations."""
    r = np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3]
    return math.degrees(math.acos(float(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))
