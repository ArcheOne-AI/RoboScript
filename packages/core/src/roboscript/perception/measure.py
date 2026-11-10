"""Measuring a registered object again (docs/concepts/perception.md): its identity fixed, its
pose and shape from a new frame, and where a wrist camera should look for a better view, and a hinge
line of one of its parts. Plain numpy over a `FrameGeometry`; the scene toolsets call it
(`toolsets.scene.SceneAccess`), storing through the registry's lock.

`measure`: the pixels of each view whose world points lie in the object's expected box (where the
registry has it now, grown by `margin_m`; when the border cut its last measurement, by
`cut_margin_m` on the footprint faces the border cut and upward, the growth kept out of the other
registered footprints beside it; its bottom raised by a little, so what it stands on stays out), off
the other registered objects' surfaces and the robot, are lifted view by view. The object's known
shape is laid under them: the turn about z and the shift whose height map best matches theirs where
they were seen, then the height (`_register`; flat faces slide under nearest-point matching, a
height map's edges do not). An object the grasps set down (`moved`) keeps its shape: the moved shape
is its new points, only its pose changes. An unmoved one takes what the frame saw, completed by the
moved points no view of the frame could have seen (outside every image, or behind what the view
shows), so a close wrist view of one side, or a view past the head image's border, adds to what was
measured instead of replacing it. The points are fitted as the object was
(`SceneRegistry.remeasure`: the frame keeps x the way nearest the old one, grasp relations are re-
recorded); its parts stay where they were in the world (moved with it), and fixed ones are fitted
again (`parts.candidates`: the candidate nearest each, within `part_m`). Confidence: the share of
the moved points the frame should see that have a new point within `inlier_m`, times how well the
new size agrees with the old (a larger size where the old measurement was cut by the border counts
as agreeing). A footprint that does not continue the old one off the cut faces (a face moved more
than `jump_m` from where the alignment put it, or an uncut axis grown by more than `grow_frac`: a
neighbor taken in) is `inconsistent`, not stored. An object an arm holds is not measured (`held`):
the grasp relation has it, and the fingers around it, which the self-filter leaves at its edges in a
close view, shift and grow it.

`relocate`: an object `measure` found `not_visible` where the head view sees past where the
registry has it (it is gone from there, not hidden behind the arm), searched for among the frame's
proposals no other registered object's points cover, within `search_m` of where it was: the one
whose size, its sides sorted (an object knocked over lies on another side), is within `size_tol`
(log ratio, each side) of its own, the nearest when several are. Its points are fitted as the
object was (`SceneRegistry.remeasure`; its parts keep their place in its frame), `relocated` with
the size agreement as confidence (stored by the caller when at least `trust`). Without a match it
stays `not_visible`.

`look_pose`: an ee pose that puts an arm's wrist camera `distance` from the object, looking down
at it at `elevation` from the side of the robot's base, aimed past the part the head image cut
off; the camera's mount on the arm is read from the frame (its extrinsics against the arm's ee).

`hinge`: a hinge line of a part on its body, as a `HingeSpec` for `rotate_about`: an upright
plate turns about an upright line on its left or right side, a lying one about a horizontal line
at its bottom; of the sides, the one the body touches along most of the line, opposite what sticks
out of the part (a handle) when that decides it. `radius_m` is how far the part's far side is from
the line; confidence the share of the line the body touches.

`turn_offset`: how far a part lies turned about a hinge line from where the registry has it (a
held part follows the gripper there, as commanded): its points, turned about the line in 1 deg
steps, are checked against each view's depth, a point confirmed where the depth at its pixel is
within `inlier_m` of its own, refuted where the view sees past it; nearer depth (the gripper, the
part's own front) and the robot's pixels judge nothing. The turn with the most confirmed less
refuted points wins, the nearest to none on a tie; confidence the share of the judged points it
confirms.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.spatial import cKDTree

from roboscript.contracts.geometry import OrientedBox, Pose
from roboscript.contracts.robot import Arm, RobotState
from roboscript.perception.config import DEFAULTS, MeasureConfig, PerceptionConfig
from roboscript.perception.lift import lift, touches_border
from roboscript.perception.parts import candidates
from roboscript.perception.proposals import Proposal
from roboscript.perception.registry import (
    WHOLE,
    ObjectGeometry,
    ObjectRecord,
    SceneRegistry,
    object_geometry,
)
from roboscript.perception.views import FrameGeometry
from roboscript.spatial.camera import project
from roboscript.spatial.cloud import voxel_downsample
from roboscript.spatial.pose import pose_to_matrix
from roboscript.spatial.rotation import quat_to_matrix

Reason = Literal["ok", "not_visible", "low_confidence", "inconsistent", "held", "relocated"]
Side = Literal["front", "back", "left", "right"]
VISIBLE: Final = 0.01
"""A point shows in a view when the depth at its pixel is not nearer than its own by more."""
CELL: Final = 0.0025
"""The height maps' cell (m)."""
STEP_CAP: Final = 0.02
"""A seen cell costs at most this (m) in the alignment: the shape missing there, or far off."""
TURNS: Final = tuple(range(-12, 13, 2))
"""The turns (deg) the alignment tries."""
FACE_FRAC: Final = 0.5
"""A footprint face is cut when the object's points at the image border lie, on average, this
share of its half length or more toward it."""
SIDES: Final[dict[Side, tuple[float, float]]] = {
    "front": (0.0, -1.0),
    "back": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}


@dataclass(frozen=True, eq=False)
class Measured:
    """A measurement: the object's record as measured (not stored), how far its frame moved,
    its confidence and why it is (not) usable, the cameras that saw it and their points."""

    record: ObjectRecord
    shift_m: float
    confidence: float
    reason: Reason
    cameras: tuple[str, ...]
    points: int


class HingeEstimate(BaseModel):
    """A hinge of a part: the spec's side and height, its line (a point and a unit direction,
    world), the part's far side's distance from it, and the share of the line the body
    touches."""

    model_config = ConfigDict(frozen=True)

    part: str
    side: Side
    at: Literal["bottom", "upright"]
    point: tuple[float, float, float]
    direction: tuple[float, float, float]
    radius_m: float
    confidence: float


class TurnEstimate(BaseModel):
    """How far a part lies turned about a hinge from where the registry has it (deg, right-handed
    about the line's direction), the share of the points judged that confirm it there, and how
    many the views could judge."""

    model_config = ConfigDict(frozen=True)

    part: str
    offset_deg: float
    confidence: float
    judged: int


def measure(
    registry: SceneRegistry,
    geometry: FrameGeometry,
    obj_id: str,
    *,
    robot: RobotState | None = None,
    cameras: Sequence[str] | None = None,
    pipeline: PerceptionConfig = DEFAULTS,
) -> Measured:
    """`obj_id` measured again in `geometry`'s views (`cameras`: those only), as the module
    says; nothing stored."""
    config = pipeline.measure
    old = registry.get(obj_id)
    if old.grasps:  # its grasp owns where it is; the fingers on it would pass for its surface
        return Measured(old, 0.0, 0.0, "held", (), 0)
    expected = object_geometry(old, robot)
    neighbors = [object_geometry(r, robot) for i, r in registry.objects.items() if i != obj_id]
    held = [
        object_geometry(r, robot).obb
        for i, r in registry.objects.items()
        if i != obj_id and (r.grasps or r.part_grasps) and config.held_margin_m > 0
    ]
    rest = [n.points for n in neighbors if len(n.points)]
    others = cKDTree(np.concatenate(rest)) if rest else None
    # the footprints the growth past the object's margin stays out of: not what it stands on or in
    around = [n.obb for n in neighbors if not _in_footprint(expected.center[:2], n.obb, 0.0)]
    faces = _cut_faces(geometry, expected) if old.partially_visible else np.zeros((2, 2), bool)
    used = [c for c in (cameras or geometry.views) if c in geometry.views]
    seen: list[NDArray[np.float64]] = []
    names = []
    cut = []
    for camera in used:
        view = geometry.view(camera)
        mask = _expected_pixels(
            view.xyz,
            view.usable,
            expected,
            others,
            around,
            config,
            old.partially_visible,
            faces,
            held,
        )
        lifted = lift(geometry, {camera: mask}, pipeline) if mask.any() else None
        if lifted is None or len(lifted.points) < config.min_points:
            continue
        seen.append(lifted.points)
        names.append(camera)
        cut.append(touches_border(lifted.masks[camera]))
    if not seen:
        return Measured(old, 0.0, 0.0, "not_visible", (), 0)
    new = voxel_downsample(np.concatenate(seen), pipeline.lift.voxel)
    turn = _register(expected.points, new, expected.center, config, old.partially_visible)
    moved = _apply(turn, expected.points)
    visible = _visible(geometry, names, moved)
    if old.moved:  # the grasps set it down: its shape is known, only where it lies is new
        points = moved
    else:  # unmoved: what the frame saw, completed by what it could not see
        points = np.concatenate([new, moved[~visible]]) if (~visible).any() else new
    record = registry.remeasure(
        obj_id,
        points,
        frame=geometry.frame,
        step=geometry.step,
        table_z=geometry.table.z,
        robot=robot,
        frames=pipeline.frames,
    )
    seen_whole = not all(cut) or ((~visible).any() and not old.partially_visible)
    record = record.model_copy(update={"partially_visible": not seen_whole})
    record = _parts(record, old, turn, robot, geometry.table.z, config)
    inliers = _coverage(moved[visible], new, config.inlier_m)
    agreement = _agreement(expected.obb, record.fit.obb, old.partially_visible)
    confidence = round(float(inliers * agreement), 3)
    shift = float(np.linalg.norm(np.subtract(record.pose.p, expected.pose[:3])))
    reason: Reason = "ok" if confidence >= config.trust else "low_confidence"
    if reason == "ok" and not _consistent(expected.obb, record.fit.obb, turn, faces, config):
        reason = "inconsistent"
    return Measured(record, shift, confidence, reason, tuple(names), len(new))


def relocate(
    registry: SceneRegistry,
    geometry: FrameGeometry,
    obj_id: str,
    proposals: Sequence[Proposal],
    *,
    robot: RobotState | None = None,
    pipeline: PerceptionConfig = DEFAULTS,
) -> Measured:
    """`obj_id`, not visible where the registry has it, searched for among `proposals` (of
    `geometry`'s frame) as the module says; nothing stored."""
    config = pipeline.measure
    old = registry.get(obj_id)
    gone = Measured(old, 0.0, 0.0, "not_visible", (), 0)
    if old.grasps:
        return Measured(old, 0.0, 0.0, "held", (), 0)
    expected = object_geometry(old, robot)
    if not _seen_past(geometry, expected.points):
        return gone
    rest = [object_geometry(r, robot).points for i, r in registry.objects.items() if i != obj_id]
    rest = [p for p in rest if len(p)]
    others = cKDTree(np.concatenate(rest)) if rest else None
    size = np.sort(np.maximum(np.asarray(expected.obb.size), 0.01))
    found: list[tuple[float, float, Proposal]] = []
    for proposal in proposals:
        far = float(np.linalg.norm(np.subtract(proposal.summary.center[:2], expected.center[:2])))
        if far > config.search_m or len(proposal.points) < config.min_points:
            continue
        if others is not None:
            near, _ = others.query(proposal.points, distance_upper_bound=2 * config.inlier_m)
            if np.isfinite(near).mean() >= 0.5:  # another object's surface
                continue
        sides = np.sort(np.maximum(np.asarray(proposal.summary.size), 0.01))
        off = float(np.abs(np.log(sides / size)).max())
        if off <= config.size_tol:
            found.append((far, off, proposal))
    if not found:
        return gone
    _, off, best = min(found, key=lambda f: f[0])
    record = registry.remeasure(
        obj_id,
        best,
        frame=geometry.frame,
        step=geometry.step,
        table_z=geometry.table.z,
        robot=robot,
        frames=pipeline.frames,
    )
    shift = float(np.linalg.norm(np.subtract(record.pose.p, expected.pose[:3])))
    cameras = tuple(c for c, m in best.masks.items() if m.any())
    confidence = round(float(np.exp(-off)), 3)
    return Measured(record, shift, confidence, "relocated", cameras, len(best.points))


def _seen_past(geometry: FrameGeometry, points: NDArray[np.float64]) -> bool:
    """Whether the anchor view sees past most of the points it could show (depth beyond them):
    the object is gone from there, not hidden behind something nearer."""
    if geometry.anchor not in geometry.views or not len(points):
        return False
    view = geometry.view(geometry.anchor)
    image = view.image
    h, w = image.shape
    uv, z, front = project(points, image.K, image.cam2world)
    ok = front & np.isfinite(uv).all(axis=1)
    pixel = np.floor(uv[ok]).astype(np.int64)
    inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
    if not inside.any():
        return False
    depth = view.depth[pixel[inside, 1], pixel[inside, 0]]
    past = (depth > 0) & (depth > z[ok][inside] + VISIBLE)
    return bool(past.mean() > 0.5)


def look_pose(
    geometry: FrameGeometry,
    expected: ObjectGeometry,
    arm: Arm,
    camera: str,
    robot: RobotState,
    config: MeasureConfig = DEFAULTS.measure,
) -> Pose:
    """An ee pose of `arm` that puts its wrist `camera` where the module says (the mount read
    from `geometry`'s view of that camera against `robot`'s ee pose, the frame's)."""
    view = geometry.view(camera)
    mount = np.linalg.inv(pose_to_matrix(robot.arms[arm].ee.as_list())) @ view.image.cam2world
    aim = expected.center.copy()
    if geometry.anchor in geometry.views:
        aim = aim + _cut_side(geometry, expected) * 0.5
    base = np.asarray(geometry.base if geometry.base is not None else (0.0, -0.65), dtype=float)
    toward = base - aim[:2]
    toward = toward / max(float(np.linalg.norm(toward)), 1e-9)
    elevation = np.radians(config.elevation)
    back = np.array([*(np.cos(elevation) * toward), np.sin(elevation)])
    eye = aim + config.distance * back
    forward = (aim - eye) / np.linalg.norm(aim - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    down = np.cross(forward, right)
    cam = np.eye(4)
    cam[:3, :3] = np.column_stack([right, down, forward])  # OpenCV: x right, y down, z forward
    cam[:3, 3] = eye
    ee = cam @ np.linalg.inv(mount)
    return Pose.from_matrix(ee)


def hinge(
    registry: SceneRegistry,
    obj_id: str,
    part: str,
    *,
    side: Side | None = None,
    robot: RobotState | None = None,
) -> HingeEstimate:
    """The hinge of `part` on the rest of `obj_id`, as the module says (`side`: that side's)."""
    geometry = object_geometry(registry.get(obj_id), robot)
    if part not in geometry.parts:
        raise LookupError(f"{obj_id} has no part {part!r}: register it first")
    shape = geometry.parts[part]
    frame = pose_to_matrix(shape.pose)
    local = (geometry.points - frame[:3, 3]) @ frame[:3, :3]
    half = np.asarray(shape.size, dtype=float) / 2
    inside = np.all(np.abs(local) <= half + 0.005, axis=1)
    body, piece = local[~inside], local[inside]
    thin = int(np.argmin(half))
    upright = thin != 2  # a plate standing on edge (a door), else lying (a lid)
    options = []
    for name, (sx, sy) in SIDES.items():
        toward = frame[:3, :3].T @ np.array([sx, sy, 0.0])
        k = int(np.argmax(np.abs(toward[:2])))
        normal = np.zeros(3)
        normal[k] = np.sign(toward[k])
        if upright and k == thin:  # an upright plate turns on its width's ends, not its faces
            continue
        edge = normal * half[k]
        if not upright:
            edge = edge - np.array([0.0, 0.0, half[2]])
        direction = np.array([0.0, 0.0, 1.0]) if upright else np.cross([0.0, 0.0, 1.0], normal)
        options.append((name, k, normal, edge, direction))
    handle = _outward(piece, half)
    best = None
    for name, k, normal, edge, direction in options:
        if side is not None and name != side:
            continue
        touch = _touching(body, edge, direction, half)
        opposite = 0.0 if handle is None else max(0.0, -float(handle @ normal))
        score = touch + 0.5 * opposite
        if best is None or score > best[0]:
            best = (score, name, k, normal, edge, direction, touch)
    if best is None:
        raise LookupError(f"part {part!r} has no {side} side to turn on")
    _, name, k, normal, edge, direction, touch = best
    radius = float(2 * half[k])
    point = frame[:3, :3] @ edge + frame[:3, 3]
    world = frame[:3, :3] @ direction
    return HingeEstimate(
        part=part,
        side=name,
        at="upright" if upright else "bottom",
        point=(float(point[0]), float(point[1]), float(point[2])),
        direction=(float(world[0]), float(world[1]), float(world[2])),
        radius_m=radius,
        confidence=round(float(touch), 3),
    )


def _expected_pixels(
    xyz: NDArray[np.float64],
    usable: NDArray[np.bool_],
    expected: ObjectGeometry,
    others: cKDTree | None,
    around: Sequence[OrientedBox],
    config: MeasureConfig,
    cut: bool,
    faces: NDArray[np.bool_],
    held: Sequence[OrientedBox] = (),
) -> NDArray[np.bool_]:
    """A view's pixels in the object's grown box, off the surfaces of the other registered
    objects (a container's walls around what was set into it, a neighbor's side): grown by
    `margin_m`, by `cut_margin_m` on the footprint's faces (`faces`, (axis, low/high)) the border
    cut, and upward when it cut the object at all; what the growth past
    `margin_m` adds stays out of the `around` footprints (a neighbor the border cut too, whose
    unregistered top would pass for the object's). Nothing in the column over what an arm holds
    counts (`held`: its box's footprint grown by `held_margin_m`, from its bottom less that
    margin, or from the object's measured top, whichever is lower): its unseen sides and the
    fingers around it, over a target the arm carries it to (a wrist look before a placement),
    would pass for the target's top (none with `held_margin_m` 0)."""
    mask = np.zeros(usable.shape, dtype=bool)
    v, u = np.nonzero(usable)
    points = xyz[v, u]
    box = expected.obb
    local = _local(points, box)
    half = np.asarray(box.size) / 2
    rise = min(config.lift_m, 0.3 * 2 * half[2])
    low = half[:2] + np.where(faces[:, 0], config.cut_margin_m, config.margin_m)
    high = half[:2] + np.where(faces[:, 1], config.cut_margin_m, config.margin_m)
    up = config.cut_margin_m if cut else config.margin_m  # a cut object's top is unknown
    inside = np.all((local[:, :2] >= -low) & (local[:, :2] <= high), axis=1)
    inside &= (local[:, 2] >= -half[2] + rise) & (local[:, 2] <= half[2] + up)
    grown = inside & ~np.all(np.abs(local[:, :2]) <= half[:2] + config.margin_m, axis=1)
    for other in around:
        grown_at = np.flatnonzero(grown)
        taken = _in_footprint(points[grown_at, :2], other, 2 * config.inlier_m)
        inside[grown_at[taken]] = False
    known_top = box.center[2] + half[2] + config.margin_m  # the object's measured top
    for other in held:  # its footprint and bottom as it lies (a held object may be tilted)
        at = np.flatnonzero(inside)
        reach = np.abs(quat_to_matrix(other.q)[2]) @ np.asarray(other.size) / 2
        bottom = other.center[2] - reach - config.held_margin_m
        over = _in_footprint(points[at, :2], other, config.held_margin_m)
        over &= points[at, 2] >= min(bottom, known_top)
        inside[at[over]] = False
    if others is not None and inside.any():
        distance, _ = others.query(points[inside], distance_upper_bound=2 * config.inlier_m)
        inside[np.flatnonzero(inside)[np.isfinite(distance)]] = False
    mask[v[inside], u[inside]] = True
    return mask


def _in_footprint(xy: NDArray[np.float64], box: OrientedBox, grow: float) -> NDArray[np.bool_]:
    """Which xy (N, 2), or one xy (2,), lie within a box's footprint grown by `grow`."""
    flat = np.atleast_2d(xy)
    local = _local(np.column_stack([flat, np.full(len(flat), box.center[2])]), box)
    half = np.asarray(box.size[:2]) / 2 + grow
    inside: NDArray[np.bool_] = np.all(np.abs(local[:, :2]) <= half, axis=1)
    return inside if np.ndim(xy) == 2 else inside[0]


def _cut_faces(geometry: FrameGeometry, expected: ObjectGeometry) -> NDArray[np.bool_]:
    """The footprint faces (axis, low/high) the anchor camera's border cuts: those whose outer
    half (`FACE_FRAC`) holds the object's points at the border, on average (a border through
    an upright object's top holds them across its middle, and cuts none)."""
    faces = np.zeros((2, 2), dtype=bool)
    if geometry.anchor not in geometry.views:
        return faces
    near = _at_border(geometry, expected)
    if near is None:
        return faces
    local = _local(expected.points[near], expected.obb)[:, :2].mean(axis=0)
    reach = local / np.maximum(np.asarray(expected.obb.size[:2]) / 2, 1e-6)
    faces[:, 0] = reach <= -FACE_FRAC
    faces[:, 1] = reach >= FACE_FRAC
    return faces


def _consistent(
    old: OrientedBox,
    new: OrientedBox,
    turn: NDArray[np.float64],
    faces: NDArray[np.bool_],
    config: MeasureConfig,
) -> bool:
    """Whether a new footprint continues the old one moved as the alignment found (`turn`) off
    the faces the border cut: no other face lies more than `jump_m` from where it went, no axis
    uncut on both faces grown by more than `grow_frac` of its length (or the depth noise,
    `2 * inlier_m`, when that is more)."""
    rotation = turn[:3, :3] @ quat_to_matrix(old.q)
    center = _apply(turn, np.asarray(old.center, dtype=float))
    corners = ((_corners(new) - center) @ rotation)[:, :2]
    half = np.asarray(old.size[:2]) / 2
    moved = np.stack([-half - corners.min(axis=0), corners.max(axis=0) - half], axis=1)
    if (np.abs(moved) > config.jump_m)[~faces].any():
        return False
    uncut = ~faces.any(axis=1)
    growth = moved.sum(axis=1)[uncut]
    limit = np.maximum(config.grow_frac * 2 * half, 2 * config.inlier_m)[uncut]
    return bool((growth <= limit).all())


def _corners(box: OrientedBox) -> NDArray[np.float64]:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    out: NDArray[np.float64] = (signs * np.asarray(box.size) / 2) @ quat_to_matrix(box.q).T
    return out + np.asarray(box.center, dtype=float)


def _local(points: NDArray[np.float64], box: OrientedBox) -> NDArray[np.float64]:
    rotation = quat_to_matrix(box.q)
    return (points - np.asarray(box.center)) @ rotation


def _register(
    known: NDArray[np.float64],
    new: NDArray[np.float64],
    center: NDArray[np.float64],
    config: MeasureConfig,
    cut: bool = False,
) -> NDArray[np.float64]:
    """The rigid motion (4x4: a turn about z through `center`, then a translation) that best
    lays the known shape's height map under the new points' (seen cells only; a seen cell the
    shape does not cover, or covers at another height, costs up to `STEP_CAP`; of a `cut` shape,
    whose top a border may have cut off, a seen cell above it costs nothing): searched over
    turns of `TURNS` and shifts on the height map's grid within `margin_m`, then the height by
    the median gap. Flat faces slide under nearest-point matching; a height map's edges do not."""
    cell = CELL
    lo = np.minimum(known[:, :2].min(axis=0), new[:, :2].min(axis=0)) - config.margin_m - cell
    shape = tuple(np.ceil((np.maximum(known[:, :2].max(axis=0), new[:, :2].max(axis=0))
                           + config.margin_m + cell - lo) / cell).astype(int) + 1)  # fmt: skip
    top = _height_map(known, lo, shape)
    cells, heights = _cells(new, lo)
    reach = int(np.ceil(config.margin_m / cell))
    best = (np.inf, 0.0, 0, 0)
    for yaw in np.radians(TURNS):
        c, s = np.cos(-yaw), np.sin(-yaw)
        xy = (cells + 0.5) * cell + lo - center[:2]  # new cells, turned back about the center
        back = xy @ np.array([[c, s], [-s, c]]) + center[:2]
        base = np.floor((back - lo) / cell).astype(int)
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                i, j = base[:, 0] - di, base[:, 1] - dj
                ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
                gap = np.full(len(i), STEP_CAP)
                diff = heights[ok] - top[i[ok], j[ok]]
                if cut:  # the shape is a lower bound
                    diff = np.minimum(diff, 0.0)
                gap[ok] = np.minimum(np.abs(diff), STEP_CAP)
                cost = float(np.nan_to_num(gap, nan=STEP_CAP).mean())
                if cost < best[0]:
                    best = (cost, yaw, di, dj)
    _, yaw, di, dj = best
    turn = np.eye(4)
    turn[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    turn[:2, 3] = center[:2] - turn[:2, :2] @ center[:2] + np.array([di, dj]) * cell
    moved = _apply(turn, known)
    moved_top = _height_map(moved, lo, shape)
    i, j = cells[:, 0], cells[:, 1]
    inside = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    gaps = heights[inside] - moved_top[i[inside], j[inside]]
    gaps = gaps[np.abs(gaps) < STEP_CAP]
    if len(gaps):
        turn[2, 3] = float(np.median(gaps))
    return turn


def _height_map(
    points: NDArray[np.float64], lo: NDArray[np.float64], shape: tuple[int, ...]
) -> NDArray[np.float64]:
    """The highest point over each grid cell (NaN where none)."""
    cells, heights = _cells(points, lo)
    top = np.full(shape, -np.inf)
    ok = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < shape[0])
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < shape[1])
    )
    np.maximum.at(top, (cells[ok, 0], cells[ok, 1]), heights[ok])
    top[~np.isfinite(top)] = np.nan
    return top


def _cells(
    points: NDArray[np.float64], lo: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Points' grid cells and, per occupied cell, the highest point's height (one per cell)."""
    ij = np.floor((points[:, :2] - lo) / CELL).astype(np.int64)
    order = np.lexsort((-points[:, 2], ij[:, 1], ij[:, 0]))
    ij, z = ij[order], points[order, 2]
    first = np.ones(len(ij), dtype=bool)
    first[1:] = np.any(ij[1:] != ij[:-1], axis=1)
    return ij[first], z[first]


def _apply(m: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = points @ m[:3, :3].T + m[:3, 3]
    return out


def _visible(
    geometry: FrameGeometry, cameras: Sequence[str], points: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Whether some of the cameras could see each point (in its image, not behind a nearer
    surface)."""
    shown = np.zeros(len(points), dtype=bool)
    for camera in cameras:
        view = geometry.view(camera)
        image = view.image
        h, w = image.shape
        uv, z, front = project(points, image.K, image.cam2world)
        ok = front & np.isfinite(uv).all(axis=1)
        pixel = np.floor(uv[ok]).astype(np.int64)
        inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
        index = np.flatnonzero(ok)[inside]
        depth = view.depth[pixel[inside, 1], pixel[inside, 0]]
        shown[index[(depth <= 0) | (depth >= z[index] - VISIBLE)]] = True
    return shown


def _coverage(expected: NDArray[np.float64], new: NDArray[np.float64], radius: float) -> float:
    """The share of the points the frame should see that it saw (a new point near each)."""
    if not len(expected):
        return 1.0
    distance, _ = cKDTree(new).query(expected, distance_upper_bound=max(radius, 0.005))
    return float(np.isfinite(distance).mean())


def _agreement(old: OrientedBox, new: OrientedBox, old_cut: bool) -> float:
    """How well two sizes agree (1: the same; a larger new one after a cut one agrees)."""
    a = np.maximum(np.asarray(old.size), 0.01)
    b = np.maximum(np.asarray(new.size), 0.01)
    ratio = np.log(b / a)
    if old_cut:
        ratio = np.minimum(ratio, 0.0)
    return float(np.exp(-np.abs(ratio).sum()))


def _parts(
    record: ObjectRecord,
    old: ObjectRecord,
    turn: NDArray[np.float64],
    robot: RobotState | None,
    table_z: float,
    config: MeasureConfig,
) -> ObjectRecord:
    """The record with its parts where they were in the world, moved by the alignment (`turn`),
    in its new frame; then each fixed part fitted again: the part candidate of the new points
    nearest it, within `part_m`."""
    if not old.parts:
        return record
    before = pose_to_matrix(object_geometry(old, robot).pose)
    after = pose_to_matrix(object_geometry(record, robot).pose)
    change = np.linalg.inv(after) @ turn @ before  # old object frame -> new object frame
    parts = {}
    for name, part in old.parts.items():
        pose = Pose.from_matrix(change @ part.pose.matrix())
        parts[name] = part.model_copy(
            update={"pose": pose, "points": _apply(change, np.asarray(part.points))}
        )
    record = record.model_copy(update={"parts": parts})
    fixed = {k: p for k, p in parts.items() if k != WHOLE and not p.articulated}
    if not fixed:
        return record
    geometry = object_geometry(record, robot)
    found = candidates(geometry.points, None, table_z)
    if not found:
        return record
    scratch = SceneRegistry(objects={record.id: record})
    for name, part in fixed.items():
        where = geometry.parts[name].pose[:3]
        near = min(found, key=lambda c: float(np.linalg.norm(c.frame[:3, 3] - where)))
        if np.linalg.norm(near.frame[:3, 3] - where) > config.part_m:
            continue
        scratch.register_part(
            record.id, name, near.points, box=(near.frame, near.extent), decided=near.decided,
            x_toward=part.x_toward, robot=robot,
        )  # fmt: skip
    return scratch.get(record.id)


def _cut_side(geometry: FrameGeometry, expected: ObjectGeometry) -> NDArray[np.float64]:
    """Toward the part of the object the anchor camera's image cuts off (world, its length the
    visible part's distance to that border; zero when nothing is cut)."""
    near = _at_border(geometry, expected)
    if near is None:
        return np.zeros(3)
    offset = expected.points[near].mean(axis=0) - expected.center
    offset[2] = 0.0
    return offset


def _at_border(geometry: FrameGeometry, expected: ObjectGeometry) -> NDArray[np.bool_] | None:
    """The object's points the anchor camera shows at its image border (None: fewer than 5)."""
    image = geometry.view(geometry.anchor).image
    h, w = image.shape
    uv, _, front = project(expected.points, image.K, image.cam2world)
    ok = front & np.isfinite(uv).all(axis=1)
    near = ok & ((uv[:, 0] < 3) | (uv[:, 0] > w - 4) | (uv[:, 1] < 3) | (uv[:, 1] > h - 4))
    return near if near.sum() >= 5 else None


def _outward(piece: NDArray[np.float64], half: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """The unit direction (part frame, horizontal, in the part's plane) of what stands out of
    either face of the part (a handle, a knob), None when nothing does."""
    thin = int(np.argmin(half))
    if len(piece) < 10:
        return None
    face = piece[:, thin] - np.median(piece[:, thin])
    extreme = face[np.argmax(np.abs(face))]
    if abs(extreme) < 0.005:
        return None
    far = np.abs(face - extreme) < 0.002
    center = piece[far].mean(axis=0)
    center[thin] = 0.0
    center[2] = 0.0
    length = float(np.linalg.norm(center))
    return None if length < 0.2 * float(np.max(half[:2])) else center / length


def _touching(
    body: NDArray[np.float64],
    edge: NDArray[np.float64],
    direction: NDArray[np.float64],
    half: NDArray[np.float64],
) -> float:
    """The share of a hinge line's length (through the part's box) the body comes within 1 cm
    of (past the plate's half thickness)."""
    if not len(body):
        return 0.0
    k = int(np.argmax(np.abs(direction)))
    length = 2 * half[k]
    offset = body - edge
    along = offset @ direction
    across = np.linalg.norm(offset - np.outer(along, direction), axis=1)
    near = 0.01 + float(np.min(half))  # the line runs through the plate's middle
    close = (across < near) & (np.abs(along) <= length / 2)
    if not close.any():
        return 0.0
    bins = np.unique(np.floor((along[close] + length / 2) / 0.005))
    return float(min(1.0, len(bins) * 0.005 / max(length, 1e-6)))


TURN_POINTS: Final = 3000
"""A turn's search checks at most this many of the part's points (evenly taken)."""


def turn_offset(
    registry: SceneRegistry,
    geometry: FrameGeometry,
    obj_id: str,
    part: str,
    hinge: tuple[Sequence[float], Sequence[float]],
    span_deg: float,
    robot: RobotState | None = None,
    config: MeasureConfig = DEFAULTS.measure,
) -> TurnEstimate:
    """How far `part` lies turned about `hinge` (a point, a direction; world) from where the
    registry has it, within `span_deg` either way, as the module says; `judged` 0 and offset 0
    when no view could judge its points."""
    geometry_of = object_geometry(registry.get(obj_id), robot)
    if part not in geometry_of.parts:
        raise LookupError(f"{obj_id} has no part {part!r}: register it first")
    points = np.asarray(geometry_of.parts[part].points, dtype=float).reshape(-1, 3)
    points = points[:: len(points) // TURN_POINTS + 1]
    point = np.asarray(hinge[0], dtype=float)
    axis = np.asarray(hinge[1], dtype=float)
    axis = axis / np.linalg.norm(axis)
    steps = np.arange(0.0, float(span_deg) + 0.5, 1.0)
    angles = [0.0, *(a for step in steps[1:] for a in (step, -step))]  # the nearest wins a tie
    best: tuple[int, float, int, int] | None = None  # score, angle, confirmed, judged
    for angle in angles:
        confirmed, refuted = _judge_turn(geometry, _turned(points, point, axis, angle), config)
        if best is None or confirmed - refuted > best[0]:
            best = (confirmed - refuted, angle, confirmed, confirmed + refuted)
    assert best is not None
    _, angle, confirmed, judged = best
    return TurnEstimate(
        part=part,
        offset_deg=angle if judged else 0.0,
        confidence=round(confirmed / judged, 3) if judged else 0.0,
        judged=judged,
    )


def _turned(
    points: NDArray[np.float64], point: NDArray[np.float64], axis: NDArray[np.float64], deg: float
) -> NDArray[np.float64]:
    """`points` turned by `deg` about the line through `point` along the unit `axis`."""
    a = np.radians(deg)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)
    turned: NDArray[np.float64] = (points - point) @ rotation.T + point
    return turned


def _judge_turn(
    geometry: FrameGeometry, points: NDArray[np.float64], config: MeasureConfig
) -> tuple[int, int]:
    """How many of `points` the views' depth confirms and how many it refutes (see the module)."""
    confirmed = refuted = 0
    for view in geometry.views.values():
        h, w = view.depth.shape
        uv, z, front = project(points, view.image.K, view.image.cam2world)
        ok = front & np.isfinite(uv).all(axis=1)
        u, v = np.rint(uv[ok, 0]).astype(int), np.rint(uv[ok, 1]).astype(int)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[inside], v[inside], z[ok][inside]
        seen = view.depth[v, u].astype(float)
        valid = (seen > 0) & ~view.robot[v, u]
        gap = seen - z
        confirmed += int(np.count_nonzero(valid & (np.abs(gap) <= config.inlier_m)))
        refuted += int(np.count_nonzero(valid & (gap > config.inlier_m)))
    return confirmed, refuted
