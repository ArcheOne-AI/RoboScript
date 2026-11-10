"""From specs to the points the pose math takes (docs/concepts/embodied-core.md): GraspSpec ->
contact-point candidates, PlaceSpec -> target pose and placement point, on a perceived-object proxy.

**Grasp candidates** are 4x4 contact matrices in the object frame, in the pose math's convention (y
against the approach, z the closing axis, origin where the TCP goes). They are the spec applied to
the part frame turned about z by each variant yaw: the object's symmetric turns (`whole` only:
180 deg for `pi`, 90 deg steps for `half_pi`, 45 deg steps for a round footprint, 90 deg with
`auto`), plus the part's turns when x's direction is unknown (`Part.turns`: 180 deg, and 90 deg
steps for a near-square footprint) and `x_toward` names nothing (a part, or `other_arm`: x
toward the other gripper's TCP); the grasp chooser orders them by the spec's reference. In each
such frame F: `top` approaches along -z, `side` along +y (entering the -y face; both faces are
candidates); the anchor is F's origin (`center`), its top or bottom face center, its -y face center
(`side_center`) or its most prominent surface point toward the gripper (`apex`: the middle of the
part's surface points within `APEX_BAND_M` of the foremost along the approach; a part without points
has none), moved by the offset (meters plus fractions of F's extents; from an apex, the depth is
taken from the surface the approach line meets there); the closing
axis is F's x (`major`) or y (`minor`) for `top`, and of x and z the longer (`major`) or shorter
one for `side`, turned by `yaw_deg` about the approach; then the approach tilts by `tilt_deg`
about the closing axis and leans by `lean_deg` about the axis across both, through the contact
point. A press (closed fingers push) also gets the other closing axis's candidates, after its
own: which extent is the longer can change from one object to the next. With the open gripper's
span (`span_m`), a pick's or a pull's closing axis that would close round more of the measured
surface than it spans (`_fits`; unmeasured, the box's extent) closes across the other one
instead where that one fits (labelled ` minor` or ` major`): fingers that cannot reach round a
solid never hold it, while a rim's wall fits however wide the container. A `pull` on a part also
gets, after its own, its own on the part's decided x turned half a turn (`_flipped`), then
candidates from around the part (`_around`): from above and into each of its
four upright faces, on its most prominent point that way, as deep as the spec's fingers go, kept
only where they pinch an edge the cameras saw, both fingers beside the object (`_edge`); the chooser
takes them only when none of the spec's own is reachable (a door's edge fitted from above where
the door stood low, reachable elsewhere only from its side or front). For a box
this gives the usual contact points (a cube's four top-down grasps, a long box's side grasps).

**Mirroring** (a spec written for the other arm): every candidate is reflected across the
robot's sagittal plane, taken through the part frame (the vertical plane of its x and z, or of
its y and z, whichever is nearer the sagittal plane, so that the grasp stays on the part), the
gripper's approach and camera axes reflected and its closing axis reversed: C -> S C D with S
negating that lateral axis in the part frame and D = diag(1, 1, -1) on the contact frame. This
maps left-arm grasp directions exactly onto the right arm's: the spec's lateral
offset and its yaw about a vertical approach change sign. A parallel gripper holds the same way
rolled half a turn about its approach, so every mirrored candidate comes in both rolls, and the
chooser's plannability and its preferred orientations pick. A placement with `mirror` reflects the
same way through its target frame (`for_arm`): its offset's lateral component and its
orientation.

**Placements**: the target frame has its origin at the anchor (an object's or part's box: top
center, center, inside bottom = bottom center, a top edge's midpoint toward the robot's front,
back, left or right; a landmark; the robot base origin) and the target's axes (z up), the world's
(`world_axes`) or the base's (`robot_frame`). The target pose is the anchor + the offset in those
axes, turned so that a placement point lands upright (z down): `keep` the held
point's current rotation, `upright` its current yaw, `yaw` = `yaw_deg` from the frame's x (for a
symmetric held object the equivalent yaw nearest its own; a round one keeps its yaw), or, tilted,
`relative`: the held object's frame at `rpy_deg` in the target frame (a symmetric object at its
equivalent turn about its own z nearest its orientation now). The placement point is the held
box's bottom or top face center or its center, with the placement_point rotation (x kept, y
and z reversed), or a part's bottom face center.

**Hinges** (`hinge_line`): a HingeSpec's line on the part's box: the side is the box face whose
outward normal is nearest the world direction the side names (as a placement's `edge:` anchors
take it), the line runs along that face horizontally at its bottom, middle or top, or upright
through its middle, then moves by the offset (across the face (up x its outward normal: along a
horizontal line), out of it, up).
"""

import math
from dataclasses import dataclass, replace
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm
from roboscript.contracts.skills import PoseConventions
from roboscript.manipulation.posemath import axangle
from roboscript.manipulation.proxy import ObjectProxy, Part
from roboscript.manipulation.specs import (
    APEX_BAND_M,
    CATCH_M,
    FREE_TURN_STEP_DEG,
    OTHER_ARM,
    RAY_RADIUS_M,
    GraspSpec,
    HingeSpec,
    PlaceSpec,
    SpecError,
)
from roboscript.perception.fit import Symmetry
from roboscript.perception.registry import WHOLE
from roboscript.spatial.pose import transform
from roboscript.spatial.rotation import matrix_to_rpy, rpy_to_matrix, yaw_matrix

Array = NDArray[np.float64]

REGION_STEP_M: Final = 0.02
"""A placement region is searched on rings this far apart, points this far apart on each."""
EXPOSED_M: Final = 0.01
"""A part's edge is exposed toward a gripper when no surface of its object lies more than this
before the edge along the approach line (a measured face's depth noise stays within it)."""
HELD_WIDTH_M: Final = 0.01
"""Surface this near the fingers' plane (across both the approach and the closing axis) lies
between them when they close."""
HELD_GAP_M: Final = 0.01
"""A gap this wide along the closing axis still joins two runs of surface into one solid the
fingers close round (measured points lie about 2-5 mm apart); a wider one is air (a container's
inside)."""
POSE_ROUND_STEP_DEG: Final = 15
"""A pose grasp of a round object is tried at turns this far apart (deg) about its axis, the
nearest to its reference orientation first (`ActionApi.grasp_object`)."""
_FLIP: Final = np.diag([1.0, 1.0, -1.0, 1.0])  # on the contact frame: reverse the closing axis
_ROLL: Final = np.diag(
    [-1.0, 1.0, -1.0, 1.0]
)  # on the contact frame: half a turn about the approach
_EDGES: Final = {"front": (0.0, -1.0), "back": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


@dataclass(frozen=True, eq=False)
class Candidate:
    """A grasp candidate: its contact matrix (object frame), a label, e.g. `top/90`, whether it
    is a pull's fallback (`_flipped`, `_around`: tried only when none of the spec's own is
    reachable), whether its fingers would close on air (`_air`: never chosen; kept so that a
    refusal can say why the spec has no candidate; `miss`: then the nearest measured surface
    point of the object from the contact, in the spec's frame, so that the refusal can say which
    way the material lies) and, for a pose grasp, whether it lies off the measured surface (`off`:
    tried after those on it)."""

    matrix: Array
    label: str
    around: bool = False
    air: bool = False
    off: bool = False
    miss: tuple[float, float, float] | None = None


def grasp_candidates(
    spec: GraspSpec,
    proxy: ObjectProxy,
    arm: Arm,
    *,
    other_tcp: ArrayLike | None = None,
    lateral: ArrayLike | None = None,
    span_m: float | None = None,
) -> list[Candidate]:
    """The spec's candidates on `proxy` for `arm` (mirrored if written for the other arm, across
    the robot's sagittal plane, `lateral` its normal in the world; None: across the part frame's
    x-z plane), part by part (every part of the kind the spec names), top approaches first, each
    by increasing variant yaw; duplicates dropped, and contacts off the object's surface (`_off`)
    or whose fingers would close on air (`_air`, with `span_m`) kept but marked `air`.
    `other_tcp`, the other gripper's TCP (world),
    serves `x_toward: other_arm`; `span_m`, the open gripper's span, swaps a closing axis it
    cannot span (see the module docstring)."""
    side = None if lateral is None else proxy.pose[:3, :3].T @ np.asarray(lateral, dtype=float)
    mirrored = spec.arm is not None and spec.arm != arm
    if spec.anchor == "pose":
        return _posed(spec, proxy, mirrored, side, other_tcp)
    specs = [(spec, "")]
    if spec.purpose == "press":
        other = "minor" if spec.closing == "major" else "major"
        specs.append((spec.model_copy(update={"closing": other}), f" {other}"))
    found: list[Candidate] = []
    tries = [(w, suffix, False, False, variants(w, proxy, other_tcp)) for w, suffix in specs]
    if spec.purpose == "pull" and spec.part != WHOLE:
        flipped = _flipped(proxy, spec)
        if flipped is not None:
            tries.append((spec, " turned", True, False, variants(spec, flipped, other_tcp)))
        tries += [
            (around, " around", True, True, variants(around, proxy, other_tcp, faces=True))
            for around in _around(spec)
        ]
    for written, suffix, fallback, edge, tried in tries:
        for variant in tried:
            named = suffix
            local = contact(
                written, variant.approach, variant.size, variant.points, variant.surface
            )
            if span_m is not None and written.purpose != "press" and local is not None:
                local, named = _spanned(written, suffix, variant, local, span_m)
            if local is None or (edge and not _edge(variant, local)):
                continue
            air = _off(variant, local) or (
                span_m is not None and written.purpose != "press" and _air(variant, local, span_m)
            )
            miss = _nearest(variant, local) if air else None
            matrix = variant.frame @ local
            label = variant.label + named
            options = [(matrix, label)]
            if mirrored:  # a parallel gripper holds the same way rolled half a turn: both
                matrix = mirror(matrix, variant.part_frame, side)
                options = [(matrix, label), (matrix @ _ROLL, f"{label} rolled")]
            for option, name in options:
                if not any(np.allclose(option, c.matrix, atol=1e-9) for c in found):
                    found.append(Candidate(option, name, fallback, air, miss=miss))
    return found


def _posed(
    spec: GraspSpec,
    proxy: ObjectProxy,
    mirrored: bool,
    side: Array | None,
    other_tcp: ArrayLike | None,
) -> list[Candidate]:
    """A pose grasp's candidates: the pose in each turned frame of the whole object (the turns
    its shape looks the same under, and both ways where its x is open), then each rolled half a
    turn about the approach (a parallel gripper holds the same way so); mirrored for the other
    arm. Contacts off the object's measured surface (`_off`) come last, not dropped: the pose is
    the demonstration's, and a registration that saw less of the object should not refuse it."""
    found: list[Candidate] = []
    off: list[Candidate] = []
    for roll in (False, True):
        for variant in variants(spec, proxy, other_tcp):
            local = contact(spec, variant.approach, variant.size)
            assert local is not None  # a pose always has its contact
            listed = off if _off(variant, local) else found
            matrix = variant.frame @ local
            if mirrored:
                matrix = mirror(matrix, variant.part_frame, side)
            if roll:
                matrix = matrix @ _ROLL
            label = f"pose/{variant.label.split('/', 1)[1]}" + (" rolled" if roll else "")
            if not any(np.allclose(matrix, c.matrix, atol=1e-9) for c in [*found, *off]):
                listed.append(Candidate(matrix, label, off=listed is off))
    return [*found, *off]


@dataclass(frozen=True, eq=False)
class Variant:
    """One approach in one turned part frame (object frame 4x4, `size` along its axes, the
    part's surface `points` in it, the whole object's `surface` points in it) of one part
    (`part_frame`: the part's own frame)."""

    approach: Literal["top", "side"]
    label: str
    frame: Array
    size: Array
    points: Array
    part_frame: Array
    surface: Array


def variants(
    spec: GraspSpec, proxy: ObjectProxy, other_tcp: ArrayLike | None = None, faces: bool = False
) -> list[Variant]:
    """The spec's approaches in each turned part frame it is tried in (see the module docstring),
    part by part, top first, each by increasing yaw; labelled `top/90`, or `handle0 top/90` when
    the spec's part name gives several parts. `faces`: a side approach enters each of the four
    upright faces, not only the two across y."""
    approaches: list[Literal["top", "side"]] = (
        ["top"]  # a pose has its own approach: one frame per turn
        if spec.anchor == "pose"
        else ["top", "side"]
        if spec.approach == "auto"
        else [spec.approach]
    )
    turns_about = (0, 90, 180, 270) if faces else (0, 180)
    found = []
    whole = proxy.surface()  # dense where measured: a hull's vertices would leave contacts off
    named = proxy.parts_of(spec.part) or [(spec.part, proxy.part(spec.part))]  # or its error
    for name, found_part in named:
        part = _toward(proxy, found_part, spec.x_toward, other_tcp)
        yaws = _variant_yaws(proxy, spec, part)
        prefix = f"{name} " if len(named) > 1 else ""
        for approach in approaches:
            turns = (
                yaws
                if approach == "top"
                else sorted({(y + d) % 360 for y in yaws for d in turns_about})
            )
            for yaw in turns:
                rotation = yaw_matrix(math.radians(yaw))
                size = np.abs(rotation).T @ part.size
                points = part.points @ rotation  # into the turned frame
                frame = part.frame @ transform(rotation)
                surface = (whole - frame[:3, 3]) @ frame[:3, :3]
                label = f"{prefix}{approach}/{yaw:g}"
                found.append(Variant(approach, label, frame, size, points, part.frame, surface))
    return found


def mirror(matrix: ArrayLike, part_frame: ArrayLike, lateral: ArrayLike | None = None) -> Array:
    """A contact matrix (object frame) reflected through `part_frame` across the vertical plane
    of its axes whose normal, its x or its y, is nearer `lateral` (the robot's lateral axis in the
    object frame; None: across its x-z plane): the same grasp by the other arm (see the module
    docstring)."""
    frame = np.asarray(part_frame, dtype=float)
    flip = np.eye(4)
    k = _lateral_axis(frame[:3, :3], lateral)
    flip[k, k] = -1.0
    result: Array = frame @ flip @ np.linalg.inv(frame) @ np.asarray(matrix, dtype=float) @ _FLIP
    return result


def for_arm(spec: PlaceSpec, arm: Arm, frame: ArrayLike, lateral: ArrayLike) -> PlaceSpec:
    """The placement as `arm` makes it: one written for the other arm with `mirror` is reflected
    through its target frame (`frame`, world) across the vertical plane whose normal, the frame's
    x or y, is nearer the robot's lateral axis (`lateral`, world): that offset component negated
    and the orientation (`yaw`, `relative`) and approach reflected; else as written."""
    if not spec.mirror or spec.arm is None or spec.arm == arm:
        return spec
    axes = np.asarray(frame, dtype=float)[:3, :3]
    k = _lateral_axis(axes, lateral)
    s = np.ones(3)
    s[k] = -1.0
    yaw = math.radians(spec.yaw_deg)
    turned_yaw = math.degrees(math.atan2(s[1] * math.sin(yaw), s[0] * math.cos(yaw)))
    rpy = matrix_to_rpy(np.diag(s) @ rpy_to_matrix(spec.rpy_deg) @ np.diag(s))
    return spec.model_copy(
        update={
            "offset_m": tuple(float(v) for v in np.asarray(spec.offset_m) * s),
            "yaw_deg": turned_yaw,
            "rpy_deg": tuple(float(v) for v in rpy),
            "approach": None
            if spec.approach is None
            else tuple(float(v) for v in np.asarray(spec.approach) * s),
            "arm": arm,
        }
    )


def placement_point_pose(
    proxy: ObjectProxy,
    placement_point: str,
    at: ArrayLike | None = None,
    *,
    conventions: PoseConventions,
) -> Array:
    """The held object's placement point (object frame): a box face center or its center, or a
    part's (the first of the kind named) bottom face center or, with `at`, its point at those
    fractions of its extent from its pivot, with the env's placement_point rotation
    (`conventions.placement_rotation`: z down when upright)."""
    m = np.eye(4)
    m[:3, :3] = np.array(conventions.placement_rotation)
    if placement_point.startswith("part:"):
        part = proxy.part(placement_point.removeprefix("part:"))
        if at is not None:
            m[:3, 3] = np.asarray(at, dtype=float) * part.size
            result: Array = part.pivot @ m
            return result
        m[:3, 3] = (0.0, 0.0, -0.5 * part.size[2])
        return part.frame @ m  # type: ignore[no-any-return]
    part = proxy.whole
    m[:3, 3] = (
        0.0,
        0.0,
        {"bottom": -0.5, "top": 0.5, "center": 0.0}[placement_point] * part.size[2],
    )
    return part.frame @ m  # type: ignore[no-any-return]


def target_frame(spec: PlaceSpec, target: ObjectProxy | Pose | None, base: Pose) -> Array:
    """The frame of a placement's target (world 4x4): origin at the anchor, axes as `frame` (and
    the kind) say. `target` is the target object's proxy, a landmark's pose, or None for the
    robot base frame."""
    where = spec.target
    if where.kind == "robot_frame":
        return base.matrix()
    if where.kind == "landmark":
        if not isinstance(target, Pose):
            raise ValueError(f"place at landmark {where.landmark}: give the landmark's pose")
        frame = np.eye(4)
        frame[:3, 3] = target.p
        return frame
    if not isinstance(target, ObjectProxy):
        raise ValueError(f"place at {where.obj}: give its proxy")
    part = target.part(where.part or WHOLE)
    world = target.pose @ part.frame
    anchor = where.anchor or "center"
    half = part.size / 2
    if anchor == "top_center":
        local = np.array([0.0, 0.0, half[2]])
    elif anchor == "inside_bottom":
        local = np.array([0.0, 0.0, -half[2]])
    elif anchor == "center":
        local = np.zeros(3)
    else:  # a top edge's midpoint toward a world direction
        toward = np.array([*_EDGES[anchor.removeprefix("edge:")], 0.0])
        along = world[:3, :3].T @ toward
        reach = np.min(
            np.divide(
                half[:2], np.abs(along[:2]), where=np.abs(along[:2]) > 1e-9, out=np.full(2, np.inf)
            )
        )
        local = np.array([*(along[:2] * reach), half[2]])
    frame = np.eye(4) if spec.frame == "world_axes" else world.copy()
    frame[:3, 3] = world[:3, :3] @ local + world[:3, 3]
    return frame


def target_pose(
    spec: PlaceSpec,
    frame: ArrayLike,
    held_point: ArrayLike,
    symmetry: Symmetry = "none",
    placement_point: ArrayLike | None = None,
) -> Pose:
    """The `target_pose` of a placement: the frame's origin + the offset in its axes,
    turned by `orient` for the held placement point now at `held_point` (world 4x4), which sits
    at `placement_point` in the held object's frame (None: at its origin). A held object with a yaw
    `symmetry` takes the `yaw`, or the `relative` turn about its own z, among its equivalent ones
    nearest its own now."""
    f, held = np.asarray(frame, dtype=float), np.asarray(held_point, dtype=float)
    position = f[:3, 3] + f[:3, :3] @ np.asarray(spec.offset_m)
    if spec.orient == "keep":
        rotation = held[:3, :3]
    elif spec.orient == "upright" or (spec.orient == "yaw" and symmetry == "continuous"):
        rotation = _down(held[:3, 0])
    elif spec.orient == "yaw":
        yaw = math.radians(spec.yaw_deg)
        x = f[:3, :3] @ np.array([math.cos(yaw), math.sin(yaw), 0.0])
        turns = {"none": 1, "pi": 2, "half_pi": 4}[symmetry]
        options = [yaw_matrix(2 * math.pi * k / turns) @ x for k in range(turns)]
        rotation = _down(max(options, key=lambda v: float(v @ held[:3, 0])))
    elif spec.orient == "relative":
        point = np.eye(4) if placement_point is None else np.asarray(placement_point, dtype=float)
        now = (held @ np.linalg.inv(point))[:3, :3]  # the held object's frame
        placed = f[:3, :3] @ rpy_to_matrix(spec.rpy_deg)
        rotation = _nearest_turn(placed, now, symmetry) @ point[:3, :3]
    else:
        raise ValueError(f"orient {spec.orient!r} is not one of keep, upright, yaw, relative")
    return Pose.from_matrix(transform(rotation, position))


def hinge_line(spec: HingeSpec, proxy: ObjectProxy) -> tuple[Array, Array]:
    """A hinge's line in the world: a point on it and its unit direction (see the module
    docstring)."""
    point, axes = hinge_axes(spec, proxy)
    return point, axes[:, 2] if spec.at == "upright" else axes[:, 0]


def hinge_axes(spec: HingeSpec, proxy: ObjectProxy) -> tuple[Array, Array]:
    """A hinge's point (world, its offset applied) and the side's axes as columns (world):
    across it (up x its outward normal), its outward normal, up; the offset's axes."""
    part = proxy.part(spec.part)
    box = proxy.pose @ part.frame
    half = part.size / 2
    toward = box[:3, :3].T @ np.array([*_EDGES[spec.side], 0.0])
    k = int(np.argmax(np.abs(toward[:2])))
    normal, up = np.zeros(3), np.array([0.0, 0.0, 1.0])
    normal[k] = math.copysign(1.0, toward[k])
    height = {"bottom": -half[2], "middle": 0.0, "top": half[2], "upright": 0.0}[spec.at]
    local = np.column_stack([np.cross(up, normal), normal, up])
    point: Array = box[:3, :3] @ (normal * half[k] + up * height + local @ spec.offset_m)
    axes: Array = box[:3, :3] @ local
    return point + box[:3, 3], axes


def _nearest_turn(placed: Array, now: Array, symmetry: Symmetry) -> Array:
    """`placed` turned about its own z by the equivalent turn of `symmetry` that brings it
    nearest `now` (any turn for a round object)."""
    m = placed.T @ now
    if symmetry == "continuous":
        return placed @ yaw_matrix(math.atan2(m[1, 0] - m[0, 1], m[0, 0] + m[1, 1]))
    turns = [yaw_matrix(turn) for turn in equivalent_turns(symmetry)]
    return placed @ max(turns, key=lambda r: float(np.trace(r.T @ m)))


def equivalent_turns(symmetry: Symmetry) -> list[float]:
    """The turns about the vertical (rad) that leave an upright object of `symmetry` looking the
    same, nearest first (a round one's in 45 deg steps)."""
    n = {"none": 1, "pi": 2, "half_pi": 4, "continuous": 8}[symmetry]
    turns = {math.remainder(2 * math.pi * k / n, 2 * math.pi) for k in range(n)}
    return sorted(turns, key=lambda a: (round(abs(a), 9), a < 0))


def free_turns(limit_deg: float) -> list[float]:
    """A placement's free turns (rad) up to `limit_deg` either way, nearest first, in
    `FREE_TURN_STEP_DEG` steps and the limit itself."""
    steps = [*np.arange(FREE_TURN_STEP_DEG, limit_deg, FREE_TURN_STEP_DEG), limit_deg]
    turns = [0.0] + [math.radians(sign * a) for a in steps if a > 0 for sign in (1, -1)]
    unique = {(round(math.cos(t), 9), round(math.sin(t), 9)): t for t in reversed(turns)}
    return sorted(unique.values(), key=turns.index)  # -180 is +180


def turned(pose: Pose, center: ArrayLike, turn: float, axis: ArrayLike | None = None) -> Pose:
    """`pose` turned by `turn` (rad) about the line through `center` along `axis` (world; None:
    the vertical)."""
    m, c = pose.matrix(), np.asarray(center, dtype=float)
    r = yaw_matrix(turn) if axis is None else axangle(axis, turn)
    m[:3, :3] = r @ m[:3, :3]
    m[:3, 3] = c + r @ (m[:3, 3] - c)
    return Pose.from_matrix(m)


def region_points(center: ArrayLike, radius: float) -> list[Array]:
    """A placement region's points: the center, then rings `REGION_STEP_M` apart out to the
    radius, points about `REGION_STEP_M` apart on each, starting toward the robot."""
    c = np.asarray(center, dtype=float)
    points = [c]
    rings = (
        [] if radius <= 0 else sorted({*np.arange(REGION_STEP_M, radius, REGION_STEP_M), radius})
    )
    for r in rings:
        n = max(6, math.ceil(2 * math.pi * r / REGION_STEP_M))
        for k in range(n):
            angle = -math.pi / 2 + 2 * math.pi * k / n
            points.append(c + r * np.array([math.cos(angle), math.sin(angle), 0.0]))
    return points


def _toward(
    proxy: ObjectProxy, part: Part, x_toward: str | None, other_tcp: ArrayLike | None = None
) -> Part:
    """The part with its frame's x toward the `x_toward` part (`ObjectProxy.aim`) or the other
    gripper's TCP (`other_arm`): a part's frame turned half a turn if x points away, the whole's
    turned exactly (its box re-measured from its points in the turned frame), as the object's
    own front need not lie along its footprint's sides (a vessel's handle); unchanged (x still
    unknown) when there is nothing to aim at."""
    if x_toward is None:
        return part
    if x_toward == OTHER_ARM:
        if other_tcp is None:
            raise SpecError(
                "x_toward other_arm aims at the other gripper, whose TCP is not known here"
            )
        point = (np.linalg.inv(proxy.pose) @ np.array([*np.asarray(other_tcp, dtype=float), 1.0]))[
            :3
        ]
    else:
        aimed = proxy.aim(x_toward)
        if aimed is None:
            return part
        point = aimed
    toward = part.frame[:3, :3].T @ (point - part.frame[:3, 3])  # in the part's frame
    if part is proxy.whole and x_toward != OTHER_ARM and np.hypot(*toward[:2]) > 1e-9:
        return part.turned(math.atan2(toward[1], toward[0]))
    if toward[0] >= 0:
        return replace(part, x_known=True)
    half = yaw_matrix(math.pi)
    return replace(
        part, frame=part.frame @ transform(half), x_known=True, points=part.points @ half
    )


def _flipped(proxy: ObjectProxy, spec: GraspSpec) -> ObjectProxy | None:
    """The proxy with the parts the spec names turned half a turn about z where their x was
    decided by their shape or hinge (not by the spec's `x_toward`): a moving part's hinge side
    read from partial points can come out reversed, and a pull whose own pre-grasps are all out
    of reach tries the other way before going around. None when no part has a decided x."""
    if spec.x_toward is not None:
        return None
    named = proxy.parts_of(spec.part)
    half = yaw_matrix(math.pi)
    turned = {
        name: replace(part, frame=part.frame @ transform(half), points=part.points @ half)
        for name, part in named
        if part.x_known
    }
    return replace(proxy, parts={**proxy.parts, **turned}) if turned else None


def _around(spec: GraspSpec) -> list[GraspSpec]:
    """A pull's grasp of a part from around it: from above and from the side (each upright face
    with `variants(faces=True)`), on the part's most prominent point toward the gripper, the
    fingers as far past that surface as the spec's go along its own approach (its meters; its
    fractions and its offsets across the approach, yaw, tilt and lean were fitted for its own
    approach and do not carry over). `_edge` keeps those that pinch an exposed edge."""
    depth = -spec.offset_m[2] if spec.approach == "top" else spec.offset_m[1]
    plain = {"anchor": "apex", "offset_frac": (0.0, 0.0, 0.0)}
    plain |= {"yaw_deg": 0.0, "tilt_deg": 0.0, "lean_deg": 0.0, "reference_q": None}
    return [
        spec.model_copy(update={**plain, "approach": approach, "offset_m": offset, "closing": c})
        for approach, offset in (("top", (0.0, 0.0, -depth)), ("side", (0.0, depth, 0.0)))
        for c in ("minor", "major")
    ]


def closed_extent(closing: str, approach: str, size: Array) -> float:
    """The extent a spec's closing axis names before its yaw (`major`: the longer of the two
    across the approach, `minor`: the shorter) on a box `size` along the variant frame's axes."""
    across = size[[0, 1]] if approach == "top" else size[[0, 2]]
    return float(across.max() if closing == "major" else across.min())


def _spanned(
    spec: GraspSpec, suffix: str, variant: "Variant", local: Array, span_m: float
) -> tuple[Array | None, str]:
    """The spec's contact `local`, or the one with the other closing axis where the fingers at
    `local` would close round more than `span_m` and at the other's they would not (`_held`)."""

    def fits(written: GraspSpec, at: Array) -> bool:
        held = _fits(variant, at, span_m)
        if held is None:
            return closed_extent(written.closing, variant.approach, variant.size) <= span_m
        return held

    if fits(spec, local):
        return local, suffix
    other = "minor" if spec.closing == "major" else "major"
    turned = spec.model_copy(update={"closing": other})
    there = contact(turned, variant.approach, variant.size, variant.points, variant.surface)
    if there is None or not fits(turned, there):
        return local, suffix
    return there, f"{suffix} {other}"


def _fits(variant: "Variant", local: Array, span_m: float) -> bool | None:
    """Whether the open fingers at contact `local` (variant frame), `span_m` apart, pass beside
    the object's measured surface they close on: of the surface within `HELD_WIDTH_M` of the
    fingers' plane and `CATCH_M` of the contact along the approach, every run along the closing
    axis (no gap over `HELD_GAP_M`) that reaches between the fingers ends between them. A vessel's
    rim wall fits, a vessel's height does not. None without such surface (the box's extent
    decides then)."""
    surface = variant.surface
    if not len(surface):
        return None
    offsets = surface - local[:3, 3]
    near = (np.abs(offsets @ local[:3, 0]) <= HELD_WIDTH_M) & (
        np.abs(offsets @ local[:3, 1]) <= CATCH_M
    )
    if not near.any():
        return None
    along = np.sort(offsets[near] @ local[:3, 2])
    half = span_m / 2
    breaks = np.flatnonzero(np.diff(along) > HELD_GAP_M)
    for run in np.split(along, breaks + 1):
        between = (run[0] <= half) and (run[-1] >= -half)
        if between and (run[0] < -half or run[-1] > half):
            return False
    return True


def _air(variant: "Variant", local: Array, span_m: float) -> bool:
    """Whether the fingers at contact `local` (variant frame), `span_m` apart, would close on air:
    no measured surface of the object lies between them, within `HELD_WIDTH_M` of their plane
    and no deeper than `CATCH_M` past the contact (a top grasp at the middle of a container's
    opening, its walls beyond both fingers). A proxy without surface points says nothing."""
    surface = variant.surface
    if not len(surface):
        return False
    offsets = surface - local[:3, 3]
    between = (
        (np.abs(offsets @ local[:3, 0]) <= HELD_WIDTH_M)
        & (np.abs(offsets @ local[:3, 2]) <= span_m / 2)
        & (offsets @ local[:3, 1] >= -CATCH_M)  # y points back along the approach
    )
    return not bool(between.any())


def _off(variant: "Variant", local: Array) -> bool:
    """Whether a contact (`local`, variant frame) lies off the object: its approach line passes
    no point of the object's measured surface within `CATCH_M` (across the approach; at any
    depth), so the fingers would close on air (an offset fitted on one model's box carried past
    the next one's). A proxy without surface points says nothing."""
    surface = variant.surface
    if not len(surface):
        return False
    across = [0, 1] if variant.approach == "top" else [0, 2]
    gaps = np.linalg.norm(surface[:, across] - local[across, 3], axis=1)
    return bool(gaps.min() > CATCH_M)


def _nearest(variant: "Variant", local: Array) -> tuple[float, float, float] | None:
    """From a contact (`local`, variant frame) to the nearest point of the object's measured
    surface, in the variant frame (the frame the spec's offsets are in); None without surface."""
    surface = variant.surface
    if not len(surface):
        return None
    gaps = surface - local[:3, 3]
    x, y, z = (float(v) for v in gaps[int(np.argmin(np.linalg.norm(gaps, axis=1)))])
    return x, y, z


def _edge(variant: "Variant", local: Array) -> bool:
    """Whether a grasp from around a part (`_around`, contact `local` in the variant frame)
    pinches an edge the cameras saw: its fingers close across the part's thinner extent across
    the approach; the part's surface the gripper meets first (its apex band, `EXPOSED_M` deep)
    spans at least half that extent; no surface of the object lies more than `EXPOSED_M` before
    it on the approach line; and the object's surface on the fingers' line, through the part's
    depth (or to the contact, if deeper), ends within that extent, so both fingers pass beside
    it (a door's free edge or end face, not its front, nor the side the rest of the door joins,
    nor a face no camera saw)."""
    points = variant.points
    if not len(points):
        return False
    closing, at = local[:3, 2], local[:3, 3]
    depth, sign = (2, -1.0) if variant.approach == "top" else (1, 1.0)
    along = sign * points[:, depth]
    band = points[along <= along.min() + EXPOSED_M] @ closing
    extent = float(np.abs(closing) @ variant.size)
    if float(band.max() - band.min()) < extent / 2:
        return False
    across = int(np.argmax(np.abs(np.cross(closing, np.eye(3)[depth]))))
    if extent > variant.size[across] + 1e-9:  # across the part's thinner extent there
        return False
    front = points[int(np.argmin(along))]
    before = surface_hit(variant.surface, variant.approach, at)
    if before is not None and sign * (front[depth] - before) > EXPOSED_M:
        return False
    surface = variant.surface
    ahead = sign * (surface[:, depth] - front[depth])
    deep = max(sign * (at[depth] - front[depth]), float(variant.size[depth]))
    slab = surface[
        (ahead >= -EXPOSED_M)
        & (ahead <= deep + EXPOSED_M)
        & (np.abs(surface[:, across] - at[across]) <= RAY_RADIUS_M)
    ]
    spread = np.abs((slab - at) @ closing)
    return bool(np.all(spread <= extent / 2 + EXPOSED_M))


def _lateral_axis(axes: Array, lateral: ArrayLike | None) -> int:
    """Which of the frame's x (0) and y (1) axes lies nearer `lateral` (in the frame's parent);
    y without one."""
    if lateral is None:
        return 1
    along = np.abs(axes[:, :2].T @ np.asarray(lateral, dtype=float))
    return int(np.argmax(along))


def _variant_yaws(proxy: ObjectProxy, spec: GraspSpec, part: Part) -> list[float]:
    """The part frame's turns (deg) the spec is tried at: symmetric ones, and the part's turns
    if x's direction is unknown (also when `x_toward` found nothing to aim at). A pose takes
    the turns its shape looks the same under however its x was decided (a cue or a model may
    decide a two-handled pot's x either way from one registration to the next): the grasp's
    reference orientation orders them."""
    if spec.anchor == "pose":
        shaped = proxy.shape_symmetry or proxy.symmetry
        step = {"pi": 180, "half_pi": 90, "continuous": POSE_ROUND_STEP_DEG}.get(shaped, 360)
        yaws = {float(a) for a in range(0, 360, step)}
        if not part.x_known:
            yaws |= {t % 360.0 for t in part.turns}
        return sorted(yaws)
    if spec.x_toward is not None and part.x_known:
        return [0.0]
    yaws = {0.0}
    if spec.part == WHOLE:
        step = {"pi": 180, "half_pi": 90, "continuous": 90 if spec.approach == "auto" else 45}
        yaws |= {float(a) for a in range(0, 360, step.get(proxy.symmetry, 360))}
    if not part.x_known:
        yaws |= {t % 360.0 for t in part.turns}
    return sorted(yaws)


def contact(
    spec: GraspSpec,
    approach: Literal["top", "side"],
    size: Array,
    points: Array | None = None,
    surface: Array | None = None,
) -> Array | None:
    """The spec's contact matrix in a variant frame (`size` along its axes, the part's surface
    `points` in it, the whole object's `surface` points); None for an `apex` without points or
    whose approach line meets neither."""
    if spec.anchor == "pose":
        return _pose_contact(spec)
    half = size / 2
    offset = np.asarray(spec.offset_m) + np.asarray(spec.offset_frac) * size
    if approach == "top":
        direction = np.array([0.0, 0.0, -1.0])
        closing = np.array([1.0, 0.0, 0.0] if spec.closing == "major" else [0.0, 1.0, 0.0])
    else:
        direction = np.array([0.0, 1.0, 0.0])
        longer = size[0] >= size[2]
        along_x = longer == (spec.closing == "major")
        closing = np.array([1.0, 0.0, 0.0] if along_x else [0.0, 0.0, 1.0])
    closing = axangle(direction, math.radians(spec.yaw_deg)) @ closing
    if spec.anchor == "apex":
        found = apex(points, approach)
        if found is None:
            return None
        position = found + offset
        depth = 2 if approach == "top" else 1
        hit = met(points, surface, approach, position)
        if hit is None:  # the fingers close on what lies between them along the closing axis
            caught = caught_at(points, approach, position, closing)
            if caught is None:
                caught = caught_at(surface, approach, position, closing)
            if caught is None:
                return None
            position, hit = caught
        position[depth] = hit + offset[depth]
    else:
        anchor = {
            "center": np.zeros(3),
            "top_center": np.array([0.0, 0.0, half[2]]),
            "bottom_center": np.array([0.0, 0.0, -half[2]]),
            "side_center": np.array([0.0, -half[1], 0.0]),
        }[spec.anchor]
        position = anchor + offset
    y = -direction
    rotation = np.column_stack([np.cross(y, closing), y, closing])
    tilt = yaw_matrix(math.radians(spec.tilt_deg))  # about the closing axis (z)
    lean = axangle([1.0, 0.0, 0.0], math.radians(spec.lean_deg))  # about the axis across both
    return transform(rotation @ tilt @ lean, position)


def _pose_contact(spec: GraspSpec) -> Array:
    """A pose grasp's contact matrix in the whole object's frame: y against its approach, z its
    closing axis made square to it, origin its offset."""
    assert spec.approach_axis is not None and spec.closing_axis is not None  # the validator's
    y = -np.asarray(spec.approach_axis, dtype=float)
    y /= np.linalg.norm(y)
    z = np.asarray(spec.closing_axis, dtype=float)
    z = z - (z @ y) * y
    z /= np.linalg.norm(z)
    return transform(np.column_stack([np.cross(y, z), y, z]), np.asarray(spec.offset_m))


def apex(points: Array | None, approach: Literal["top", "side"]) -> Array | None:
    """The most prominent point of surface `points` (variant frame) toward a gripper coming
    along the approach (from above: down its -z; from the side: along its +y): the middle of the
    points within `APEX_BAND_M` of the foremost, at the foremost depth. None without points."""
    if points is None or not len(points):
        return None
    depth, sign = (2, -1.0) if approach == "top" else (1, 1.0)
    along = sign * points[:, depth]  # grows away from the gripper
    near = points[along <= along.min() + APEX_BAND_M]
    found = near.mean(axis=0)
    found[depth] = points[int(np.argmin(along)), depth]
    result: Array = found
    return result


def surface_hit(
    points: Array | None, approach: Literal["top", "side"], position: ArrayLike
) -> float | None:
    """Where a gripper coming along the approach through `position` (variant frame) meets the
    surface `points`: the depth coordinate (z from above, y from the side) of the foremost of the
    points within `RAY_RADIUS_M` of its line; None if none is."""
    if points is None or not len(points):
        return None
    depth, sign = (2, -1.0) if approach == "top" else (1, 1.0)
    across = [k for k in range(3) if k != depth]
    near = points[
        np.linalg.norm(points[:, across] - np.asarray(position)[across], axis=1) <= RAY_RADIUS_M
    ]
    if not len(near):
        return None
    return float(near[int(np.argmin(sign * near[:, depth])), depth])


def caught_at(
    points: Array | None, approach: Literal["top", "side"], position: ArrayLike, closing: Array
) -> tuple[Array, float] | None:
    """Where an apex grasp through `position` (variant frame) whose approach line meets no
    surface closes instead: moved along the closing axis onto the nearest surface points within
    `CATCH_M` (those within `RAY_RADIUS_M` of the line across both axes), and the depth
    coordinate there (`surface_hit`'s); None if there are none."""
    if points is None or not len(points):
        return None
    depth = 2 if approach == "top" else 1
    axis = np.array(closing, dtype=float)
    axis[depth] = 0.0
    axis /= np.linalg.norm(axis)
    offsets = points - np.asarray(position, dtype=float)
    offsets[:, depth] = 0.0
    along = offsets @ axis
    across = np.linalg.norm(offsets - along[:, None] * axis, axis=1)
    near = (across <= RAY_RADIUS_M) & (np.abs(along) <= CATCH_M)
    if not near.any():
        return None
    shift = float(along[near][np.argmin(np.abs(along[near]))])
    moved = np.asarray(position, dtype=float) + shift * axis
    hit = surface_hit(points, approach, moved)
    assert hit is not None  # a point within the ray's radius of the moved line
    return moved, hit


def met(
    points: Array | None,
    surface: Array | None,
    approach: Literal["top", "side"],
    position: ArrayLike,
) -> float | None:
    """The surface an apex grasp's approach line meets (`surface_hit`): the part's `points`, else
    the whole object's `surface` points: a part's measured points may cover less than its box
    (a door's rim where the fitted offset lands past its points), and the fingers stop at
    whatever surface is there."""
    hit = surface_hit(points, approach, position)
    return surface_hit(surface, approach, position) if hit is None else hit


def _down(x: ArrayLike) -> Array:
    """A rotation with z down and x along the horizontal part of `x`."""
    v = np.asarray(x, dtype=float)
    horizontal = np.array([v[0], v[1], 0.0])
    x_axis = horizontal / np.linalg.norm(horizontal)
    z_axis = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
