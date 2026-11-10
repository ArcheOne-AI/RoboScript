"""Spec fitting (docs/concepts/embodied-core.md): the GraspSpec or PlaceSpec that reproduces an
observed pose on a proxy, the core of `demo_relation`.

A grasp is fitted against every discrete combination the spec could take (approach, turned part
frame, anchor, closing axis): the observed contact frame is the combination's nominal one turned
by a yaw about the approach, then a tilt about the closing axis, then a lean about the axis across
both. The offset from the anchor is written in meters along the approach (how deep the fingers
go past the anchor's face or apex, which the part's size does not change) and in fractions of the
part's extents across it (where on the part: a rocker's lower half; meters across an extent under
`THIN_M`), kept within half an extent (`FIT_FRACTION_LIMIT`: farther out a fraction mostly
measures the box's error, not the grasp). The nearest combination wins: the least position
residual to 5 mm (a clipped offset, or an apex rebuilt elsewhere, reproduces the grasp worse),
then a closing axis the open gripper spans (`span_m`: the extent `major` or `minor` names before
the yaw; a wider one only where none spans), the least lean and tilt together, the least yaw,
the shortest depth to the millimeter, the smallest fractions with an apex counting `APEX_MARGIN`
more than a box anchor (an apex moves with every measured point, a box anchor with the box; a
press at a rocker's prominent edge is still written from its apex), then the minor axis (on a
near-square box either names the same extent). The offset, yaw, tilt and lean are what is
left over, so a pose from a spec's candidate comes back as that spec written in the canonical
way. What the spec cannot express is the residual: a tilt or lean beyond 75 deg and an offset
beyond its bounds. A receiving grasp of a hand-over (the other gripper holding the
object) is written with x toward the other gripper (`x_toward: other_arm`), a grasp of a whole
object with a single decided part with x toward it (`x_toward: handle`: a vessel's handle need not
lie along its footprint's sides, nor the same way on the next model, whose box may leave x open
where this one's decided it). A
placement is fitted
from the held object's pose once placed: its placement point (bottom, top or center) is expressed
from the target's nearest anchor in the target's axes and in the world's, or from the robot base,
and its orientation as a yaw (upright), as `keep` when the object keeps the orientation it is
held at, or else, tilted, `relative` (roll, pitch and yaw in the target frame). A hinge turn is
fitted from the TCP poses of an arc (`fit_hinge`): the rigid motion from the first to the last is
a turn about a line (its axis from the rotation, a point on it from the motion's fixed points);
of the part's HingeSpec lines (four sides, four heights) the one nearest parallel wins (then the
smallest offset), the offset moving it onto the fitted line; the angle is signed about the
spec's direction, and the residual is how far the poses stray from the spec's arc.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm
from roboscript.contracts.skills import PoseConventions
from roboscript.manipulation.api import arc_steps
from roboscript.manipulation.proxy import ObjectProxy
from roboscript.manipulation.specs import (
    OTHER_ARM,
    TILT_LIMIT_DEG,
    GraspSpec,
    HingeSpec,
    PlaceSpec,
    PlaceTarget,
    Purpose,
)
from roboscript.manipulation.synthesis import (
    closed_extent,
    contact,
    hinge_axes,
    hinge_line,
    met,
    placement_point_pose,
    target_frame,
    target_pose,
    variants,
)
from roboscript.perception.registry import WHOLE
from roboscript.spatial.pose import as_matrix
from roboscript.spatial.rotation import matrix_to_rpy, quat_angle_deg, quat_from_matrix

Array = NDArray[np.float64]

OFFSET_LIMIT_M: Final = 0.3
"""A fitted offset component is kept within this (Memory's default spec bounds); beyond it the
fit reports a position error."""
_ANCHORS: Final = ("top_center", "center", "side_center", "bottom_center", "apex")
_TARGET_ANCHORS: Final = ("top_center", "center", "inside_bottom")
PLAIN_ANCHORS: Final = ("center", "top_center", "apex")
"""The anchors of a grasp's plain form: the whole object's center, top or apex, with no offset."""
_BUCKET_DEG: Final = 1.0  # rolls, tilts and yaws this close count as equal when choosing
_BUCKET_M: Final = 0.001  # and depths this close
_RESIDUAL_M: Final = 0.005  # and position residuals this close
FIT_FRACTION_LIMIT: Final = 0.5
"""A fitted offset fraction is kept within this (half the part's extent)."""
APEX_MARGIN: Final = 0.25
"""An apex anchor wins over a centre anchor only with fractions smaller by more than this."""
THIN_M: Final = 0.01
"""A part extent under this takes a grasp offset across it in meters, not as a fraction."""


@dataclass(frozen=True)
class GraspFit:
    """A fitted grasp: the spec, the variant it matched (e.g. `top/180`), and what the spec
    cannot reproduce."""

    spec: GraspSpec
    variant: str
    position_error_m: float
    angle_error_deg: float


@dataclass(frozen=True)
class PlaceFit:
    """A fitted placement: the spec, what it cannot reproduce (the held object's pose), and
    whether the demonstration bears its form out (`consistent`, with the evidence: see
    `fit_place`); `tcp_gaps`, where a caller made them: how far the
    TCP that `place_object` commands with the spec at `stop_back_m` 0 lies from the demonstration's,
    (meters, degrees) by the call's `constrain`."""

    spec: PlaceSpec
    position_error_m: float
    angle_error_deg: float
    consistent: bool = True
    evidence: str = ""
    tcp_gaps: Mapping[str, tuple[float, float]] = field(default_factory=dict)


MIN_TURN_DEG: Final = 5.0
"""A gripper turning less than this from the first pose to the last made no hinge turn."""


@dataclass(frozen=True)
class HingeFit:
    """A fitted hinge turn: the spec, the angle about it (deg, right-handed about its line's
    direction), its straight moves (one per `ARC_STEP_DEG`), how far the poses stray from the
    spec's arc (m) and the angle between the fitted axis and the spec's line (deg)."""

    spec: HingeSpec
    angle_deg: float
    steps: int
    position_error_m: float
    angle_error_deg: float


def fit_grasp(
    tcp: Pose,
    proxy: ObjectProxy,
    *,
    part: str = WHOLE,
    purpose: Purpose = "pick",
    arm: Arm | None = None,
    other_tcp: ArrayLike | None = None,
    base: ArrayLike | None = None,
    plain: bool = False,
    span_m: float | None = None,
) -> GraspFit:
    """The GraspSpec (written for `arm`) whose candidate on `proxy` is nearest the TCP pose of a
    grasp (the fingertips at the contact point, the gripper's x along the approach); with the
    other gripper's TCP (`other_tcp`, world: it holds the object too) x points toward it, else on
    a whole object whose own front its geometry fixes x points toward that (a vessel's handle, a
    pan's handle as its footprint's far end: `ObjectProxy.front`), which the next model's box may
    lie elsewhere from;
    with the robot base (`base`, world 4x4) the grasp's orientation in it as `reference_q`;
    with the open gripper's span (`span_m`) a closing axis across a wider extent only where no
    combination spans its extent. `plain` fits the plain form instead (`PLAIN_ANCHORS`, no offset):
    the orientation as fitted and the anchor whose zero-offset contact is nearest, which a scene's
    noise does not move."""
    observed = np.linalg.inv(proxy.pose) @ contact_of(tcp)
    toward = OTHER_ARM if other_tcp is not None else None
    if toward is None and part == WHOLE:
        toward = proxy.front()  # the object's own front, where its asymmetry fixes one
    best: tuple[tuple[float, ...], GraspFit] | None = None
    for approach in ("top", "side"):
        shape = GraspSpec(part=part, approach=approach, x_toward=toward)
        for index, variant in enumerate(variants(shape, proxy, other_tcp)):
            local = np.linalg.inv(variant.frame) @ observed
            for closing in ("major", "minor"):
                for anchor in PLAIN_ANCHORS if plain else _ANCHORS:
                    spec = shape.model_copy(update={"closing": closing, "anchor": anchor})
                    nominal = contact(
                        spec, variant.approach, variant.size, variant.points, variant.surface
                    )
                    if nominal is None:  # an apex needs the part's points
                        continue
                    yaw, tilt, lean = _yaw_tilt_lean(nominal, local)
                    offset = local[:3, 3] - nominal[:3, 3]
                    if anchor == "apex":  # the depth from the surface met at the contact
                        depth = 2 if variant.approach == "top" else 1
                        hit = met(variant.points, variant.surface, variant.approach, local[:3, 3])
                        if hit is None:
                            continue
                        offset[depth] = local[depth, 3] - hit
                    metric, fraction = _split(offset, variant.size, variant.approach)
                    kept = np.clip(metric, -OFFSET_LIMIT_M, OFFSET_LIMIT_M)
                    kept_frac = np.clip(fraction, -FIT_FRACTION_LIMIT, FIT_FRACTION_LIMIT)
                    if plain:
                        kept, kept_frac = np.zeros(3), np.zeros(3)
                    fitted = spec.model_copy(
                        update={
                            "offset_m": tuple(float(v) for v in kept),
                            "offset_frac": tuple(float(v) for v in kept_frac),
                            "yaw_deg": yaw,
                            "tilt_deg": float(np.clip(tilt, -TILT_LIMIT_DEG, TILT_LIMIT_DEG)),
                            "lean_deg": float(np.clip(lean, -TILT_LIMIT_DEG, TILT_LIMIT_DEG)),
                            "purpose": purpose,
                            "arm": arm,
                        }
                    )
                    rebuilt = contact(
                        fitted, variant.approach, variant.size, variant.points, variant.surface
                    )
                    if rebuilt is None:  # an apex whose clipped offset leaves its line off the part
                        continue
                    again = variant.frame @ rebuilt
                    fit = GraspFit(
                        fitted,
                        variant.label,
                        float(np.linalg.norm(again[:3, 3] - observed[:3, 3])),
                        _angle(again, observed),
                    )
                    named = closed_extent(closing, variant.approach, variant.size)
                    wide = span_m is not None and named > span_m
                    fractions = float(np.abs(kept_frac).sum())
                    key = (
                        round(fit.position_error_m / _RESIDUAL_M),
                        int(wide),
                        round((abs(tilt) + abs(lean)) / _BUCKET_DEG),
                        round(abs(yaw) / _BUCKET_DEG),
                        round(
                            (fit.position_error_m if plain else float(np.linalg.norm(kept)))
                            / _BUCKET_M
                        ),
                        round(fractions + (APEX_MARGIN if anchor == "apex" else 0.0), 6),
                        int(closing == "major"),  # minor on a tie (a near-square box)
                        index,
                    )
                    if best is None or key < best[0]:
                        best = (key, fit)
    assert best is not None
    if base is None:
        return best[1]
    local = np.linalg.inv(as_matrix(base)) @ contact_of(tcp)
    reference = tuple(round(float(v), 3) for v in quat_from_matrix(local[:3, :3]))
    spec = best[1].spec.model_copy(update={"reference_q": reference})
    return replace(best[1], spec=spec)


def fit_grasp_pose(
    tcp: Pose,
    proxy: ObjectProxy,
    *,
    purpose: Purpose = "pick",
    arm: Arm | None = None,
    other_tcp: ArrayLike | None = None,
    base: ArrayLike | None = None,
) -> GraspFit:
    """The pose form of a grasp (`GraspSpec` anchor `pose`): the TCP's pose in the whole
    object's frame as its candidates take it (x toward the other gripper with `other_tcp`, else
    toward the object's own front where its geometry fixes one, as `fit_grasp`), in the first
    of the frame's turns; the other turns are the candidates' to try. With the robot base
    (`base`) the grasp's orientation in it as `reference_q`, which puts the turn that reproduces
    this one first. Nothing is left over but rounding: the TCP is written as it is."""
    observed = np.linalg.inv(proxy.pose) @ contact_of(tcp)
    toward = OTHER_ARM if other_tcp is not None else proxy.front()
    shape = GraspSpec(approach="top", x_toward=toward)  # the whole's frames, as a pose's turn
    variant = variants(shape, proxy, other_tcp)[0]
    local = np.linalg.inv(variant.frame) @ observed
    offset = np.clip(local[:3, 3], -OFFSET_LIMIT_M, OFFSET_LIMIT_M)
    spec = GraspSpec(
        anchor="pose",
        x_toward=toward,
        offset_m=_vec(offset),
        approach_axis=_vec(-local[:3, 1]),
        closing_axis=_vec(local[:3, 2]),
        purpose=purpose,
        arm=arm,
    )
    local_again = contact(spec, "top", variant.size)
    assert local_again is not None  # a pose always has its contact
    rebuilt = variant.frame @ local_again
    if base is not None:
        mine = np.linalg.inv(as_matrix(base)) @ contact_of(tcp)
        reference = tuple(round(float(v), 3) for v in quat_from_matrix(mine[:3, :3]))
        spec = spec.model_copy(update={"reference_q": reference})
    label = f"pose/{variant.label.split('/', 1)[1]}"
    error = float(np.linalg.norm(rebuilt[:3, 3] - observed[:3, 3]))
    return GraspFit(spec, label, error, _angle(rebuilt, observed))


def _vec(v: Array) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _split(offset: Array, size: Array, approach: str) -> tuple[Array, Array]:
    """An offset (variant frame) as meters along the approach and fractions of the extents
    across it (meters across an extent under `THIN_M`)."""
    depth = 2 if approach == "top" else 1
    across = np.array([k != depth and size[k] >= THIN_M for k in range(3)])
    metric = np.where(across, 0.0, offset)
    fraction = np.divide(offset, size, where=across, out=np.zeros(3))
    return metric, fraction


def fit_place(
    placed: ArrayLike,
    held: ObjectProxy,
    target: ObjectProxy | None,
    base: Pose,
    *,
    placement_point: str = "bottom",
    placement_point_at: ArrayLike | None = None,
    held_now: ArrayLike | None = None,
    part: str | None = None,
    arm: Arm | None = None,
    plain: bool = False,
    conventions: PoseConventions,
) -> dict[str, PlaceFit]:
    """The PlaceSpecs (written for `arm`) that put `held`'s placement point where it is when
    the object's frame is at `placed` (world): from `target`'s nearest anchor (its `part`, else
    the whole), in its axes (`target`) and the world's (`world_axes`), or with no target from the
    robot base (`robot_frame`), which also stands in for a target form whose offset the bounds
    (`OFFSET_LIMIT_M`) would cut: such a placement is not relative to the target (a target only
    partly in view, whose perceived box ends short of its true extent). The orientation is an
    upright yaw where that reproduces the placement, else the one it is held at (`keep`, given
    `held_now`, the object's frame while held, world) where that does, else `relative`; a form whose
    orientation is not the yaw comes with its upright yaw form too (`<form>/yaw`): `place_object`
    under `constrain` auto or free turns the object about its up by its own rule, so a tilted form
    may reproduce no call where the yaw form under `align` does. A held object whose own
    front its geometry fixes is written facing it (`x_toward`: `ObjectProxy.front`,
    `ObjectProxy.facing`). With a target, the forms come as `_borne_out` orders them, each
    marked with whether the demonstration bears it out; with `plain`, the plain one (`plain`):
    from the target's inside bottom along the world axes, with no offset where the point lies
    over the target's footprint (the placement is on or in it), else the offset beside it (a
    relation such as left of it). `conventions`: the embodiment profile's (its placement_point
    rotation)."""
    toward = held.front()
    if toward is not None:
        faced = held.facing(toward)
        turn = np.linalg.inv(held.pose) @ faced.pose
        placed = as_matrix(placed) @ turn
        held_now = None if held_now is None else as_matrix(held_now) @ turn
        held = faced
    fp = placement_point_pose(held, placement_point, placement_point_at, conventions=conventions)
    at = None if placement_point_at is None else tuple(float(v) for v in placement_point_at)
    point = as_matrix(placed) @ fp
    now = None if held_now is None else as_matrix(held_now) @ fp
    frames: dict[str, tuple[PlaceTarget, str]] = {
        "robot_frame": (PlaceTarget(kind="robot_frame"), "target")
    }
    if target is not None:
        where = min(
            (
                PlaceTarget(kind="object", obj=target.name, part=part, anchor=anchor)
                for anchor in _TARGET_ANCHORS
            ),
            key=lambda t: float(np.linalg.norm(point[:3, 3] - _origin(t, target, base))),
        )
        inside = PlaceTarget(kind="object", obj=target.name, part=part, anchor="inside_bottom")
        plain_form = {"plain": (inside, "world_axes")} if plain else {}
        frames = {
            **plain_form,
            "target": (where, "target"),
            "world_axes": (where, "world_axes"),
            **frames,
        }
    fits: dict[str, PlaceFit] = {}
    yaws: dict[str, PlaceFit] = {}
    clipped = set()
    for name, (place_at, axes) in frames.items():
        spec = PlaceSpec(
            target=place_at,
            frame=axes,  # type: ignore[arg-type]
            placement_point=placement_point,
            placement_point_at=at,  # type: ignore[arg-type]
            x_toward=toward,
            arm=arm,
        )
        frame = target_frame(spec, target, base)
        offset = frame[:3, :3].T @ (point[:3, 3] - frame[:3, 3])
        relative = place_at.kind != "robot_frame"
        kept = np.clip(offset, -OFFSET_LIMIT_M, OFFSET_LIMIT_M) if relative else offset
        if name == "plain" and target is not None and _over(point[:3, 3], target, part):
            kept = np.zeros(3)  # on the target: its inside bottom, however this scene put it
        elif relative and np.any(kept != offset):
            clipped.add(name)
        x = frame[:3, :3].T @ point[:3, 0]
        yaw = math.degrees(math.atan2(x[1], x[0]))
        rpy = matrix_to_rpy(frame[:3, :3].T @ as_matrix(placed)[:3, :3])
        options = [spec.model_copy(update={"orient": "yaw", "yaw_deg": yaw})]
        if now is not None:
            options.append(spec.model_copy(update={"orient": "keep"}))
        tilted = {"orient": "relative", "rpy_deg": tuple(float(v) for v in rpy)}
        options.append(spec.model_copy(update=tilted))
        candidates = []
        for option in options:
            option = option.model_copy(update={"offset_m": tuple(float(v) for v in kept)})
            held_point = point if now is None else now
            again = target_pose(option, frame, held_point, placement_point=fp).matrix()
            candidates.append(
                PlaceFit(
                    option,
                    float(np.linalg.norm(again[:3, 3] - point[:3, 3])),
                    _angle(again, point),
                )
            )
        fits[name] = min(candidates, key=lambda f: round(f.angle_error_deg / _BUCKET_DEG))
        yaws[name] = candidates[0]  # the upright yaw form, offered beside a tilted one
    if target is None:
        return _with_yaws(fits, yaws)
    # A placement beyond the offset bounds from the target's nearest anchor is not one relative
    # to it (a target only partly in view, whose perceived box ends short of its true extent):
    # it is written from the robot base.
    names = [name for name in ("target", "world_axes", "robot_frame", "plain") if name in fits]
    fits = {name: fits["robot_frame"] if name in clipped else fits[name] for name in names}
    yaws = {name: yaws["robot_frame"] if name in clipped else yaws[name] for name in names}
    return _borne_out(_with_yaws(fits, yaws), point, as_matrix(placed), target, part, base)


def _with_yaws(fits: dict[str, PlaceFit], yaws: dict[str, PlaceFit]) -> dict[str, PlaceFit]:
    """Each form, followed by its upright yaw form (`<form>/yaw`) where it is not one itself."""
    out = {}
    for name, fit in fits.items():
        out[name] = fit
        if fit.spec.orient != "yaw":
            out[f"{name}/yaw"] = yaws[name]
    return out


ON_M: Final = 0.05
"""A placed point this far above its target's top at most, over its footprint, rests on it."""
BESIDE_M: Final = 0.15
"""A placed point this far beside its target's footprint at most, no higher than resting on it,
is placed beside it (a relation such as left of it)."""
ALIGNED_DEG: Final = 5.0
"""A placed object's axes this close to a frame's (up to a signed permutation) are aligned."""


def _borne_out(
    fits: dict[str, PlaceFit],
    point: Array,
    placed: Array,
    target: ObjectProxy,
    part: str | None,
    base: Pose,
) -> dict[str, PlaceFit]:
    """The forms marked by what the demonstration itself shows (every form reproduces it on its
    own scene; this says which rule it bears out). Position: the placed point resting on the
    target (over its footprint, at most `ON_M` above its top) bears out a target-relative
    position; one beside it (within `BESIDE_M` of its footprint, no higher than that) both a
    target-relative and a robot-frame one; one elsewhere (in the air, far off) only a
    robot-frame one. Orientation: the placed object's axes aligned (`ALIGNED_DEG`) with the
    target's bear out the target's axes, aligned with the world's the world's (the robot base's
    are the world's turned a quarter); an object aligned with neither bears out none over the
    other, nor does an orientation kept as held. The consistent forms come first, in their
    order (the plain form, which drops this scene's offset, last)."""
    box = target.part(part or WHOLE)
    frame = target.pose @ box.frame
    local = np.linalg.inv(frame) @ np.array([*point[:3, 3], 1.0])
    half = box.size / 2
    outside = float(np.linalg.norm(np.maximum(np.abs(local[:2]) - half[:2], 0.0)))
    low = local[2] <= half[2] + ON_M
    where = "on" if outside == 0.0 and low else "beside" if outside <= BESIDE_M and low else "away"
    by_target = _aligned(frame[:3, :3], placed[:3, :3])
    by_world = _aligned(np.eye(3), placed[:3, :3])
    position = {"on": "rests on the target", "beside": "lies beside the target"}.get(
        where, "is away from the target (in the air or far off)"
    )
    turn = (
        "aligned with the target's axes and the world's"
        if by_target and by_world
        else "aligned with the target's axes"
        if by_target
        else "aligned with the world's axes"
        if by_world
        else "aligned with neither the target's axes nor the world's"
    )
    evidence = f"the placed object {position}, {turn}"
    marked = {}
    for name, fit in fits.items():
        spec = fit.spec
        relative = spec.target.kind != "robot_frame"
        in_world = not relative or spec.frame == "world_axes"
        turned = (
            spec.orient in ("keep", "upright")  # the held orientation: no frame's
            or not (by_target or by_world)
            or (by_world if in_world else by_target)
        )
        placed_ok = where != "away" if relative else where != "on"
        marked[name] = replace(fit, consistent=placed_ok and turned, evidence=evidence)
    return dict(sorted(marked.items(), key=lambda item: not item[1].consistent))


def _aligned(frame: Array, rotation: Array) -> bool:
    """Whether `rotation`'s axes lie along `frame`'s, each within `ALIGNED_DEG` of one."""
    m = np.abs(frame.T @ rotation)
    return bool(np.all(m.max(axis=0) >= math.cos(math.radians(ALIGNED_DEG))))


def fit_hinge(tcps: Sequence[Pose], proxy: ObjectProxy, part: str = WHOLE) -> HingeFit:
    """The HingeSpec on `part`'s box (of `proxy`: the object as measured, the hinge's body) and
    the angle the TCP poses turned about it, first to last (see the module docstring); a
    ValueError for fewer than two poses or a turn under `MIN_TURN_DEG`."""
    if len(tcps) < 2:
        raise ValueError("a hinge turn needs the poses of its arc, two at least")
    first, last = tcps[0].matrix(), tcps[-1].matrix()
    rotation = last[:3, :3] @ first[:3, :3].T
    angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))
    if angle < MIN_TURN_DEG:
        raise ValueError(
            f"the gripper turned {angle:.1f} deg from the first pose to the last: no hinge turn"
        )
    skew = rotation - rotation.T
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    if np.linalg.norm(axis) < 1e-9:  # a half turn
        values, vectors = np.linalg.eigh(rotation + rotation.T)
        axis = vectors[:, int(np.argmax(values))]
    axis = axis / np.linalg.norm(axis)
    shift = last[:3, 3] - rotation @ first[:3, 3]
    shift -= (shift @ axis) * axis
    fixed = np.linalg.lstsq(np.eye(3) - rotation, shift, rcond=None)[0]  # a point on the axis
    best: tuple[tuple[float, float], HingeFit] | None = None
    for side in get_args(HingeSpec.model_fields["side"].annotation):
        for at in get_args(HingeSpec.model_fields["at"].annotation):
            spec = HingeSpec(part=part, side=side, at=at)
            point, direction = hinge_line(spec, proxy)
            _, axes = hinge_axes(spec, proxy)
            along = float(direction @ axis)
            off = math.degrees(math.acos(min(1.0, abs(along))))
            nearest = fixed + ((point - fixed) @ axis) * axis  # on the fitted line
            gap = nearest - point
            offset = tuple(float(v) for v in axes.T @ (gap - (gap @ direction) * direction))
            fitted = spec.model_copy(update={"offset_m": offset})
            turn = math.copysign(angle, along)
            fit = HingeFit(
                fitted,
                turn,
                arc_steps(turn),
                _arc_error(tcps, *hinge_line(fitted, proxy)),
                off,
            )
            key = (round(off / _BUCKET_DEG), round(float(np.linalg.norm(offset)), 6))
            if best is None or key < best[0]:
                best = (key, fit)
    assert best is not None
    return best[1]


def _arc_error(tcps: Sequence[Pose], point: Array, direction: Array) -> float:
    """How far TCP positions stray from the circle about a line through the first one."""
    rel = np.array([tcp.p for tcp in tcps]) - point
    along = rel @ direction
    radius = np.linalg.norm(rel - np.outer(along, direction), axis=1)
    return float(np.max(np.hypot(radius - radius[0], along - along[0])))


def _over(point: Array, target: ObjectProxy, part: str | None) -> bool:
    """Whether a world point lies above or inside the footprint of `target`'s part (or whole)."""
    box = target.part(part or WHOLE)
    local = np.linalg.inv(target.pose @ box.frame) @ np.array([*point, 1.0])
    return bool(np.all(np.abs(local[:2]) <= box.size[:2] / 2))


def _origin(where: PlaceTarget, target: ObjectProxy, base: Pose) -> Array:
    return target_frame(PlaceSpec(target=where), target, base)[:3, 3]


def contact_of(tcp: Pose) -> Array:
    """The contact frame (4x4, as the TCP's) of a grasp from its TCP pose: y against the
    gripper's x (the approach), z against its y (the closing axis), x its z."""
    grasp = tcp.matrix()
    m = np.eye(4)
    m[:3, :3] = np.column_stack([grasp[:3, 2], -grasp[:3, 0], -grasp[:3, 1]])
    m[:3, 3] = grasp[:3, 3]
    return m


def _yaw_tilt_lean(nominal: Array, observed: Array) -> tuple[float, float, float]:
    """The observed contact frame as the nominal one turned by a yaw about its approach, then a
    tilt about its closing axis, then a lean about the axis across both (deg): M = Ry(-yaw)
    Rz(tilt) Rx(lean) in the nominal frame (`synthesis.contact`)."""
    m = nominal[:3, :3].T @ observed[:3, :3]
    tilt = math.degrees(math.asin(float(np.clip(m[1, 0], -1.0, 1.0))))
    yaw = -math.degrees(math.atan2(-m[2, 0], m[0, 0]))
    lean = math.degrees(math.atan2(-m[1, 2], m[1, 1]))
    return yaw, tilt, lean


def _angle(a: Array, b: Array) -> float:
    return float(quat_angle_deg(quat_from_matrix(a[:3, :3]), quat_from_matrix(b[:3, :3])))
