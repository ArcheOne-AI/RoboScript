"""Read-only predicates over a `Scene`: the robot state, the table and the registered objects
where the registry has them (identified at the start, moved only by grasp relations; built by
`perception.locate.scene`). They mark stage completion; a failing predicate never ends an
episode. Thresholds are arguments without defaults: values from the Memory's parameters, which the
Demo Reader measures from the demonstration; physical constants come from the embodiment profile
(what holding and an open gripper mean).

Each returns its value (None when the inputs cannot decide it, e.g. no measured opening), the
measured quantities and a one-line reason. Lengths are meters, angles degrees. An object frame's
x sign is not known, so yaws compare modulo 180 deg at least (90 deg for square footprints; a
round object is always aligned, and nothing aligns with a round reference).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.robot import Arm, RobotState
from roboscript.perception.registry import ObjectGeometry, ObjectRecord
from roboscript.perception.relations import relation
from roboscript.perception.sides import resolve
from roboscript.perception.table import Table
from roboscript.spatial.pose import pose_to_matrix
from roboscript.spatial.rotation import quat_to_matrix

Direction = Literal["left", "right", "front", "back", "arm_side", "other_side"]
"""The robot's left, right, front (nearer it) or back; or relative to the arm that acted on the
object (`perception.sides`): its side of the robot, or the other arm's."""
_DIRECTIONS: dict[str, tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "front": (0.0, -1.0),
    "back": (0.0, 1.0),
}


class PredicateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicate: str
    value: bool | None
    measured: dict[str, float | None]
    reason: str


@dataclass(frozen=True, eq=False)
class Scene:
    """What predicates read: the robot now, the table, the registered objects (held ones where
    their TCP holds them) and the registry's records (for measurement history)."""

    robot: RobotState
    table: Table
    objects: Mapping[str, ObjectGeometry]
    records: Mapping[str, ObjectRecord]

    def object(self, obj_id: str) -> ObjectGeometry:
        if obj_id not in self.objects:
            raise LookupError(
                f"no registered object {obj_id!r}; registered: {sorted(self.objects)}"
            )
        return self.objects[obj_id]


def _result(
    name: str, value: bool | None, reason: str, **measured: float | None
) -> PredicateResult:
    return PredicateResult(predicate=name, value=value, measured=measured, reason=reason)


def holding(scene: Scene, arm: Arm, threshold: float) -> PredicateResult:
    """The gripper is commanded closed and its settled opening is at least `threshold` (the
    embodiment's `Gripper.holding_threshold`)."""
    state = scene.robot.arms[arm]
    opening, command = state.gripper, state.gripper_cmd
    if command >= 0.5:
        return _result(
            "holding",
            False,
            f"the {arm} gripper is commanded open ({command:.2f})",
            opening=opening,
            command=command,
        )
    if opening is None:
        return _result("holding", None, "the env measures no finger opening", command=command)
    value = opening >= threshold
    reason = f"measured opening {opening:.4f} {'>=' if value else '<'} {threshold:.4f}"
    return _result("holding", value, reason, opening=opening, command=command, threshold=threshold)


def gripper_open(scene: Scene, arm: Arm, fraction: float, full: float) -> PredicateResult:
    """The gripper's measured opening is at least `fraction` of its full opening `full` (the
    embodiment's `Gripper.open`)."""
    opening = scene.robot.arms[arm].gripper
    if opening is None:
        return _result("gripper_open", None, "the env measures no finger opening")
    share = opening / full
    value = share >= fraction
    reason = f"measured opening {share:.0%} of full ({'>=' if value else '<'} {fraction:.0%})"
    return _result("gripper_open", value, reason, opening=opening, share=share)


def lifted(scene: Scene, obj: str, height: float) -> PredicateResult:
    """The object's bottom is at least `height` above the table."""
    above = float(scene.table.height(scene.object(obj).bottom_center))
    value = above >= height
    reason = (
        f"bottom {above * 100:.1f} cm above the table ({'>=' if value else '<'} {height * 100:.1f})"
    )
    return _result("lifted", value, reason, bottom_above_table=above)


def on(scene: Scene, a: str, b: str, tol: float, min_overlap: float) -> PredicateResult:
    """a's bottom is within `tol` of b's top and at least `min_overlap` of a's footprint is
    over b's."""
    r = relation(scene.object(a), scene.object(b), tol)
    value = abs(r.gap_z) <= tol and r.overlap >= min_overlap
    reason = f"gap {r.gap_z * 100:+.1f} cm, {r.overlap:.0%} of {a}'s footprint over {b}"
    return _result("on", value, reason, gap_z=r.gap_z, overlap=r.overlap)


def inside(scene: Scene, a: str, b: str) -> PredicateResult:
    """a's center is over b's footprint and a's bottom below b's top."""
    r = relation(scene.object(a), scene.object(b))
    where = "over" if r.center_over else "outside"
    reason = (
        f"{a}'s center is {where} {b}'s footprint, its bottom {r.gap_z * 100:+.1f} cm from the top"
    )
    return _result("inside", r.inside, reason, gap_z=r.gap_z, overlap=r.overlap)


def beside(
    scene: Scene, a: str, b: str, direction: Direction, dmin: float, dmax: float
) -> PredicateResult:
    """a's center lies `dmin` to `dmax` from b's in `direction` (the robot's left, right, front
    (nearer the robot) or back; `arm_side` / `other_side`: the side of the arm that took a last,
    or of the other arm; undecidable before an arm took it)."""
    side = resolve(direction, scene.object(a).handled_by)
    if side is None:
        return _result("beside", None, f"no arm has taken {a} yet: {direction} is not decided")
    d = np.asarray(relation(scene.object(a), scene.object(b)).displacement[:2])
    unit = np.asarray(_DIRECTIONS[side])
    along = float(d @ unit)
    across = float(abs(d @ [-unit[1], unit[0]]))
    value = dmin <= along <= dmax
    named = side if side == direction else f"{side} ({direction})"
    reason = f"{a} is {along * 100:+.1f} cm to the {named} of {b} ({across * 100:.1f} cm off)"
    return _result("beside", value, reason, along=along, across=across)


def near(scene: Scene, arm: Arm, obj: str, distance: float) -> PredicateResult:
    """The arm's TCP is within `distance` of the object's box."""
    o = scene.object(obj)
    tcp = np.asarray(scene.robot.arms[arm].tcp.p)
    local = (tcp - np.asarray(o.obb.center)) @ quat_to_matrix(o.obb.q)
    gap = float(np.linalg.norm(np.maximum(np.abs(local) - np.asarray(o.obb.size) / 2, 0.0)))
    value = gap <= distance
    reason = f"the {arm} TCP is {gap * 100:.1f} cm from {obj}'s box"
    return _result("near", value, reason, distance=gap)


def pressed(scene: Scene, obj: str, distance: float) -> PredicateResult:
    """An empty gripper was last stopped pushing on the object within `distance`, horizontally, of
    its center (`press_offset`: as an action's receipt reported the press); None
    until something pressed it."""
    offset, why = press_offset(scene.records[obj], scene.object(obj), None)
    if offset is None:
        return _result("pressed", None, f"{obj}: {why}")
    reason = f"{obj} was last pressed {offset * 100:.1f} cm from its center, seen from above"
    return _result("pressed", offset <= distance, reason, offset=offset)


def moved(scene: Scene, obj: str, distance: float, since: int | None = None) -> PredicateResult:
    """The object's frame moved more than `distance` since its first measurement (or its last
    measurement at or before step `since`)."""
    history = scene.records[obj].history if obj in scene.records else []
    earlier = [m for m in history if since is None or m.step <= since]
    if not earlier:
        return _result("moved", None, f"{obj} has no measurement at or before step {since}")
    then = earlier[0] if since is None else earlier[-1]
    shift = float(np.linalg.norm(scene.object(obj).pose[:3] - np.asarray(then.pose.p)))
    reason = f"{obj} moved {shift * 100:.1f} cm since step {then.step}"
    return _result("moved", shift > distance, reason, displacement=shift)


def yaw_aligned(scene: Scene, obj: str, ref: str | float, tol: float) -> PredicateResult:
    """The object's yaw is within `tol` degrees of `ref` (another object's yaw, or degrees),
    folded by the object's symmetry; None against a round object, which has no yaw."""
    o = scene.object(obj)
    if isinstance(ref, str) and scene.object(ref).symmetry == "continuous":
        return _result("yaw_aligned", None, f"{ref} is round: it has no yaw to align with")
    target = scene.object(ref).yaw_deg if isinstance(ref, str) else float(ref)
    if o.symmetry == "continuous":
        return _result("yaw_aligned", True, f"{obj} is round: every yaw is aligned", difference=0.0)
    period = 90.0 if o.symmetry == "half_pi" else 180.0
    diff = abs((o.yaw_deg - target + period / 2) % period - period / 2)
    value = diff <= tol
    reason = f"{obj}'s yaw differs by {diff:.1f} deg (modulo {period:.0f}) from {target:.1f} deg"
    return _result("yaw_aligned", value, reason, difference=diff, yaw=o.yaw_deg)


PREDICATES: dict[str, Callable[..., PredicateResult]] = {
    "holding": holding,
    "gripper_open": gripper_open,
    "lifted": lifted,
    "on": on,
    "inside": inside,
    "beside": beside,
    "near": near,
    "pressed": pressed,
    "moved": moved,
    "yaw_aligned": yaw_aligned,
}


def check(scene: Scene, predicate: str, **arguments: Any) -> PredicateResult:
    """Evaluate a predicate by name (the `check` tool's library side)."""
    if predicate not in PREDICATES:
        raise LookupError(f"no predicate {predicate!r}; predicates: {sorted(PREDICATES)}")
    return PREDICATES[predicate](scene, **arguments)


def press_offset(
    record: ObjectRecord, now: ObjectGeometry, part: str | None
) -> tuple[float | None, str]:
    """`press`: how far (m) horizontally from the center of `now` (the object `record` as it is
    now), or of its part `part`'s box, its last press was (`ObjectRecord.presses`, kept in its
    frame), with why it is undecidable."""
    if not record.presses:
        return None, "nothing pressed it: no push of an empty gripper was stopped on it"
    if part is not None and part not in now.parts:
        return None, f"it has no part {part}"
    point = pose_to_matrix(now.pose) @ np.array([*record.presses[-1].point, 1.0])
    center = now.center if part is None else now.parts[part].pose[:3]
    return float(np.linalg.norm(point[:2] - center[:2])), ""
