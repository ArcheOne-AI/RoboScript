"""Pose math (docs/concepts/embodied-core.md): `approach_pose`, `first_plannable_tilt`,
`choose_grasp` and `place_ee_pose`, with the helpers they use. Pure numpy; plannability is
asked of the caller (`Plannable`, a batch plan from the arm's drive target). The contact frame to
the grasp frame, an asset's up axis and the number of approach tilts are the env's, bound as data
(`contracts.skills.PoseConventions`) and passed in by the caller, along with the ee-to-TCP distance,
the tilt range and the preferred grasp orientations. The rest is the core's own pose convention,
which a binding maps its env onto: the ee's x is the approach, tilts turn about its y toward world
-y, and `constrain="auto"` aligns with fixed world axes and takes the ee's z as the camera axis.

Adapted from RoboTwin (`envs/_base_task.py`, `envs/utils/transforms.py`, `envs/robot/robot.py`,
MIT);
`axangle` and `mat2quat` follow transforms3d (BSD-2-Clause). See NOTICE and LICENSES/.

`align_matrix` leaves antiparallel vectors unaligned, `first_plannable_tilt` takes the first
plannable tilt, not the shortest path, and quaternions come from rotation matrices as
`transforms3d.quaternions.mat2quat` takes them (the nearest quaternion; annotated frames are not
always exactly orthonormal). The functions touch an object only through the `SceneObject`
surface (its pose and its frames of a kind), so a perceived-object proxy (`manipulation.proxy`)
and a simulator's object (through an adapter in its benchmark package) run the same code.
"""

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm
from roboscript.contracts.skills import PoseConventions
from roboscript.embodiments.profile import GraspPreference
from roboscript.spatial.rotation import quat_multiply

Array = NDArray[np.float64]
Plannable = Callable[[list[Pose]], Awaitable[list[bool]]]
"""Which ee poses the arm can plan to from where it is, as one batch (`Planner.plan_batch`)."""
Constrain = Literal["free", "align", "auto"]
CONTACT: Final = "contact"
"""The frame kind a grasp closes on: the fingertips' point, in the env's contact convention
(`PoseConventions.contact_to_grasp` turns it into the ee's grasp frame)."""
PLACEMENT: Final = "placement"
"""The frame kind a placement puts at its target: the object's point that lands there."""


class SceneObject(Protocol):
    """What the pose functions read of an object: its pose and its frames of a kind, each a 4x4
    world matrix, in a fixed order (a frame's index is its position in it). The core reads the
    `CONTACT` and `PLACEMENT` kinds; an object has no frames of a kind it does not carry (an
    empty sequence), and an implementation may carry kinds of its own."""

    def get_pose(self) -> Pose: ...

    def frames(self, kind: str) -> Sequence[Array]: ...


@dataclass(frozen=True)
class GraspChoice:
    """`choose_grasp`'s result: the pre-grasp and grasp ee poses and the index of the contact
    frame used."""

    pre: Pose
    grasp: Pose
    frame: int


def axangle(axis: ArrayLike, angle: float) -> Array:
    """Rotation by `angle` about `axis` (any length)."""
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cc = 1 - c
    return np.array(
        [
            [x * x * cc + c, x * y * cc - z * s, z * x * cc + y * s],
            [x * y * cc + z * s, y * y * cc + c, y * z * cc - x * s],
            [z * x * cc - y * s, y * z * cc + x * s, z * z * cc + c],
        ]
    )


def mat2quat(rotation: ArrayLike) -> Array:
    """`transforms3d.quaternions.mat2quat` (Bar-Itzhack): the quaternion (w >= 0) nearest a
    3x3 matrix that may be slightly off a rotation."""
    xx, yx, zx, xy, yy, zy, xz, yz, zz = np.asarray(rotation, dtype=float).flat
    k = np.array(
        [
            [xx - yy - zz, 0, 0, 0],
            [yx + xy, yy - xx - zz, 0, 0],
            [zx + xz, zy + yz, zz - xx - yy, 0],
            [yz - zy, zx - xz, xy - yx, xx + yy + zz],
        ]
    )
    values, vectors = np.linalg.eigh(k / 3.0)
    q: Array = vectors[[3, 0, 1, 2], np.argmax(values)]
    return -q if q[0] < 0 else q


def quat_dis(q1: ArrayLike, q2: ArrayLike) -> float:
    """The angle between two orientations over pi (0 same, 1 opposite)."""
    a = np.asarray(q1, dtype=float)
    delta = quat_multiply(a * np.array([1.0, -1, -1, -1]) / float(a @ a), q2)
    return float(2 * np.arccos(np.fabs(delta[0] / np.linalg.norm(delta))) / np.pi)


def align_matrix(v1: ArrayLike, v2: ArrayLike) -> Array:
    """The rotation taking direction v1 to v2 (the identity when they are
    parallel or antiparallel)."""
    a = np.asarray(v1, dtype=float).reshape(3)
    b = np.asarray(v2, dtype=float).reshape(3)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    return axangle(axis, float(np.arccos(np.dot(a, b))))


def product_vector(v1: ArrayLike, v2: ArrayLike) -> Array:
    """The projection of v2 on v1."""
    a = np.asarray(v1, dtype=float).reshape(3)
    a = a / np.linalg.norm(a)
    result: Array = np.dot(a, np.asarray(v2, dtype=float).reshape(3)) * a
    return result


def rotate_along_axis(
    target: Pose, center: ArrayLike, axis: ArrayLike, theta: float, towards: ArrayLike
) -> Pose:
    """`target` turned by `theta`
    about its own `axis` through the point `center`, the way that moves it along `towards`."""
    if theta == 0:
        return target
    target_mat, center_p = target.matrix(), np.asarray(center, dtype=float)[:3]
    world_axis = target_mat[:3, :3] @ np.asarray(axis, dtype=float)
    rotate = axangle(world_axis, theta)
    p = rotate @ (target_mat[:3, 3] - center_p) + center_p
    if np.dot(p - center_p, np.asarray(towards, dtype=float)) < 0:
        rotate = axangle(world_axis, -theta)
        p = rotate @ (target_mat[:3, 3] - center_p) + center_p
    return _pose(p, rotate @ target_mat[:3, :3])


def tilted_poses(
    origin: Pose, center: ArrayLike, tilt: tuple[float, float], conventions: PoseConventions
) -> list[Pose]:
    """The grasp tilted about its closing axis (ee y) through the contact frame's point, toward
    the robot, `conventions.approach_tilts` steps from `tilt[0]` (rad) on."""
    tilts = conventions.approach_tilts
    step = (tilt[1] - tilt[0]) / tilts
    return [
        rotate_along_axis(origin, center, [0, 1, 0], step * i + tilt[0], [0, -1, 0])
        for i in range(tilts)
    ]


async def first_plannable_tilt(
    pose: Pose,
    center: ArrayLike,
    tilt: tuple[float, float],
    plannable: Plannable,
    conventions: PoseConventions,
) -> Pose | None:
    """The first plannable of the tilted poses, None if none is."""
    targets = tilted_poses(pose, center, tilt, conventions)
    ok = await plannable(targets)
    return next((target for target, fine in zip(targets, ok, strict=True) if fine), None)


async def approach_pose(
    obj: SceneObject,
    frame: int,
    approach_m: float,
    preference: GraspPreference,
    tcp_offset: float,
    plannable: Plannable,
    conventions: PoseConventions,
) -> Pose | None:
    """The ee pose `tcp_offset` + `approach_m` behind the object's contact frame `frame` along
    its approach, at the first plannable tilt; None without that frame or a plannable tilt."""
    contacts = obj.frames(CONTACT)
    if not 0 <= frame < len(contacts):
        return None
    contact = np.asarray(contacts[frame], dtype=float)
    grasp = contact @ np.array(conventions.contact_to_grasp)
    position = grasp[:3, 3] + grasp[:3, :3] @ np.array([-tcp_offset - approach_m, 0, 0])
    return await first_plannable_tilt(
        _pose(position, grasp[:3, :3]), contact[:3, 3], preference.tilt, plannable, conventions
    )


async def choose_grasp(
    obj: SceneObject,
    arm: Arm,
    approach_m: float,
    grasp_offset_m: float,
    frames: Sequence[int] | None,
    preference: GraspPreference,
    tcp_offset: float,
    plannable: Plannable,
    conventions: PoseConventions,
) -> GraspChoice | None:
    """Over the object's contact frames (all, or the listed indices), the plannable one
    whose grasp is nearest the arm's preferred top-down orientation if within 0.15 (27 deg),
    else nearest its preferred side orientation if within 0.15, else the best 0.7/0.3 blend; the
    grasp is the pre-grasp moved `approach_m - grasp_offset_m` along its approach. None if no
    contact frame has a plannable pre-grasp."""
    ids = list(frames) if frames is not None else list(range(len(obj.frames(CONTACT))))
    top_down: tuple[float, GraspChoice] | None = None
    side: tuple[float, GraspChoice] | None = None
    blend: tuple[float, GraspChoice] | None = None
    for i in ids:
        pre = await approach_pose(
            obj, i, approach_m, preference, tcp_offset, plannable, conventions
        )
        if pre is None:
            continue
        rotation = pre.matrix()[:3, :3]
        forward = np.array([approach_m - grasp_offset_m, 0, 0]) @ np.linalg.inv(rotation)
        choice = GraspChoice(pre, _pose(np.add(pre.p, forward), rotation, pre.q), i)
        to_top_down = quat_dis(choice.grasp.q, preference.top_down)
        to_side = quat_dis(choice.grasp.q, preference.side)
        if top_down is None or to_top_down < top_down[0]:
            top_down = (to_top_down, choice)
        if side is None or to_side < side[0]:
            side = (to_side, choice)
        mixed = 0.7 * to_top_down + 0.3 * to_side
        if blend is None or mixed < blend[0]:
            blend = (mixed, choice)
    if top_down is None or side is None or blend is None:
        return None
    if top_down[0] < 0.15:
        return top_down[1]
    return side[1] if side[0] < 0.15 else blend[1]


def placed_object_pose(
    obj_pose: Pose,
    target_pose: Pose,
    constrain: Literal["free", "align"],
    align_axis: ArrayLike | None,
    object_axis: ArrayLike,
    object_axis_type: Literal["object", "world"],
    z_transform: bool,
    conventions: PoseConventions,
) -> Pose:
    """`obj_pose` moved to the target's position, turned so that
    its z (or with `z_transform` its local up, `conventions.object_up`) points along the target's
    z, and with `align` turned about that axis so that `object_axis` points along the nearest
    `align_axis` (default: the target's x) in the target's xy plane."""
    obj_mat, target_mat = obj_pose.matrix(), target_pose.matrix()
    obj_mat[:3, 3] = target_mat[:3, 3]
    target_x, target_y, target_z = target_mat[:3, 0], target_mat[:3, 1], target_mat[:3, 2]
    up = obj_mat[:3, :3] @ np.array(conventions.object_up) if z_transform else obj_mat[:3, 2]
    obj_mat[:3, :3] = align_matrix(up, target_z) @ obj_mat[:3, :3]
    if constrain == "align":
        if align_axis is None:
            axes = target_mat[:3, :3] @ np.array([[1.0], [0.0], [0.0]])
        elif isinstance(align_axis, list):
            axes = np.array(align_axis, dtype=float).reshape((-1, 3)).T
        else:
            axes = np.array(align_axis, dtype=float).reshape((3, -1))
        axes = axes / np.linalg.norm(axes, axis=0)
        if object_axis_type == "object":
            axis = obj_mat[:3, :3] @ np.asarray(object_axis, dtype=float).reshape(3, 1)
        else:
            axis = np.asarray(object_axis, dtype=float)
        align = axes[:, int(np.argmax(axis.reshape(3) @ axes))]
        obj_xy = product_vector(target_x, axis) + product_vector(target_y, axis)
        align_xy = product_vector(target_x, align) + product_vector(target_y, align)
        obj_mat[:3, :3] = align_matrix(obj_xy, align_xy) @ obj_mat[:3, :3]
    return _pose(obj_mat[:3, 3], obj_mat[:3, :3])


def place_ee_pose(
    obj: SceneObject,
    arm: Arm,
    target_pose: Pose,
    ee_pose: Pose,
    conventions: PoseConventions,
    *,
    constrain: Constrain = "auto",
    align_axis: ArrayLike | None = None,
    object_axis: ArrayLike = (1.0, 0.0, 0.0),
    object_axis_type: Literal["object", "world"] = "object",
    placement_frame: int | None = None,
    approach_m: float = 0.1,
    approach_axis: Literal["grasp", "target_z"] | ArrayLike = "grasp",
) -> Pose:
    """The ee pose (the arm's ee now at `ee_pose`, holding `obj`) that puts its placement frame
    `placement_frame` (else the object's origin, `conventions.object_up` up) at `target_pose`,
    turned by `constrain` (`auto`: align the grasp direction with the arm's diagonal when it is
    horizontal, else the camera axis with the world's y), `approach_m` back along the direction from
    the ee to the object (`grasp`) or along the target's `approach_axis` (`target_z`: its z)."""
    obj_mat = obj.get_pose().matrix()
    if placement_frame is not None:
        point = np.asarray(obj.frames(PLACEMENT)[placement_frame], dtype=float)
        start = _pose(point[:3, 3], point[:3, :3])
    else:
        start = obj.get_pose()
    z_transform = placement_frame is None
    ee = ee_pose.matrix()
    if constrain == "auto":
        direction = np.asarray(start.p) - ee[:3, 3]
        if np.abs(np.dot(direction, [0, 0, 1])) <= 0.1:
            align, axis = [1.0, 1.0, 0.0] if arm == "left" else [-1.0, 1.0, 0.0], direction
        else:
            align, axis = [0.0, 1.0, 0.0], ee[:3, 2]
        placed = placed_object_pose(
            start, target_pose, "align", align, axis, "world", z_transform, conventions
        )
    else:
        placed = placed_object_pose(
            start,
            target_pose,
            constrain,
            align_axis,
            object_axis,
            object_axis_type,
            z_transform,
            conventions,
        )
    start2target = placed.matrix()[:3, :3] @ start.matrix()[:3, :3].T
    target_point = start2target @ (obj_mat[:3, 3] - np.asarray(start.p)) + np.asarray(placed.p)
    grasp_rotation = start2target @ ee[:3, :3]
    grasp_offset = np.linalg.inv(ee[:3, :3]) @ (obj_mat[:3, 3] - ee[:3, 3])
    grasp_bias = grasp_rotation @ grasp_offset
    if isinstance(approach_axis, str) and approach_axis == "grasp":
        back = grasp_bias / np.linalg.norm(grasp_bias)
    else:
        local = [0.0, 0.0, 1.0] if isinstance(approach_axis, str) else approach_axis
        local = np.asarray(local, dtype=float) / np.linalg.norm(local)
        back = target_pose.matrix()[:3, :3] @ local
        back = back / np.linalg.norm(back)
    return _pose(target_point - grasp_bias - approach_m * back, grasp_rotation)


def arc_poses(
    axis: tuple[ArrayLike, ArrayLike], start_pose: Pose, angle_deg: float, steps: int
) -> list[Pose]:
    """Ee poses turning `start_pose` rigidly about a hinge, `axis` = (a point on it, its
    direction; right-handed), by `angle_deg` in `steps` equal steps (for successive
    `move_to_pose` calls opening a lid or a door)."""
    point = np.asarray(axis[0], dtype=float)
    direction = np.asarray(axis[1], dtype=float)
    start = start_pose.matrix()
    poses = []
    for k in range(1, steps + 1):
        turn = axangle(direction, math.radians(angle_deg) * k / steps)
        poses.append(_pose(turn @ (start[:3, 3] - point) + point, turn @ start[:3, :3]))
    return poses


def _pose(position: ArrayLike, rotation: ArrayLike, q: ArrayLike | None = None) -> Pose:
    """A pose from a position and a rotation matrix (or, kept exactly, its quaternion `q`)."""
    x, y, z = np.asarray(position, dtype=float).tolist()
    w, qx, qy, qz = (mat2quat(rotation) if q is None else np.asarray(q)).tolist()
    return Pose(p=(x, y, z), q=(w, qx, qy, qz))
