"""The pose math against hand-computed cases."""

import math

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.embodiments import MOCK_BIMANUAL, GraspPreference, load_embodiment
from roboscript.manipulation.posemath import (
    CONTACT,
    PLACEMENT,
    align_matrix,
    approach_pose,
    axangle,
    place_ee_pose,
    placed_object_pose,
    quat_dis,
)
from roboscript.manipulation.proxy import ObjectProxy, Part
from roboscript.spatial.rotation import quat_from_axis_angle, quat_from_matrix

TOP_DOWN = (-0.61239, 0.353523, -0.61239, -0.353524)  # a top-down gripper orientation
SIDE = (-0.924, 0.0, 0.0, -0.383)  # a side gripper orientation
PREFERENCE = GraspPreference(tilt=(0.0, 1.0), top_down=TOP_DOWN, side=SIDE)

POSES = load_embodiment(MOCK_BIMANUAL).poses  # the pose conventions


def pose(p: tuple[float, float, float], rotation: np.ndarray) -> Pose:
    return Pose(p=p, q=tuple(quat_from_matrix(rotation).tolist()))  # type: ignore[arg-type]


def rigid(rotation: np.ndarray, position: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, position
    return m


def box(
    contact: list[np.ndarray], placement: list[np.ndarray] | None = None, yaw: float = 0.0
) -> ObjectProxy:
    """A 4 cm cube at (0.1, 0.2, 0.8), turned by `yaw` (rad) about z."""
    part = Part(frame=np.eye(4), size=np.full(3, 0.04), x_known=False)
    frame = rigid(axangle([0, 0, 1], yaw), (0.1, 0.2, 0.8))
    proxy = ObjectProxy("cube", frame, "half_pi", {"whole": part})
    return proxy.with_frames({CONTACT: contact, PLACEMENT: placement or []})


def always(ok: bool = True):  # type: ignore[no-untyped-def]
    asked: list[list[Pose]] = []

    async def plannable(poses: list[Pose]) -> list[bool]:
        asked.append(poses)
        return [ok] * len(poses)

    plannable.asked = asked  # type: ignore[attr-defined]
    return plannable


def test_rotation_helpers_are_transforms3d_s() -> None:
    assert axangle([0, 0, 2], math.pi / 2) @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0])
    assert quat_dis((1, 0, 0, 0), (1, 0, 0, 0)) == pytest.approx(0.0)
    assert quat_dis((1, 0, 0, 0), (0, 1, 0, 0)) == pytest.approx(1.0)
    half = quat_from_axis_angle([0, 0, 1], math.pi / 2)
    assert quat_dis((2, 0, 0, 0), half) == pytest.approx(0.5)  # scale-free
    assert align_matrix([1, 0, 0], [0, 2, 0]) @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0])
    assert np.array_equal(align_matrix([1, 0, 0], [3, 0, 0]), np.eye(3))
    assert np.array_equal(align_matrix([1, 0, 0], [-1, 0, 0]), np.eye(3))  # opposite vectors


async def test_approach_pose_backs_off_along_the_approach() -> None:
    # Contact frame: y up (the approach is down), z along x (the closing axis).
    contact = rigid(np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), (0.0, 0.0, 0.02))
    cube = box([contact])
    plannable = always()
    pre = await approach_pose(cube, 0, 0.1, PREFERENCE, 0.12, plannable, POSES)
    assert pre is not None and pre.p == pytest.approx((0.1, 0.2, 0.82 + 0.22))
    assert pre.matrix()[:3, 0] == pytest.approx([0, 0, -1])  # the gripper points down
    assert pre.matrix()[:3, 1] == pytest.approx([1, 0, 0])  # its y along the closing axis
    assert len(plannable.asked) == 1 and len(plannable.asked[0]) == POSES.approach_tilts  # type: ignore[attr-defined]
    assert await approach_pose(cube, 3, 0.1, PREFERENCE, 0.12, plannable, POSES) is None


def test_placed_object_pose_aligns_z_then_x() -> None:
    # A placement frame, z down, its x turned 0.3 rad from the world's.
    held = pose((0.5, 0.5, 1.0), axangle([0, 0, 1], 0.3) @ np.diag([1.0, -1, -1]))
    down = pose((0.0, 0.0, 0.8), np.diag([1.0, -1, -1]))
    free = placed_object_pose(held, down, "free", None, [1, 0, 0], "object", False, POSES)
    assert free.p == pytest.approx((0.0, 0.0, 0.8))
    assert free.matrix()[:3, 2] == pytest.approx([0, 0, -1])
    aligned = placed_object_pose(held, down, "align", None, [1, 0, 0], "object", False, POSES)
    assert aligned.matrix()[:3, 0] == pytest.approx([1, 0, 0])
    assert aligned.matrix()[:3, 2] == pytest.approx([0, 0, -1])
    # Without a placement frame the asset's local up (`object_up`) points up; here the held
    # asset lies on its side, its up a quarter turn from the world's.
    up = np.array(POSES.object_up)
    side = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    lying = pose((0.5, 0.5, 1.0), axangle(side / np.linalg.norm(side), math.pi / 2))
    upright = placed_object_pose(
        lying, pose((0, 0, 0.8), np.eye(3)), "free", None, [1, 0, 0], "object", True, POSES
    )
    assert upright.matrix()[:3, :3] @ np.array(POSES.object_up) == pytest.approx([0, 0, 1])


def test_place_ee_pose_carries_the_grasp_to_the_target() -> None:
    # The cube held 12 cm below a downward gripper; its bottom face center is the placement
    # frame, z down.
    ee = pose((0.1, 0.2, 0.8 + 0.12), np.column_stack([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]))
    bottom = rigid(np.diag([1.0, -1, -1]), (0.0, 0.0, -0.02))
    cube = box([], [bottom])
    target = pose((0.3, -0.1, 0.75), np.diag([1.0, -1, -1]))
    kwargs = {"constrain": "free", "placement_frame": 0}
    placed = place_ee_pose(cube, "left", target, ee, POSES, approach_m=0.0, **kwargs)  # type: ignore[arg-type]
    # The bottom lands on the target: the cube's center 2 cm above it, the ee 12 cm above that.
    assert placed.p == pytest.approx((0.3, -0.1, 0.75 + 0.02 + 0.12))
    assert placed.matrix()[:3, :3] == pytest.approx(ee.matrix()[:3, :3])
    back = place_ee_pose(cube, "left", target, ee, POSES, approach_m=0.05, **kwargs)  # type: ignore[arg-type]
    assert back.p == pytest.approx((0.3, -0.1, 0.75 + 0.02 + 0.12 + 0.05))  # along ee -> object
    side = place_ee_pose(
        cube, "left", target, ee, POSES, approach_m=0.05, approach_axis=[1, 0, 0], **kwargs
    )  # type: ignore[arg-type]
    assert side.p == pytest.approx((0.3 - 0.05, -0.1, 0.75 + 0.02 + 0.12))  # along the target's x
