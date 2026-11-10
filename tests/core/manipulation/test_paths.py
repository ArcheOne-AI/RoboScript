"""roboscript.manipulation.paths: a held path fitted in the held object's body frame comes back
as the same rigid motion whichever way the box is measured turned, from wherever the gripper
holds; a mirrored path reflects the lateral axis and keeps proper orientations."""

import math

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.manipulation import ObjectProxy, Part, PathSpec
from roboscript.manipulation.paths import (
    body_frames,
    choose_frame,
    fit_path,
    motions,
)
from roboscript.manipulation.posemath import axangle
from roboscript.perception.registry import WHOLE
from roboscript.spatial.rotation import yaw_matrix

DOWN = (0.0, 0.7071068, 0.7071068, 0.0)  # a TCP pointing down, closing along y


def box(yaw_deg: float = 25.0, symmetry: str = "pi") -> ObjectProxy:
    at = np.eye(4)
    at[:3, :3] = yaw_matrix(math.radians(yaw_deg))
    at[:3, 3] = (0.05, 0.1, 0.78)
    part = Part(frame=np.eye(4), size=np.array([0.3, 0.2, 0.04]), x_known=False)
    return ObjectProxy("box", at, symmetry, {WHOLE: part})  # type: ignore[arg-type]


def turned_about(point: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = axangle(axis, math.radians(deg))
    m[:3, 3] = point - m[:3, :3] @ point
    return m


@pytest.mark.parametrize("turn_deg", [90.0, 180.0])
def test_a_path_is_the_same_motion_on_a_box_measured_turned(turn_deg: float) -> None:
    """Fitted on one measurement of the body, followed on another that turned the box's frame by
    a symmetric turn: the frame chosen by the grasp gives the same world motion."""
    body = box(symmetry="half_pi" if turn_deg == 90.0 else "pi")
    grip = Pose(p=(0.1, 0.0, 0.8), q=DOWN).matrix()
    hinge = (np.array([0.1, 0.25, 0.76]), np.array([-1.0, 0.0, 0.0]))
    arc = [grip] + [turned_about(hinge[0], hinge[1], a) @ grip for a in (10.0, 20.0, 31.0)]
    fit = fit_path([Pose.from_matrix(m) for m in arc], body)
    assert fit.position_error_m < 1e-4 and fit.turn_deg == pytest.approx(31.0, abs=0.01)
    again = box(symmetry=body.symmetry)
    turn = np.eye(4)
    turn[:3, :3] = yaw_matrix(math.radians(turn_deg))
    again = ObjectProxy("box", again.pose @ turn, again.symmetry, again.parts)
    frames = body_frames(again)
    assert len(frames) > 1  # the box allows turns: the grasp picks the frame
    frame, off, deg = choose_frame(frames, fit.spec, [Pose.from_matrix(grip)])
    assert (off, deg) == pytest.approx((0.0, 0.0), abs=1e-3)
    for world, pose in zip(motions(frame, fit.spec), arc[1:], strict=True):
        assert world @ grip == pytest.approx(pose, abs=5e-4)


def _tcp(p: tuple[float, float, float]) -> np.ndarray:
    return Pose(p=p, q=DOWN).matrix()


def test_a_mirrored_path_reflects_the_lateral_axis_and_keeps_proper_orientations() -> None:
    """The other arm's path: in the body frame the position's component along the axis nearest
    the robot's lateral one is negated, the rest kept, every orientation a proper rotation, and
    the motion it makes the world motion reflected across that plane."""
    from roboscript.manipulation.paths import _TCP_FLIPS, mirror_path
    from roboscript.manipulation.paths import path_matrix as as_matrix

    turned = (0.9238795, 0.0, 0.3826834, 0.0)  # a tilt about y, not along any axis's plane
    spec = PathSpec(poses=[(0.01, 0.02, 0.0, *DOWN), (0.03, 0.15, 0.06, *turned)], arm="right")
    frame = np.eye(4)
    frame[:3, :3] = yaw_matrix(math.radians(10.0))
    lateral = (0.0, 1.0, 0.0)  # nearest the frame's y
    for flip in _TCP_FLIPS:
        mirrored = mirror_path(spec, frame, lateral, flip)
        for pose, before in zip(mirrored.poses, spec.poses, strict=True):
            assert pose[:3] == pytest.approx((before[0], -before[1], before[2]), abs=1e-4)
            assert np.linalg.det(as_matrix(pose)[:3, :3]) == pytest.approx(1.0, abs=1e-3)
        plane = np.diag([1.0, -1.0, 1.0, 1.0])
        world = frame @ plane @ np.linalg.inv(frame)
        expected = [world @ m @ world for m in motions(frame, spec)]
        assert motions(frame, mirrored)[0] == pytest.approx(expected[0], abs=1e-3)
