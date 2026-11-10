"""The keypoint referee's paths: the TCP between two recorded steps is the forward kinematics of
the joint line between their drive targets (here `MockPlanner`'s), not the chord between the
measured TCPs."""

import math

import numpy as np
import pytest

from roboscript.demos.paths import ArmPaths
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.motion import MockPlanner

PROFILE = load_embodiment(MOCK_BIMANUAL)


def samples() -> dict[str, np.ndarray]:
    """The start, a quarter turn of the left arm's yaw in place (turn 1), then its gripper closing
    without joint motion (turn 2). The right arm stays put, open."""
    left = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, math.pi / 2]]
    right = [0.2, 0.0, 1.0, 0.0, 0.0, 0.0]
    qpos = [
        [*left[0], 1.0, *right, 1.0],
        [*left[1], 1.0, *right, 1.0],
        [*left[1], 0.0, *right, 1.0],
    ]
    rest, turned = [1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    return {
        "step": np.array([0, 1, 2]),
        "turn": np.array([0, 1, 2]),
        "qpos_cmd": np.array(qpos),
        "left.tcp": np.array(
            [[0.1, 0.0, 1.0, *rest], [0.0, 0.1, 1.0, *turned], [0.0, 0.1, 1.0, *turned]]
        ),
        "left.gripper_cmd": np.array([1.0, 1.0, 0.0]),
        "right.tcp": np.array([[0.3, 0.0, 1.0, *rest]] * 3),
        "right.gripper_cmd": np.array([1.0, 1.0, 1.0]),
    }


@pytest.mark.parametrize("step_rad", [0.01, 1e-4])  # 1e-4: over MAX_JOINT_VECTORS, in chunks
async def test_the_path_follows_the_joint_line(step_rad: float) -> None:
    paths = ArmPaths(MockPlanner(PROFILE), PROFILE, step_rad=step_rad, step_m=0.005)
    await paths.extend(samples())
    left, right = paths.paths()["left"], paths.paths()["right"]
    n = math.ceil(math.pi / 2 / step_rad)  # joint samples in the quarter turn
    assert len(left.tcp) == 1 + n + 1
    # On the arc the TCP draws 10 cm around the ee, not on the chord between its ends.
    radius = np.linalg.norm(left.tcp[:, :3] - [0.0, 0.0, 1.0], axis=1)
    np.testing.assert_allclose(radius, 0.1, atol=1e-9)
    yaw = math.pi / 2 * (n // 2) / n  # the joint line's point halfway
    np.testing.assert_allclose(left.tcp[n // 2, :3], [0.1 * math.cos(yaw), 0.1 * math.sin(yaw), 1])
    gaps = np.linalg.norm(np.diff(left.tcp[:, :3], axis=0), axis=1)
    assert gaps.max() <= 0.1 * step_rad + 1e-9
    # The gripper command changes where the step that changed it ends.
    assert left.gripper.tolist() == [1.0] * (n + 1) + [0.0]
    assert paths.turns()["left"].tolist() == [0] + [1] * n + [2]
    assert len(right.tcp) == 3 and np.allclose(right.tcp[:, :3], [0.3, 0.0, 1.0])
