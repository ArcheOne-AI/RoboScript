"""RoboTwin captures as observations (`roboscript_robotwin.frames`)."""

from pathlib import Path

import numpy as np

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import ArmState, RobotState
from roboscript_robotwin.frames import MAX_DEPTH_M, Frame, observation


def _robot() -> RobotState:
    ee = Pose(p=(0.0, 0.0, 1.0), q=(1.0, 0.0, 0.0, 0.0))
    arm = ArmState(ee=ee, tcp=ee, joints=None, joints_cmd=[0.0] * 6, gripper=None, gripper_cmd=1.0)
    return RobotState(
        step=3, step_limit=None, arms={"left": arm, "right": arm}, qpos_cmd=[0.0] * 14
    )


def test_depth_that_hit_nothing_is_stored_as_no_return(tmp_path: Path) -> None:
    depth = np.full((4, 6), 0.8, dtype=np.float32)
    depth[0, 0], depth[1, 1], depth[2, 2], depth[3, 3] = np.inf, np.nan, MAX_DEPTH_M + 1, -0.1
    frame = Frame(K=np.eye(3), cam2world=np.eye(4), rgb=np.zeros((4, 6, 3), np.uint8), depth=depth)
    store = LocalBlobStore(tmp_path)
    seen = observation({"wrist": frame}, _robot(), None, store)
    ref = seen.cameras["wrist"].depth
    assert ref is not None
    stored = store.get_array(ref)
    expected = np.full((4, 6), 0.8)
    expected[[0, 1, 2, 3], [0, 1, 2, 3]] = 0.0
    np.testing.assert_allclose(stored, expected, atol=0.0006)
    assert seen.step == 3
