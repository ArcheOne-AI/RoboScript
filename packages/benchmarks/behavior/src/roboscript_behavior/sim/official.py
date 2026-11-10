"""The only module of this adapter that imports OmniGibson, and only inside functions, so the
adapter's agent side imports without it.

ASSUMED OFFICIAL SURFACE, to check against OmniGibson's and the BEHAVIOR challenge's source when
integrating (the tests run on a fake module that mimics exactly this surface,
`tests/benchmarks/behavior/fakes.py`):

- `omnigibson.Environment(configs=<dict>)` with a `BehaviorTask` (`activity_name`,
  `activity_definition_id`, `activity_instance_id`, `online_object_sampling` False) and one
  `R1Pro` robot; `reset()`, `step(action) -> obs, reward, terminated, truncated, info`, `close()`;
- the robot, `env.robots[0]`: `joint_names`, `get_joint_positions()`, `get_eef_position(arm)`,
  `get_eef_orientation(arm)` (x, y, z, w) and `sensors[<camera>]`, each with `get_obs()`
  (`rgb`, `depth_linear`), `intrinsic_matrix` and `get_position_orientation()`, the camera's pose
  in OpenGL's axes;
- the 23-vector action: base velocity (3), torso (4), left arm (7), left gripper (1), right arm
  (7), right gripper (1), the arms and torso by absolute joint position;
- `env.task.goal_status`: `{"satisfied": [...], "unsatisfied": [...]}` goal predicates, whose
  share satisfied now stands in for the q-score (`Official.q_score`);
- `env.scene.objects`, each with `name` and `get_position_orientation()` (position, (x, y, z, w)).
"""

import importlib
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from roboscript.contracts.geometry import Pose
from roboscript.envs.skeleton import Frame

ACTION_DIM: Final = 23
BASE, TORSO = slice(0, 3), slice(3, 7)
ARM_AND_GRIPPER: Final = {"left": (slice(7, 14), 14), "right": (slice(15, 22), 22)}
"""The official action vector's slices: base, torso, and each arm's joints and gripper."""
_GL_TO_CV: Final = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class Official:
    """OmniGibson, imported."""

    omnigibson: Any

    def make_env(self, activity: str, scene: str, instance: int) -> Any:
        config = {
            "scene": {"type": "InteractiveTraversableScene", "scene_model": scene},
            "robots": [{"type": "R1Pro", "obs_modalities": ["rgb", "depth_linear"]}],
            "task": {
                "type": "BehaviorTask",
                "activity_name": activity,
                "activity_definition_id": 0,
                "activity_instance_id": instance,
                "online_object_sampling": False,
            },
        }
        env = self.omnigibson.Environment(configs=config)
        env.reset()
        return env

    def joint_positions(self, env: Any) -> dict[str, float]:
        robot = env.robots[0]
        values = np.asarray(robot.get_joint_positions(), dtype=float)
        return dict(zip(robot.joint_names, values.tolist(), strict=True))

    def eef_pose(self, env: Any, arm: str) -> Pose:
        robot = env.robots[0]
        x, y, z, w = (float(v) for v in np.asarray(robot.get_eef_orientation(arm)))
        position = tuple(float(v) for v in np.asarray(robot.get_eef_position(arm)))
        return Pose(p=position, q=(w, x, y, z))

    def frame(self, env: Any, camera: str) -> Frame:
        """One camera's capture: depth in meters, the pose turned from OpenGL's camera axes to
        OpenCV's."""
        sensor = env.robots[0].sensors[camera]
        obs, _ = sensor.get_obs()
        position, (x, y, z, w) = sensor.get_position_orientation()
        cam2world = Pose(p=tuple(float(v) for v in position), q=(w, x, y, z)).matrix()
        return Frame(
            K=np.asarray(sensor.intrinsic_matrix, dtype=float),
            cam2world=cam2world @ _GL_TO_CV,
            rgb=np.ascontiguousarray(np.asarray(obs["rgb"])[..., :3], dtype=np.uint8),
            depth=np.ascontiguousarray(np.asarray(obs["depth_linear"], dtype=np.float32)),
        )

    def object_poses(self, env: Any) -> dict[str, Pose]:
        poses = {}
        for obj in env.scene.objects:
            position, (x, y, z, w) = obj.get_position_orientation()
            poses[str(obj.name)] = Pose(p=tuple(float(v) for v in position), q=(w, x, y, z))
        return poses

    def q_score(self, env: Any) -> tuple[float, bool]:
        """The share of goal predicates satisfied now, and whether all are. A stand-in for the
        official q-score, which scores the episode's end and counts a predicate already
        satisfied at the start in the denominator only."""
        status = env.task.goal_status
        satisfied, unsatisfied = len(status["satisfied"]), len(status["unsatisfied"])
        return satisfied / (satisfied + unsatisfied), unsatisfied == 0


def load() -> Official:
    """Import OmniGibson (it has to be installed in this interpreter)."""
    return Official(omnigibson=importlib.import_module("omnigibson"))
