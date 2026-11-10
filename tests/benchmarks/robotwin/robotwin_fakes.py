"""A stand-in for a RoboTwin task object in the evaluation loop (`TASK_ENV`), with only what the
shim reads and drives: `get_obs()` (three cameras, RGB, depth in millimeters, calibration),
`get_instruction()`, `take_action(qpos)` (counted, a no-op at the limit or after the success
latch), the step count and limit, the latch, and the robot's accessors. A closed gripper stops
where RoboTwin's does on nothing (the finger's lower limit)."""

from typing import Any

import numpy as np

from roboscript_robotwin.benchmark import ROBOTWIN

PROFILE = ROBOTWIN.profile()
REST = [0.0] * 6 + [1.0]


class Joint:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class Entity:
    """An arm's articulation: its joints and the gripper's finger joint."""

    def __init__(self, robot: "FakeRobot", arm: str) -> None:
        spec = PROFILE.arms[arm]  # type: ignore[index]
        self.robot, self.arm = robot, arm
        self.joints = [Joint(name) for name in [*spec.joints, spec.gripper.joint]]

    def get_active_joints(self) -> list[Joint]:
        return self.joints

    def get_qpos(self) -> np.ndarray:
        state = self.robot.targets[self.arm]
        low, high = PROFILE.arms[self.arm].gripper.scale  # type: ignore[index]
        finger = max(0.0, low + (high - low) * state[-1])  # closed on nothing: stops at 0
        return np.array([*state[:-1], finger])


class FakeRobot:
    def __init__(self) -> None:
        self.targets = {"left": list(REST), "right": list(REST)}
        self.left_entity, self.right_entity = Entity(self, "left"), Entity(self, "right")

    def __getattr__(self, name: str) -> Any:
        arm = "left" if "_left_" in name else "right"
        if name.endswith("jointState"):  # drive targets and measured joints alike
            return lambda: list(self.targets[arm])
        if name.endswith("ee_pose"):
            return lambda: [0.3 if arm == "right" else -0.3, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0]
        if name.endswith("tcp_pose"):
            return lambda: [0.42 if arm == "right" else -0.18, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0]
        raise AttributeError(name)


class FakeTaskEnv:
    """`success_at`: the latch fires on that counted action."""

    def __init__(
        self, step_lim: int = 20, success_at: int | None = None, instruction: str = "lift it"
    ) -> None:
        self.robot = FakeRobot()
        self.take_action_cnt, self.step_lim, self.eval_success = 0, step_lim, False
        self.success_at, self.instruction = success_at, instruction
        self.observed = 0  # get_obs() calls
        self.actions: list[list[float]] = []

    def get_obs(self) -> dict[str, Any]:
        self.observed += 1
        return {"observation": {name: _camera(name) for name in PROFILE.cameras}}

    def get_instruction(self) -> str:
        return self.instruction

    def take_action(self, action: Any, action_type: str = "qpos") -> None:
        assert action_type == "qpos"
        if self.take_action_cnt == self.step_lim or self.eval_success:
            return
        self.take_action_cnt += 1
        qpos = [float(v) for v in action]
        self.actions.append(qpos)
        self.robot.targets = {"left": qpos[:7], "right": qpos[7:]}
        if self.take_action_cnt == self.success_at:
            self.eval_success = True


def _camera(name: str) -> dict[str, Any]:
    spec = PROFILE.cameras[name]
    height, width = spec.height, spec.width
    return {
        "rgb": np.full((height, width, 3), 90, dtype=np.uint8),
        "depth": np.full((height, width), 700.0),  # millimeters, float64 as RoboTwin's
        "intrinsic_cv": np.array([[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0, 0, 1]]),
        "extrinsic_cv": np.hstack([np.eye(3), [[0.0], [0.0], [1.0]]]),
    }
