"""FAKE official modules for the placeholder adapters' tests: tiny stand-ins that have the surface
each adapter's `sim/official.py` documents, so the adapters' env servers run under the env
conformance suite without OmniGibson installed. A fake is a
scene that holds still, records the actions it is given, and reports success (or a score) once
the gripper has been commanded closed, which is how a test makes an episode succeed.

`STEP_DELAY_S` makes a step take a little time, so that the conformance suite's concurrent
`execute` finds the first batch still running."""

import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

STEP_DELAY_S = 0.01
HOME7 = [0.0, 0.196, 0.0, -2.618, 0.0, 2.94, 0.785]
EE_POS = (0.5, 0.0, 1.0)


def _pause() -> None:
    if STEP_DELAY_S:
        time.sleep(STEP_DELAY_S)


def _install(monkeypatch: pytest.MonkeyPatch, modules: dict[str, ModuleType]) -> None:
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _module(name: str, **attributes: Any) -> ModuleType:
    module = ModuleType(name)
    vars(module).update(attributes)
    return module


def _intrinsics(width: int, height: int) -> np.ndarray:
    return np.array([[200.0, 0, width / 2], [0, 200.0, height / 2], [0, 0, 1.0]])


def _cam2world(near: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = near
    return matrix


R1_JOINTS = (
    [f"left_arm_joint{i}" for i in range(1, 8)]
    + ["left_gripper_finger_joint1"]
    + [f"right_arm_joint{i}" for i in range(1, 8)]
    + ["right_gripper_finger_joint1"]
    + [f"torso_joint{i}" for i in range(1, 5)]
)
GOALS = 4


class FakeSensor:
    def __init__(self, near: tuple[float, float, float]) -> None:
        self.near = near
        self.intrinsic_matrix = _intrinsics(224, 224)

    def get_obs(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        rgba = np.full((224, 224, 4), 9, dtype=np.uint8)
        return {"rgb": rgba, "depth_linear": np.full((224, 224), 0.6, dtype=np.float32)}, {}

    def get_position_orientation(self) -> tuple[tuple[float, float, float], tuple[float, ...]]:
        return self.near, (0.0, 0.0, 0.0, 1.0)


class FakeR1Pro:
    """Joint positions follow the 23-vector action; the left gripper closing counts a goal."""

    def __init__(self) -> None:
        self.joint_names = list(R1_JOINTS)
        self.positions = dict.fromkeys(R1_JOINTS, 0.0)
        for name in ("left_gripper_finger_joint1", "right_gripper_finger_joint1"):
            self.positions[name] = 0.05
        self.sensors = {
            "head": FakeSensor((0.4, 0.0, 1.4)),
            "left_wrist": FakeSensor((0.4, 0.2, 1.0)),
            "right_wrist": FakeSensor((0.4, -0.2, 1.0)),
        }

    def get_joint_positions(self) -> np.ndarray:
        return np.array([self.positions[n] for n in self.joint_names])

    def get_eef_position(self, arm: str) -> np.ndarray:
        return np.array((0.4, 0.2 if arm == "left" else -0.2, 1.0))

    def get_eef_orientation(self, arm: str) -> np.ndarray:
        return np.array((0.0, 0.0, 0.0, 1.0))  # x, y, z, w


class FakeBehaviorEnv:
    """`omnigibson.Environment`: four goal predicates, one more satisfied per step with the left
    gripper commanded closed."""

    instances: ClassVar[list["FakeBehaviorEnv"]] = []

    def __init__(self, configs: dict[str, Any]) -> None:
        self.configs = configs
        self.robot = FakeR1Pro()
        self.robots = [self.robot]
        self.actions: list[np.ndarray] = []
        self.satisfied = 0
        self.closed = False
        self.scene = SimpleNamespace(
            objects=[
                SimpleNamespace(
                    name="radio", get_position_orientation=lambda: ((1.0, 0.0, 0.8), (0, 0, 0, 1))
                ),
                SimpleNamespace(
                    name="table", get_position_orientation=lambda: ((1.2, 0.1, 0.4), (0, 0, 0, 1))
                ),
            ]
        )
        self.task = SimpleNamespace(goal_status=None)
        self._status()
        FakeBehaviorEnv.instances.append(self)

    def reset(self) -> None:
        pass

    def _status(self) -> None:
        met = list(range(self.satisfied))
        self.task.goal_status = {"satisfied": met, "unsatisfied": list(range(GOALS - len(met)))}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        _pause()
        assert action.shape == (23,)
        self.actions.append(action.copy())
        positions = self.robot.positions
        for arm, joints, gripper in (("left", slice(7, 14), 14), ("right", slice(15, 22), 22)):
            for i, value in enumerate(action[joints], start=1):
                positions[f"{arm}_arm_joint{i}"] = float(value)
            positions[f"{arm}_gripper_finger_joint1"] = (float(action[gripper]) + 1.0) / 2 * 0.05
        for i, value in enumerate(action[3:7], start=1):
            positions[f"torso_joint{i}"] = float(value)
        if action[14] < 0 and self.satisfied < GOALS:
            self.satisfied += 1
            self._status()
        return {}, 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


def install_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeBehaviorEnv.instances.clear()
    _install(monkeypatch, {"omnigibson": _module("omnigibson", Environment=FakeBehaviorEnv)})
