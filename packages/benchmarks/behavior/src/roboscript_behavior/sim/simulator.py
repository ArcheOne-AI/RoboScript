"""BEHAVIOR behind the planes' `SimBackend`: everything done with an OmniGibson environment, on the
one simulation thread. The official code is reached through `sim.official` only.

The official action is a 23-vector (base velocity, torso, arms and grippers). A joint-position
step from the wire fills the torso, arm and gripper slots and leaves the base still. The wire
carries no base step, so `drive_base` raises `NotImplementedError`
(`roboscript.envs.skeleton.NOT_CARRIED`).

The verdict is terminal: the q-score is the share of goal predicates satisfied when it is asked,
and `success` all of them. There is no latch (`success_step` stays None), so a batch never stops
at success; the episode ends at the step limit.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import RobotState
from roboscript.embodiments import EmbodimentProfile
from roboscript.envs.skeleton import (
    ArmReading,
    Frame,
    Opened,
    Verdict,
    build_robot_state,
    not_carried,
)
from roboscript_behavior.benchmark import BEHAVIOR, INSTANCES, SCENES
from roboscript_behavior.config import BehaviorConfig
from roboscript_behavior.sim import official


class BehaviorOptions(BaseModel):
    """BEHAVIOR's reset options (`EpisodeSpec.options`): whether the episode has the task's step
    limit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_limit: bool = True


class BehaviorSimulator:
    """One OmniGibson environment, reset to one numbered instance of its activity."""

    options = BehaviorOptions

    def __init__(self, config: BehaviorConfig, profile: EmbodimentProfile) -> None:
        self.config, self.profile = config, profile
        self._official: official.Official | None = None
        self._env: Any = None
        self._cmd = np.zeros(profile.action_dim)
        self._steps = 0
        self._limit: int | None = None

    def start(self) -> str | None:
        self._official = official.load()
        return None  # return OmniGibson's version here when integrating

    def open(self, task: TaskName, index: int, options: BehaviorOptions) -> Opened:
        assert self._official is not None  # `start` ran before serving
        if not 0 <= index < INSTANCES:
            raise ProtocolError(
                "invalid_request", f"{task} has {INSTANCES} instances, not {index + 1}"
            )
        self.close()
        self._env = self._official.make_env(task, SCENES[task], index)
        self._steps = 0
        self._limit = BEHAVIOR.tasks[task].step_limit if options.step_limit else None
        self._cmd = self._joint_vector()
        return Opened(instruction=task.replace("_", " "), step_limit=self._limit)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _joint_vector(self) -> np.ndarray:
        """The profile's joint vector from the measured joints: each arm's joints and its
        gripper (normalized opening), and the torso's joints."""
        assert self._official is not None
        measured = self._official.joint_positions(self._env)
        vector = np.zeros(self.profile.action_dim)
        for spec in self.profile.arms.values():
            start, stop = spec.action
            vector[start:stop] = [measured[j] for j in spec.joints]
            vector[spec.gripper_index] = spec.gripper.opening(measured[spec.gripper.joint])
        for group in self.profile.joint_groups.values():
            start, stop = group.action
            vector[start:stop] = [measured[j] for j in group.joints]
        return vector

    def robot(self) -> RobotState:
        assert self._official is not None
        measured = self._joint_vector()
        arms = {}
        for arm, spec in self.profile.arms.items():
            start, stop = spec.action
            arms[arm] = ArmReading(
                ee=self._official.eef_pose(self._env, arm),
                joints=measured[start:stop].tolist(),
                joints_cmd=self._cmd[start:stop].tolist(),
                gripper=float(measured[spec.gripper_index]),
                gripper_cmd=float(self._cmd[spec.gripper_index]),
            )
        return build_robot_state(self.profile, self._steps, self._limit, arms, self._cmd.tolist())

    def frames(self, cameras: list[str]) -> dict[str, Frame]:
        assert self._official is not None
        return {name: self._official.frame(self._env, name) for name in cameras}

    def step(self, qpos: list[float]) -> None:
        action = np.zeros(official.ACTION_DIM)  # the base stays still
        for arm, spec in self.profile.arms.items():
            joints, gripper = official.ARM_AND_GRIPPER[arm]
            start, stop = spec.action
            action[joints] = qpos[start:stop]
            action[gripper] = 2.0 * qpos[spec.gripper_index] - 1.0  # normalized to -1..1
        torso = self.profile.joint_groups["torso"].action
        action[official.TORSO] = qpos[torso[0] : torso[1]]
        self._env.step(action)
        self._cmd = np.asarray(qpos, dtype=float)
        self._steps += 1

    def drive_base(self, forward: float, sideways: float, turn: float) -> None:
        """Command the base's planar velocity for one step. The wire does not carry it."""
        raise not_carried("BEHAVIOR's base moves by velocity")

    def verdict(self) -> Verdict:
        assert self._official is not None
        score, success = self._official.q_score(self._env)
        return Verdict(
            success=success,
            success_step=None,
            steps_used=self._steps,
            step_limit=self._limit,
            score=score,
        )

    def object_poses(self) -> dict[str, Pose]:
        assert self._official is not None
        return self._official.object_poses(self._env)
