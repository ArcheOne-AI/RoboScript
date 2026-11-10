"""Proprioception. Measured values are None when unavailable, never replaced by commands."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
)

from roboscript.contracts.geometry import Pose

ArmName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
"""An arm's name as its embodiment declares it (`EmbodimentProfile.arms`): a Panda's `arm`, a
bimanual robot's `left` and `right`. How many arms a robot has and what they are called is its
profile's data, never assumed."""
Arm = Literal["left", "right"]
"""The arms of a two-arm embodiment that names them left and right (the mock robot, for one): the
names the core's manipulation entries, their receipts and the motion plane take.
A skill bound to them says so (`SkillNeeds.arms`), so it does not mount on a robot whose arms are
named otherwise."""


class ArmState(BaseModel):
    """One arm. `ee` and `tcp` are measured world poses (TCP = fingertip midpoint); joints in
    radians; the gripper is normalized, 1 open and 0 closed."""

    model_config = ConfigDict(frozen=True)

    ee: Pose
    tcp: Pose
    joints: list[FiniteFloat] | None
    joints_cmd: list[FiniteFloat]
    gripper: FiniteFloat | None
    gripper_cmd: FiniteFloat


class RobotState(BaseModel):
    """The robot after `step` counted actions; `arms` by the names its embodiment declares;
    `qpos_cmd` is the full commanded joint vector (`EmbodimentProfile.action_dim`)."""

    model_config = ConfigDict(frozen=True)

    step: NonNegativeInt
    step_limit: PositiveInt | None
    arms: dict[ArmName, ArmState]
    qpos_cmd: list[FiniteFloat]
