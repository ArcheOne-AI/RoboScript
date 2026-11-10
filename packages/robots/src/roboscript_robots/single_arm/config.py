"""The single-arm configuration: one arm with its gripper, cameras, no base."""

from pydantic import model_validator
from typing_extensions import Self

from roboscript_robots.base.config import RobotConfig


class SingleArmConfig(RobotConfig):
    """One arm (any name; `arm` by convention), a gripper, cameras. It may take joint positions
    (the default), end-effector deltas (`ee_delta`, naming the arm), or both; it has no torso
    and no base."""

    @model_validator(mode="after")
    def _one_arm(self) -> Self:
        if len(self.arms) != 1:
            raise ValueError(f"a single-arm robot has one arm, got {sorted(self.arms)}")
        if self.torso is not None or self.base is not None:
            raise ValueError("a single-arm robot has no torso and no base: use mobile_manipulator")
        return self
