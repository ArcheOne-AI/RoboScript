"""The dual-arm configuration: two arms with grippers, cameras, no base."""

from pydantic import model_validator
from typing_extensions import Self

from roboscript_robots.base.config import RobotConfig


class DualArmConfig(RobotConfig):
    """Two arms (by name, `left` and `right` by convention), grippers, cameras. It has no torso
    and no base."""

    @model_validator(mode="after")
    def _two_arms(self) -> Self:
        if len(self.arms) != 2:
            raise ValueError(f"a dual-arm robot has two arms, got {sorted(self.arms)}")
        if self.torso is not None or self.base is not None:
            raise ValueError("a dual-arm robot has no torso and no base: use mobile_manipulator")
        return self
