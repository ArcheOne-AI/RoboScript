"""The mobile-manipulator configuration: arms on a torso and a mobile base."""

from pydantic import model_validator
from typing_extensions import Self

from roboscript_robots.base.config import RobotConfig


class MobileManipulatorConfig(RobotConfig):
    """One or more arms, usually a torso, and a mobile base (`base`, required: how it is
    commanded and how fast it may go). The base is declared and its limits are data, but base
    steps are not carried by `env.agent.execute` (`backend.NOT_CARRIED`): the robot is driven by
    joint positions."""

    @model_validator(mode="after")
    def _has_a_base(self) -> Self:
        if self.base is None:
            raise ValueError("a mobile manipulator declares its base (`base:`)")
        return self
