"""The mobile-manipulator hardware backend skeleton."""

from roboscript_robots.base.backend import HardwareBackend
from roboscript_robots.base.config import RobotConfig
from roboscript_robots.mobile_manipulator.config import MobileManipulatorConfig


class MobileManipulatorBackend(HardwareBackend):
    """Subclass it in the robot's model module (`mobile_manipulator/<model>.py`) and implement:
    `connect`, `disconnect`, `read_arm` (once per arm), `write_arms`, `read_torso` and
    `write_torso` (when the configuration has a torso), `read_camera`, `forward_kinematics` and
    `halt`. Each raises `NotImplementedError` until it is. `execute` carries joint positions only,
    so the server never drives the base, and `command_base` raises `NotImplementedError`."""

    def __init__(self, config: RobotConfig) -> None:
        if not isinstance(config, MobileManipulatorConfig):
            raise TypeError(
                "MobileManipulatorBackend takes a MobileManipulatorConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__(config)
