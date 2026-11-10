"""The single-arm hardware backend skeleton."""

from roboscript_robots.base.backend import HardwareBackend
from roboscript_robots.base.config import RobotConfig
from roboscript_robots.single_arm.config import SingleArmConfig


class SingleArmBackend(HardwareBackend):
    """Subclass it in the arm's model module (`single_arm/<model>.py`) and implement:
    `connect`, `disconnect`, `read_arm` (the arm's joints, gripper opening and ee pose),
    `write_arms` (called with the one arm's joints and gripper command each control tick),
    `read_camera`, `forward_kinematics` and `halt`. Each raises `NotImplementedError` until it
    is."""

    def __init__(self, config: RobotConfig) -> None:
        if not isinstance(config, SingleArmConfig):
            raise TypeError(
                f"SingleArmBackend takes a SingleArmConfig, got {type(config).__name__}"
            )
        super().__init__(config)
