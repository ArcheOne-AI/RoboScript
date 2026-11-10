"""The dual-arm hardware backend skeleton."""

from roboscript_robots.base.backend import HardwareBackend
from roboscript_robots.base.config import RobotConfig
from roboscript_robots.dual_arm.config import DualArmConfig


class DualArmBackend(HardwareBackend):
    """Subclass it in the robot's model module (`dual_arm/<model>.py`) and implement:
    `connect`, `disconnect`, `read_arm` (called once per arm), `write_arms` (called with both arms'
    targets each control tick: send them so that both arms move on the same tick),
    `read_camera`, `forward_kinematics` and `halt`. Each raises `NotImplementedError` until it
    is."""

    def __init__(self, config: RobotConfig) -> None:
        if not isinstance(config, DualArmConfig):
            raise TypeError(f"DualArmBackend takes a DualArmConfig, got {type(config).__name__}")
        super().__init__(config)
