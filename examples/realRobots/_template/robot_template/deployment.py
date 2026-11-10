"""A real robot's deployment, built on `packages/robots` (docs/how-to/add-a-real-robot.md): its
configuration is the robots package's schema (`SingleArmConfig` here; pick `DualArmConfig` or
`MobileManipulatorConfig` for another shape) plus the settings only its backend reads, its
backend subclasses the configuration's skeleton, and `MY_ROBOT` is the `RobotSpec` the robot
server runs. Everything a real robot declares (success judged from outside, a person's reset, no
ground truth, no expert), its embodiment profile and its server's description come from the
robots package, so the template cannot drift from it.
"""

from pathlib import Path
from typing import Final

from pydantic import model_validator
from typing_extensions import Self

from roboscript_robots.base import RobotSpec, load_robot_config
from roboscript_robots.single_arm import SingleArmBackend, SingleArmConfig

HERE: Final = Path(__file__).resolve().parent
EXAMPLE: Final = HERE.parent / "configs" / "robot.example.yaml"
EMBODIMENT: Final = HERE.parent / "embodiments" / "my_robot.yaml"
"""The embodiment profile, as `MyRobotConfig.profile_yaml()` writes it (a test keeps them equal)."""


class MyRobotConfig(SingleArmConfig):
    """The robots package's single-arm configuration plus this robot's hardware settings, which
    only `MyRobotBackend` reads: the controller's `address` and each camera's device."""

    address: str
    camera_devices: dict[str, str]

    @model_validator(mode="after")
    def _a_device_per_camera(self) -> Self:
        if set(self.camera_devices) != set(self.cameras):
            raise ValueError(
                f"camera_devices {sorted(self.camera_devices)} are not the cameras "
                f"{sorted(self.cameras)}"
            )
        return self


class MyRobotBackend(SingleArmBackend):
    """The hardware backend: implement the hardware points of `SingleArmBackend` (`connect`,
    `disconnect`, `read_arm`, `write_arms`, `read_camera`, `forward_kinematics`, `halt`) with the
    vendor SDK, reading `self.config.address` and `self.config.camera_devices`. Until then each
    raises `NotImplementedError` naming itself (docs/how-to/add-a-real-robot.md section 3)."""


def load_deployment(path: Path) -> MyRobotConfig:
    """The robot's configuration in the YAML file `path`, checked as the server checks it."""
    return load_robot_config(path, MyRobotConfig)


CONFIG: Final = load_deployment(EXAMPLE)
MY_ROBOT: Final = RobotSpec(CONFIG.name, CONFIG, hardware=MyRobotBackend)
"""The robot the server runs (`roboscript_robots.base.build_server(config, MY_ROBOT)`): the
example configuration, which a server's `robot_file` replaces with this machine's."""
