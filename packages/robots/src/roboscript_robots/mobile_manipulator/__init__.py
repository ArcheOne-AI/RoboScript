"""The mobile-manipulator configuration (docs/how-to/add-a-real-robot.md): a schema, a hardware
backend skeleton and a fake robot. A model module (`mobile_manipulator/<model>.py`) subclasses
`MobileManipulatorBackend`."""

from pathlib import Path

from roboscript_robots.base.config import load_robot_config
from roboscript_robots.base.spec import RobotSpec
from roboscript_robots.mobile_manipulator.backend import MobileManipulatorBackend
from roboscript_robots.mobile_manipulator.config import MobileManipulatorConfig

FAKE_CONFIG = load_robot_config(Path(__file__).with_name("fake.yaml"), MobileManipulatorConfig)
FAKE = RobotSpec("fake_mobile_manipulator", FAKE_CONFIG, MobileManipulatorBackend)
FAKE_PROFILE = FAKE_CONFIG.profile()
EMBODIMENT = Path(__file__).with_name("embodiment.yaml")
FAKE_BENCHMARK = FAKE_CONFIG.benchmark(EMBODIMENT)

__all__ = [
    "EMBODIMENT",
    "FAKE",
    "FAKE_BENCHMARK",
    "FAKE_CONFIG",
    "FAKE_PROFILE",
    "MobileManipulatorBackend",
    "MobileManipulatorConfig",
]
