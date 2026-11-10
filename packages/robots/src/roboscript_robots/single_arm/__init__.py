"""The single-arm configuration (docs/how-to/add-a-real-robot.md): a schema, a hardware backend
skeleton and a fake robot. A model module (`single_arm/<model>.py`) subclasses
`SingleArmBackend`."""

from pathlib import Path

from roboscript_robots.base.config import load_robot_config
from roboscript_robots.base.spec import RobotSpec
from roboscript_robots.single_arm.backend import SingleArmBackend
from roboscript_robots.single_arm.config import SingleArmConfig

FAKE_CONFIG = load_robot_config(Path(__file__).with_name("fake.yaml"), SingleArmConfig)
FAKE = RobotSpec("fake_single_arm", FAKE_CONFIG, SingleArmBackend)
FAKE_PROFILE = FAKE_CONFIG.profile()
EMBODIMENT = Path(__file__).with_name("embodiment.yaml")
FAKE_BENCHMARK = FAKE_CONFIG.benchmark(EMBODIMENT)

__all__ = [
    "EMBODIMENT",
    "FAKE",
    "FAKE_BENCHMARK",
    "FAKE_CONFIG",
    "FAKE_PROFILE",
    "SingleArmBackend",
    "SingleArmConfig",
]
