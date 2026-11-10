"""The dual-arm configuration (docs/how-to/add-a-real-robot.md): a schema, a hardware backend
skeleton and a fake robot. A model module (`dual_arm/<model>.py`) subclasses `DualArmBackend`."""

from pathlib import Path

from roboscript_robots.base.config import load_robot_config
from roboscript_robots.base.spec import RobotSpec
from roboscript_robots.dual_arm.backend import DualArmBackend
from roboscript_robots.dual_arm.config import DualArmConfig

FAKE_CONFIG = load_robot_config(Path(__file__).with_name("fake.yaml"), DualArmConfig)
FAKE = RobotSpec("fake_dual_arm", FAKE_CONFIG, DualArmBackend)
FAKE_PROFILE = FAKE_CONFIG.profile()
EMBODIMENT = Path(__file__).with_name("embodiment.yaml")
FAKE_BENCHMARK = FAKE_CONFIG.benchmark(EMBODIMENT)

__all__ = [
    "EMBODIMENT",
    "FAKE",
    "FAKE_BENCHMARK",
    "FAKE_CONFIG",
    "FAKE_PROFILE",
    "DualArmBackend",
    "DualArmConfig",
]
