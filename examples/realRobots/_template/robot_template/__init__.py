"""The real-robot template (examples/realRobots/_template, docs/how-to/add-a-real-robot.md):
what one robot's deployment adds to `packages/robots`. Copy this directory to
`examples/realRobots/<model>/`, rename `robot_template`, and fill it in."""

from robot_template.deployment import (
    CONFIG,
    EMBODIMENT,
    MY_ROBOT,
    MyRobotBackend,
    MyRobotConfig,
    load_deployment,
)
from robot_template.skills import ROBOT_SKILLS

__all__ = [
    "CONFIG",
    "EMBODIMENT",
    "MY_ROBOT",
    "ROBOT_SKILLS",
    "MyRobotBackend",
    "MyRobotConfig",
    "load_deployment",
]
