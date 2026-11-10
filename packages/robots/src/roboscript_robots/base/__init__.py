"""The robot server skeleton (docs/how-to/add-a-real-robot.md): the env contract's required
planes plus the real-robot extensions (emergency stop, homing, control rate, workspace and
speed limits, manual reset, external success judgment), the hardware seam (`RobotBackend`,
`HardwareBackend`), a software-only fake backend and the safety conventions (`safety`).

- `config`: the configuration schema every robot configuration builds on (`RobotConfig`).
- `backend`: `RobotBackend`, the hardware seam, and `HardwareBackend`, the skeleton to subclass.
- `server`: `RobotServer`, its apps and its process entry
  (`python -m roboscript_robots.base.server`).
- `safety`: the e-stop, limit and heartbeat conventions.
- `fake`: `FakeBackend`. `operator`: the person who resets and judges. `spec`: `RobotSpec`, what a
  robot registers in `roboscript.robots`.
"""

from roboscript_robots.base.backend import (
    NOT_CARRIED,
    ArmReading,
    ArmTarget,
    Capture,
    HardwareBackend,
    Measured,
    RobotBackend,
)
from roboscript_robots.base.config import (
    ArmHardware,
    RobotConfig,
    RobotTask,
    TorsoHardware,
    load_robot_config,
)
from roboscript_robots.base.fake import FakeBackend
from roboscript_robots.base.operator import ConsoleOperator, Operator, ScriptedOperator
from roboscript_robots.base.safety import Safety
from roboscript_robots.base.server import (
    Clock,
    RobotServer,
    RobotServerConfig,
    build_server,
    robot_apps,
    serve_robot,
)
from roboscript_robots.base.spec import GROUP, RobotSpec, load_robot

__all__ = [
    "GROUP",
    "NOT_CARRIED",
    "ArmHardware",
    "ArmReading",
    "ArmTarget",
    "Capture",
    "Clock",
    "ConsoleOperator",
    "FakeBackend",
    "HardwareBackend",
    "Measured",
    "Operator",
    "RobotBackend",
    "RobotConfig",
    "RobotServer",
    "RobotServerConfig",
    "RobotSpec",
    "RobotTask",
    "Safety",
    "ScriptedOperator",
    "TorsoHardware",
    "build_server",
    "load_robot",
    "load_robot_config",
    "robot_apps",
    "serve_robot",
]
