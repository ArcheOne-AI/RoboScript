"""What a robot package registers in `roboscript.robots` (docs/how-to/add-a-real-robot.md)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from roboscript.kernel import entrypoints
from roboscript_robots.base.backend import RobotBackend
from roboscript_robots.base.config import RobotConfig
from roboscript_robots.base.fake import FakeBackend

GROUP: Final = "roboscript.robots"

BackendKind = Literal["fake", "hardware"]


@dataclass(frozen=True)
class RobotSpec:
    """A robot: its configuration and the backends that drive it. `hardware`: builds the
    backend that talks to the real robot (a `HardwareBackend` subclass of its configuration);
    `fake`: builds a software stand-in for tests and tutorials (default `FakeBackend`, which takes
    arms of six joints; a robot with other arms supplies its own)."""

    name: str
    config: RobotConfig
    hardware: Callable[[RobotConfig], RobotBackend]
    fake: Callable[[RobotConfig], RobotBackend] = FakeBackend

    def __post_init__(self) -> None:
        if self.name != self.config.name:
            raise ValueError(
                f"robot {self.name!r} carries the configuration of {self.config.name!r}"
            )

    def backend(self, kind: BackendKind, config: RobotConfig | None = None) -> RobotBackend:
        """A backend of `kind` over `config` (default: the registered one)."""
        make = self.fake if kind == "fake" else self.hardware
        return make(self.config if config is None else config)


def load_robot(name: str) -> RobotSpec:
    """The robot registered as `name` in `roboscript.robots`."""
    found = entrypoints.load(GROUP, name)
    if not isinstance(found, RobotSpec):
        raise TypeError(f"{name!r} in {GROUP} is a {type(found).__name__}, not a RobotSpec")
    return found
