"""The real-robot server: the env contract's required faces plus the real-robot extensions
(docs/concepts/embodied-core.md, docs/how-to/add-a-real-robot.md).

A robot server is its own process. Its agent face is `contracts.env.EnvAgent` unchanged (robot
state, observations, action execution); `RobotControl` is its control face. Safety order: an
estop wins over everything, limits are checked before every action, and a missed heartbeat
stops the robot. `roboscript_robots` serves both faces (its `fake` backend runs them without
hardware); no runner drives `RobotControl`.
"""

from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, PositiveFloat, model_validator
from typing_extensions import Self

from roboscript.contracts.capabilities import EnvCapabilities, OpenSpec
from roboscript.contracts.env import EpisodeOpened, JudgeResult
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.ids import EpisodeId, TaskName
from roboscript.contracts.versions import VERSIONS


class RobotLimits(BaseModel):
    """`control_hz`: the rate the server commands the hardware at; `workspace_min` and
    `workspace_max`: the box (world frame) a TCP may not leave; speed caps on the TCP and on any
    joint; `heartbeat_timeout_s`: silence from the client after which the robot stops."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control_hz: PositiveFloat
    workspace_min: Vec3
    workspace_max: Vec3
    max_tcp_speed_m_s: PositiveFloat
    max_joint_speed_rad_s: PositiveFloat
    heartbeat_timeout_s: PositiveFloat

    @model_validator(mode="after")
    def _box(self) -> Self:
        if any(
            low >= high for low, high in zip(self.workspace_min, self.workspace_max, strict=True)
        ):
            raise ValueError("workspace_min must be below workspace_max on every axis")
        return self


class RobotDescription(BaseModel):
    """What `describe()` returns: the task table, the embodiment (a registered profile name), the
    cameras, the capabilities (always a real robot's) and the limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    tasks: list[TaskName]
    embodiment: str
    cameras: list[str]
    capabilities: EnvCapabilities
    limits: RobotLimits

    @model_validator(mode="after")
    def _real(self) -> Self:
        if not self.capabilities.real_robot:
            raise ValueError("a robot server's capabilities must declare real_robot")
        return self


class RobotControl(Protocol):
    """The control face. `open` with a manual reset shows its prompt and returns once a person
    confirms the scene; `judge` returns the outside judge's verdict (`success_step` None);
    `estop` stops at once and holds until the server is restarted; `home` drives to the
    embodiment's home pose under the limits; `heartbeat` keeps the episode's robot moving."""

    PLANE: ClassVar[str] = "robot.control"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def describe(self) -> RobotDescription: ...

    async def open(self, spec: OpenSpec) -> EpisodeOpened: ...

    async def judge(self, episode_id: EpisodeId) -> JudgeResult: ...

    async def close(self, episode_id: EpisodeId) -> None: ...

    async def estop(self, reason: str) -> None: ...

    async def home(self, episode_id: EpisodeId) -> None: ...

    async def heartbeat(self, episode_id: EpisodeId) -> None: ...
