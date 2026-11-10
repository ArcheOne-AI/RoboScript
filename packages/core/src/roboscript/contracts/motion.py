"""The motion plane: joint paths, IK and FK for one arm per call (docs/concepts/embodied-core.md).

Poses are the embodiment's ee link in the world frame; the TCP, the fingertip midpoint, is
`ee_to_tcp` ahead of it. `base` is the world pose of the robot root the arm's planner frame hangs
off (an embodiment's `robot_pose`); requests carry it, so a planner keeps no episode state. Joint
vectors hold the arm's joints in the order of `PlannerInfo.arms[arm].joints`. A plan should start
where the next move will: at the drive target, which `take_action('qpos')` moves from, or at the
measured joints after a blocked move.

A planner may share random state among the episodes it serves (cuRobo does), so identical
requests may return different paths to the same end.
"""

from types import MappingProxyType
from typing import ClassVar, Final, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeFloat,
    PositiveFloat,
    computed_field,
)

from roboscript.contracts.geometry import OrientedBox, Pose, Vec3
from roboscript.contracts.robot import Arm
from roboscript.contracts.versions import VERSIONS

Joints = list[FiniteFloat]

MAX_GOALS: Final = 64
MAX_JOINT_VECTORS: Final = 4096
MAX_OBSTACLES: Final = 32

TIMEOUTS: Final = MappingProxyType(
    {"info": 10.0, "fk": 10.0, "ik": 30.0, "plan": 30.0, "plan_batch": 120.0}
)
"""Client timeouts in seconds, e.g. `rpc.client(Planner, url, token=None, timeout=10,
timeouts=TIMEOUTS)`: a plan takes about 60 ms, but cuRobo may spend its 10 s budget on a hard
goal, and calls wait for other episodes' on the service's single worker."""

Constraint = Literal["free", "orientation"]
"""`orientation` holds the ee orientation along the path as a soft cost and leaves the position
free. A locked plan can still swing far: verify its `ee_path`. The path is not
straight, and the start orientation must already be within about 5.7 deg of the goal's."""

PlanReason = Literal["ok", "unreachable", "no_path", "invalid_start", "orientation_mismatch"]
"""`unreachable`: no IK solution for the goal within joint limits and free of collisions;
`no_path`: no trajectory found (in a batch, any failure but an invalid start or no IK solution
for any goal);
`invalid_start`: the start violates joint limits or collides; `orientation_mismatch`: an
orientation-locked plan whose start orientation differs from the goal's."""


class ArmInfo(BaseModel):
    """An arm's joints in planning order with their limits, the links FK can report, and where
    the TCP is in the ee frame (with the ee's orientation), as in its embodiment profile."""

    model_config = ConfigDict(frozen=True)

    joints: list[str]
    lower: Joints
    upper: Joints
    links: list[str]
    ee_to_tcp: Vec3


class PlannerInfo(BaseModel):
    """Paths have a point every `dt` seconds."""

    model_config = ConfigDict(frozen=True)

    provider: str
    arms: dict[Arm, ArmInfo]
    dt: PositiveFloat


class ArmRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: Arm
    base: Pose


class PlanRequest(ArmRequest):
    """A path from `start` to the ee pose `goal`; `ee_path` asks for the ee pose at every point.
    `obstacles` are boxes in the world the path must keep the arm out of, for this
    request only, besides what the planner always avoids: the scene as perceived, say. A start
    or goal inside one fails the plan (`invalid_start`, `unreachable`); IK, FK and batch plans
    never see them."""

    start: Joints
    goal: Pose
    constraint: Constraint = "free"
    ee_path: bool = False
    obstacles: list[OrientedBox] = Field(default=[], max_length=MAX_OBSTACLES)


class BatchPlanRequest(ArmRequest):
    """Several goals planned independently from one start; results come back in the order of
    `goals`."""

    start: Joints
    goals: list[Pose] = Field(min_length=1, max_length=MAX_GOALS)
    constraint: Constraint = "free"
    ee_path: bool = False


class PlanResult(BaseModel):
    """`path` runs from the start to the goal with a point every `dt` seconds; `detail` holds the
    planner's own status text when the plan failed; `time_s` is the planner's wall time for the
    request (for a batch: the whole batch). `status` is derived from `reason`."""

    model_config = ConfigDict(frozen=True)

    reason: PlanReason
    detail: str = ""
    path: list[Joints] | None = None
    dt: PositiveFloat
    ee_path: list[Pose] | None = None
    time_s: NonNegativeFloat = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> Literal["success", "failed"]:
        return "success" if self.reason == "ok" else "failed"


class IKRequest(ArmRequest):
    """Joint vectors reaching each ee pose in `goals`, near `seed`, which the solver starts from."""

    goals: list[Pose] = Field(min_length=1, max_length=MAX_GOALS)
    seed: Joints


class IKSolution(BaseModel):
    """`joints` is None when no solution within limits and free of collisions was found; the
    errors are between the goal and the pose `joints` reach."""

    model_config = ConfigDict(frozen=True)

    joints: Joints | None
    position_error_m: NonNegativeFloat | None = None
    rotation_error_deg: NonNegativeFloat | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success(self) -> bool:
        return self.joints is not None


Sphere = tuple[FiniteFloat, FiniteFloat, FiniteFloat, NonNegativeFloat]
"""A collision sphere: its world center (m) and radius (m)."""


class FKRequest(ArmRequest):
    """`links` names links from `ArmInfo.links` whose poses to report too; `spheres` asks for
    the arm's collision spheres (the spheres the planner keeps the arm clear with)."""

    joints: list[Joints] = Field(min_length=1, max_length=MAX_JOINT_VECTORS)
    links: list[str] = []
    spheres: bool = False


class FKResult(BaseModel):
    """One ee pose, TCP pose and pose per requested link for each joint vector, in order, and
    the arm's collision spheres per joint vector when asked for."""

    model_config = ConfigDict(frozen=True)

    ee: list[Pose]
    tcp: list[Pose]
    links: dict[str, list[Pose]] = Field(default_factory=dict)
    spheres: list[list[Sphere]] = Field(default_factory=list)


class Planner(Protocol):
    """Plans, IK and FK for the arms of one embodiment, shared by any number of episodes."""

    PLANE: ClassVar[str] = "motion"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def info(self) -> PlannerInfo: ...

    async def plan(self, request: PlanRequest) -> PlanResult: ...

    async def plan_batch(self, request: BatchPlanRequest) -> list[PlanResult]: ...

    async def ik(self, request: IKRequest) -> list[IKSolution]: ...

    async def fk(self, request: FKRequest) -> FKResult: ...
