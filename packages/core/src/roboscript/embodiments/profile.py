"""`EmbodimentProfile`: arms, action layout and kinds, TCP, grippers, cameras and planner assets,
as data."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final, Literal

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeInt,
    PositiveInt,
    SerializerFunctionWrapHandler,
    StringConstraints,
    ValidationError,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.action import ActionSpace
from roboscript.contracts.geometry import Pose, Quat, Vec3
from roboscript.contracts.robot import ArmName
from roboscript.contracts.skills import PoseConventions
from roboscript.kernel import entrypoints
from roboscript.settings import expand_path
from roboscript.spatial.pose import pose_compose

GROUP = "roboscript.embodiments"
STOP_SLACK_RAD: Final = 1e-4
"""A joint this close past its stop is at it: the planner works in float32, and a solution it
clamps to a stop comes back rounded past it (-1.85 as -1.8500000238), and a strict comparison
would refuse every path that brings a joint to its stop."""


class Gripper(BaseModel):
    """Commands are normalized: `open` and `closed` are the commands of an open and a closed
    gripper. The measured opening is `(q - scale[0]) / (scale[1] - scale[0])` for the position q
    of `joint`; `empty` is that opening once settled after closing on nothing, and a settled
    opening >= `holding_threshold` means something is between the fingers. `span_m`, if known, is
    the gap between the open fingers: a grasp closing across more cannot hold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    joint: str
    scale: tuple[FiniteFloat, FiniteFloat]
    open: FiniteFloat = 1.0
    closed: FiniteFloat = 0.0
    empty: FiniteFloat
    holding_threshold: FiniteFloat
    span_m: FiniteFloat | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _threshold_above_empty(self) -> Self:
        if not self.empty < self.holding_threshold < self.open:
            raise ValueError(
                f"holding_threshold {self.holding_threshold} must lie between the empty reading "
                f"{self.empty} and open {self.open}"
            )
        return self

    def opening(self, q: float) -> float:
        low, high = self.scale
        return (q - low) / (high - low)


class PlannerAssets(BaseModel):
    """One arm's motion-planner inputs, relative to the profile's `assets_root`; `base` is the
    world pose of the robot root the planner config is expressed in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    curobo: str | None = None
    urdf: str
    srdf: str | None = None
    base: Pose


class GraspPreference(BaseModel):
    """How the grasp chooser picks for an arm (`manipulation.posemath.choose_grasp`):
    each grasp is tried at approach tilts from `tilt[0]` to `tilt[1]` rad toward the robot, and
    among contact frames the one nearest `top_down` (within 27 deg) wins, else the one nearest
    `side`; ee orientations, wxyz."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tilt: tuple[FiniteFloat, FiniteFloat] = (0.0, 0.0)
    top_down: Quat
    side: Quat


class ArmProfile(BaseModel):
    """One arm: `joints` fill `action[0]:action[1]` of the action vector and the gripper command
    sits at `gripper_index`. The ee frame is the pose of `ee_link`; the TCP (fingertip midpoint)
    is at `ee_to_tcp` in that frame, with the ee's orientation. `home` holds the joints the arm
    starts every episode at, whose ee pose `return_home` returns to (None: the robot has no
    such pose). `stops` gives the range each joint really turns through, low and high (rad),
    where the arm stops before the planner model's limits (a mechanical or self-contact stop
    the model lacks); None: the model's limits hold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    joints: list[str]
    action: tuple[NonNegativeInt, NonNegativeInt]
    gripper_index: NonNegativeInt
    ee_link: str
    ee_to_tcp: Vec3
    gripper: Gripper
    planner: PlannerAssets | None = None
    grasp: GraspPreference | None = None
    home: list[FiniteFloat] | None = None
    stops: list[tuple[FiniteFloat, FiniteFloat]] | None = None

    @model_validator(mode="after")
    def _home_fits(self) -> Self:
        if self.home is not None and len(self.home) != len(self.joints):
            raise ValueError(f"home has {len(self.home)} joints, the arm {len(self.joints)}")
        if self.stops is not None:
            if len(self.stops) != len(self.joints):
                raise ValueError(f"stops has {len(self.stops)} joints, the arm {len(self.joints)}")
            if any(low >= high for low, high in self.stops):
                raise ValueError("every joint's stops must be low < high")
        return self

    def past_stops(self, joints: Sequence[Sequence[float]]) -> str | None:
        """Which joint of these joint vectors turns past its stop (by more than
        `STOP_SLACK_RAD`), in words; None when none."""
        if self.stops is None:
            return None
        q = np.asarray(joints, dtype=float).reshape(-1, len(self.joints))
        low, high = np.asarray(self.stops, dtype=float).T
        low, high = low - STOP_SLACK_RAD, high + STOP_SLACK_RAD
        for i in range(len(self.joints)):
            lowest, highest = float(q[:, i].min()), float(q[:, i].max())
            if lowest < low[i] or highest > high[i]:
                turn, stop = (lowest, low[i]) if lowest < low[i] else (highest, high[i])
                return f"joint {i + 1} would turn to {turn:.2f} rad, past its stop at {stop:.2f}"
        return None


class CameraSpec(BaseModel):
    """A camera: fixed (`head`) or carried by an arm (`wrist`), image size and camera type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["head", "wrist"]
    type: str
    width: PositiveInt
    height: PositiveInt
    arm: ArmName | None = None

    @model_validator(mode="after")
    def _wrist_names_its_arm(self) -> Self:
        if (self.kind == "wrist") != (self.arm is not None):
            raise ValueError("a wrist camera names the arm carrying it, a head camera none")
        return self


class JointGroup(BaseModel):
    """Joints of the joint vector that belong to no arm (a torso, a head): their names and where
    they sit, `action[0]:action[1]`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    joints: list[str] = Field(min_length=1)
    action: tuple[NonNegativeInt, NonNegativeInt]


class EmbodimentProfile(BaseModel):
    """A robot as toolsets see it. `arms` by the names it declares (one or more; the mock's are
    `left` and `right`); `action_dim`: the length of its joint vector, which its
    arms' joints and grippers and its `joint_groups` (a torso's joints) fill once each (the
    commanded vector of proprioception, `RobotState.qpos_cmd`); `actions`: the kinds of action it
    takes (`contracts.action.ActionSpace`; default: joint positions only, the joint vector).
    `world_axes` says in words how the world frame's axes lie relative to the robot, as the
    prompts tell a model. `poses`: how its env frames grasps and assets, which the core's pose
    math follows (`contracts.skills.PoseConventions`). `assets_root` may hold
    `${NAME}`-style variables, resolved only when an asset is asked for (`asset`).
    `description`: the robot and where it works in a prompt's words (the Actor's first line).
    `joint_groups`, `actions` and `description` are left out of a dump while at their defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    action_dim: PositiveInt
    world_axes: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    arms: dict[ArmName, ArmProfile] = Field(min_length=1)
    cameras: dict[str, CameraSpec]
    poses: PoseConventions
    assets_root: str | None = None
    joint_groups: dict[str, JointGroup] = Field(default_factory=dict)
    actions: ActionSpace = ActionSpace()
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = "a robot"

    @model_serializer(mode="wrap")
    def _leave_defaults_out(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if not self.joint_groups:
            data.pop("joint_groups", None)
        if self.actions == ActionSpace():
            data.pop("actions", None)
        if self.description == type(self).model_fields["description"].default:
            data.pop("description", None)
        return data

    @model_validator(mode="after")
    def _action_layout(self) -> Self:
        used: list[int] = []
        for arm, spec in self.arms.items():
            start, stop = spec.action
            if stop - start != len(spec.joints):
                raise ValueError(f"arm {arm}: action {spec.action} does not fit {spec.joints}")
            used += [*range(start, stop), spec.gripper_index]
        for name, group in self.joint_groups.items():
            start, stop = group.action
            if stop - start != len(group.joints):
                raise ValueError(f"group {name}: action {group.action} does not fit {group.joints}")
            used += range(start, stop)
        if sorted(used) != list(range(self.action_dim)):
            raise ValueError(
                f"the arms' joints and grippers and the joint groups must fill action indices "
                f"0..{self.action_dim - 1} once each, got {sorted(used)}"
            )
        return self

    @model_validator(mode="after")
    def _ee_arms(self) -> Self:
        ee = self.actions.ee_delta
        if ee is not None and (unknown := sorted(set(ee.arms) - set(self.arms))):
            raise ValueError(f"ee_delta control names arms {unknown} the profile does not have")
        return self

    def tcp_pose(self, arm: ArmName, ee: Pose) -> Pose:
        """Where `arm`'s TCP is when its ee is at `ee` (same orientation)."""
        return Pose.from_list(pose_compose(ee.as_list(), [*self.arms[arm].ee_to_tcp, 1, 0, 0, 0]))

    def ee_pose(self, arm: ArmName, tcp: Pose) -> Pose:
        """The ee pose that puts `arm`'s TCP at `tcp`; the action API's targets are ee poses."""
        back = [-v for v in self.arms[arm].ee_to_tcp]
        return Pose.from_list(pose_compose(tcp.as_list(), [*back, 1, 0, 0, 0]))

    def asset(self, path: str) -> Path:
        """`path`, relative to `assets_root`, as an absolute path."""
        if self.assets_root is None:
            raise ValueError(f"embodiment {self.name!r} has no assets_root")
        return expand_path(self.assets_root) / path


def load_embodiment(ref: str | Path) -> EmbodimentProfile:
    """A profile from a YAML file, or by a name registered in the `roboscript.embodiments`
    entry-point group (whose object is a profile or the path of its YAML file)."""
    if isinstance(ref, Path) or ref.endswith((".yaml", ".yml")):
        return _read(Path(ref))
    found = entrypoints.load(GROUP, ref)
    return found if isinstance(found, EmbodimentProfile) else _read(Path(found))


def _read(path: Path) -> EmbodimentProfile:
    try:
        return EmbodimentProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from None
