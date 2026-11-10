"""The configuration schema every robot configuration builds on (docs/how-to/add-a-real-robot.md).

A robot's configuration is data: its arms with their grippers, an optional torso, an optional
mobile base, its cameras, the tasks a person can set up, and the safety limits. From it the
package derives everything the core wants to know about a robot: the `EmbodimentProfile` the
toolsets read, the `EnvCapabilities` a real robot declares, the `RobotDescription` its server
answers `describe()` with and the `Benchmark` descriptor a run names it by.

What a real robot declares is not configurable (docs/how-to/add-a-real-robot.md): success is judged
from outside, a person resets the scene, and there is never a truth snapshot or an expert pre-run.
The configuration has no field for any of it, so no configuration can claim otherwise.

The action vector is laid out as the simulated robots' are: per arm its joints then its gripper
command (1 open, 0 closed), arms in the order the configuration lists them, then the torso's
joints.
"""

from pathlib import Path
from typing import Annotated, Final, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PositiveInt,
    StringConstraints,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.action import ActionSpace, BaseControl, EeDeltaControl
from roboscript.contracts.capabilities import EnvCapabilities, SensingCapabilities
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import ArmName
from roboscript.contracts.robot_server import RobotDescription, RobotLimits
from roboscript.contracts.skills import PoseConventions
from roboscript.embodiments import (
    ArmProfile,
    CameraSpec,
    EmbodimentProfile,
    GraspPreference,
    Gripper,
    JointGroup,
)
from roboscript.envs import Benchmark, TaskInfo
from roboscript.settings import load_config_file

C = TypeVar("C", bound="RobotConfig")

RobotName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
"""A robot's name: the name of its embodiment profile, its benchmark descriptor and its entry in
`roboscript.robots`."""

TORSO: Final = "torso"
"""The name of the joint group that holds a torso's joints (`EmbodimentProfile.joint_groups`)."""
SERVER_MODULE: Final = "roboscript_robots.base.server"
"""The env server module of every robot's `Benchmark` descriptor: it runs as `python -m
<module> --config <file> --ready-file <path>`."""


class RobotTask(BaseModel):
    """A task a person can set up at the robot: the instruction the Agent gets and the step
    budget of one episode (counted actions, repeats included)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    step_limit: PositiveInt


class ArmHardware(BaseModel):
    """One arm as hardware: the names of its `joints`, the `ee_link` whose pose is the ee, the
    `ee_to_tcp` offset of the fingertip midpoint, its `gripper`, the joints it rests at (`home`,
    required: a real robot has a pose it can be sent to safely) and the range each joint may
    turn through (`stops`, rad; required for a real arm, since no model of the arm's limits is
    consulted). `grasp` is how the core's grasp chooser picks, if the robot uses it. Fields are
    those of `roboscript.embodiments.ArmProfile`; the action layout is the configuration's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    joints: list[str] = Field(min_length=1)
    ee_link: str
    ee_to_tcp: Vec3
    gripper: Gripper
    home: list[FiniteFloat]
    stops: list[tuple[FiniteFloat, FiniteFloat]]
    grasp: GraspPreference | None = None

    @model_validator(mode="after")
    def _per_joint(self) -> Self:
        for what, values in (("home", self.home), ("stops", self.stops)):
            if len(values) != len(self.joints):
                raise ValueError(f"{what} has {len(values)} values, the arm {len(self.joints)}")
        return self


class TorsoHardware(BaseModel):
    """The joints of a torso (or a head) that belong to no arm, with the pose they rest at and
    the range each may turn through or slide over."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    joints: list[str] = Field(min_length=1)
    home: list[FiniteFloat]
    stops: list[tuple[FiniteFloat, FiniteFloat]]

    @model_validator(mode="after")
    def _per_joint(self) -> Self:
        for what, values in (("home", self.home), ("stops", self.stops)):
            if len(values) != len(self.joints):
                raise ValueError(
                    f"torso {what} has {len(values)} values, {len(self.joints)} joints"
                )
        if any(low >= high for low, high in self.stops):
            raise ValueError("every torso joint's stops must be low < high")
        return self


class RobotConfig(BaseModel):
    """A real robot: arms (by name, in the order of the action vector), a torso, cameras, the
    kinds of action it takes, its tasks and its safety limits. `depth`: the cameras give depth.
    `home_on_open`: opening an episode first sends the robot home, so that the arm is out of the
    way while a person sets the scene up. The configurations in `single_arm`, `dual_arm` and
    `mobile_manipulator` fix the shape (how many arms, whether there is a base)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: RobotName
    world_axes: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    arms: dict[ArmName, ArmHardware] = Field(min_length=1)
    torso: TorsoHardware | None = None
    cameras: dict[str, CameraSpec] = Field(min_length=1)
    depth: bool = True
    poses: PoseConventions
    joint_position: bool = True
    ee_delta: EeDeltaControl | None = None
    base: BaseControl | None = None
    tasks: dict[TaskName, RobotTask] = Field(min_length=1)
    limits: RobotLimits
    home_on_open: bool = True

    @model_validator(mode="after")
    def _the_profile_builds(self) -> Self:
        # the profile's own checks (action layout, wrist cameras on arms it has, ee-delta arms)
        # run at load, not when the first server asks for it
        self.profile()
        return self

    @property
    def action_dim(self) -> int:
        return sum(len(arm.joints) + 1 for arm in self.arms.values()) + (
            len(self.torso.joints) if self.torso else 0
        )

    def actions(self) -> ActionSpace:
        return ActionSpace(
            joint_position=self.joint_position, ee_delta=self.ee_delta, base=self.base
        )

    def profile(self) -> EmbodimentProfile:
        """The embodiment profile toolsets read: per arm its joints then its gripper, then the
        torso's joints."""
        arms: dict[ArmName, ArmProfile] = {}
        offset = 0
        for name, arm in self.arms.items():
            stop = offset + len(arm.joints)
            arms[name] = ArmProfile(
                joints=arm.joints,
                action=(offset, stop),
                gripper_index=stop,
                ee_link=arm.ee_link,
                ee_to_tcp=arm.ee_to_tcp,
                gripper=arm.gripper,
                grasp=arm.grasp,
                home=arm.home,
                stops=arm.stops,
            )
            offset = stop + 1
        groups = {}
        if self.torso is not None:
            groups[TORSO] = JointGroup(
                joints=self.torso.joints, action=(offset, offset + len(self.torso.joints))
            )
        return EmbodimentProfile(
            name=self.name,
            action_dim=self.action_dim,
            world_axes=self.world_axes,
            arms=arms,
            cameras=self.cameras,
            poses=self.poses,
            joint_groups=groups,
            actions=self.actions(),
        )

    def home_vector(self) -> list[float]:
        """The action vector that sends the robot home: every arm at its home joints with its
        gripper open, the torso at its home."""
        home: list[float] = []
        for arm in self.arms.values():
            home += [*arm.home, arm.gripper.open]
        return home + (list(self.torso.home) if self.torso else [])

    def torso_range(self) -> tuple[int, int] | None:
        """Where the torso's joints sit in the action vector (None: no torso)."""
        if self.torso is None:
            return None
        start = self.action_dim - len(self.torso.joints)
        return start, self.action_dim

    def capabilities(self) -> EnvCapabilities:
        """What every real robot declares, whatever its shape: success judged from outside, a
        person's reset, the real-robot extensions, no truth, no expert. Fixed here so that no
        configuration can claim otherwise."""
        return EnvCapabilities(
            success=frozenset({"external"}), reset=frozenset({"manual"}), real_robot=True
        )

    def sensing(self) -> SensingCapabilities:
        """What its sensors give: measured proprioception, no reproducible scene identity (a
        person sets the scene up), the cameras it has, steps counted in actions."""
        return SensingCapabilities(
            scene_identity=False,
            measured_proprio=True,
            depth=self.depth,
            cameras=list(self.cameras),
            step_unit="action",
        )

    def description(self) -> RobotDescription:
        return RobotDescription(
            name=self.name,
            tasks=list(self.tasks),
            embodiment=self.name,
            cameras=list(self.cameras),
            capabilities=self.capabilities(),
            limits=self.limits,
        )

    def profile_yaml(self) -> str:
        """The embodiment profile as the YAML file `load_embodiment` reads: what a robot package
        ships next to its configuration (`embodiment.yaml`) for its `Benchmark` descriptor and
        its `roboscript.embodiments` entry to point at. A test keeps the file equal to this."""
        body = yaml.safe_dump(self.profile().model_dump(mode="json"), sort_keys=False)
        return (
            f"# Generated from the {self.name} configuration by RobotConfig.profile_yaml().\n{body}"
        )

    def benchmark(self, embodiment: Path) -> Benchmark:
        """The descriptor a run names this robot by (`env: {name: <robot>}`): its tasks, the robot
        server as the env server, and the embodiment profile's YAML (`profile_yaml`, kept as a
        file because a `Benchmark` read back from JSON turns an embodiment name into a path)."""
        return Benchmark(
            name=self.name,
            tasks={name: TaskInfo(step_limit=task.step_limit) for name, task in self.tasks.items()},
            env_server=SERVER_MODULE,
            interpreter="agent",
            embodiment=embodiment,
            env_config={"robot": self.name},
            capabilities=self.capabilities(),
            env_config_model=f"{SERVER_MODULE}:RobotServerConfig",
        )


def load_robot_config(path: Path, model: type[C]) -> C:
    """A robot's configuration from a YAML file, validated as `model` (one of the configurations'
    classes)."""
    return load_config_file(path, model)
