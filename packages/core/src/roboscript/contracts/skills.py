"""The skillset manifest: operation APIs as one artifact kind (docs/how-to/add-a-skillset.md).

A skillset is a set of Agent-facing physical tools the Hub mounts. Its manifest says where it
sits, what it fits, what each skill needs, which bindings it provides to the core's entries and
how it was verified; it never records how it was built. The bindings travel with it as data, so
a skillset served by another process binds the same values: `PlannerProfile` (the planner's
collision table and the arms' turns), the executor's tuning (`ExecutorConfig`) and the surface
the scenes stand on (`SupportSurface`). An entry whose
bindings no mounted skillset provides is not mounted. The embodiment and the pose conventions are
not among them: they are the env's embodiment profile's (`Benchmark.embodiment`,
`EmbodimentProfile.poses`), which every episode names, so every caller of the pose math (the
action API, the scene registry's grasp relations, a demonstration's relations, a method's checks)
has them with the profile it already holds.
"""

from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.action import ActionKind
from roboscript.contracts.artifact import Artifact
from roboscript.contracts.executor import ExecutorConfig
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.ids import ToolName
from roboscript.contracts.robot import ArmName

SkillSetLayer = Literal["core", "benchmark", "robot", "built"]
"""Core entries, a benchmark's or a real robot's bindings and skills, or a skill builder's output
not yet written as code."""
BindingName = Literal["planner_profile", "executor", "support_surface"]
"""The fields of `Bindings`."""
BINDING_NAMES: tuple[BindingName, ...] = ("planner_profile", "executor", "support_surface")
Matrix3 = tuple[Vec3, Vec3, Vec3]
Matrix4 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
]
ChainStep = Literal["parameters", "perception", "planning", "executor", "env_execute", "receipt"]
CHAIN: tuple[ChainStep, ...] = (
    "parameters",
    "perception",
    "planning",
    "executor",
    "env_execute",
    "receipt",
)
"""A skill's implementation chain runs through these in this order, skipping any."""
VerifyStep = Literal["interface", "boundary", "ceiling", "freeze"]


class PoseConventions(BaseModel):
    """How an env's assets and grasps are framed, which the core's pose math follows
    (`manipulation.posemath`, `manipulation.synthesis`), part of its embodiment profile
    (`EmbodimentProfile.poses`): `contact_to_grasp`, the contact frame (y against the approach, z
    the closing axis) to the grasp frame (x the approach), row-major; `tcp_closing_axis`, the
    fingers' closing axis in the ee and TCP frames (the grasp frame's y); `object_up`, an object's
    up axis in its local frame, which a placement without a placement point aligns with the
    target's z; `placement_rotation`, a placement point's rotation in the frame of the box face
    it sits on (z down when upright); `approach_tilts`, how many tilts about the closing axis a
    grasp is tried at."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contact_to_grasp: Matrix4
    tcp_closing_axis: Vec3
    object_up: Vec3
    placement_rotation: Matrix3
    approach_tilts: PositiveInt


class PlannerTable(BaseModel):
    """The one cuboid of the planner's collision world: its `size` (m) and its top's height
    (world z); placed in the planner frame from an arm's root pose (`motion.curobo.table_world`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    size: Vec3
    top_z: FiniteFloat


class PlannerProfile(BaseModel):
    """The env's planning setup: the collision table and each arm's turn about z after the frame
    bias (the yaw its URDF mounts it with), which an arm's planner config may override."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: PlannerTable
    yaw_rad: dict[ArmName, FiniteFloat] = Field(default_factory=dict)


class SupportSurface(BaseModel):
    """The fixed surface an env's scenes stand on, where it has one (an env
    whose objects rest on many surfaces binds none): `top_z`, its top's nominal
    height (world z), which a scene may set up to `max_drop_m` lower; `extent` (x0, x1, y0, y1,
    world), the area it spans, where known; `landmarks`, named points on it (world); and
    `refuse_below`, whether a TCP target or path below the top is refused. The `manipulation`
    toolset
    keeps targets above the table perception measured, or until then above its config's
    `table_top_z`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_z: FiniteFloat
    max_drop_m: NonNegativeFloat = 0.0
    extent: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None = None
    landmarks: dict[str, Vec3] = Field(default_factory=dict)
    refuse_below: bool = True

    @model_validator(mode="after")
    def _extent(self) -> Self:
        if self.extent is not None:
            x0, x1, y0, y1 = self.extent
            if not (x0 < x1 and y0 < y1):
                raise ValueError(f"extent {self.extent} needs x0 < x1 and y0 < y1")
        return self


class Bindings(BaseModel):
    """What an env binds for the core's entries (None: not bound)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner_profile: PlannerProfile | None = None
    executor: ExecutorConfig | None = None
    support_surface: SupportSurface | None = None

    def provided(self) -> frozenset[BindingName]:
        return frozenset(name for name in BINDING_NAMES if getattr(self, name) is not None)

    def merged(self, other: "Bindings") -> "Bindings":
        """These and `other`'s; a binding both give is an error (two skillsets disagree on who
        binds it)."""
        if twice := sorted(self.provided() & other.provided()):
            raise ValueError(f"two skillsets bind {twice}")
        return self.model_copy(update={name: getattr(other, name) for name in other.provided()})


class SkillNeeds(BaseModel):
    """What a skill needs of the embodiment: at least `min_arms` arms, each named among `arms`
    (the names its parameters take; empty: any names), and the kinds of action it executes
    (`actions`, `EmbodimentProfile.actions`; empty: it moves nothing)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_arms: PositiveInt = 1
    gripper: bool = True
    arms: list[ArmName] = []
    actions: list[ActionKind] = []


class SkillSpec(BaseModel):
    """One skill: the tool's name, description and parameter schema as the model sees them,
    what it needs of the embodiment, the perception inputs it reads (e.g. registered objects or
    parts), its implementation chain, the bindings its entry asks of the env, and the names of
    its verification cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[ToolName, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
    description: str
    parameters: dict[str, Any]
    needs: SkillNeeds = SkillNeeds()
    perception: list[str] = []
    chain: list[ChainStep]
    bindings: list[BindingName] = []
    cases: list[str] = []

    @field_validator("chain")
    @classmethod
    def _in_order(cls, chain: list[ChainStep]) -> list[ChainStep]:
        positions = [CHAIN.index(step) for step in chain]
        if positions != sorted(set(positions)):
            raise ValueError(f"chain {chain} must follow {list(CHAIN)}, each step once")
        return chain


class StepResult(BaseModel):
    """One verification step: whether it passed, why not, and its measure (the reachable
    ceiling's success share)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    detail: str = ""
    value: FiniteFloat | None = None


class VerificationReport(BaseModel):
    """The four checks of skillset validation: interface, information boundary, reachable ceiling,
    freeze."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: dict[VerifyStep, StepResult]

    @property
    def passed(self) -> bool:
        return all(step.passed for step in self.steps.values())


class SkillSet(Artifact):
    """A skillset manifest: its `name`, `version` and `layer`; the `envs` and `embodiments` it
    fits (empty: any); its `skills`, each with its schema, what it needs of the robot and of
    perception; `bindings`: what it binds for the core's entries, as data; `guide`: its guide's
    Markdown (its `guide.md`, read in; a template over the robot it runs on, `skills.guide`),
    which a stack that mounts the skillset hands the method's prompts (`learning.runtime`
    `PromptFactory`); `report`: how it was verified. Nothing records how it was built."""

    kind: ClassVar[str] = "skillset"

    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
    version: str
    layer: SkillSetLayer
    envs: list[str] = Field(default_factory=list)
    embodiments: list[str] = Field(default_factory=list)
    skills: list[SkillSpec] = Field(default_factory=list)
    bindings: Bindings = Bindings()
    guide: str | None = None
    report: VerificationReport | None = None

    @field_validator("skills")
    @classmethod
    def _unique(cls, skills: list[SkillSpec]) -> list[SkillSpec]:
        names = [skill.name for skill in skills]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"skills named twice: {twice}")
        return skills
