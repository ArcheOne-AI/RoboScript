"""The env contract: public, `env.control` and `env.agent` planes
(docs/concepts/embodied-core.md).

Only the orchestrator holds `env.control` (ground truth lives there); toolsets reach the env
through `env.agent`, which serves what a real robot has.
"""

import json
from typing import ClassVar, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    FiniteFloat,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.capabilities import (
    EnvCapabilities,
    EpisodeReset,
    SeedReset,
    StateReset,
)
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.ids import EpisodeId, PhaseName, TaskName, WireSecret
from roboscript.contracts.motion import Constraint
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import Arm, RobotState
from roboscript.contracts.versions import VERSIONS


class EnvDiscovery(BaseModel):
    """What an env server is; `contracts` maps each plane to its version, e.g. `env.agent: 1.0`.
    `capabilities`: its optional faces and its sensing, which a server always declares (the
    sensing fields sent flat also parse)."""

    model_config = ConfigDict(frozen=True)

    contracts: dict[str, str]
    benchmark: str
    embodiment: str
    upstream_commit: str | None
    config_hash: str
    capabilities: EnvCapabilities

    @model_validator(mode="after")
    def _sensing(self) -> Self:
        if self.capabilities.sensing is None:
            raise ValueError("an env server declares its sensing capabilities")
        return self


class EpisodeSpec(BaseModel):
    """Reset one task at one seed. `instruction` reuses an earlier try's text, and `repeat`
    reopens the reset this env opened last with it (a later try at the same seed): an env may skip
    the checks it ran on that reset, such as an expert pre-run, which decides whether the seed
    counts and sets attributes some success checks read. A pre-run's planning may not be
    deterministic, so it gets `prerun_attempts` tries.
    `options` are the benchmark's own reset options (which of its instruction sets the text comes
    from, say): the env's plugin declares them, their defaults and their checks, and an env
    refuses one it does not declare (`roboscript.envs.reset_options`).
    `trace_expert` records the expert's API calls of the pre-run that passed, for
    `EnvControl.expert_trace`: ground truth, so only for episodes of no agent stage
    (control-plane tools and demo validation); envs without an expert refuse it. `reset_by` says
    how `seed` sets the scene: as a
    seed, or as the index of one of the env's fixed initial states; an env refuses a
    kind it does not offer (`EnvCapabilities.reset`, `unoffered_reset`). A person's reset is a
    real robot's (`contracts.robot_server`, `OpenSpec`). `reset_by` and `options` are left out of
    a dump at their defaults. A spec hashes by its content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    seed: NonNegativeInt
    stage: PhaseName | None = None  # the method's phase, or `demo` for a recording
    instruction: str | None = None
    repeat: bool = False
    prerun_attempts: PositiveInt = 1
    options: dict[str, JsonValue] = {}
    record: bool = False
    trace_expert: bool = False
    reset_by: EpisodeReset = "seed"

    @model_serializer(mode="wrap")
    def _leave_defaults_out(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if self.reset_by == "seed":
            data.pop("reset_by", None)
        if not self.options:
            data.pop("options", None)
        return data

    def __hash__(self) -> int:  # `options` is a dict: hash the canonical dump
        return hash(json.dumps(self.model_dump(mode="json"), sort_keys=True))

    def reset(self) -> SeedReset | StateReset:
        """The episode's reset as the generic spec (`contracts.capabilities.ResetSpec`)."""
        if self.reset_by == "initial_state":
            return StateReset(index=self.seed)
        return SeedReset(seed=self.seed)

    @model_validator(mode="after")
    def _repeat_reuses_the_instruction(self) -> Self:
        if self.repeat and self.instruction is None:
            raise ValueError("a repeat reuses the earlier open's instruction: pass it")
        if self.trace_expert and self.repeat:
            raise ValueError("trace_expert records the expert pre-run, which a repeat skips")
        if self.trace_expert and self.stage not in (None, "demo"):
            detail = f"the expert trace is ground truth: no trace_expert in stage {self.stage}"
            raise ValueError(detail)
        return self


class EpisodeOpened(BaseModel):
    """An open env episode. Toolsets reach it on its own agent plane (`agent_url` with
    `agent_token`, via the orchestrator's `EnvBinding`); `instruction` is None when the seed is not
    `solvable`. `prerun_attempts` counts the expert pre-runs made (0 when none ran)."""

    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    agent_url: str
    agent_token: WireSecret
    instruction: str | None
    step_limit: PositiveInt | None
    embodiment: str
    cameras: list[str]
    solvable: bool
    prerun_attempts: NonNegativeInt = 0


class JudgeResult(BaseModel):
    """The benchmark's verdict, as the env judges (`EnvCapabilities.success`): by a latch,
    `success_step` is when it first fired; by a terminal score, `score` is the score at the end
    and `success` the env's reading of it; by an
    outside judge (a real robot), `success` is that judge's verdict. `success_step` and `score`
    are None where the env does not judge so; `score` is left out of a dump while None.
    `faulted`: the simulator failed during the episode, an infrastructure error rather than a
    policy failure. `ended`: why the benchmark ended the episode before
    its step limit without success (a failure it checks for fired; batches stop `ended`), None
    while it runs; left out of a dump while None."""

    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    success: bool
    success_step: NonNegativeInt | None
    steps_used: NonNegativeInt
    step_limit: PositiveInt | None
    faulted: bool = False
    score: FiniteFloat | None = None
    ended: str | None = None

    @model_serializer(mode="wrap")
    def _leave_defaults_out(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        for name in ("score", "ended"):
            if getattr(self, name) is None:
                data.pop(name, None)
        return data


class JointTruth(BaseModel):
    """One joint of an articulated object: its position at the start and at the end of the
    episode, and its limits."""

    model_config = ConfigDict(frozen=True)

    initial: FiniteFloat
    final: FiniteFloat
    low: FiniteFloat
    high: FiniteFloat


class ObjectTruth(BaseModel):
    """Ground truth of one object at the start and the end of the episode: world poses, the
    centers of its geometry (the middle of the box around it, as perception's centers are of the
    box around its points) and its joints, if it is articulated."""

    model_config = ConfigDict(frozen=True)

    name: str
    initial: Pose
    final: Pose
    displacement_m: NonNegativeFloat
    initial_center: Vec3 | None = None
    final_center: Vec3 | None = None
    joints: list[JointTruth] = []


class ConditionMargin(BaseModel):
    """How far one numeric argument of a benchmark success check is from flipping the check's
    result now (`ConditionTruth`): its value, and the nearest value at which the check's
    result would flip (None: none within the factors searched). A tolerance's margin, in its
    own units, read without knowing what the check computes."""

    model_config = ConfigDict(frozen=True)

    arg: str
    value: FiniteFloat
    flips_at: FiniteFloat | None = None


class ConditionTruth(BaseModel):
    """One of the benchmark's success checks, evaluated at the end of an episode:
    which stage it belongs to (`pending`: the first not yet held; `latched`:
    the last held, read again now; `final`: judged at the end only), the check as text (it names
    the task's objects: a skillset build only, never an agent), whether it holds now and its numeric
    arguments' margins (`ConditionMargin`)."""

    model_config = ConfigDict(frozen=True)

    stage: str
    check: str
    holds: bool
    margins: list[ConditionMargin] = []


class Diagnosis(BaseModel):
    """Ground truth after an episode, for the Reviser only, who reads it in the try's own terms
    (a method's explainer, such as wicl's `diagnosis`); `notes` name no object (e.g. a
    fault). `conditions` (where the env can read its success checks):
    the benchmark's success checks at the end (`ConditionTruth`), for a skillset build only, left
    out
    of the record when there are none."""

    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    success: bool
    objects: list[ObjectTruth]
    notes: list[str]
    conditions: list[ConditionTruth] = []

    @model_serializer(mode="wrap")
    def _leave_defaults_out(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if not self.conditions:
            data.pop("conditions", None)
        return data


class GtJoint(BaseModel):
    """A movable link's joint: revolute or prismatic, its axis through
    `origin` along `axis` (world, unit; its position grows turning right-handed about the axis or
    sliding along it), the position now and its limits (None: unbounded)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["revolute", "prismatic"]
    origin: Vec3
    axis: Vec3
    position: FiniteFloat
    limits: tuple[FiniteFloat | None, FiniteFloat | None]


class GtLink(BaseModel):
    """A movable link of an articulated object: its world pose, its
    convex collision hulls in the world (one vertex list per collision shape) and its joint to
    its parent."""

    model_config = ConfigDict(frozen=True)

    pose: Pose
    hulls: list[list[Vec3]] = []
    joint: GtJoint


class EndRule(BaseModel):
    """A failure the benchmark ends the episode for: `object` moving
    farther than `limit_m` from where the episode put it (`moved`) or rising more than that
    (`lifted`). Privileged, as the rest of the truth: the check of an API keeps its calls within
    them; no agent sees them."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["moved", "lifted"]
    object: str
    limit_m: NonNegativeFloat


class Progress(BaseModel):
    """How far the benchmark's own success checks have got: the
    stages its success needs (checks it pops in order, conditional checks, the counts its
    queries must reach) and how many are still to come. Privileged, as the rest of the truth: the
    check of an API reads it to tell which call moved the task on; no agent sees it."""

    model_config = ConfigDict(frozen=True)

    stages: NonNegativeInt
    left: NonNegativeInt


class GtSnapshot(BaseModel):
    """Ground-truth world poses of the scene objects at one step (offline use), the table top's
    height when the env knows it, each articulated object's movable links by link name and, when
    asked for, each object's and link's convex hulls in the world (one vertex list per collision
    shape; a visual-only object's render shapes), which a control-plane tool measures boxes from
    and samples surfaces of; the failures the benchmark ends the episode for that it can state as
    an object's motion (`EndRule`), left out of the record when there are none; and its success
    checks' progress where
    the benchmark tracks stages (`Progress`), left out when it does not."""

    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    step: NonNegativeInt
    objects: dict[str, Pose]
    table_top_z: FiniteFloat | None = None
    hulls: dict[str, list[list[Vec3]]] = {}
    links: dict[str, dict[str, GtLink]] = {}
    end_rules: list[EndRule] = []
    progress: Progress | None = None

    @model_serializer(mode="wrap")
    def _leave_defaults_out(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if not self.end_rules:
            data.pop("end_rules", None)
        if self.progress is None:
            data.pop("progress", None)
        return data


class ExpertAction(BaseModel):
    """One entry of an arm's action list: the ee moves to `target`, planned under `constraint`,
    or the gripper is commanded to `gripper` (normalized, 1 open and 0 closed)."""

    model_config = ConfigDict(frozen=True)

    target: Pose | None = None
    constraint: Constraint = "free"
    gripper: FiniteFloat | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        if (self.target is None) == (self.gripper is None):
            raise ValueError("an expert action has either a target or a gripper command")
        return self


class ExpertBuilt(BaseModel):
    """How the data-generation API built one arm's action list: the
    task object it acts on, its other arguments (JSON; poses as 7-vectors), every task object's
    true pose then and its moving links' (by link name), and for `place_object` where the object
    would be once placed (the placement at `approach_m` 0)."""

    model_config = ConfigDict(frozen=True)

    obj: str | None = None
    args: dict[str, JsonValue] = {}
    objects: dict[str, Pose] = {}
    links: dict[str, dict[str, Pose]] = {}
    placed: Pose | None = None


class ExpertCall(BaseModel):
    """One motion call of the expert's run. `move`: each arm's action list, run pair by pair (the
    k-th actions of both arms at once), built by the data-generation API named in `api` (e.g.
    `grasp_object`, `place_object`, or `raw` for a list the task built itself); `grippers`: both
    grippers commanded at once; `hold`: nothing moves. `sim_steps`
    counts the simulation steps the call took; `refused`: the expert's planner refused it before
    anything moved; `partial`: it refused a later pair, and `actions` hold only what ran;
    `openings`: the measured finger opening of each arm in `actions` after the call, which tells
    whether a close held something (a task may close on nothing, to press); `built`: how each
    API-built list was built."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["move", "grippers", "hold"]
    actions: dict[Arm, list[ExpertAction]] = {}
    api: dict[Arm, str] = {}
    sim_steps: NonNegativeInt
    refused: bool = False
    partial: bool = False
    openings: dict[Arm, FiniteFloat] = {}
    built: dict[Arm, ExpertBuilt] = {}


class ExpertTrace(BaseModel):
    """The expert's API calls, in order, of the pre-run that made the episode's seed
    count: ground truth for control-plane tools (replaying an expert's calls, demo validation),
    never
    for agents."""

    model_config = ConfigDict(frozen=True)

    episode_id: EpisodeId
    calls: list[ExpertCall]


class ObserveRequest(BaseModel):
    """`cameras` None means every whitelisted camera."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cameras: list[str] | None = None
    depth: bool = True


class EnvPublic(Protocol):
    """Anyone may call it."""

    PLANE: ClassVar[str] = "env.public"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def discover(self) -> EnvDiscovery: ...


class EnvControl(Protocol):
    """The orchestrator's plane: episodes, the referee and ground truth."""

    PLANE: ClassVar[str] = "env.control"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def open(self, spec: EpisodeSpec) -> EpisodeOpened: ...

    async def judge(self, episode_id: EpisodeId) -> JudgeResult: ...

    async def diagnose(self, episode_id: EpisodeId) -> Diagnosis: ...

    async def gt_snapshot(self, episode_id: EpisodeId, geometry: bool = False) -> GtSnapshot:
        """Ground truth now; with `geometry` also the objects' and moving links' convex hulls,
        which a control-plane tool measures once per episode."""
        ...

    async def expert_trace(self, episode_id: EpisodeId) -> ExpertTrace: ...

    async def close(self, episode_id: EpisodeId) -> None: ...


class EnvAgent(Protocol):
    """The toolsets' plane: proprioception, whitelisted observations and action execution."""

    PLANE: ClassVar[str] = "env.agent"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def robot(self, episode_id: EpisodeId) -> RobotState: ...

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation: ...

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult: ...
