"""The `manipulation` toolset (docs/concepts/embodied-core.md,
docs/concepts/agent-runtime.md): the Actor's physical actions, the action API as tools
(its names, parameters and semantics, for the episode's embodiment profile), `plan_arms`, which arm
does which of the Memory's stages in this scene (`arms`: one assignment over the workflow's holds,
by the instruction, the holds open at once and which arm plans each), and `check_reach`, which
plans an action without executing it. What a model reads never names the simulator.

A thin wrapper over the episode's `manipulation.ActionApi` (one per episode, in the toolset's
episode state, with its one `Executor`): each action tool
builds the action with it and runs it through the env's agent plane, one action at a time under
the Hub's action lock, and returns the `ActionReceipt`, which the Hub's stage rules read (a hard
failure ends an S2 try; P3 recovers). A displacement's target comes from the measured ee
pose; the executor clamps it to the per-action limits by the distance from the drive target, the
move it will actually make (the two differ only while an arm is blocked). `together` keeps the
TCPs the executor's `min_arm_gap_m` apart along the planned paths (the executor's check). The
table the executor keeps targets above is the episode's `TABLE` fact (perception's estimate) or,
until one is set, the configured `table_top_z`, a fallback that may be a floor (a scene
variant's lowest table): never the simulator's own. After every physical action the toolset runs
its `after_action` hook with the receipt: by default the scene registry's grasp relations
(`toolsets.scene.after_action`: a settled close attaches the object the grasp aimed at, else the
nearest one, and an open or a lost hold detaches and settles it; nothing to do without registered
objects), `no_after_action` to keep the registry as it is; a failing hook is logged, and the
receipt goes on to the stage rules and the model.

Grasps, placements and hinge turns act on the episode's registered objects (the scene registry,
read once per call): `grasp_object` builds its candidates on the object (or its registered part)
as the registry has it now, `place_object` carries the object the arm holds (where its grasp holds
it) to a registered object or part found by the role the PlaceSpec names, a landmark of the
episode's table or the robot base, and a hinge turn (`together`'s `rotate_about` action; a skillset
may add tools on this toolset's handles `act`, `api` and `obstacles`) turns about a line on a
registered part's box as it was measured, since a hinge stays with the object it belongs to. Each
hands the hook the object it acts on (`targets`), so a settled close attaches that object. What a
model reads names objects by their aliases and parts by name, never the registry's ids.
"""

import logging
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Annotated, Any, Final, Protocol

import numpy as np
from numpy.typing import ArrayLike
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import ToolError
from roboscript.contracts.executor import executor_overrides, with_overrides
from roboscript.contracts.geometry import OrientedBox, Pose
from roboscript.contracts.motion import TIMEOUTS, Planner
from roboscript.contracts.receipt import HARD_FAILURES, ActionReceipt, GraspChoice, ReceiptReason
from roboscript.contracts.robot import Arm, RobotState
from roboscript.contracts.skills import BindingName
from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.manipulation import (
    GEOMETRIC_ENDS,
    OTHER_ARM,
    ActionApi,
    Built,
    Executor,
    ExecutorConfig,
    GraspCall,
    GraspSpec,
    HingeSpec,
    ObjectProxy,
    PathSpec,
    PlaceSpec,
    PlanningLog,
    Reach,
    Segment,
    SpecError,
    body_frames,
    chained,
    hinge_line,
    hold_frames,
    reach,
    robot_base,
)
from roboscript.memory.chains import Chains, Hold
from roboscript.memory.objects import directed_arm, side_rule
from roboscript.motion import MockPlanner
from roboscript.perception.config import DEFAULTS, PerceptionConfig
from roboscript.perception.registry import (
    WHOLE,
    ObjectGeometry,
    ObjectRecord,
    SceneRegistry,
    object_geometry,
    start_geometry,
)
from roboscript.perception.relations import overlap
from roboscript.perception.table import Table
from roboscript.rpc import aclose, client, read_ready
from roboscript.settings import repo_root
from roboscript.spatial.pose import pose_to_matrix
from roboscript.spatial.rotation import (
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_to_matrix,
)
from roboscript.tools import (
    ARM_PLAN,
    ARM_SIDES,
    DEMO_ARMS,
    DONE,
    IDENTIFIED,
    MEMORY_SPECS,
    REGISTRY,
    TABLE,
    Rationale,
    ToolContext,
    Toolset,
    ToolsetRuntime,
    tool,
)
from roboscript.toolsets.manipulation import misses
from roboscript.toolsets.manipulation.actions import (
    AlignAxis,
    ApproachAxis,
    ArmAction,
    ArmMove,
    ArmName,
    CloseGripper,
    Constrain,
    EePose,
    EndQuat,
    GraspApproachM,
    GraspArg,
    GraspObject,
    GraspOffsetM,
    HeldObject,
    MoveAxis,
    MoveBy,
    MoveToPose,
    ObjectAxis,
    ObjectAxisType,
    OpenGripper,
    PathArg,
    PathArms,
    PlaceApproachM,
    PlaceArg,
    PlaceObject,
    Position,
    ReturnHome,
    RotateAbout,
    ShiftXArg,
    ShiftYArg,
    ShiftZArg,
    SpecName,
    StageArm,
    StopBackM,
    TheObject,
    spec_name,
)
from roboscript.toolsets.manipulation.arms import (
    FIXED,
    ArmChoice,
    ArmPlan,
    HoldArm,
    demo_arm,
    named_arms,
    nearest_arm,
    rank,
    render_plan,
)
from roboscript.toolsets.manipulation.misses import MissConfig
from roboscript.toolsets.manipulation.refine import (
    RefineConfig,
    SceneMeasure,
    ToolMeasure,
    hinge,
    refine,
    remeasure,
)
from roboscript.toolsets.manipulation.results import (
    receipt_text,
    render_reach,
    render_receipt,
)
from roboscript.toolsets.scene import after_action as scene_after_action
from roboscript.toolsets.scene import reference_table, stored_table

__all__ = [
    "ACTION_TIMEOUT_S",
    "EXECUTOR",
    "AfterAction",
    "ArmAction",
    "ArmChoice",
    "CloseGripper",
    "GraspObject",
    "Manipulation",
    "ManipulationConfig",
    "MoveBy",
    "MoveToPose",
    "OpenGripper",
    "PlaceObject",
    "PlannerConfig",
    "Reach",
    "ReturnHome",
    "RotateAbout",
    "holding",
    "in_the_way",
    "no_after_action",
    "nudge_moves",
    "receipt_text",
    "release_on_table",
    "render_reach",
    "render_receipt",
    "resolve",
]

log = logging.getLogger(__name__)

ACTION_TIMEOUT_S: Final = 300.0
PLAN_TIMEOUT_S: Final = 900.0
"""`plan_arms` checks every hold of the workflow, each within the planning budget."""
"""An action plans (cuRobo may take its 10 s budget twice per move, and a grasp or a placement
searches batches of candidates first) and then runs for seconds."""
_OBJECT_ACTIONS: Final = (GraspObject, PlaceObject, RotateAbout)
_STOPS: Final = frozenset(
    {*HARD_FAILURES, ReceiptReason.invalid_args, ReceiptReason.rejected_repeat}
    | {ReceiptReason.budget_exhausted}
)
"""Where a coarse move ends the action: nothing fine follows, nothing is measured again."""
_SAME_FIELDS: Final = (
    "`action` names the action tool; its other fields mean what that tool's do (no rationale)."
)
_UNPLANNED: Final = frozenset({ReceiptReason.plan_failed, ReceiptReason.unreachable})
"""Where a grasp's chosen candidate did not run (nothing held yet): another may be tried."""
_KNOCKED: Final = frozenset(
    {
        ReceiptReason.deviated,
        ReceiptReason.stopped_short,
        ReceiptReason.lost_hold,
        ReceiptReason.empty_grasp,
    }
)
"""Where a grasp's fingers reached the object without holding it: it is measured again."""
_MISSED: Final = frozenset(
    {
        ReceiptReason.deviated,
        ReceiptReason.stopped_short,
        ReceiptReason.lost_hold,
        ReceiptReason.empty_grasp,
        ReceiptReason.unreachable,
        ReceiptReason.plan_failed,
    }
)
"""Receipts after which the other arm may have been in the way."""
NUDGE_BACKOFF_M: Final = 0.05
NUDGE_TURN_DEG: Final = 45.0
"""A nudge turns in moves of at most this much: the ee swings about the object's center, and the
per-action limits are 90 degrees and 0.2 m."""
EXECUTOR: Final[tuple[BindingName, ...]] = ("executor",)
"""What every tool here asks of the env: the executor's tuning (`contracts.skills.Bindings`)."""
_API: Final = "api"  # the episode's ActionApi in the toolset's episode state
_LOOKED: Final = "looked"  # the border-cut objects an arm has looked at, in the episode state
_NUDGES: Final = "nudges"  # per object: the last nudge's start yaw, turn and lost-turn run
_MIRRORED: Final = "mirrored"  # the episode's arms mirror the Memory's specs' (`_note_mirrored`)


def nudge_moves(
    ee: Sequence[float], center: Sequence[float], x: float, y: float, turn_deg: float, lift: float
) -> list[tuple[float, float, float, tuple[float, float, float, float]]]:
    """The displacements (x, y, z, end quat) that carry a held object's center by (x, y) and
    turn it by `turn_deg` about the vertical through it, from the ee pose `ee` (7: position and
    qw, qx, qy, qz) and the object's center: the ee turns about that vertical with the object,
    in moves of at most `NUDGE_TURN_DEG`, the first also lifting by `lift`."""
    n = max(1, int(np.ceil(abs(turn_deg) / NUDGE_TURN_DEG - 1e-9)))
    position = np.asarray(ee[:3], dtype=float)
    quat = np.asarray(ee[3:], dtype=float)
    pivot = np.asarray(center[:3], dtype=float)
    moves = []
    for i in range(n):
        q = quat_from_axis_angle([0.0, 0.0, 1.0], np.radians(turn_deg / n))
        shift = np.array([x, y, 0.0]) / n
        arm = position - pivot
        target = pivot + shift + quat_rotate(q, arm)
        quat = quat_normalize(quat_multiply(q, quat))
        delta = target - position
        delta[2] = lift if i == 0 else 0.0
        target[2] = position[2] + delta[2]
        moves.append(
            (float(delta[0]), float(delta[1]), float(delta[2]), tuple(float(v) for v in quat))
        )
        position, pivot = target, pivot + shift + np.array([0.0, 0.0, delta[2]])
    return moves  # type: ignore[return-value]


def nudge_made(
    before: Pose, after: Pose, center: ArrayLike, turn_deg: float, shift: Sequence[float]
) -> str:
    """What a nudge's turn moves made of the held object, from the ee before and after them
    (proprioception): the turn about the vertical and how far its center went, against what was
    asked (`center`: the object's center before)."""
    rotation = quat_to_matrix(quat_multiply(after.q, quat_conjugate(before.q)))
    turned = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    pivot = np.asarray(center, dtype=float)[:3]
    moved = rotation @ (pivot - np.asarray(before.p)) + np.asarray(after.p) - pivot
    moved = np.round(moved, 3) + 0.0  # no "-0.000"
    return (
        f"turned it {turned:+.1f} deg about the vertical (asked {turn_deg:+.1f}) and moved its "
        f"center x {moved[0]:+.3f}, y {moved[1]:+.3f} m (asked {shift[0]:+.3f}, {shift[1]:+.3f})"
    )


def nudge_repeat(
    starts: dict[str, tuple[float, float, int]],
    obj: str,
    start_deg: float,
    turn_deg: float,
    tolerance_deg: float,
) -> str:
    """From the third nudge in a row of `obj` that starts where the one before started instead
    of where it turned the object to (more than half its turn off, and more than
    `tolerance_deg`): the turns are made, but something turns it back between them (each
    regrasp's close turning the object back as far). `starts` (the episode state's) keeps, per
    object, the last nudge's start yaw, turn and how many in a row lost their turn."""
    last = starts.get(obj)
    lost = 0
    if last is not None and abs(last[1]) >= tolerance_deg:
        missed = _wrap_deg(start_deg - (last[0] + last[1]))
        if abs(missed) > max(abs(last[1]) / 2, tolerance_deg):
            lost = last[2] + 1
    starts[obj] = (start_deg, turn_deg, lost)
    if lost < 2:
        return ""
    return (
        f"the turns of the last {lost} nudges of {obj} were undone before the next one (it "
        f"starts at yaw {start_deg:.0f} deg again, not where they turned it): the turns are "
        "made, but something turns it back between nudges (the regrasp's close, if each "
        "follows a regrasp); repeating changes nothing: grasp it so the close does not turn "
        "it, or place it again with the turn instead"
    )


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class PlannerConfig(BaseModel):
    """The motion planner: the motion service at `url`, or at the URL in `ready_file` (the file
    a stack's supervisor has the service write, `<run_dir>/.roboscript/ready/<name>.json`; read
    when the toolset starts, so the service starts first; relative to the repository root), or
    `mock`: MockPlanner in-process, the planner of MockEnv's robot (tests)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str | None = None
    ready_file: Path | None = None
    mock: bool = False

    @model_validator(mode="after")
    def _one_planner(self) -> Self:
        if sum((self.url is not None, self.ready_file is not None, self.mock)) != 1:
            raise ValueError("set exactly one of planner.url, planner.ready_file, planner.mock")
        return self


class ManipulationConfig(BaseModel):
    """`configs/manipulation.example.yaml`: the planner; the table height used until the
    episode's `TABLE` fact is set (None: no table check until then), a floor where the table's
    height varies; overrides of the executor's tuning, which the env's skillset binds (among them
    how far apart `together` keeps the TCPs);
    how grasps and placements go coarse, then fine (`refine`); `clearance_m`: an arm whose move
    missed within this distance of the other arm's TCP has its receipt say so."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    planner: PlannerConfig
    table_top_z: FiniteFloat | None = None
    executor: Annotated[dict[str, Any], AfterValidator(executor_overrides)] = {}
    """Overrides of single fields of the executor's tuning, which the env's skillset binds
    (`contracts.executor.ExecutorConfig`, `with_overrides`)."""
    refine: RefineConfig = RefineConfig()
    clearance_m: PositiveFloat = 0.15
    misses: MissConfig = MissConfig()
    """From how many empty closes in a row on one grasp target, nothing measured in between,
    the receipt asks for a new measurement (`toolsets.manipulation.misses`)."""
    pin_grasp_frames: bool = True
    """A grasp pins the held object's frame it was made in where a choice left its front open
    (`SceneRegistry.pin`); false: the frame stays as perception left it."""
    square_grasped: bool = False
    """A close holding a box turns its registered pose about the TCP the way flat pads square
    it against them (`SceneRegistry.attach`'s `square_axis`); false (the default): the pose
    before the close stays."""
    table_releases: bool = True
    """A placement at a robot-frame constant that sets the object down onto the table (as the
    demonstration's table had it: `release_on_table`) keeps its height over this episode's table,
    not over the robot base, where the `DONE` fact names the demonstration's frame; false: as
    written, over the base."""
    table_perception: PerceptionConfig = DEFAULTS
    """The perception pipeline that finds the demonstration's table for `table_releases` (its
    anchor camera among the env's: the scene toolsets' `pipeline`, as the stack configures it)."""
    arm_end_margin_m: PositiveFloat = 0.05
    """`plan_arms` with a placement and a done criterion relating the object to another one:
    when both arms' whole chains plan, the arm whose chain leaves the object nearer where the
    criterion wants it (horizontally) by more than this takes it over the nearer arm (a
    robot-frame placement written for one arm lands at its mirror image for the other)."""
    arm_end_near_m: PositiveFloat = 0.1
    """... and only where that arm's chain leaves it within this of where the criterion wants
    it: the stage completes that relation (a first stack's placement, which a later stage
    builds on, leaves it far off with either arm and decides nothing)."""
    plan_budget_s: Annotated[PositiveFloat, Field(lt=ACTION_TIMEOUT_S)] = 150.0
    """How long (s) one tool call's grasp and placement searches may plan before they stop and
    refuse (plan_failed, a hard failure the Agent can act on) instead of running into the
    tool's `ACTION_TIMEOUT_S`, which ends the episode; the rest of that time is the execution's."""


class AfterAction(Protocol):
    """What runs after every physical action with its receipt: the scene registry's grasp
    relations (attach on a settled close, detach and settle on an open or a lost hold);
    `targets` names the object each arm's action aimed at, when it aimed at one, `parts` the
    part of it a grasp aimed at, `presses` the arms whose grasp is a press (it holds nothing),
    `pin` whether a grasp pins the frame it was made in (`ManipulationConfig.pin_grasp_frames`),
    `square` whether a close squares a box against the pads
    (`ManipulationConfig.square_grasped`)."""

    async def __call__(
        self,
        ctx: ToolContext,
        receipt: ActionReceipt,
        *,
        targets: Mapping[Arm, str] | None = None,
        parts: Mapping[Arm, str] | None = None,
        presses: Collection[Arm] = (),
        pin: bool = True,
        square: bool = False,
    ) -> None: ...


async def no_after_action(
    ctx: ToolContext,
    receipt: ActionReceipt,
    *,
    targets: Mapping[Arm, str] | None = None,
    parts: Mapping[Arm, str] | None = None,
    presses: Collection[Arm] = (),
    pin: bool = True,
    square: bool = False,
) -> None:
    """A hook that does nothing: the scene registry does not follow the actions."""


class Manipulation(Toolset):
    """Physical actions of the episode's robot and reach checks."""

    name = "manipulation"
    Config = ManipulationConfig
    config: ManipulationConfig

    def __init__(
        self, after_action: AfterAction = scene_after_action, measure: SceneMeasure | None = None
    ) -> None:
        self.after_action = after_action
        self.measure: SceneMeasure = measure or ToolMeasure()
        self.planner: Planner | None = None  # set by `start`
        self._profiles: dict[str, EmbodimentProfile] = {}

    async def start(self, runtime: ToolsetRuntime) -> None:
        self.planner = _planner(self.config.planner)

    @property
    def executor_config(self) -> ExecutorConfig:
        """The env's executor tuning (its skillset's binding) with this config's overrides."""
        binding = self.bindings.executor
        if binding is None:
            raise LookupError("no mounted skillset binds the executor: its tools are not mounted")
        return with_overrides(binding, self.config.executor)

    async def stop(self) -> None:
        if self.planner is not None:
            await aclose(self.planner)
            self.planner = None

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def move_to_pose(
        self, ctx: ToolContext, rationale: Rationale, arm: ArmName, pose: EePose
    ) -> ActionReceipt:
        """Move one arm's ee (the wrist frame, not the TCP) to a pose in the world frame along a
        planned path. The planner keeps the robot clear of itself and, if possible, of the
        registered objects no arm holds (the receipt notes when it did not); an empty gripper
        starting low beside one first rises straight up."""
        return await self.act(ctx, {arm: MoveToPose(action="move_to_pose", pose=pose)})

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def move_by(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        arm: ArmName,
        x: ShiftXArg = 0.0,
        y: ShiftYArg = 0.0,
        z: ShiftZArg = 0.0,
        quat: EndQuat = None,
        move_axis: MoveAxis = "world",
    ) -> ActionReceipt:
        """Move one arm's ee from where it is now along a straight line: by x, y, z meters along
        the world axes, or with move_axis arm by z meters along the gripper's pointing axis,
        positive z backing away from where it points (x and y must then be 0). One call moves at
        most the per-action limits (0.2 m and 90 degrees unless configured otherwise); a longer
        displacement is shortened to them (reason clamped). A shift may name a Memory number
        parameter ({name}); see `displacement`."""
        x, y, z = displacement(ctx, (x, y, z), move_axis, self.api(ctx).lateral)
        action = MoveBy(action="move_by", x=x, y=y, z=z, quat=quat, move_axis=move_axis)
        return await self.act(ctx, {arm: action})

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def open_gripper(
        self, ctx: ToolContext, rationale: Rationale, arm: ArmName, pos: Position = 1.0
    ) -> ActionReceipt:
        """Open one arm's gripper to pos (1 fully open)."""
        return await self.act(ctx, {arm: OpenGripper(action="open_gripper", pos=pos)})

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def close_gripper(
        self, ctx: ToolContext, rationale: Rationale, arm: ArmName, pos: Position = 0.0
    ) -> ActionReceipt:
        """Close one arm's gripper to pos (0 fully closed). Closing on nothing is no failure here
        (to push with closed fingers, say); the receipt says whether the fingers hold something."""
        return await self.act(ctx, {arm: CloseGripper(action="close_gripper", pos=pos)})

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def return_home(
        self, ctx: ToolContext, rationale: Rationale, arm: ArmName
    ) -> ActionReceipt:
        """Move one arm's ee back to its pose at the start of the episode along a planned
        path."""
        return await self.act(ctx, {arm: ReturnHome(action="return_home")})

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def grasp_object(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        arm: ArmName,
        obj: TheObject,
        grasp: GraspArg,
        approach_m: GraspApproachM = 0.1,
        grasp_offset_m: GraspOffsetM = 0.0,
        gripper_pos: Position = 0.0,
    ) -> ActionReceipt:
        """Grasp a registered object where a grasp spec says: the spec's candidate grasps on the
        object (or on its registered part) as it is now, the first one the arm reaches in the
        preferred orientations; the gripper goes to approach_m back from it, comes straight in
        to grasp_offset_m back and closes. The receipt names the candidate taken, k of n. The spec's
        purpose decides what fails: pick, an empty grasp or a lost hold; pull, an empty grasp;
        press, nothing (the fingers push)."""
        action = GraspObject(
            action="grasp_object",
            obj=obj,
            grasp=grasp,
            approach_m=approach_m,
            grasp_offset_m=grasp_offset_m,
            gripper_pos=gripper_pos,
        )
        return await self.act(ctx, {arm: action})

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def place_object(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        arm: ArmName,
        obj: HeldObject,
        place: PlaceArg,
        approach_m: PlaceApproachM = 0.1,
        stop_back_m: StopBackM = 0.02,
        is_open: Annotated[bool, Field(description="Open the gripper at the end.")] = True,
        constrain: Constrain = "auto",
        align_axis: AlignAxis = None,
        object_axis: ObjectAxis = (1.0, 0.0, 0.0),
        object_axis_type: ObjectAxisType = "object",
        approach_axis: ApproachAxis = "grasp",
    ) -> ActionReceipt:
        """Place the object the arm holds where a place spec says: its placement point at the
        spec's target (a registered object or one of its parts, found by the role the spec
        names; a landmark of the table; or the robot base), the held object being where the
        arm's grasp holds it. The gripper goes to approach_m back from the placement, moves
        straight to stop_back_m back (a negative stop_back_m: past the placement, pushing the
        object on) and opens, unless is_open is false. A region, or an object that
        lands upright and looks the same turned, is searched for the first placement the arm
        reaches; the receipt names it."""
        action = PlaceObject(
            action="place_object",
            obj=obj,
            place=place,
            approach_m=approach_m,
            stop_back_m=stop_back_m,
            is_open=is_open,
            constrain=constrain,
            align_axis=align_axis,
            object_axis=object_axis,
            object_axis_type=object_axis_type,
            approach_axis=approach_axis,
        )
        return await self.act(ctx, {arm: action})

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def follow_path(
        self, ctx: ToolContext, rationale: Rationale, arm: PathArms, path: PathArg
    ) -> ActionReceipt:
        """Carry what the arm holds (an object, or a part such as a lid, a door or a drawer) along
        a held path fitted on the demonstration, keeping the hold: what is held makes the
        demonstration's motion relative to its body (its whole object as measured now), each
        gripper from where it holds it; with both, two arms holding one object move together,
        neither retreating along its grasp. An arc about a hinge needs no hinge here. The receipt
        says how many of the path's waypoints were reached, how far each arm was from each, and
        whether, and where, a hold was lost."""
        # The path and the hold are the core's action API; what is env-specific, the hold frames, is
        # the executor binding's (`ExecutorConfig.hold_frames`).
        arms: list[Arm] = ["left", "right"] if arm == "both" else [arm]
        spec: PathSpec = resolve(ctx, path, PathSpec)
        api = self.api(ctx)
        for one in arms:
            if one not in api.profile.arms:
                raise ToolError(f"this robot has no {one} arm")
        registry = await ctx.frames.registry()
        robot = await ctx.env.robot(ctx.env_episode)
        held = {one: holding(registry, one) for one in arms}
        if empty := [one for one, record in held.items() if record is None]:
            raise ToolError(
                f"the {empty[0]} arm holds no registered object: a path carries what the arm "
                "holds; grasp it first"
            )
        records = {record.id: record for record in held.values() if record is not None}
        if len(records) > 1:
            raise ToolError("both arms follow one path only holding one object together")
        (record,) = records.values()
        body = record.model_copy(update={"part_grasps": {}, "part_poses": {}})
        proxy = ObjectProxy.from_geometry(object_geometry(body, robot))
        if api.executor.config.hold_frames:
            frames = hold_frames(proxy, spec.x_toward)
        else:
            frames = body_frames(proxy, spec.x_toward)
        await self.obstacles(ctx, api)
        articulated = any(one in record.part_grasps for one in arms)  # a registered part
        receipt = await api.follow_path(arms, frames, spec, articulated)
        config = self.config
        try:
            await self.after_action(
                ctx, receipt, pin=config.pin_grasp_frames, square=config.square_grasped
            )
        except Exception as error:  # the action happened: its receipt must reach the Hub
            log.exception("the post-action hook failed in episode %s", ctx.episode_id)
            ctx.emit("hook_error", {"error": type(error).__name__, "detail": str(error)})
        return receipt

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        terse=("moves",),
        needs=EXECUTOR,
    )
    async def together(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        moves: Annotated[
            list[ArmMove],
            Field(
                min_length=2,
                max_length=2,
                description="Each arm's action, one per arm: {arm, action}, the arm as "
                f"plan_arms gives it for the role it acts on. {_SAME_FIELDS}",
            ),
        ],
    ) -> ActionReceipt:
        """Run one action of each arm at once, in the same steps: one action and one receipt. It
        is refused, and nothing moves, if the TCPs would come closer than the configured gap (3
        cm unless configured otherwise), unless they start closer and come at most 5 mm closer
        (leaving a shared hold)."""
        if moves[0].arm == moves[1].arm:
            raise ToolError(f"both moves name the {moves[0].arm} arm: give one move per arm")
        return await self.act(ctx, {move.arm: move.action for move in moves})

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def nudge(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        arm: ArmName,
        obj: HeldObject,
        x: Annotated[
            FiniteFloat, Field(ge=-0.1, le=0.1, description="Meters along world x.")
        ] = 0.0,
        y: Annotated[
            FiniteFloat, Field(ge=-0.1, le=0.1, description="Meters along world y.")
        ] = 0.0,
        turn_deg: Annotated[
            FiniteFloat,
            Field(
                ge=-180.0,
                le=180.0,
                description="Degrees about the vertical through the object's center, "
                "counterclockwise seen from above.",
            ),
        ] = 0.0,
        lift: Annotated[
            FiniteFloat, Field(ge=0.0, le=0.1, description="Meters to lift it before moving.")
        ] = 0.01,
        release: Annotated[
            bool, Field(description="Lower it back, open the gripper and back off 5 cm up.")
        ] = True,
    ) -> ActionReceipt:
        """Correct where an object the arm holds ends, as check_done's corrections say: lift it
        by lift, move its center by x, y along the world axes while turning it by turn_deg about
        the vertical through its center, then (release) lower it by lift, open the gripper and
        back the gripper 5 cm straight up. A small correction of an object that is nearly where
        it should be, instead of a new placement; grasp it first if no arm holds it. Stops at
        the first part that fails, whose receipt it returns."""
        registry = await ctx.frames.registry()
        record = registry.objects.get(obj)
        if record is None or arm not in record.grasps:
            raise ToolError(f"the {arm} arm does not hold that object: grasp it first")
        robot = await ctx.env.robot(ctx.env_episode)
        geometry = object_geometry(record, robot)
        center, ee = geometry.center, robot.arms[arm].ee
        tolerance = self.api(ctx).executor.config.repeat_tolerance_deg
        nudges = ctx.state.setdefault(_NUDGES, {})
        repeated = nudge_repeat(nudges, obj, geometry.yaw_deg, turn_deg, tolerance)
        moves = nudge_moves(ee.as_list(), center, x, y, turn_deg, lift)
        receipt: ActionReceipt | None = None
        steps: list[ArmAction] = [
            MoveBy(action="move_by", x=dx, y=dy, z=dz, quat=q) for dx, dy, dz, q in moves
        ]
        if release:
            steps += [
                MoveBy(action="move_by", z=-lift),
                OpenGripper(action="open_gripper", pos=1.0),
                MoveBy(action="move_by", z=NUDGE_BACKOFF_M),
            ]
        made = ""
        for n, step in enumerate(steps):
            done = await self.act(ctx, {arm: step})
            receipt = done if receipt is None else chained(receipt, done)
            if done.reason in _STOPS:
                break
            if n + 1 == len(moves):
                after = (await ctx.env.robot(ctx.env_episode)).arms[arm].ee
                made = nudge_made(ee, after, center, turn_deg, (x, y))
        assert receipt is not None
        rest = [receipt.note]
        if made:
            # the last part (the back-off a release's withdrawal already made) moved nothing:
            # its "did not move" is not the nudge's outcome
            idle = f"{arm} did not move"
            rest = [part for part in receipt.note.split("; ") if part != idle]
            if release and receipt.reason not in _STOPS:
                made += ", then lowered and released it"
        note = "; ".join(filter(None, [made, *rest, repeated]))
        return receipt.model_copy(update={"note": note})

    @tool(
        kind="perception", tier="D", timeout_s=ACTION_TIMEOUT_S, render=render_reach, needs=EXECUTOR
    )
    async def check_reach(
        self,
        ctx: ToolContext,
        left: Annotated[
            ArmAction | None, Field(description="The left arm's action, or null.")
        ] = None,
        right: Annotated[
            ArmAction | None, Field(description="The right arm's action, or null.")
        ] = None,
    ) -> Reach:
        """Plan an action without executing it: one arm's, or both as in together. Nothing moves
        and no steps are used. Says whether it would run, the reason its receipt would give if
        not, each arm's TCP target and the steps it would take."""
        given: dict[Arm, ArmAction | None] = {"left": left, "right": right}
        actions = {arm: a for arm, a in given.items() if a is not None}
        if not actions:
            raise ToolError("give an action for the left arm, the right arm or both")
        api = self.api(ctx)
        built = await self._builds(ctx, api, expand_specs(ctx, actions))
        await self.obstacles(ctx, api)
        found = await api.check_reach(*built)
        _log_planning(ctx, api, "check_reach")
        return found

    @tool(
        kind="perception",
        tier="D",
        timeout_s=PLAN_TIMEOUT_S,
        render=render_plan,
        needs=EXECUTOR,
    )
    async def plan_arms(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        assign: Annotated[
            list[StageArm] | None,
            Field(
                description="Your own choice for the stages you assign otherwise than the plan "
                "(each stage's arm; the rationale says why). Null: take the plan."
            ),
        ] = None,
    ) -> ArmPlan:
        """Which arm does which stage of the Memory's workflow in this scene, planned once for
        the whole episode: each hold (a grasp of a role until its release) gets an arm; holds
        open at the same time get different arms, a hand-over's receiving hold the arm that
        does not hold the object, an arm the instruction names the hold it names. Each hold's
        arm is checked by planning its grasp, carry and placement as the calls would (nothing
        moves); on the demonstration's scene it is the demonstration's. Call it after the
        objects are identified and before the first action, and again after a hard failure
        when the scene has changed. With `assign` the plan takes your arms (never against the
        instruction's) and says what that breaks."""
        api = self.api(ctx)
        given = None if not assign else {a.stage: a.arm for a in assign}
        plan = await self._plan_arms(ctx, api, given, rationale.plan if given else None)
        ctx.frames.set_fact(ARM_PLAN, plan.fact())
        ctx.emit("arm_plan", plan.model_dump(mode="json"))
        _log_planning(ctx, api, "plan_arms")
        return plan

    async def _plan_arms(
        self,
        ctx: ToolContext,
        api: ActionApi,
        assign: Mapping[str, Arm] | None,
        reason: str | None,
    ) -> ArmPlan:
        arms: list[Arm] = [a for a in ("left", "right") if a in api.profile.arms]
        chains = Chains.model_validate((ctx.frames.fact(MEMORY_SPECS) or {}).get("chains") or {})
        roles = (ctx.frames.fact(IDENTIFIED) or {}).get("roles", {})
        objects = {
            role: found.get("object")
            for role, found in roles.items()
            if isinstance(found, Mapping) and found.get("object")
        }
        registry = await ctx.frames.registry()
        before = ctx.frames.fact(ARM_PLAN) or {}
        done_before = {
            i: h["arm"] for i, h in enumerate(before.get("holds", [])) if isinstance(h, Mapping)
        }
        if not chains.holds:
            obj = next(iter(objects.values()), None)
            if obj is None or len(arms) == 1:
                return ArmPlan(note="the workflow grasps nothing" if arms else "")
            choice = await self._choose_arm(ctx, api, obj, None, 0.1, None, None, 0.1, 0.02)
            return ArmPlan(
                note=f"the workflow grasps nothing: the {choice.arm} arm does every stage that "
                f"acts ({choice.why})"
            )
        held: list[_Planned] = []
        for i, hold in enumerate(chains.holds):
            obj = objects.get(hold.role)
            record = None if obj is None else registry.objects.get(obj)
            if len(arms) == 1:
                held.append(_Planned(hold, arms[0], "only"))
                continue
            if hold.receives_from is not None:
                held.append(_Planned(hold, arms[0], "receiver"))
                continue
            if (
                record is not None
                and record.taken
                and not record.grasps
                and hold.closes is not None
                and i in done_before
                and not any(h.role == hold.role for h in chains.holds[i + 1 :])
            ):
                held.append(_Planned(hold, done_before[i], "done"))
                continue
            if obj is None:
                held.append(_Planned(hold, arms[0], "nearer", note="the role is not identified"))
                continue
            api.deadline = time.monotonic() + self.config.plan_budget_s  # per hold
            try:
                choice = await self._choose_arm(
                    ctx, api, obj, hold.grasp, 0.1, hold.path, hold.place, 0.1, 0.02
                )
            except ToolError as error:
                held.append(_Planned(hold, arms[0], "nearer", note=str(error)))
                continue
            checked = {a: ("ok" if ok else "does not plan") for a, ok in choice.reachable.items()}
            held.append(_Planned(hold, choice.arm, choice.why, checked, choice.side, choice.note))
        if len(arms) > 1:
            await self._apart(ctx, api, chains, held, objects)
        override = None
        if assign:
            override = _assigned(chains, held, assign, reason)
        plan_holds = [
            HoldArm(
                role=p.hold.role,
                arm=p.arm,
                why=p.why,
                stages=list(p.hold.stages),
                checked=p.checked,
                side=p.side,
                note=p.note,
            )
            for p in held
        ]
        stages = {
            stage: list(dict.fromkeys(held[i].arm for i in indices))
            for stage, indices in chains.stages.items()
        }
        return ArmPlan(stages=stages, holds=plan_holds, override=override)

    async def _apart(
        self,
        ctx: ToolContext,
        api: ActionApi,
        chains: Chains,
        held: list["_Planned"],
        objects: Mapping[str, str],
    ) -> None:
        """Receiving holds take the arm that does not give; holds open at the same time on one
        arm: the weaker choice (`arms.rank`) moves to the other arm, where that arm plans it (else
        the stronger one moves, else both stay and the note says so)."""

        def receivers() -> None:
            for p in held:
                if p.hold.receives_from is not None:
                    p.arm = _other(held[p.hold.receives_from].arm)

        receivers()
        for a, b in chains.together:
            if held[a].arm != held[b].arm:
                continue
            order = sorted((a, b), key=lambda i: rank(held[i].why))
            moved = False
            for i in order:
                p = held[i]
                if p.why in FIXED or p.hold.receives_from is not None:
                    continue
                other = _other(p.arm)
                if await self._plans(ctx, api, p, other, objects):
                    p.arm, p.why = other, "apart"
                    moved = True
                    break
            if not moved:
                for i in order:
                    if held[i].why not in FIXED and held[i].hold.receives_from is None:
                        held[i].arm, held[i].why = _other(held[i].arm), "apart"
                        held[i].note = "; ".join(
                            filter(None, [held[i].note, "the other arm did not plan it either"])
                        )
                        moved = True
                        break
            if not moved:
                held[b].note = "; ".join(
                    filter(None, [held[b].note, f"open with hold {a + 1} on the same arm"])
                )
            receivers()

    async def _plans(
        self, ctx: ToolContext, api: ActionApi, p: "_Planned", arm: Arm, objects: Mapping[str, str]
    ) -> bool:
        """Whether `arm` plans the hold's grasp (and carry and placement), as `choose_arm`'s
        rehearsal does; the check is recorded."""
        if arm in p.checked:
            return p.checked[arm] == "ok"
        api.deadline = time.monotonic() + self.config.plan_budget_s
        obj = objects.get(p.hold.role)
        if obj is None or p.hold.grasp is None:
            p.checked[arm] = "unchecked"
            return True
        grasp = GraspObject(
            action="grasp_object", obj=obj, grasp=resolve(ctx, p.hold.grasp, GraspSpec)
        )
        if p.hold.place is not None:
            placing = PlaceObject(
                action="place_object", obj=obj, place=resolve(ctx, p.hold.place, PlaceSpec)
            )
            path = None if p.hold.path is None else resolve(ctx, p.hold.path, PathSpec)
            failed = (await self._rehearse(ctx, api, arm, grasp, placing, path)).failed
        else:
            built = await self._builds(ctx, api, {arm: grasp})
            await self.obstacles(ctx, api)
            failed = None if (await api.check_reach(*built)).reachable else "its grasp"
        p.checked[arm] = "ok" if failed is None else failed
        return failed is None

    async def choose_arm(
        self,
        ctx: ToolContext,
        obj: str,
        grasp: GraspSpec | SpecName | None = None,
        path: PathSpec | SpecName | None = None,
        place: PlaceSpec | SpecName | None = None,
    ) -> ArmChoice:
        """The arm for one object alone (`plan_arms`' single-hold check, not a tool): the
        demonstration's on its scene, else the instruction's, the holding arm, the role's `arm
        side:`, else the nearer arm unless only the other plans the grasp (with a placement:
        the grasp, carry and placement)."""
        api = self.api(ctx)
        return await self._choose_arm(ctx, api, obj, grasp, 0.1, path, place, 0.1, 0.02)

    async def _choose_arm(
        self,
        ctx: ToolContext,
        api: ActionApi,
        obj: TheObject,
        grasp: GraspSpec | SpecName | None,
        grasp_approach_m: float,
        path: PathSpec | SpecName | None,
        place: PlaceSpec | SpecName | None,
        place_approach_m: float,
        stop_back_m: float,
    ) -> ArmChoice:
        grasp = None if grasp is None else resolve(ctx, grasp, GraspSpec)
        arms: list[Arm] = [a for a in ("left", "right") if a in api.profile.arms]
        if len(arms) == 1:
            return ArmChoice(arm=arms[0], why="only")
        live = ctx.frames.live()
        instruction = None if live is None else (await ctx.frames.get(live.alias)).instruction
        named = named_arms(instruction)
        registry = await ctx.frames.registry()
        record = registry.objects.get(obj)
        shown = ctx.frames.fact(DEMO_ARMS)
        lateral = api.lateral
        x_axis = None
        if record is not None and record.x_known:
            x_axis = tuple(pose_to_matrix(record.pose.as_list())[:3, 0])
        staged = _stage_arms(ctx, obj)
        if shown is not None and record is not None and not record.grasps:
            # the demonstration's own scene: its evidence first
            starts = {o.id: start_geometry(o).center for o in registry.objects.values()}
            found = demo_arm(shown.get("grasps", []), starts, obj)
            if found is not None and len(staged) > 1:
                # grasped by both arms in turn: the demonstration answers per stage, as the
                # stage rule does (a regrasp takes its own spec's arm), not by the object
                found = demo_stage_arm(shown.get("grasps", []), found, staged, record, grasp)
            elif found is not None and record.taken:
                # taken and set down since: a regrasp's spec arm, at the demonstration's later
                # grasp by that arm (the workflow's stage arms may be unknown)
                found = demo_spec_arm(shown.get("grasps", []), found, grasp)
            if found is not None:
                side = None if x_axis is None else side_rule(found[0], x_axis, lateral)
                return ArmChoice(
                    arm=found[0], why="demonstration", side=side, note=f"at {found[1]}"
                )
        if staged and record is not None and grasp is not None and len(record.grasps) == 1:
            return _receiving_choice(staged, record, grasp)
        instructed = record.arm if record is not None else None
        if instructed is None and named:
            instructed = named[0]
        if staged and record is not None and instructed is not None:
            return _stage_choice(staged, record, grasp, instructed, named)
        if record is not None and record.arm is not None:
            note = "the identification read the instruction to have this arm act on it"
            return ArmChoice(arm=record.arm, why="instruction", note=note)
        if len(named) == 1:
            return ArmChoice(arm=named[0], why="instruction")
        if record is None:
            raise ToolError("that object is not registered in this episode")
        note = ""
        if named and not record.grasps:  # both arms named and identify read none for it
            return ArmChoice(
                arm=named[0],
                why="both_named",
                candidates=named,
                note="the instruction names both arms and its reading gave none for this "
                "object: decide from the instruction's wording",
            )
        if record.grasps:  # the arm that grasped last (`grasps` is in the order they took it)
            return ArmChoice(arm=list(record.grasps)[-1], why="holding", note=note)
        rule = (ctx.frames.fact(ARM_SIDES) or {}).get("roles", {}).get(record.name)
        if rule is not None and rule != "either":
            directed = None if x_axis is None else directed_arm(rule, x_axis, lateral)
            if directed is not None:
                return ArmChoice(arm=directed, why="direction", side=rule)
            note = (
                f"its arm side is {rule!r}, but its x "
                + ("is not decided" if x_axis is None else "points to neither side")
                + ": the nearer arm, which may carry it the wrong way round"
            )
        robot = await ctx.env.robot(ctx.env_episode)
        center = object_geometry(record, robot).center
        starts = {
            arm: api.profile.tcp_pose(arm, await api.executor.home_pose(arm)).p for arm in arms
        }
        order = nearest_arm(center, starts)
        distance = {
            arm: round(float(np.linalg.norm(np.subtract(starts[arm][:2], center[:2]))), 3)
            for arm in arms
        }
        if grasp is None:
            return ArmChoice(arm=order[0], why="nearer", distance_m=distance, note=note)
        reachable: dict[Arm, bool] = {}
        action = GraspObject(
            action="grasp_object", obj=obj, grasp=grasp, approach_m=grasp_approach_m
        )
        if place is not None:
            held = None if path is None else resolve(ctx, path, PathSpec)
            placing = PlaceObject(
                action="place_object", obj=obj, place=resolve(ctx, place, PlaceSpec),
                approach_m=place_approach_m, stop_back_m=stop_back_m,
            )  # fmt: skip
            ending = _done_partner(ctx, registry, record, robot)
            rehearsed: dict[Arm, _Rehearsal] = {}
            for arm in order:
                rehearsed[arm] = await self._rehearse(ctx, api, arm, action, placing, held)
                failed = rehearsed[arm].failed
                reachable[arm] = failed is None or not failed.startswith("its grasp")
                if failed is None and (
                    ending is None
                    or abs(_xy_m(rehearsed[arm].end, ending[1]) - ending[2])
                    <= self.config.arm_end_margin_m
                ):
                    break  # no done criterion to tell whole chains apart, or this one meets it
            whole = [arm for arm, r in rehearsed.items() if r.failed is None]
            failed_notes = [f"{a}: {r.failed}" for a, r in rehearsed.items() if r.failed]
            if whole:
                arm, why = whole[0], "nearer" if whole[0] == order[0] else "place"
                ends = ""
                if ending is not None and len(whole) > 1:
                    partner, at, wanted = ending
                    away = {a: _xy_m(rehearsed[a].end, at) for a in whole}
                    off = {a: abs(away[a] - wanted) for a in whole}  # from where done wants it
                    each = ", ".join(f"{a} {away[a]:.2f} m from {partner}" for a in whole)
                    ends = (
                        f"where its grasp, carry and placement leave {record.name}: {each} "
                        f"(the done criteria want {wanted:.2f} m)"
                    )
                    best = min(whole, key=lambda a: off[a])
                    completes = off[best] <= self.config.arm_end_near_m
                    if completes and off[arm] - off[best] > self.config.arm_end_margin_m:
                        arm, why = best, "end"
                note = "; ".join(filter(None, [note, ends, *failed_notes]))
                return ArmChoice(
                    arm=arm, why=why, distance_m=distance, reachable=reachable, note=note
                )
            why_not = "; ".join(f"{arm}: {rehearsed[arm].failed}" for arm in order)
            note = "; ".join(
                filter(
                    None,
                    [
                        note,
                        "neither arm grasps, carries and places it alone "
                        f"({why_not}): hand it over (one arm grasps, the other places)",
                    ],
                )
            )
            grasping = [arm for arm in order if reachable[arm]]
            first = grasping[0] if grasping else order[0]
            return ArmChoice(
                arm=first, why="neither", distance_m=distance, reachable=reachable, note=note
            )
        for arm in order:
            built = await self._builds(ctx, api, {arm: action})
            await self.obstacles(ctx, api)
            reachable[arm] = (await api.check_reach(*built)).reachable
            if reachable[arm]:
                why = "nearer" if arm == order[0] else "reach"
                return ArmChoice(
                    arm=arm, why=why, distance_m=distance, reachable=reachable, note=note
                )
        note = "; ".join(filter(None, [note, "neither arm plans this grasp now"]))
        return ArmChoice(
            arm=order[0], why="nearer", distance_m=distance, reachable=reachable, note=note
        )

    async def act(self, ctx: ToolContext, actions: Mapping[Arm, ArmAction]) -> ActionReceipt:
        actions = expand_specs(ctx, actions)
        _note_mirrored(ctx, actions)
        api = self.api(ctx)
        refining = self.config.refine.enabled
        notes: list[str] = []
        if refining:
            actions, notes = await self._hinges(ctx, actions)
        built = await self._builds(ctx, api, actions)
        staged = {arm: a for arm, a in actions.items() if isinstance(a, GraspObject | PlaceObject)}
        looked = None
        refused = any(b.refused for b in built)
        if refining and staged and (not refused or self.config.refine.look_refused):
            looked = await self._look(ctx, api, staged)
            if looked is not None and looked.reason not in _STOPS:
                built = await self._builds(ctx, api, actions)  # from where they were seen whole
                refused = any(b.refused for b in built)
        if looked is not None and looked.reason in _STOPS:
            receipt = looked
        elif refining and staged and not refused:
            receipt = await self._coarse_fine(ctx, api, staged, built)
            receipt = receipt if looked is None else chained(looked, receipt, [looked.note])
        elif looked is not None:  # refused again on what the look saw
            receipt = chained(looked, await self._run(ctx, api, built), [looked.note])
        else:
            receipt = await self._run(ctx, api, built)
            if len(built) == 1 and built[0].grasp is not None:
                receipt = await self._fall_through(ctx, api, actions, built[0].grasp, receipt)
        _log_planning(ctx, api, ", ".join(f"{arm} {a.action}" for arm, a in actions.items()))
        if len(actions) == 1 and receipt.reason in _MISSED:
            notes += await self._in_the_way(ctx, receipt)
        if notes:
            receipt = receipt.model_copy(
                update={"note": "; ".join(filter(None, [receipt.note, *notes]))}
            )
        targets = {arm: a.obj for arm, a in actions.items() if isinstance(a, _OBJECT_ACTIONS)}
        if targets and (acting := _acting(ctx, targets)):
            receipt = receipt.model_copy(
                update={"note": "; ".join(filter(None, [acting, receipt.note]))}
            )
        parts = {
            arm: a.grasp.part
            for arm, a in actions.items()
            if isinstance(a, GraspObject) and a.grasp.part != WHOLE
        }
        presses = {
            arm
            for arm, a in actions.items()
            if isinstance(a, GraspObject) and a.grasp.purpose == "press"
        }
        try:
            await self.after_action(
                ctx,
                receipt,
                targets=targets or None,
                parts=parts or None,
                presses=presses,
                pin=self.config.pin_grasp_frames,
                square=self.config.square_grasped,
            )
        except Exception as error:  # the action happened: its receipt must reach the Hub
            log.exception("the post-action hook failed in episode %s", ctx.episode_id)
            ctx.emit("hook_error", {"error": type(error).__name__, "detail": str(error)})
        if refining and receipt.reason not in _STOPS:
            for a in staged.values():
                if isinstance(a, PlaceObject) and a.is_open:
                    await remeasure(ctx, self.measure, a.obj)
        if refining and receipt.reason in _KNOCKED:  # the fingers may have pushed it away
            found = [
                await remeasure(ctx, self.measure, a.obj)
                for a in staged.values()
                if isinstance(a, GraspObject)
            ]
            if any(found):
                note = "; ".join(filter(None, [receipt.note, *found]))
                receipt = receipt.model_copy(update={"note": note})
        return await self._misses(ctx, actions, receipt)

    async def _misses(
        self, ctx: ToolContext, actions: Mapping[Arm, ArmAction], receipt: ActionReceipt
    ) -> ActionReceipt:
        """Count an arm's empty closes in a row on its grasp target (`misses`); from
        `MissConfig.after` on, the receipt asks for a new measurement."""
        after = self.config.misses.after
        if after is None or len(actions) != 1:
            return receipt
        ((arm, action),) = actions.items()
        version = int((ctx.frames.fact(REGISTRY) or {}).get("version", 0))
        if isinstance(action, GraspObject) and isinstance(action.grasp, GraspSpec):
            part = None if action.grasp.part == WHOLE else action.grasp.part
            point = None if receipt.commanded is None else receipt.commanded.tcp.p
            misses.aim(ctx.state, arm, action.obj, part, point, version)
        gripper = receipt.gripper
        if receipt.arm != arm or gripper is None or gripper.commanded > 0.5:
            return receipt  # no close (an open, or a move with the gripper open)
        target = misses.closed(ctx.state, arm, gripper.holding, version)
        if target is None or target.count < after:
            return receipt
        registry = await ctx.frames.registry()
        if target.obj not in registry.objects:
            return receipt
        robot = await ctx.env.robot(ctx.env_episode)
        aimed = target.point is not None and target.point_version == version
        if aimed:
            located: Any = target.point
        else:
            geometry = object_geometry(registry.objects[target.obj], robot)
            part = target.part
            found = part is not None and part in geometry.parts
            located = geometry.parts[part].pose if found else geometry.center  # type: ignore[index]
        name = registry.objects[target.obj].name
        tcp = robot.arms[arm].tcp.p
        text = misses.note(arm, name, target.part, target.count, located, tcp, aimed)
        event = {"arm": arm, "object": target.obj, "part": target.part, "count": target.count}
        ctx.emit("misses", {**event, "aimed": aimed})
        return receipt.model_copy(update={"note": "; ".join(filter(None, [receipt.note, text]))})

    async def _in_the_way(self, ctx: ToolContext, receipt: ActionReceipt) -> list[str]:
        if receipt.arm is None or receipt.commanded is None:
            return []
        robot = await ctx.env.robot(ctx.env_episode)
        note = in_the_way(receipt, robot, self.config.clearance_m)
        if note is not None:
            ctx.emit("in_the_way", {"arm": receipt.arm, "note": note})
        return [] if note is None else [note]

    async def _hinges(
        self, ctx: ToolContext, actions: Mapping[Arm, ArmAction]
    ) -> tuple[dict[Arm, ArmAction], list[str]]:
        """Each hinge turn's part measured again from where the gripper is (the turn starts
        there), and its hinge estimated on that geometry (`refine.hinge`); the receipt's notes."""
        staged: dict[Arm, ArmAction] = dict(actions)
        notes: list[str] = []
        for arm, a in actions.items():
            if not isinstance(a, RotateAbout) or a.hinge.part == WHOLE:
                continue
            config = self.config.refine
            notes += await refine(ctx, self.measure, config, {arm: [a.obj]}, a.hinge.part)
            spec, note = await hinge(ctx, self.measure, a.obj, a.hinge)
            staged[arm] = a.model_copy(update={"hinge": spec})
            notes.append(note)
        return staged, notes

    async def _look(
        self, ctx: ToolContext, api: ActionApi, staged: Mapping[Arm, GraspObject | PlaceObject]
    ) -> ActionReceipt | None:
        """Before an episode's first grasp of, or placement on, an object whose registration an
        image border cut: the arm to where its wrist camera sees it whole (the `look_pose`
        tool), and the object measured again there (stored when sure). None: none to look at,
        or the look does not plan (it would pass through what the other arm holds, say): the
        action goes on unlooked rather than fail on a nicety (`manipulation.look_skipped`)."""
        looked: set[str] = ctx.state.setdefault(_LOOKED, set())
        registry = await ctx.frames.registry()
        poses: dict[Arm, tuple[str, Pose]] = {}
        for arm, a in staged.items():
            obj = a.obj if isinstance(a, GraspObject) else _target(registry, a)
            record = None if obj is None else registry.objects.get(obj)
            if record is None or not record.partially_visible or record.id in looked:
                continue
            looked.add(record.id)
            pose = await self.measure.look_pose(ctx, record.id, arm)
            ctx.emit("look", {"arm": arm, "object": record.id, "found": pose is not None})
            if pose is not None:
                poses[arm] = (record.id, pose)
        if not poses:
            return None
        moves = [await api.move_to_pose(arm, p) for arm, (_, p) in poses.items()]
        receipt = await self._run(ctx, api, moves)
        if receipt.unmoved and receipt.steps_used == 0:  # a look is a nicety: act unlooked
            objects = [obj for obj, _ in poses.values()]
            ctx.emit("look_skipped", {"objects": objects, "note": receipt.note})
            return None
        if receipt.reason in _STOPS:
            return receipt
        for obj, _ in poses.values():
            await remeasure(ctx, self.measure, obj)
        names = ", ".join(registry.objects[obj].name for obj, _ in poses.values())
        note = f"looked at {names} whole first"
        return receipt.model_copy(update={"note": note, "look_steps": receipt.steps_used})

    async def _coarse_fine(
        self,
        ctx: ToolContext,
        api: ActionApi,
        staged: Mapping[Arm, GraspObject | PlaceObject],
        built: list[Built],
    ) -> ActionReceipt:
        """The staged arms to their pre-poses (the other arm's action runs whole meanwhile), the
        objects measured again there, then the rest built again on that geometry, among the
        grasp candidate or placement option the coarse build chose."""
        coarse = [
            Built(b.arm, b.segments[:1], call=b.call) if b.arm in staged else b for b in built
        ]
        first = await self._run(ctx, api, coarse)
        lone = built[0].grasp if len(built) == 1 else None
        if first.reason in _STOPS:
            if lone is None:
                return first
            return await self._fall_through(ctx, api, staged, lone, first)
        registry = await ctx.frames.registry()
        targets = {}
        for arm, a in staged.items():  # a held object follows its grasp: not measured again
            target = None if isinstance(a, GraspObject) else _target(registry, a)
            found = [registry.objects.get(obj) for obj in (a.obj, target) if obj is not None]
            targets[arm] = [
                record.id for record in found if record is not None and not record.grasps
            ]
        doubts: dict[str, str] = {}
        notes = await refine(ctx, self.measure, self.config.refine, targets, doubts=doubts)
        grasped = {a.obj for a in staged.values() if isinstance(a, GraspObject)}
        if doubted := [doubts[obj] for obj in grasped if obj in doubts]:
            ctx.emit("grasp_doubted", {"objects": sorted(grasped & set(doubts))})
            note = "; ".join(filter(None, [first.note, *notes, *doubted]))
            return first.model_copy(update={"reason": ReceiptReason.stopped_short, "note": note})
        near = {b.arm: b for b in built}
        fine = await self._builds(ctx, api, staged, near)
        done = await self._run(ctx, api, fine)
        if lone is not None:
            done = await self._fall_through(ctx, api, staged, lone, done)
        return chained(first, done, notes, moved_last=api.executor.config.coarse_stops)

    async def _fall_through(
        self,
        ctx: ToolContext,
        api: ActionApi,
        actions: Mapping[Arm, ArmAction],
        chosen: GraspChoice,
        receipt: ActionReceipt,
    ) -> ActionReceipt:
        """A lone grasp whose chosen candidate could not be planned past its pre-grasp (its
        straight approach turns the wrist over, a locked plan strays, it no longer reaches on
        refined geometry) built again among the candidates not tried yet, from where the arm
        is, until one runs or none is left; the receipts chained. The chooser checks only a
        pre-grasp's plan, so an approach may fail where another candidate's would not."""
        tried = {chosen.candidate}
        while receipt.reason in _UNPLANNED and len(tried) < chosen.n_candidates:
            untried = [i for i in range(chosen.n_candidates) if i not in tried]
            rest = {arm: untried for arm in actions}
            (again,) = await self._builds(ctx, api, actions, among=rest)
            if again.grasp is None:  # no other candidate's pre-grasp is plannable
                break
            tried.add(again.grasp.candidate)
            note = (
                f"grasp candidate {chosen.candidate + 1} could not be planned "
                f"({receipt.reason.value}): tried candidate {again.grasp.candidate + 1}"
            )
            moved = {"from": chosen.candidate, "to": again.grasp.candidate}
            ctx.emit("fall_through", {"arm": again.arm, **moved, "reason": receipt.reason.value})
            api.forget()  # another candidate, not a repeat of the failed call
            receipt = chained(receipt, await self._run(ctx, api, [again]), [note])
            chosen = again.grasp
        return receipt

    async def _builds(
        self,
        ctx: ToolContext,
        api: ActionApi,
        actions: Mapping[Arm, ArmAction],
        near: Mapping[Arm, Built] | None = None,
        among: Mapping[Arm, Sequence[int]] | None = None,
    ) -> list[Built]:
        """Each arm's action built by the API, on the scene as the registry has it now when an
        action acts on an object (`near`: among what an earlier build of it chose; `among`: a
        grasp among these candidates)."""
        for arm in actions:
            if arm not in api.profile.arms:
                raise ToolError(f"this robot has no {arm} arm")
        scene = None
        if any(isinstance(a, _OBJECT_ACTIONS) for a in actions.values()):
            robot = await ctx.env.robot(ctx.env_episode)
            scene = _Scene(await ctx.frames.registry(), robot, stored_table(ctx))
            actions = await self._on_table(ctx, api, actions, scene)
        grasps = [a for a in actions.values() if isinstance(a, GraspObject)]
        try:
            if scene is not None and len(grasps) == 2 and grasps[0].obj == grasps[1].obj:
                return await _co_grasp(api, actions, scene, near, among)
            return [
                await _built(api, arm, action, scene, (near or {}).get(arm), (among or {}).get(arm))
                for arm, action in actions.items()
            ]
        except SpecError as error:  # the Agent's spec, not a fault: a refusal it can act on
            raise ToolError(f"the spec cannot be made here: {error}") from error

    async def _on_table(
        self, ctx: ToolContext, api: ActionApi, actions: Mapping[Arm, ArmAction], scene: "_Scene"
    ) -> Mapping[Arm, ArmAction]:
        """`actions` with each robot-frame placement that sets its object down onto the table
        moved to the same height over this episode's table as over the demonstration's
        (`release_on_table`); as given without both tables or with `table_releases` off."""
        robot_frame = {
            arm: a
            for arm, a in actions.items()
            if isinstance(a, PlaceObject)
            and isinstance(a.place, PlaceSpec)
            and a.place.target.kind == "robot_frame"
        }
        done = ctx.frames.fact(DONE)
        observation = None if done is None else done.get("reference")
        if not robot_frame or not self.config.table_releases or observation is None:
            return actions
        if scene.table is None:
            return actions
        found = BlobRef.model_validate(observation)
        demo = await reference_table(ctx, found, self.config.table_perception)
        base = robot_base(api.profile).matrix()
        moved = dict(actions)
        for arm, action in robot_frame.items():
            assert isinstance(action.place, PlaceSpec)  # robot_frame holds only specs
            held = scene.held(arm, action.obj, action.place)
            standing = [
                object_geometry(record, scene.robot)
                for record in scene.registry.objects.values()
                if record.id != action.obj and not record.grasps and not record.part_grasps
            ]
            place = release_on_table(
                action.place, held.whole.size[2], base, demo, scene.table, standing
            )
            if place is not action.place:
                ctx.emit(
                    "release_on_table",
                    {"arm": arm, "from": list(action.place.offset_m), "to": list(place.offset_m)},
                )
                moved[arm] = action.model_copy(update={"place": place})
        return moved

    async def _run(self, ctx: ToolContext, api: ActionApi, built: Sequence[Built]) -> ActionReceipt:
        await self.obstacles(ctx, api)
        return await api.run(*built)

    async def obstacles(self, ctx: ToolContext, api: ActionApi) -> None:
        """The executor's obstacles: the registered objects as they are now, but those an arm
        holds (they move with it: the other arm's moves keep clear of them, `Executor.held`)."""
        registry = await ctx.frames.registry()
        robot = await ctx.env.robot(ctx.env_episode)
        scene = _Scene(registry, robot, None)
        api.executor.obstacles, api.executor.held = scene.boxes(), scene.held_boxes()

    async def _rehearse(
        self,
        ctx: ToolContext,
        api: ActionApi,
        arm: Arm,
        grasp: GraspObject,
        place: PlaceObject,
        path: PathSpec | None,
    ) -> "_Rehearsal":
        """Plan `arm`'s grasp, the carry along `path` and the placement as the calls would make
        them, each from where the one before leaves the arm and the object held as the grasp
        would hold it (`ActionApi.rehearsal`, `Executor.after`), nothing executed: where the
        object's center ends when all three plan, else what does not (a refusal's note). The
        same builders and planner checks as the calls; what only the run decides (refined
        geometry, the held object's slip) can still differ."""
        robot = await ctx.env.robot(ctx.env_episode)
        scene = _Scene(await ctx.frames.registry(), robot, stored_table(ctx))
        api.executor.obstacles, api.executor.held = scene.boxes(), scene.held_boxes()
        rehearsed = api.rehearsal(scene.robot)
        built = await _built(rehearsed, arm, grasp, scene)
        planned = await rehearsed.executor.plan(*_by_arm(built))
        if (refused := built.refused or planned.refused) is not None:
            return _Rehearsal(f"its grasp: {refused.note}")
        gripper = api.profile.arms[arm].gripper
        robot = await rehearsed.executor.after(planned, {arm: (gripper.holding_threshold + 1) / 2})
        registry = scene.registry.clone()
        chosen = built.grasp
        registry.on_close(  # as `after_action` attaches it once the close holds
            arm,
            robot,
            threshold=gripper.holding_threshold,
            target=grasp.obj,
            part=None if grasp.grasp.part == WHOLE else grasp.grasp.part,
            turn_deg=0.0 if chosen is None else chosen.frame_turn_deg or 0.0,
            pin=self.config.pin_grasp_frames,
            hold_frame=chosen is not None and chosen.hold_frame,
            square_axis=api.profile.poses.tcp_closing_axis if self.config.square_grasped else None,
            closing_axis=api.profile.poses.tcp_closing_axis,
            span_m=gripper.span_m,
        )
        scene = _Scene(registry, robot, scene.table)
        if path is not None:
            body = registry.objects[grasp.obj].model_copy(update={"part_grasps": {}})
            proxy = ObjectProxy.from_geometry(object_geometry(body, robot))
            frames = (hold_frames if api.executor.config.hold_frames else body_frames)(
                proxy, path.x_toward
            )
            carried = api.rehearsal(robot)
            moves = await carried.path_moves(arm, frames, path)
            planned = await carried.executor.plan(*_by_arm(Built(arm, moves)))
            if planned.refused is not None:
                return _Rehearsal(f"its carry: {planned.refused.note}")
            robot = await carried.executor.after(planned)
            scene = _Scene(registry, robot, scene.table)
        placing = api.rehearsal(robot)
        placing.executor.obstacles, placing.executor.held = scene.boxes(), scene.held_boxes()
        actions = await self._on_table(ctx, placing, {arm: place}, scene)
        built = await _built(placing, arm, actions[arm], scene)
        if built.refused is not None:
            return _Rehearsal(f"its placement: {built.refused.note}")
        planned = await placing.executor.plan(*_by_arm(built))
        reached = reach(planned)
        if not reached.reachable:
            return _Rehearsal(f"its placement: {reached.note}")
        robot = await placing.executor.after(planned)
        return _Rehearsal(end=object_geometry(registry.objects[grasp.obj], robot).center)

    def api(self, ctx: ToolContext) -> ActionApi:
        """The episode's action API, made on its first action; the table it keeps targets above
        is set again for every action (perception may set the TABLE fact at any time)."""
        assert self.planner is not None  # the Hub starts a toolset before its first call
        api: ActionApi | None = ctx.state.get(_API)
        if api is None:
            executor = Executor(
                ctx.env, ctx.env_episode, self.planner, self._profile(ctx), self.executor_config
            )
            api = ctx.state[_API] = ActionApi(executor)
        # one planning budget and log per tool call (`api` is each tool's first step)
        api.deadline = time.monotonic() + self.config.plan_budget_s
        api.planning = PlanningLog()
        table = ctx.frames.fact(TABLE)
        api.executor.table_top_z = self.config.table_top_z if table is None else float(table["z"])
        return api

    def _profile(self, ctx: ToolContext) -> EmbodimentProfile:
        name = ctx.context.embodiment
        if name is None:
            raise LookupError(f"episode {ctx.episode_id} has no OpenEpisode.embodiment")
        if name not in self._profiles:
            self._profiles[name] = load_embodiment(name)
        return self._profiles[name]


@dataclass(frozen=True)
class _Rehearsal:
    """`Manipulation._rehearse`'s result: what does not plan (None: all of it does) and, when
    all does, where the held object's center ends."""

    failed: str | None = None
    end: np.ndarray | None = None


def _done_partner(
    ctx: ToolContext, registry: SceneRegistry, record: ObjectRecord, robot: RobotState
) -> tuple[str, np.ndarray, float] | None:
    """The object the Memory's done criteria want `record` relative to, as (role, its center
    now, the horizontal distance wanted between the two centers: a `distance_xy` criterion's
    value, else 0 for one over or in it): the first criterion naming `record`'s role with another
    registered object; None without one."""
    done = ctx.frames.fact(DONE)
    criteria = [] if done is None else done.get("criteria", [])
    found: str | None = None
    wanted: dict[str, float] = {}
    for c in criteria:
        roles = {c.get("a"), c.get("b")}
        if record.name not in roles or c.get("metric") not in _PLACED_METRICS:
            continue
        other = next((r for r in roles if r not in (record.name, None, "table")), None)
        if other is None:
            continue
        found = found or other
        if c["metric"] == "distance_xy":
            wanted.setdefault(other, float(c["value"]))
    if found is None:
        return None
    partner = next(
        (o for o in registry.objects.values() if o.name == found and o.id != record.id), None
    )
    if partner is None:
        return None
    center = object_geometry(partner, robot).center
    return found, np.asarray(center, dtype=float), wanted.get(found, 0.0)


_PLACED_METRICS: Final = frozenset({"distance_xy", "offset_x", "offset_y", "overlap", "inside"})
"""The done metrics that say where an object ends relative to another one."""


def _xy_m(a: ArrayLike | None, b: ArrayLike) -> float:
    assert a is not None  # a whole chain's end
    return float(np.linalg.norm(np.subtract(a, b)[:2]))


def _log_planning(ctx: ToolContext, api: ActionApi, call: str) -> None:
    """The planner work of this tool call as a `manipulation.planning` event (goals planned and
    solved, cache hits, seconds waiting on the planner), when there was any."""
    work = api.planning
    if work.plans or work.iks:
        left = None if api.deadline is None else round(api.deadline - time.monotonic(), 1)
        ctx.emit(
            "planning",
            {
                "call": call,
                "plans": work.plans,
                "iks": work.iks,
                "cached": work.cached,
                "seconds": round(work.seconds, 2),
                "budget_left_s": left,
            },
        )


RESTING_M: Final = 0.03
"""A release whose object's bottom is at most this far over the demonstration's table sets it
down onto the table (as wicl's `perception.endstate.LIFTED_M`)."""
TABLE_NOISE_M: Final = 0.002
"""Two measurements of one table's plane agree within this: a smaller difference is no lower
table, so on the demonstration's table a release stays exactly as written."""


def release_on_table(
    spec: PlaceSpec,
    height: float,
    base: np.ndarray,
    demo: Table,
    now: Table,
    standing: Sequence[ObjectGeometry] = (),
) -> PlaceSpec:
    """A robot-frame placement (`spec`, its placement point at the base origin plus its offset)
    that sets an object `height` m tall down onto the table, or onto what stands on it (one of
    `standing`, the objects no arm holds, under the point: a block on the first of a stack), as
    the demonstration's table `demo` had it (its bottom at most `RESTING_M` over the table or
    that object's top), with its offset moved along the vertical by how much lower or higher this
    episode's table `now` stands under it, so that it is released as high over the table as
    written; `spec` itself for one released in the air (a hand-over), a placement point on a
    part (its height over the object's bottom is unknown here), or a table within
    `TABLE_NOISE_M`."""
    below = {"bottom": 0.0, "center": height / 2, "top": height}.get(spec.placement_point)
    if below is None:
        return spec
    offset = np.asarray(spec.offset_m, dtype=float)
    point = base[:3, :3] @ offset if spec.frame == "target" else offset
    point = point + base[:3, 3]
    over = float(demo.height(point[None])[0])
    tops = [
        float(now.height(np.asarray(g.top_center, dtype=float)[None])[0])
        for g in standing
        if overlap(point[None], g) > 0.0
    ]
    if over - below > max([0.0, *tops]) + RESTING_M:
        return spec
    dz = over - float(now.height(point[None])[0])
    if abs(dz) <= TABLE_NOISE_M:
        return spec
    lift = np.array([0.0, 0.0, dz])
    shift = base[:3, :3].T @ lift if spec.frame == "target" else lift
    moved = tuple(float(v) for v in offset + shift)
    return spec.model_copy(update={"offset_m": moved})


@dataclass(frozen=True)
class _Scene:
    """What an object action reads, once per call: a copy of the registry, the robot now and the
    episode's table (None: not estimated yet)."""

    registry: SceneRegistry
    robot: RobotState
    table: Table | None

    def proxy(
        self, obj: str, parts: Sequence[str | None] = (), measured: bool = False
    ) -> ObjectProxy:
        """A registered object's proxy: where the registry has it now (a
        held object where its grasp holds it), or `measured`, where it was last measured; each
        of `parts` must be registered."""
        record = self.registry.objects.get(obj)
        if record is None:
            raise ToolError("that object is not registered in this episode")
        if measured:
            unmoved = {"pose": record.fit.frame, "grasps": {}, "part_grasps": {}, "part_poses": {}}
            record = record.model_copy(update=unmoved)
        geometry = object_geometry(record, self.robot)
        missing = sorted({p for p in parts if p is not None and p not in geometry.parts})
        if missing:
            raise ToolError(
                f"{record.name} has no registered part {', '.join(missing)} (its parts: "
                f"{', '.join(sorted(geometry.parts))}): locate it with locate_part or register "
                "it with register_part first"
            )
        return ObjectProxy.from_geometry(geometry)

    def boxes(self, but: str | None = None) -> list[OrientedBox]:
        """The boxes of the registered objects no arm holds (nor `but`), as they are now."""
        return [
            object_geometry(record, self.robot).obb
            for record in self.registry.objects.values()
            if not record.grasps and not record.part_grasps and record.id != but
        ]

    def held_boxes(self) -> dict[Arm, list[OrientedBox]]:
        """The boxes of the registered objects each arm holds whole, where its grasp holds them
        now (a held part's body is an articulated fixture: not one the other arm must avoid)."""
        held: dict[Arm, list[OrientedBox]] = {}
        for record in self.registry.objects.values():
            for arm in record.grasps:
                held.setdefault(arm, []).append(object_geometry(record, self.robot).obb)
        return held

    def held(self, arm: Arm, obj: str, spec: PlaceSpec) -> ObjectProxy:
        """The object `arm` holds, with the part its placement point names."""
        record = self.registry.objects.get(obj)
        if record is not None and arm not in record.grasps:
            holder = f"the {record.held_by} arm holds it" if record.grasps else "no arm holds it"
            raise ToolError(f"the {arm} arm does not hold {record.name} ({holder}): grasp it first")
        named = spec.placement_point.startswith("part:")
        return self.proxy(obj, [spec.placement_point.removeprefix("part:") if named else None])

    def target(self, spec: PlaceSpec, held: str) -> ObjectProxy | Pose | None:
        """A placement's target: the object registered under the role the spec names (with its
        part), a landmark of the table, or None for the robot base."""
        where = spec.target
        if where.kind == "robot_frame":
            return None
        if where.kind == "landmark":
            assert where.landmark is not None  # the kind requires it
            known = {} if self.table is None else self.table.landmarks
            if where.landmark not in known:
                raise ToolError(
                    f"the table's {where.landmark} is not known yet (call table first): or place "
                    "relative to an object, or at the robot base (a robot_frame target, the form "
                    "a demonstrated spot takes)"
                )
            return Pose(p=known[where.landmark], q=(1.0, 0.0, 0.0, 0.0))
        assert where.obj is not None  # the kind requires it
        record = self.registry.named(where.obj)
        if record is None:
            roles = ", ".join(sorted(o.name for o in self.registry.objects.values())) or "none"
            raise ToolError(
                f"no object is registered as {where.obj!r}: a place spec names its target by "
                f"role (registered: {roles})"
            )
        if record.id == held:
            raise ToolError("the place spec's target is the object the arm holds")
        return self.proxy(record.id, [where.part])


_SPEC_FIELDS: Final = {GraspObject: "grasp", PlaceObject: "place", RotateAbout: "hinge"}
_KINDS: Final = {
    GraspSpec: "grasp_spec",
    PlaceSpec: "place_spec",
    HingeSpec: "hinge_spec",
    PathSpec: "path_spec",
}
_USED_BY: Final = {
    "grasp_spec": "grasp_object takes (its grasp)",
    "place_spec": "place_object takes (its place)",
    "hinge_spec": "an action on a hinged part takes (its hinge)",
    "path_spec": "follow_path takes: it carries what the arm holds, one arm at a time, or both "
    "arms on one object they hold together; together does not run it",
}
"""For a spec argument naming a Memory parameter of another kind: the tool that takes that kind."""


@dataclass
class _Planned:
    """A hold as `plan_arms` assigns it (mutable while the holds are kept apart)."""

    hold: Hold
    arm: Arm
    why: str
    checked: dict[Arm, str] = dataclass_field(default_factory=dict)
    side: str | None = None
    note: str = ""


def _acting(ctx: ToolContext, targets: Mapping[Arm, str]) -> str:
    """Which arm acted on which role (the `IDENTIFIED` fact's), for the receipt's note."""
    roles = (ctx.frames.fact(IDENTIFIED) or {}).get("roles", {})
    by_object = {
        found.get("object"): role for role, found in roles.items() if isinstance(found, Mapping)
    }
    named = [
        f"the {arm} arm on {by_object[obj]}" for arm, obj in targets.items() if obj in by_object
    ]
    return ("acted: " + ", ".join(named)) if named else ""


def _other(arm: Arm) -> Arm:
    return "right" if arm == "left" else "left"


def _assigned(
    chains: Chains, held: list[_Planned], assign: Mapping[str, Arm], reason: str | None
) -> str:
    """The Agent's arms (`plan_arms(assign=...)`) on the holds of the stages it names: never
    against an instruction's arm (ToolError), the receiving holds following; what breaks the
    plan (holds open at once on one arm, an arm that did not plan) is noted, not refused."""
    unknown = sorted(set(assign) - set(chains.stages))
    if unknown:
        raise ToolError(
            f"no stage {unknown[0]!r} in the workflow's holds: the stages are "
            + ", ".join(chains.stages)
        )
    for stage, arm in assign.items():
        for i in chains.stages[stage][:1]:  # the stage's first hold: its <arm>
            p = held[i]
            if p.arm == arm:
                continue
            if p.why == "instruction":
                raise ToolError(
                    f"the instruction names the {p.arm} arm for {p.hold.role} ({stage}): keep it"
                )
            p.arm, p.why = arm, "agent"
            if p.checked.get(arm, "ok") != "ok":
                p.note = "; ".join(filter(None, [p.note, f"the {arm} arm did not plan it"]))
    for p in held:
        if p.hold.receives_from is not None:
            p.arm = _other(held[p.hold.receives_from].arm)
    for a, b in chains.together:
        if held[a].arm == held[b].arm:
            held[b].note = "; ".join(
                filter(None, [held[b].note, f"open with hold {a + 1} on the same arm"])
            )
    return reason or "the Agent's assignment"


def _stage_arms(ctx: ToolContext, obj: str) -> list[Arm]:
    """The arms the Memory's workflow grasps `obj`'s role with in turn (`stage_arms` of the
    `MEMORY_SPECS` fact, the role found through the `IDENTIFIED` fact); empty when it grasps it
    with one arm, or the role is not known."""
    staged = (ctx.frames.fact(MEMORY_SPECS) or {}).get("stage_arms") or {}
    roles = (ctx.frames.fact(IDENTIFIED) or {}).get("roles", {})
    for role, found in roles.items():
        if isinstance(found, Mapping) and found.get("object") == obj and role in staged:
            return list(staged[role])
    return []


def demo_stage_arm(
    grasps: Sequence[Mapping[str, object]],
    first: tuple[Arm, str],
    staged: Sequence[Arm],
    record: ObjectRecord,
    grasp: GraspSpec | None,
) -> tuple[Arm, str]:
    """On the demonstration's scene, the arm of the stage at hand of an object its workflow
    grasps with both arms in turn (`staged`, in order; `first`: the demonstration's first grasp
    of it, arm and keypoint): before the object was taken, the first grasp's; after (a set-down
    and a regrasp), the grasp spec's arm when it is one of the stages', else the first stage's,
    as `_stage_choice` answers; with the demonstration's first close by that arm after `first`
    as where."""
    arm, where = first
    if not record.taken:
        return first
    if grasp is None or grasp.arm is None or grasp.arm not in staged or grasp.arm == arm:
        return first  # as the stage rule: no stage's spec, the first stage's arm (its retreat)
    chosen = grasp.arm
    keypoints = [str(g["keypoint"]) for g in grasps]
    start = keypoints.index(where) + 1 if where in keypoints else len(grasps)
    for g in grasps[start:]:
        if g["arm"] == chosen:
            return chosen, str(g["keypoint"])
    return chosen, f"the {chosen} arm's stage"


def demo_spec_arm(
    grasps: Sequence[Mapping[str, object]], first: tuple[Arm, str], grasp: GraspSpec | None
) -> tuple[Arm, str]:
    """On the demonstration's scene, for a grasp spec of an arm other than the one the
    demonstration first grasped the object with (`first`: arm and keypoint), once the object was
    taken and set down: that arm, at the demonstration's first close by it after `first` (a
    regrasp's spec, fitted there); `first` when the spec names no arm, the same arm, or one the
    demonstration did not close after it."""
    if grasp is None or grasp.arm is None or grasp.arm == first[0]:
        return first
    keypoints = [str(g["keypoint"]) for g in grasps]
    start = keypoints.index(first[1]) + 1 if first[1] in keypoints else len(grasps)
    for g in grasps[start:]:
        if g["arm"] == grasp.arm:
            return grasp.arm, str(g["keypoint"])
    return first


def _receiving_choice(staged: Sequence[Arm], record: ObjectRecord, grasp: GraspSpec) -> ArmChoice:
    """A grasp of an object one arm holds, in a workflow that grasps it with both arms in turn:
    the hand-over's receiving stage, made by the arm that does not hold it, as the scene has it,
    whatever arm the instruction names (that arm names the stage it describes, the first) or the
    spec was fitted for (a spec fitted on the right arm's receiving grasp may get the instruction's
    left arm while the left holds the object)."""
    holder: Arm = next(iter(record.grasps))
    arm: Arm = "right" if holder == "left" else "left"
    note = (
        f"the {holder} arm holds this object, so a grasp of it now is the hand-over's receiving "
        f"stage: the {arm} arm makes it"
    )
    if grasp.arm is not None and grasp.arm != arm:
        note += f"; the grasp given was written for the {grasp.arm} arm: it is mirrored"
    return ArmChoice(arm=arm, why="stage", candidates=list(staged), note=note)


def _stage_choice(
    staged: Sequence[Arm],
    record: ObjectRecord,
    grasp: GraspSpec | None,
    instructed: Arm,
    named: Sequence[Arm],
) -> ArmChoice:
    """The arm of a stage of a workflow that grasps one object with both arms in turn
    (`staged`, in order): the instruction's arm names the stage that arm does, so the
    demonstration's per-stage arms stand. The stage is the held object's (its holder), else the
    grasp spec's when the object was taken before (a later stage's regrasp), else the first.
    The whole sequence is mirrored only when the instruction names both arms in the reverse
    order of the stages."""
    if record.grasps:
        arm: Arm = list(record.grasps)[-1]
    elif record.taken and grasp is not None and grasp.arm in staged:
        arm = grasp.arm
    else:
        arm = staged[0]
    mirrored = len(named) > 1 and list(named[: len(staged)]) == list(reversed(staged))
    if mirrored and not record.grasps:
        arm = "right" if arm == "left" else "left"
    order = " then ".join(("right" if a == "left" else "left") if mirrored else a for a in staged)
    note = (
        f"the workflow grasps this object with the {order} arm in turn; the instruction's "
        f"{instructed} arm names the stage that arm does, "
        + ("and it names both arms the other way round, so the sequence is mirrored"
           if mirrored else "so each stage keeps its own arm")
    )  # fmt: skip
    if grasp is not None and grasp.arm is not None and grasp.arm != arm and not record.grasps:
        note += f"; the grasp given was written for the {grasp.arm} arm's stage"
    return ArmChoice(arm=arm, why="stage", candidates=list(staged), note=note)


def _note_mirrored(ctx: ToolContext, actions: Mapping[Arm, ArmAction]) -> None:
    """Whether this scene's arms are the mirror image of the ones the Memory's specs were written
    for (`_MIRRORED` in the episode state): the last action whose spec names an arm, acted by
    the other arm (or by it)."""
    for arm, action in actions.items():
        field = _SPEC_FIELDS.get(type(action))
        spec = None if field is None else getattr(action, field)
        written = getattr(spec, "arm", None)
        if written is not None:
            ctx.state[_MIRRORED] = arm != written


def displacement(
    ctx: ToolContext,
    shift: tuple[float | str, float | str, float | str],
    move_axis: str,
    lateral: ArrayLike,
) -> tuple[float, float, float]:
    """`move_by`'s x, y, z in meters: a number as given; a Memory number parameter's
    name (`{name}`, `MEMORY_SPECS` numbers) its value. A world displacement taken from the Memory
    was fitted at the demonstration's arms: where this scene's arms are their mirror image (the
    last spec that names an arm was acted by the other one, `_note_mirrored`), the named part
    of it is reflected across the robot's sagittal plane (normal `lateral`), as a spec is for
    the other arm; numbers given as numbers come from this scene and stay. ToolError for a name
    the Memory has no number under."""
    numbers = (ctx.frames.fact(MEMORY_SPECS) or {}).get("numbers", {})
    given = np.zeros(3)
    named = np.zeros(3)
    names = {}
    for k, value in enumerate(shift):
        name = spec_name(value)
        if name is None:
            given[k] = float(value)
            continue
        if name not in numbers:
            listed = ", ".join(sorted(numbers)) or "none"
            raise ToolError(
                f"the Memory has no number parameter {name!r} (its numbers: {listed}); name one "
                "of them or give the number"
            )
        named[k] = float(numbers[name])
        names["xyz"[k]] = name
    if not names:
        return float(given[0]), float(given[1]), float(given[2])
    mirrored = move_axis == "world" and ctx.state.get(_MIRRORED) is True
    if mirrored:
        normal = np.asarray(lateral, dtype=float)
        normal /= np.linalg.norm(normal)
        named = named - 2.0 * float(named @ normal) * normal
    total = np.round(given + named, 6) + 0.0
    ctx.emit("displacement_named", {"names": names, "mirrored": mirrored, "shift": total.tolist()})
    return float(total[0]), float(total[1]), float(total[2])


def expand_specs(ctx: ToolContext, actions: Mapping[Arm, ArmAction]) -> dict[Arm, ArmAction]:
    """Each action whose spec names a Memory parameter (`{name}`) with that parameter's stored
    spec, every field of it (the acting arm mirrors it as for a spec written out); each expansion
    is logged (`spec_expanded`). ToolError for a name the Memory has no such spec under."""
    out: dict[Arm, ArmAction] = {}
    for arm, action in actions.items():
        field = _SPEC_FIELDS.get(type(action))
        given = None if field is None else getattr(action, field)
        if field is None or spec_name(given) is None:
            out[arm] = action
            continue
        kind = {"grasp": GraspSpec, "place": PlaceSpec, "hinge": HingeSpec}[field]
        out[arm] = action.model_copy(update={field: resolve(ctx, given, kind, arm)})
    return out


def resolve(
    ctx: ToolContext,
    given: object,
    kind: type[GraspSpec] | type[PlaceSpec] | type[HingeSpec] | type[PathSpec],
    arm: Arm | None = None,
) -> Any:
    """`given` as a spec: itself, or the Memory's spec it names (`MEMORY_SPECS`)."""
    name = spec_name(given)
    if name is None:
        return given
    specs = (ctx.frames.fact(MEMORY_SPECS) or {}).get("specs", {})
    found = specs.get(name)
    label = _KINDS[kind]
    if found is None or found.get("kind") != label:
        same = sorted(n for n, s in specs.items() if s.get("kind") == label)
        listed = ", ".join(same) if same else "none"
        other = None if found is None else found.get("kind")
        hint = "name one of them or write the spec out"
        if other is not None:
            used_by = _USED_BY.get(other, "another tool takes")
            hint = f"{name!r} is a {other} parameter, which {used_by}"
        raise ToolError(
            f"the Memory has no {label} parameter {name!r} (its {label} parameters: {listed}); "
            f"{hint}"
        )
    spec = kind.model_validate(found["value"])
    ctx.emit("spec_expanded", {"arm": arm, "name": name, "kind": label, "spec": found["value"]})
    return spec


def in_the_way(receipt: ActionReceipt, robot: RobotState, clearance_m: float) -> str | None:
    """The note of a missed move when the other arm's TCP is within `clearance_m` of where it
    was going (the other arm may have blocked it or pushed what it carried); None otherwise."""
    if receipt.arm is None or receipt.commanded is None or receipt.reason not in _MISSED:
        return None
    other: Arm = "right" if receipt.arm == "left" else "left"
    if other not in robot.arms:
        return None
    target = np.asarray(receipt.commanded.tcp.p)
    gap = float(np.linalg.norm(np.asarray(robot.arms[other].tcp.p) - target))
    if gap > clearance_m:
        return None
    return (
        f"the {other} arm's TCP is {gap * 100:.0f} cm from the {receipt.arm} arm's target and "
        f"may have blocked this move: move the {other} arm clear (up and away) first"
    )


def _by_arm(built: Built) -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    """One arm's built segments as `Executor.plan`'s (left, right)."""
    return (built.segments, ()) if built.arm == "left" else ((), built.segments)


def holding(registry: SceneRegistry, arm: Arm) -> ObjectRecord | None:
    """The registered object `arm` holds, or one of whose parts it holds; None: none."""
    return next(
        (o for o in registry.objects.values() if arm in o.grasps or arm in o.part_grasps), None
    )


def _target(registry: SceneRegistry, action: PlaceObject) -> str | None:
    """The registered object a placement places on (None: a landmark or the robot base)."""
    if action.place.target.kind != "object":
        return None
    named = registry.named(action.place.target.obj or "")
    return None if named is None else named.id


async def _built(
    api: ActionApi,
    arm: Arm,
    action: ArmAction,
    scene: _Scene | None,
    near: Built | None = None,
    among: Sequence[int] | None = None,
) -> Built:
    """One arm's action built by the API (`scene`: given for an action on an object; `near`:
    an earlier build whose grasp candidate or placement option it keeps; `among`: the grasp
    candidates it may choose)."""
    if isinstance(action, _OBJECT_ACTIONS):
        assert scene is not None  # `_builds` loads it for these
        return await _object_action(api, arm, action, scene, near, among)
    if isinstance(action, MoveToPose):
        return await api.move_to_pose(arm, Pose.from_list(action.pose))
    if isinstance(action, MoveBy):
        return await api.move_by(arm, action.x, action.y, action.z, action.quat, action.move_axis)
    if isinstance(action, ReturnHome):
        if api.profile.arms[arm].home is None:
            raise ToolError(f"this robot's {arm} arm has no start pose to go back to")
        return await api.return_home(arm)
    if isinstance(action, OpenGripper):
        return api.open_gripper(arm, action.pos)
    return api.close_gripper(arm, action.pos)


async def _co_grasp(
    api: ActionApi,
    actions: Mapping[Arm, ArmAction],
    scene: _Scene,
    near: Mapping[Arm, Built] | None = None,
    among: Mapping[Arm, Sequence[int]] | None = None,
) -> list[Built]:
    """Both arms' grasps of one object built as a pair (`ActionApi.co_grasp`); `near`: on the
    frame and among the candidates an earlier build of the pair chose."""
    calls = {}
    for arm, action in actions.items():
        assert isinstance(action, GraspObject)  # `_builds` routes only a pair of grasps here
        grasp = action.grasp
        toward = None if grasp.x_toward in (OTHER_ARM, *GEOMETRIC_ENDS) else grasp.x_toward
        kept = (among or {}).get(arm)
        earlier = (near or {}).get(arm)
        if kept is None and earlier is not None and earlier.grasp is not None:
            kept = [earlier.grasp.candidate]
        calls[arm] = GraspCall(
            scene.proxy(action.obj, [grasp.part, toward]),
            grasp,
            action.approach_m,
            action.grasp_offset_m,
            action.gripper_pos,
            kept,
        )
    turned = None if not near else next(iter(near.values())).turned
    pair = await api.co_grasp(calls, turned=turned)
    return [pair[arm] for arm in actions]


async def _object_action(
    api: ActionApi,
    arm: Arm,
    action: GraspObject | PlaceObject | RotateAbout,
    scene: _Scene,
    near: Built | None = None,
    among: Sequence[int] | None = None,
) -> Built:
    if isinstance(action, GraspObject):
        if among is None and near is not None and near.grasp is not None:
            among = [near.grasp.candidate]
        grasp = action.grasp
        toward = (
            None if grasp.x_toward in (OTHER_ARM, *GEOMETRIC_ENDS) else grasp.x_toward
        )  # directions
        return await api.grasp_object(
            arm,
            scene.proxy(action.obj, [grasp.part, toward]),
            grasp,
            action.approach_m,
            action.grasp_offset_m,
            action.gripper_pos,
            among=among,
        )
    if isinstance(action, PlaceObject):
        place = action.place
        axes = None if action.align_axis is None else [list(v) for v in action.align_axis]
        approach = action.approach_axis
        held = scene.held(arm, action.obj, place)
        start = None if near is None else near.placed_from
        if start is not None:
            # A rebuild among an earlier build's option (refined geometry, the arm at its
            # pre-place since) computes from the ee that build did, the object held as it is
            # held now: the placement turns with the ee (`constrain=auto`), so from the
            # pre-place the same option would name another pose.
            now = scene.robot.arms[arm].ee.matrix()
            held = held.at(start.matrix() @ np.linalg.inv(now) @ held.pose)
        return await api.place_object(
            arm,
            held,
            place,
            scene.target(place, action.obj),
            action.approach_m,
            action.stop_back_m,
            action.is_open,
            action.constrain,
            axes,
            action.object_axis,
            action.object_axis_type,
            approach if isinstance(approach, str) else list(approach),
            option=None if near is None else near.option,
            scene=scene.boxes(but=action.obj),
            ee=start,
            target_turn_deg=None if near is None else near.target_turn_deg,
        )
    hinge = hinge_line(action.hinge, scene.proxy(action.obj, [action.hinge.part], measured=True))
    return await api.rotate_about(arm, hinge, action.angle_deg, action.steps)


def _planner(config: PlannerConfig) -> Planner:
    if config.mock:
        return MockPlanner()
    url = config.url
    if config.ready_file is not None:
        path = config.ready_file
        ready = read_ready(path if path.is_absolute() else repo_root() / path)
        if ready is None:
            raise ValueError(f"no motion service has written {path}: start it first")
        url = ready.url_for(Planner.PLANE)
    assert url is not None  # PlannerConfig gives one of the three
    return client(  # type: ignore[type-abstract]  # rpc interfaces are Protocols
        Planner, url, token=None, timeout=10.0, timeouts=TIMEOUTS
    )
