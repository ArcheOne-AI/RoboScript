"""What a physical action did (docs/concepts/agent-runtime.md)."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, NonNegativeInt, computed_field

from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm


class ReceiptReason(str, Enum):
    ok = "ok"
    clamped = "clamped"
    unreachable = "unreachable"
    plan_failed = "plan_failed"
    stopped_short = "stopped_short"
    deviated = "deviated"
    empty_grasp = "empty_grasp"
    lost_hold = "lost_hold"
    rejected_repeat = "rejected_repeat"
    budget_exhausted = "budget_exhausted"
    invalid_args = "invalid_args"

    def __str__(self) -> str:  # format() of str mixins changed in 3.11; keep the value on both
        return self.value


HARD_FAILURES = frozenset(
    {
        ReceiptReason.unreachable,
        ReceiptReason.plan_failed,
        ReceiptReason.empty_grasp,
        ReceiptReason.lost_hold,
    }
)
"""Reasons that end an episode under `PhaseRules.fail_fast`."""
UNMOVED = frozenset({ReceiptReason.unreachable, ReceiptReason.plan_failed})
"""Hard failures a planner or a builder gives before anything moves (`ActionReceipt.unmoved`)."""

_REACHED = frozenset({ReceiptReason.ok, ReceiptReason.clamped})


class CommandedPoses(BaseModel):
    """The target as the action API sees it (`ee`) and as the fingertips see it (`tcp`)."""

    model_config = ConfigDict(frozen=True)

    ee: Pose
    tcp: Pose


class AchievedPose(BaseModel):
    """Measured TCP after the action and its orientation error to the commanded TCP."""

    model_config = ConfigDict(frozen=True)

    tcp: Pose
    orient_err_deg: FiniteFloat


class GripperReport(BaseModel):
    """Normalized gripper command and settled measured opening (None when not measured)."""

    model_config = ConfigDict(frozen=True)

    commanded: FiniteFloat
    measured: FiniteFloat | None
    holding: bool | None


class GraspChoice(BaseModel):
    """Which of the synthesized grasp candidates was executed; `frame_turn_deg`: for a pose
    grasp, the turn (deg, about z) of the object's frame the candidate was made in (the grasp
    relation keeps it: `SceneRegistry.pin`); `hold_frame`: the close pins the object's frame to
    that turn whenever its shape looks the same turned (`ExecutorConfig.hold_frames`), not only
    where a choice went against a weak cue."""

    model_config = ConfigDict(frozen=True)

    candidate: NonNegativeInt
    n_candidates: NonNegativeInt
    frame_turn_deg: FiniteFloat | None = None
    hold_frame: bool = False
    unplannable: NonNegativeInt = 0
    """Candidates the chooser preferred to this one that the arm could not reach (no plannable
    pre-grasp, or no IK solution for the grasp): where the arm reaches, not the spec, picked
    this grasp."""
    blocked: NonNegativeInt = 0
    """Candidates it preferred whose approach ran into the other arm."""


class GripEvent(BaseModel):
    """A gripper change during an action, what grasp relations follow: an arm's close whose
    opening settled, an open where it was first commanded, or a closed gripper whose settled
    opening fell below the holding threshold (a lost hold); at executed step `step` (an index into
    `ActionReceipt.tcp_path`), with the measured opening then (None: not measured)."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    step: NonNegativeInt
    closed: bool
    opening: FiniteFloat | None


class Press(BaseModel):
    """A move meant to touch, with nothing held, that the arm could not finish: what the fingers
    pushed on stopped them `short_m` short of the target, on the way to it, at executed step
    `step` (an index into `ActionReceipt.tcp_path`), with the TCP at `tcp` (world); `closed`:
    the gripper was closed on nothing (else open). Proprioception tells it: the commanded pose
    against the measured one."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    step: NonNegativeInt
    tcp: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    short_m: FiniteFloat
    closed: bool


class JointReport(BaseModel):
    """The joint a pulled part moved on, as fitted to the achieved TCP positions of a followed
    pull (`ActionApi`): `revolute` about the line through `point` along `direction` (unit,
    right-handed) at `radius_m` from it, or `prismatic` along `direction`; `residual_m` the RMS
    distance of the `points` positions to it, `source` `guess` (the caller's hinge or direction),
    `circle` or `line`. `travel` and `target` are the pull's achieved and asked travel along it:
    degrees about a revolute joint, meters along a prismatic one."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["revolute", "prismatic"]
    point: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    direction: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    radius_m: FiniteFloat | None
    residual_m: FiniteFloat
    source: Literal["guess", "circle", "line"]
    travel: FiniteFloat
    target: FiniteFloat
    points: NonNegativeInt


class PathHold(BaseModel):
    """One arm's hold over a followed path: `kept`, else lost after waypoint `lost_at` (1-based)
    with its TCP then at `tcp` (world)."""

    model_config = ConfigDict(frozen=True)

    kept: bool
    lost_at: NonNegativeInt | None = None
    tcp: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None


class PathReport(BaseModel):
    """A followed held path (`follow_path`): `reached` of its `waypoints` (after the start) run,
    per arm the measured TCP's distance (`deviation_m`) and turn (`angle_deg`) from each
    reached waypoint's target, and its hold (`holds`); `start_m` and `start_deg`: how far the
    path's first pose lay from the nearest holding TCP in the frame chosen for it (a grasp that
    far off the demonstration's is carried along)."""

    model_config = ConfigDict(frozen=True)

    waypoints: NonNegativeInt
    reached: NonNegativeInt
    deviation_m: dict[Arm, list[FiniteFloat]] = Field(default_factory=dict)
    angle_deg: dict[Arm, list[FiniteFloat]] = Field(default_factory=dict)
    holds: dict[Arm, PathHold] = Field(default_factory=dict)
    start_m: FiniteFloat = 0.0
    start_deg: FiniteFloat = 0.0


class ActionReceipt(BaseModel):
    """`deviation_m` is the distance between the achieved and the commanded TCP; `tcp_path` holds
    the TCP after every executed step, per arm the action commanded; `grips` the gripper changes,
    in step order; `presses` the moves meant to touch, nothing held, that what they met stopped,
    in step order; `joint` the joint a followed pull fitted; `path` what a followed held path
    did; `look_steps` the steps of `steps_used` an arm spent moving to look at an object before
    the action (a camera move, nothing touched). Fields that do not apply (nothing planned or
    executed) are None. `ok` is derived from `reason` and serialized with the rest."""

    model_config = ConfigDict(frozen=True)

    reason: ReceiptReason
    arm: Arm | None = None
    commanded: CommandedPoses | None = None
    achieved: AchievedPose | None = None
    deviation_m: FiniteFloat | None = None
    steps_used: NonNegativeInt = 0
    look_steps: NonNegativeInt = 0
    steps_left: NonNegativeInt | None = None
    gripper: GripperReport | None = None
    grasp: GraspChoice | None = None
    tcp_path: dict[Arm, list[Pose]] = Field(default_factory=dict)
    grips: list[GripEvent] = Field(default_factory=list)
    presses: list[Press] = Field(default_factory=list)
    joint: JointReport | None = None
    path: PathReport | None = None
    note: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """The action reached its target: reason `ok` or `clamped`."""
        return self.reason in _REACHED

    @property
    def hard_failure(self) -> bool:
        return self.reason in HARD_FAILURES

    @property
    def unmoved(self) -> bool:
        """A rejection before anything moved: unreachable or plan_failed with no step used but a
        look's (the planner, IK or a builder refused the call; the scene is as it was, an arm at
        most moved its camera)."""
        return self.reason in UNMOVED and self.steps_used == self.look_steps
