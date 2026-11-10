"""The executor's tuning (docs/how-to/add-a-skillset.md): an env's skillset binds it as data,
and a deployment's config overrides single fields of it (the `executor:` section of
`configs/manipulation.example.yaml`). Most fields have no default, the values being the env's. A
field at its default stays out of a stored hash (`skills.store` leaves defaults out)."""

from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    TypeAdapter,
)


class ExecutorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    waypoint_tolerance_m: PositiveFloat
    """Waypoints are the fewest planned points keeping the TCP of every joint-space line between
    them within this distance of the planned path."""
    straight_step_m: PositiveFloat
    straight_step_deg: PositiveFloat
    """A straight move is solved by IK at poses this far apart along its line (twice and four
    times as densely where the TCP between neighboring solutions leaves the line)."""
    max_joint_step_rad: PositiveFloat
    """The largest joint change between neighboring IK solutions of a straight move: more is a
    wrist spin, which the TCP does not show (near a singularity a grasp's approach may start
    with 0.7 rad in 2 cm, continuously)."""
    lock_tolerance_m: PositiveFloat
    lock_tolerance_deg: PositiveFloat
    """A straight move without an IK line takes an orientation-locked plan only if its TCP keeps
    this close to the line and its orientation to the target's."""
    max_joint_travel_rad: PositiveFloat
    """The largest distance a joint may travel in one planned move. cuRobo sometimes wraps a
    joint a full turn (joint limits are +-10 rad). A plan that travels farther is requested once
    more; IK solutions along a straight line are first
    moved by whole turns to the equivalent nearest the previous one."""
    replans: NonNegativeInt
    """A move whose plan fails is planned again up to this many times before it is refused:
    cuRobo's refusals are not deterministic, and every call advances its seed state, so each
    attempt starts from fresh seeds (a straight move: its IK line, then locked plans)."""
    settle_holds: NonNegativeInt
    """After a close, at most this many holds until the measured opening settles (most closes
    settle within a few; the rest are creeping onto an object)."""
    open_holds: NonNegativeInt
    """Holds after opening a closed gripper, for the fingers to finish opening."""
    place_step_m: PositiveFloat | None
    """A place, a contact move made while the gripper holds something, is sent in steps moving
    the TCP at most this far (each `take_action` starts and ends at rest: shorter is slower);
    None sends it as any other move."""
    release_holds: NonNegativeInt
    """Holds before opening a gripper whose last move placed what it holds, for it to settle."""
    retry_deviated: bool
    """A straight move that ends an action `deviated` is planned once more, from where the arm
    stopped to the same target."""
    stop_tolerance_m: PositiveFloat
    stop_tolerance_deg: PositiveFloat
    """A non-contact move ending farther than this from its target stopped short or deviated."""
    contact_off_path_m: PositiveFloat
    """A move meant to touch that ends farther than this off its planned path, and from its
    target, never got there (stuck or blocked on the way) and is `deviated`; nearer, what it touched
    may have pushed it aside."""
    press_m: PositiveFloat
    """A move meant to touch, with nothing held, that ends at least this far short of its target
    on its path was stopped by what it pushed: a press (`ActionReceipt.presses`). A free move
    ends within a few millimeters of its target; a firm press stops centimeters short. A light touch
    that latches ends the episode anyway; one that does not must not pass for a press."""
    max_translation_m: PositiveFloat
    max_rotation_deg: PositiveFloat
    """A bounded (displacement) move is clamped to this translation and rotation."""
    min_arm_gap_m: PositiveFloat
    """A two-arm action (`together`) keeps the TCPs this far apart along the planned paths, unless
    they start closer and come no more than 5 mm closer (arms that hand over or work side by
    side can start that close)."""
    workspace: Path | None
    """A probed workspace (`manipulation.workspace`, JSON): targets whose TCP lies farther than
    `workspace_margin_m` from every cell the arm reaches are refused as `unreachable`."""
    workspace_margin_m: PositiveFloat
    obstacle_margin_m: NonNegativeFloat
    """Each obstacle box (`Executor.obstacles`: the scene as perceived) is grown by this on every
    side before a free move is planned around it."""
    release_withdraw_m: NonNegativeFloat
    release_withdraw_max_m: PositiveFloat
    """After a release, the arm's next move first withdraws the TCP straight back along its
    pointing axis (the approach, reversed) at least `release_withdraw_m` (0: never), farther
    until it is out of every obstacle box it is in (grown by `obstacle_margin_m`), at most
    `release_withdraw_max_m`: a move away with the open fingers still around what they
    released drags it along (a straight move ignores obstacles; a free plan exempts what lies
    a hand's length from its start). A move that itself starts back along the pointing axis
    (within `WITHDRAW_CONE_DEG`, at least that far; a straight one only, with
    `withdraw_any_move`) needs none; a withdraw that cannot be made is skipped. Which action
    withdraws: `withdraw_any_move`."""
    lift_clearance_m: PositiveFloat
    """A free move with the gripper empty whose TCP starts lower than this above the top of an
    obstacle within this distance of its way (seen from above) first rises straight up to that
    height, by at most `max_lift_m` (where no IK line rises, it plans from where it is). A
    carry's first free move rises likewise until what the arm holds is this far over its
    support (the table top or a perceived object under it)."""
    max_lift_m: PositiveFloat
    reject_repeats: bool
    """Refuse, without executing, an action whose segments all match those of the action
    that just failed (targets within `repeat_tolerance_m` and `repeat_tolerance_deg`, the same
    gripper commands): `rejected_repeat`."""
    repeat_tolerance_m: PositiveFloat
    repeat_tolerance_deg: PositiveFloat
    follow_step_m: PositiveFloat
    """A pull whose gripper holds a part by a pull's close follows the part's joint in straight
    moves taking the TCP this far (a turn at most `ARC_STEP_DEG` each), refitting the joint to
    the achieved TCP positions after each (`manipulation.articulation`)."""
    follow_tolerance_m: PositiveFloat
    """A followed pull is done once what is left of it moves the TCP less than this."""
    follow_stall: float = Field(ge=0.0, lt=1.0)
    """A followed step that took the TCP along the joint less than this share of the step is a
    stall: the part is at its limit, or the first step went the way it does not move (tried the
    other way once)."""
    follow_regrasps: NonNegativeInt
    """A followed pull that loses its hold opens, backs off and closes again where the part
    was last held this many times."""
    follow_regrasp_reported: bool
    """A followed step the executor reports `lost_hold` for, whose state shows the slip, is
    taken again like any slip (`follow_regrasps`); false: that report ends the pull with it."""
    follow_joint_paths: bool
    """A one-arm held path whose end moves on one joint (`manipulation.paths.joint_tail`)
    follows that end on the joint, from the measured TCP (as a pull), instead of replaying it as
    rigid motions from the path's start."""
    follow_past: NonNegativeFloat
    """Such an end, on a part known to turn or slide (a hinge's turn; a slide held by a pull's
    close or on a registered articulated part), goes on along the joint past the path's last
    waypoint until the part stops following, by at most this share of the end's travel: the
    demonstration may end just where the task's success begins. 0: it stops at the last
    waypoint."""
    hold_frames: bool
    """Frames follow the hold on an object whose shape looks the same turned (a half or quarter
    turn): a held path is chosen among the whole's frame in every such turn too
    (`manipulation.paths.hold_frames`), and a pose grasp's close pins the object's frame to the
    turn its candidate was made in (`GraspChoice.hold_frame`, `SceneRegistry.pin`), so placements
    are written in the frame the hold reproduces. False: a path in the frames the registry's
    decision gives, and a frame pinned only where a choice went against a weak cue."""
    max_action_joint_step_rad: PositiveFloat | None = None
    """The largest joint change one step may ask of an arm: a waypoint farther than this (any
    joint) from the step before is reached through steps along the joint-space line between
    them, evenly spaced, so the path stays the same, in more calls. None: a step goes to its
    waypoint, for an env whose `take_action` moves until it gets there; an env whose `take_action`
    runs a fixed number of physics steps binds the step a tracking probe finds keeps the TCP
    within 3 mm of the waypoint."""
    honor_front: bool
    """A placement whose spec names a front (`PlaceSpec.x_toward`) the held object resolves (a
    part, or a geometric end, `ObjectProxy.aim`) takes the spec's turn as written: the object
    is not taken to look the same turned (no equivalent turn nearest how it is held, none
    searched), since its front tells the turns apart. False: a half-turn look-alike takes the
    turn nearest how it is held whatever front the spec names."""
    tilted_turns: bool
    """A symmetric held object placed tilted (`relative`) is tried at its equivalent
    turns about its own axis as placed until one plans (`manipulation.api.placement`): its
    measured yaw is arbitrary. False: the turn as measured only."""
    coarse_stops: bool
    """A grasp or placement run coarse, then fine, whose fine part moved no arm (its targets were
    where the coarse move left the drive targets: a placement with `approach_m` equal to
    `stop_back_m`)
    reports the coarse move's outcome, a stop short included (`receipts.chained`). False: the fine
    part's empty `ok`, "did not move"."""
    drop_start_colliders: bool
    """A free move the planner refuses as starting in collision with the perceived objects is
    planned again clear of all of them but those the arm's collision spheres reach into at its
    start (`Executor`), before the plan without any. False: straight to the plan without any."""
    near_bare: bool
    """A free move that keeps clear of no grown boxes is planned clear of those within a hand's
    length of its start or goal as they are (not grown) and of the rest grown, before it lets the
    near ones go (`Executor`). False: they are let go at once."""
    withdraw_any_move: bool = False
    """After a release, the arm's next action that moves it withdraws first, whatever comes
    before its first move (a gripper-only action in between leaves the withdraw due), and only a
    straight first move back along the approach is the withdraw itself: a free move is planned
    in joint space and may sweep the open fingers through what they let go of. False: only an action
    whose first segment is a move withdraws, any such move back along the approach is the withdraw,
    and the arm's next action of any kind ends the withdraw's being due."""


def executor_overrides(values: dict[str, Any]) -> dict[str, Any]:
    """`values` checked as overrides of single `ExecutorConfig` fields (names and each value's
    type and bounds), to lay over an env's binding (`with_overrides`)."""
    fields = ExecutorConfig.model_fields
    if unknown := sorted(set(values) - set(fields)):
        raise ValueError(f"no executor settings {unknown}; known: {sorted(fields)}")
    checked = {}
    for name, value in values.items():
        info = fields[name]
        kind = Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation
        checked[name] = TypeAdapter(kind).validate_python(value)
    return checked


def with_overrides(binding: ExecutorConfig, overrides: dict[str, Any]) -> ExecutorConfig:
    """The env's executor tuning with a config's single-field overrides laid over it."""
    return ExecutorConfig.model_validate({**binding.model_dump(), **overrides})
