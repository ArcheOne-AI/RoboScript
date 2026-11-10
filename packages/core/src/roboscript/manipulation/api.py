"""The action API (docs/concepts/embodied-core.md): grasp, place, displacement and pose builders on
perceived-object proxies and specs instead of assets. It is the one API every caller acts through:
the Actor's `manipulation` toolset wraps it, and a replay of an expert's calls goes through it too.

Each builder returns a `Built` action, one arm's segments as an action list (pre-grasp, approach,
close; pre-place, place, open; ...), after planning only what the pose functions plan (which grasp
candidate and tilt, which point of a placement region); `run` executes one action, or two at once
(`together`), through the episode's executor, and `check_reach` plans them without executing
(`Reach`). A builder that finds nothing plannable returns the refusal instead, and nothing moves.
A grasp's receipt names the candidate
chosen, k of n, a searched placement's the point and turn chosen. A displacement is a
target from the measured ee pose, straight and bounded (the executor clamps it to the per-action
limits); with `move_axis` arm it moves by z only. `return_home` goes to the profile's home
pose. `rotate_about` turns the gripper about a hinge (a lid, a door) in steps, holding what it
holds. A closed gripper's turn, and a displacement of an arm holding a part by a pull, follow the
part's joint instead (`Follow`, `manipulation.articulation`): short steps along the joint as
estimated from the achieved TCP positions so far, the receipt's `joint` the one fitted.
`follow_path` carries what one arm, or both, hold along a held path fitted on the demonstration
(`manipulation.paths`): the same rigid motion for every holding gripper, from where it holds.

With `ExecutorConfig.reject_repeats`, `run` refuses, without executing, an immediate repeat of
the call that just failed or was refused: the same builder and arguments, or segments within the
repeat tolerances (`rejected_repeat`). Any other call, and `hold`, forgets it; `check_reach`
plans only, so it is never refused so and forgets nothing.
"""

import math
import re
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import islice, pairwise
from typing import Final, Literal

import anyio
import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.env import ObserveRequest
from roboscript.contracts.geometry import OrientedBox, Pose
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.motion import (
    MAX_GOALS,
    BatchPlanRequest,
    FKRequest,
    IKRequest,
    IKSolution,
)
from roboscript.contracts.observation import Observation
from roboscript.contracts.receipt import (
    HARD_FAILURES,
    ActionReceipt,
    GraspChoice,
    JointReport,
    PathHold,
    PathReport,
    ReceiptReason,
)
from roboscript.contracts.robot import Arm, RobotState
from roboscript.contracts.skills import PoseConventions
from roboscript.embodiments import EmbodimentProfile
from roboscript.manipulation.articulation import Joint, estimate
from roboscript.manipulation.executor import Executor
from roboscript.manipulation.paths import (
    FRAME_M_PER_DEG,
    JointTail,
    choose_frame,
    choose_mirrored,
    joint_tail,
    legs,
    motions,
)
from roboscript.manipulation.plan import Grip, Move, Reach, Segment, reach
from roboscript.manipulation.posemath import (
    CONTACT,
    PLACEMENT,
    Constrain,
    Plannable,
    arc_poses,
    choose_grasp,
    place_ee_pose,
)
from roboscript.manipulation.proxy import ObjectProxy
from roboscript.manipulation.receipts import chained, holding, steps_left
from roboscript.manipulation.specs import GraspSpec, PathSpec, PlaceSpec
from roboscript.manipulation.synthesis import (
    Candidate,
    equivalent_turns,
    for_arm,
    free_turns,
    grasp_candidates,
    placement_point_pose,
    region_points,
    target_frame,
    target_pose,
    turned,
)
from roboscript.perception.registry import WHOLE
from roboscript.spatial.rotation import quat_angle_deg, quat_rotate, quat_to_matrix

ARM_AXIS_TAKES_Z: Final = "move_axis arm moves by z only (along the gripper): leave x and y 0"
PAST_PLACE_M: Final = 0.06
"""A placement's `stop_back_m` may go this far below 0: the TCP stops past the placement, pushing
the held object on along its approach, onto a target that takes the object only when pushed.
`approach_m` stays at or behind the placement."""
PAST_PLACE: Final = (
    f"stop_back_m goes at most {PAST_PLACE_M} m past the placement (stop_back_m >= -{PAST_PLACE_M})"
)
PRESS_M: Final = 0.01
"""A displacement or pose move lowering the TCP this much or more is meant to touch."""
LAG_M: Final = 0.005
"""An arm this near where it is driven to lags its drive (a load's sag); farther, something
stopped it (a press)."""
NEAR_M: Final = 0.01
"""Grasp candidates this much farther from the arm than the nearest one are tried only when no
nearer one is plannable."""
FINGER_M: Final = 0.06
"""A gripper's fingers reach this far back from its TCP along its pointing axis; `CLEAR_M` is
kept between the two grippers' fingers, the approaching one's swept over its approach."""
CLEAR_M: Final = 0.04
FACING_DEG: Final = 60.0
"""Grasp candidates this far from a spec's reference orientation are tried after nearer ones."""
POSE_TURN_BUCKET_DEG: Final = 5.0
"""A pose grasp's candidates are ordered by how far each turns from its reference orientation,
in steps of this (deg), each step tried before the next (the preferred orientation within one)."""
BATCH: Final = 10
"""Goals per plan batch (the size the planner's cuRobo batch is warmed up for)."""
MAX_HANDOVER_CHECKS: Final = 20
"""A hand-over placement checks the other arm's grasp at this many options at most (each plans
that arm's candidates); past them it takes the one where that arm runs least into this one,
else the first option within this arm's reach."""
MAX_PLACE_OPTIONS: Final = 400
"""A placement searches at most this many options (region points x equivalent turns, lazily, in
batches of `BATCH`), so a search stays within the action's time: 40 batch plans."""
ARM_GAP_M: Final = 0.01
"""An arm at a goal stands in the other arm's way when their collision spheres (the planner's,
`FKRequest.spheres`) come closer than this: a centimeter for what the spheres leave out (a held
object's edge, the approximation). The planner plans one arm at a time and does not see the
other."""
HOME_M: Final = 0.01
"""An arm whose ee is this close to its home pose is at home."""
ARC_STEP_DEG: Final = 10.0
"""A hinge turn's straight moves turn at most this much each by default."""
_FAILED: Final = frozenset({*HARD_FAILURES, ReceiptReason.invalid_args})
_OTHER: Final[dict[Arm, Arm]] = {"left": "right", "right": "left"}
Array = NDArray[np.float64]


class PlanningTimeout(Exception):
    """A grasp's or a placement's search ran past `ActionApi.deadline`: its builder refuses it
    (plan_failed) instead of searching on until the tool's own timeout ends the episode."""


@dataclass
class PlanningLog:
    """What the planner was asked during one tool call (`ActionApi.planning`): the pose goals
    planned (batch plans), the IK goals solved, and the seconds spent waiting on the planner;
    goals a search had already planned in the same call are answered from its cache and not
    counted."""

    plans: int = 0
    iks: int = 0
    seconds: float = 0.0
    cached: int = 0


@dataclass(frozen=True)
class Follow:
    """A pull `run` follows on the part's joint instead of executing the segments: `guess` (the
    caller's hinge, or the displacement's line), `target` the travel asked along it (degrees
    about a revolute guess, meters along a prismatic one; positive, the guess oriented the way
    the pull goes); `turn_back`: a first step that makes no headway is tried the other way (not
    for a path's end, whose way the demonstration showed); `past`: how much farther (in the
    target's units) the pull goes on until the part stops following, once the target is reached
    (a slip or a stall there ends it, done)."""

    guess: Joint
    target: float
    turn_back: bool = True
    past: float = 0.0


@dataclass(frozen=True)
class Built:
    """One arm's action: its segments, the grasp candidate chosen (k of n), what was chosen in
    words for the receipt (`label`: the grasp candidate, the placement point and turn), or the
    refusal that replaces it; `call` names the builder and its arguments (a repeat's key);
    `follow` a pull to follow on the part's joint (the segments are its open-loop plan, what
    `check_reach` plans)."""

    arm: Arm
    segments: tuple[Segment, ...] = ()
    grasp: GraspChoice | None = None
    label: str = ""
    refused: ActionReceipt | None = None
    call: str = ""
    option: int | None = None
    follow: Follow | None = None
    crowded: bool = False
    turned: bool = False
    placed_from: Pose | None = None
    """A placement's: the ee pose it was computed from (the placement depends on it
    under `constrain=auto`); a rebuild among its option computes from there again."""
    press: bool = False
    """A press's grasp (closed fingers push): one its push did not stop pressed nothing."""
    target_turn_deg: float = 0.0
    """A placement's: the turn of its target's frame it was made in (an open frame's, where the
    frame as registered had no placement within reach); a rebuild among its option keeps it."""


def passed_over(choice: GraspChoice) -> str:
    """The preferred candidates a grasp choice passed over, in words ("" when none): a choice
    made by where the arm reaches, which turning the spec does not change."""
    parts = [
        f"{n} out of reach" if why == "plan" else f"{n} into the other arm"
        for why, n in (("plan", choice.unplannable), ("arm", choice.blocked))
        if n
    ]
    return f"; {' and '.join(parts)} preferred before it" if parts else ""


@dataclass(frozen=True)
class GraspCall:
    """One arm's `grasp_object` arguments, for `ActionApi.co_grasp`."""

    obj: ObjectProxy
    grasp: GraspSpec
    approach_m: float = 0.1
    grasp_offset_m: float = 0.0
    gripper_pos: float = 0.0
    among: Sequence[int] | None = None


@dataclass
class ActionApi:
    """The API for one episode, acting through the episode's `executor`."""

    executor: Executor
    profile: EmbodimentProfile = field(init=False)
    _failed: tuple[Built, ...] | None = field(init=False, default=None)
    _released: set[Arm] = field(init=False, default_factory=set)
    """The arms whose last action opened the gripper: their next move withdraws first."""
    _touches: list[tuple[Arm, Pose, Pose, str]] = field(init=False, default_factory=list)
    """The moves meant to touch that held and pressed nothing, as they ended (`_unchanged`)."""
    deadline: float | None = field(init=False, default=None)
    """`time.monotonic()` past which a grasp's or a placement's search stops and its builder
    refuses (`PlanningTimeout`); the caller sets it per tool call, None: no limit."""
    planning: PlanningLog = field(init=False, default_factory=PlanningLog)
    """The planner work of the current tool call (the caller resets it per call)."""
    _searching: int = field(init=False, default=0)
    _searched: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.profile = self.executor.profile

    def _in_time(self) -> None:
        if self._searching and self.deadline is not None and time.monotonic() > self.deadline:
            raise PlanningTimeout

    def _budget_note(self, what: str) -> str:
        log = self.planning
        return (
            f"planning stopped: {what} ran past the planning time budget ({log.seconds:.0f} s "
            f"waiting on the planner, {log.plans} poses planned, {log.iks} IK goals) before "
            "the tool's own time ran out"
        )

    async def grasp_object(
        self,
        arm: Arm,
        obj: ObjectProxy,
        grasp: GraspSpec,
        approach_m: float = 0.1,
        grasp_offset_m: float = 0.0,
        gripper_pos: float = 0.0,
        among: Sequence[int] | None = None,
    ) -> Built:
        """`_grasp_object` (see there), refused (plan_failed) where its search runs past the
        `deadline`."""
        self._searching += 1
        self._searched = ""
        try:
            return await self._grasp_object(
                arm, obj, grasp, approach_m, grasp_offset_m, gripper_pos, among
            )
        except PlanningTimeout:
            call = _call("grasp_object", arm, obj, grasp, approach_m, grasp_offset_m, gripper_pos)
            note = self._budget_note(f"the search for a grasp{self._searched}") + (
                ": move the other arm clear of the object, or try another grasp or arm"
            )
            refused = await self._refusal(ReceiptReason.plan_failed, arm, note)
            return Built(arm, refused=refused, call=call)
        finally:
            self._searching -= 1

    async def _grasp_object(
        self,
        arm: Arm,
        obj: ObjectProxy,
        grasp: GraspSpec,
        approach_m: float,
        grasp_offset_m: float,
        gripper_pos: float,
        among: Sequence[int] | None,
    ) -> Built:
        """A grasp with the spec's candidates as contact frames (`among`: only
        these, e.g. the one a coarse estimate chose, again on refined geometry): the chooser's
        pick, then pre-grasp, a straight approach (unless `approach_m == grasp_offset_m`) and the
        close. The chooser picks among the candidates clear of the other gripper first (its
        fingers `CLEAR_M` from this one's over the approach: a hand-over's free side, the other
        end of an object held by both), and within them on the arm's side first (a symmetric
        object's end, handle or face nearest the arm's start; a pose grasp with a reference
        orientation instead tries its turns one by one, the demonstration's first), then among
        all; a pull's candidates
        from around the part (`synthesis.grasp_candidates`) only when none of the spec's own
        pre-grasps is plannable; a candidate whose
        pre-grasp is plannable but whose grasp no IK solution reaches is passed over. A `press`
        closes without expecting to hold; a `pull` may lose its hold."""
        state = await self.executor.env.robot(self.executor.episode_id)
        other = next(a for a in state.arms if a != arm)
        candidates = grasp_candidates(
            grasp,
            obj,
            arm,
            other_tcp=state.arms[other].tcp.p,
            lateral=self.lateral,
            span_m=self.profile.arms[arm].gripper.span_m,
        )
        proxy = obj.with_frames({CONTACT: [c.matrix for c in candidates]})
        preference = self.profile.arms[arm].grasp
        if preference is None:
            raise ValueError(f"the {self.profile.name} profile has no grasp preference for {arm}")
        plannable = await self.plannable(arm)
        start = np.asarray((await self.executor.home_pose(arm)).p)
        world = [obj.pose @ c.matrix for c in candidates]
        away = [float(np.linalg.norm(w[:3, 3] - start)) for w in world]
        near = [d <= min(away) + NEAR_M for d in away] if away else []
        clear = [clearance(w, state.arms[other].tcp, approach_m) >= CLEAR_M for w in world]
        facing = [True] * len(world)
        order = [(0, 0.0)] * len(world)
        allowed = [
            i
            for i in (range(len(candidates)) if among is None else among)
            if i < len(candidates) and not candidates[i].air
        ]
        if grasp.reference_q is not None and grasp.arm in (None, arm):
            want = self.base.matrix()[:3, :3] @ quat_to_matrix(grasp.reference_q)
            turned = [_angle_deg(want, w[:3, :3]) for w in world]
            facing = [t <= FACING_DEG for t in turned]
            if grasp.anchor == "pose":
                # On the measured surface, the demonstration's turn first, each turn tried on its
                # own: the turns are one grasp of one shape, so neither the preferred orientation
                # nor the arm's side may pick another (an offset off a round object's axis moves
                # the TCP by up to twice the offset from turn to turn, and a nearer turn's
                # approach comes from elsewhere).
                order = [
                    (int(c.off), t // POSE_TURN_BUCKET_DEG)
                    for c, t in zip(candidates, turned, strict=True)
                ]
                near = [True] * len(world)
        call = _call("grasp_object", arm, obj, grasp, approach_m, grasp_offset_m, gripper_pos)
        air = [c.label for c in candidates if c.air]
        if candidates and len(air) == len(candidates):
            note = (
                f"the fingers would close on air at every one of the spec's {len(air)} "
                f"{_plural('candidate', len(air))}: no measured surface of the object lies "
                "between them at the contact (the anchor and offset put it in an opening, or "
                "beside or past the object); move the contact onto material: another anchor, a "
                "smaller offset or one toward a wall or edge, or another part"
                + _nearest_material(candidates)
            )
            refused = await self._refusal(ReceiptReason.unreachable, arm, note)
            return Built(arm, refused=refused, call=call)
        groups = [
            [
                i
                for i in allowed
                if candidates[i].around == around
                and clear[i] == keep
                and facing[i] == face
                and near[i] == side
                and order[i] == rank
            ]
            for around in (False, True)
            for keep in (True, False)
            for face in (True, False)
            for side in (True, False)
            for rank in sorted(set(order))
        ]
        into_other = await self.into_other(arm)
        chosen = blocked = None
        unplannable = in_other = 0  # the candidates passed over before the chosen one, by why
        for ids in (group for group in groups if group):
            while ids and chosen is None:  # a candidate whose grasp no IK reaches is dropped
                self._searched = (
                    f" ({unplannable + in_other} of its {len(candidates)} candidates passed over:"
                    f" {unplannable} out of reach, {in_other} into the other arm)"
                )
                chosen = await choose_grasp(
                    proxy,
                    arm,
                    approach_m,
                    grasp_offset_m,
                    ids,
                    preference,
                    self.profile.arms[arm].ee_to_tcp[0],
                    plannable,
                    self.profile.poses,
                )
                if chosen is None:
                    unplannable += len(ids)
                    break
                depth = await into_other([chosen.pre, chosen.grasp])
                if depth[1] is None or depth[0] is None or max(depth[0], depth[1]) > 0:
                    if depth[1] is not None and depth[0] is not None and blocked is None:
                        blocked = (chosen, max(depth[0], depth[1]))  # if nothing else is clear
                    ids = [i for i in ids if i != chosen.frame]
                    if depth[1] is None or depth[0] is None:
                        unplannable += 1  # no IK solution reaches its pre-grasp or grasp
                    else:
                        in_other += 1
                    chosen = None
            if chosen is not None:
                break
        in_way = ""
        if chosen is None and blocked is not None:
            chosen, depth_m = blocked
            in_way = f"; the other arm stands in its way ({depth_m * 100:.0f} cm)"
        if chosen is None:
            note = (
                f"no pre-grasp of the {len(candidates)} {_plural('candidate', len(candidates))} "
                f"is plannable at any approach tilt ({grasp.approach} approach)"
                + _pose_turns_note(grasp, candidates, obj)
            )
            if world:  # how far its contact nearest the arm lies beyond each arm's reach
                tcp = np.asarray(state.arms[arm].tcp.p)
                contact = min((w[:3, 3] for w in world), key=lambda p: np.linalg.norm(p - tcp))
                note += "; " + await self.executor.reach_note(state, arm, contact.tolist())
            refused = await self._refusal(ReceiptReason.unreachable, arm, note)
            return Built(arm, refused=refused, call=call)
        close = Grip(gripper_pos, grasp=grasp.purpose != "press", pull=grasp.purpose == "pull")
        moves: list[Segment] = [Move(chosen.pre, contact=approach_m == grasp_offset_m)]
        if approach_m != grasp_offset_m:
            moves.append(Move(chosen.grasp, straight=True, contact=True))
        label = candidates[chosen.frame].label
        turn = _frame_turn(grasp, label)
        held = turn is not None and self.executor.config.hold_frames
        if held:
            turn = hold_turn(candidates, chosen.frame)
        if in_way:  # the one kept although the other arm stands in its way is no pass-over
            in_other -= 1
        choice = GraspChoice(
            candidate=chosen.frame,
            n_candidates=len(candidates),
            frame_turn_deg=turn,
            hold_frame=held,
            unplannable=unplannable,
            blocked=in_other,
        )
        note = (
            f"grasp candidate {chosen.frame + 1} of {len(candidates)} ({label}"
            f"{passed_over(choice)}){in_way}"
        )
        return Built(
            arm,
            (*moves, close),
            choice,
            note,
            call=call,
            crowded=bool(in_way),
            press=grasp.purpose == "press",
        )

    async def co_grasp(
        self, calls: Mapping[Arm, GraspCall], turned: bool | None = None
    ) -> dict[Arm, Built]:
        """Both arms' grasps of one object, to run at once (`together`): each built as
        `grasp_object` builds it, on the object's frame and, when its shape looks the same turned
        half a turn about z (`ObjectProxy.symmetric_shape`) and a spec grasps the whole, on that
        turned frame too: there a frame's x is a choice, so each spec's end or side may lie on the
        other arm's side, and the pair swapped is the same grasp. The pair kept has the fewest
        refusals, then the fewest arms standing in the other's way, then its grasps nearest the
        arms' own home poses in sum (arms that do not cross). `turned`: only that frame (a fine
        build keeping the coarse one's)."""
        if set(calls) != set(self.profile.arms) or len(calls) != 2:
            raise ValueError(f"co_grasp takes one grasp per arm, got arms {sorted(calls)}")
        symmetric = all(c.obj.symmetric_shape for c in calls.values()) and any(
            c.grasp.part == WHOLE for c in calls.values()
        )
        frames = [False, True] if turned is None else [turned]
        if not symmetric:
            frames = [f for f in frames if not f] or [False]
        homes = {arm: np.asarray((await self.executor.home_pose(arm)).p) for arm in calls}
        pairs: list[tuple[tuple[int, int, float], dict[Arm, Built]]] = []
        for flip in frames:
            pair = {}
            for arm, c in calls.items():
                obj = c.obj.turned(math.pi) if flip else c.obj
                built = await self.grasp_object(
                    arm, obj, c.grasp, c.approach_m, c.grasp_offset_m, c.gripper_pos, c.among
                )
                choice = built.grasp
                if flip and choice is not None and choice.hold_frame:
                    assert choice.frame_turn_deg is not None  # a hold frame has its turn
                    turn = (choice.frame_turn_deg + 180.0) % 360.0  # on the object's own frame
                    choice = choice.model_copy(update={"frame_turn_deg": turn})
                pair[arm] = replace(built, turned=flip, grasp=choice)
            refused = sum(b.refused is not None for b in pair.values())
            crowded = sum(b.crowded for b in pair.values())
            away = sum(_reach_m(b, homes[arm]) for arm, b in pair.items())
            pairs.append(((refused, crowded, away), pair))
        _, best = min(pairs, key=lambda p: p[0])
        if next(iter(best.values())).turned:
            arm = next(iter(best))
            note = "; ".join(
                filter(None, [best[arm].label, "the pair on the frame turned 180 deg"])
            )
            best[arm] = replace(best[arm], label=note)
        return best

    async def place_object(
        self,
        arm: Arm,
        obj: ObjectProxy,
        place: PlaceSpec,
        target: ObjectProxy | Pose | None,
        approach_m: float = 0.1,
        stop_back_m: float = 0.02,
        is_open: bool = True,
        constrain: Constrain = "auto",
        align_axis: ArrayLike | None = None,
        object_axis: ArrayLike = (1.0, 0.0, 0.0),
        object_axis_type: Literal["object", "world"] = "object",
        approach_axis: Literal["grasp", "target_z"] | Sequence[float] = "grasp",
        option: int | None = None,
        scene: Sequence[OrientedBox] = (),
        ee: Pose | None = None,
        target_turn_deg: float | None = None,
    ) -> Built:
        """`_place` (see there) on the target's frame as registered. Where nothing is within
        reach there (refused, or no IK solution reaches its pre-place or place) and the spec is
        written in the axes of a target object whose frame's x is open (`Part.x_known`: a square
        rack, a round plate), each of the frame's other turns (`Part.turns`) is tried in order,
        searched so that one out of reach is refused: the frame the spec was written in may be
        any of them, and a placement in another is another place. Where none is, the registered
        turn's build
        stands. `target_turn_deg`: only that turn (a rebuild among an earlier build's option).
        Refused (plan_failed) where its search runs past the `deadline`."""
        args = (
            approach_m,
            stop_back_m,
            is_open,
            constrain,
            align_axis,
            object_axis,
            object_axis_type,
        )
        self._searching += 1
        try:
            return await self._place_turns(
                arm, obj, place, target, args, approach_axis, option, scene, ee, target_turn_deg
            )
        except PlanningTimeout:
            call = _call(
                "place_object", arm, obj, place, target, approach_m, stop_back_m, is_open, constrain
            )
            note = self._budget_note("the search for a placement") + (
                ": move the other arm clear, or try another placement or arm"
            )
            refused = await self._refusal(ReceiptReason.plan_failed, arm, note)
            return Built(arm, refused=refused, call=call)
        finally:
            self._searching -= 1

    async def _place_turns(
        self,
        arm: Arm,
        obj: ObjectProxy,
        place: PlaceSpec,
        target: ObjectProxy | Pose | None,
        args: tuple[
            float, float, bool, Constrain, ArrayLike | None, ArrayLike, Literal["object", "world"]
        ],
        approach_axis: Literal["grasp", "target_z"] | Sequence[float],
        option: int | None,
        scene: Sequence[OrientedBox],
        ee: Pose | None,
        target_turn_deg: float | None,
    ) -> Built:
        if target_turn_deg not in (None, 0.0) and isinstance(target, ObjectProxy):
            turned = target.turned(math.radians(target_turn_deg))
            built = await self._place(
                arm, obj, place, turned, *args, approach_axis, option, scene, ee
            )
            note = f"{built.label}; on the target's frame turned {target_turn_deg:.0f} deg"
            return replace(built, target_turn_deg=target_turn_deg, label=note.lstrip("; "))
        built = await self._place(arm, obj, place, target, *args, approach_axis, option, scene, ee)
        open_frame = (
            target_turn_deg is None
            and isinstance(target, ObjectProxy)
            and place.target.kind == "object"
            and place.target.part is None
            and place.frame == "target"
            and not target.whole.x_known
        )
        if not open_frame or await self._within_reach(built):
            return built
        assert isinstance(target, ObjectProxy)  # an open frame's
        for turn in [t % 360.0 for t in target.whole.turns if t % 360.0 != 0.0]:
            turned = target.turned(math.radians(turn))
            other = await self._place(
                arm, obj, place, turned, *args, approach_axis, option, scene, ee, check=True
            )
            if other.refused is None:
                note = (
                    f"{other.label}; on the target's frame turned {turn:.0f} deg (its x is open: "
                    "as registered no placement is within reach)"
                )
                return replace(other, target_turn_deg=turn, label=note)
        return built

    async def _within_reach(self, built: Built) -> bool:
        """Whether a build stands: not refused, and some IK solution reaches each of its moves'
        targets (a search already checked its option's)."""
        if built.refused is not None:
            return False
        if built.option is not None:
            return True
        targets = [s.target for s in built.segments if isinstance(s, Move)]
        return all(await (await self.reachable(built.arm))(targets))

    async def _place(
        self,
        arm: Arm,
        obj: ObjectProxy,
        place: PlaceSpec,
        target: ObjectProxy | Pose | None,
        approach_m: float = 0.1,
        stop_back_m: float = 0.02,
        is_open: bool = True,
        constrain: Constrain = "auto",
        align_axis: ArrayLike | None = None,
        object_axis: ArrayLike = (1.0, 0.0, 0.0),
        object_axis_type: Literal["object", "world"] = "object",
        approach_axis: Literal["grasp", "target_z"] | Sequence[float] = "grasp",
        option: int | None = None,
        scene: Sequence[OrientedBox] = (),
        ee: Pose | None = None,
        check: bool = False,
    ) -> Built:
        """A placement of the held `obj` with the spec's target (`target`: the
        target object's proxy, a landmark's pose, or None for the robot base frame) and
        placement point: pre-place, place, then open unless `is_open` is false. With a region,
        or for an object that lands upright and has a yaw symmetry, the first option whose
        pre-place is plannable and whose place some IK solution reaches: region points from the
        center out, each as `place_ee_pose` puts the object, then turned by its
        equivalent turns, nearest first (it looks the same), or with a free turn by that about
        its approach; a directed placement (`PlaceSpec.approach`) moves onto it along its
        approach (instead of `approach_axis`); `option`: only that one (the one a
        coarse estimate chose, again on refined geometry). A spec written for the other arm and
        not mirrored, or a tilted one onto an object that names no arm, is made as written if some
        option of it is within reach, else at its mirror image (`PlaceSpec.mirror`), whose options
        follow (the label says so). A negative `stop_back_m`
        stops past the placement (`PAST_PLACE_M` at most), along the same approach as the
        pre-place. `scene`: the boxes of the objects not held; a placement written over or right
        by one of them (into a bin, onto a plate; `_under`) falls back to its mirror image only
        where that lands by the same one. `ee`: the ee pose the placement is computed from,
        `obj` held there (default: where the arm is now); the result records it
        (`Built.placed_from`). `check`: search even a single option, so that one out of reach is
        refused before anything moves (another turn of an open target frame may be tried)."""
        if stop_back_m < -PAST_PLACE_M or approach_m < 0:
            raise ValueError(
                PAST_PLACE if stop_back_m < -PAST_PLACE_M else "approach_m must be >= 0"
            )
        call = _call(
            "place_object", arm, obj, place, target, approach_m, stop_back_m, is_open, constrain
        )
        state = await self.executor.env.robot(self.executor.episode_id)
        held_from = state.arms[arm].ee if ee is None else ee
        found_at = placement(
            obj,
            place,
            target,
            arm,
            held_from,
            self.base,
            self.lateral,
            constrain=constrain,
            align_axis=align_axis,
            object_axis=object_axis,
            object_axis_type=object_axis_type,
            approach_axis=approach_axis,
            honor_front=self.executor.config.honor_front,
            tilted_turns=self.executor.config.tilted_turns,
            conventions=self.profile.poses,
        )
        obj, forms, centers, turns = found_at.obj, found_at.forms, found_at.centers, found_at.turns
        poses = found_at.ee
        unmirrored = ""
        # The mirror image stands in for a placement written over an object (into a bin, onto a
        # plate) only where it lands over that same object: else it is another placement, on
        # bare table or on something else, made silently in its stead.
        margin = self.executor.config.obstacle_margin_m
        boxes = [_grown_box(b, margin) for b in scene]
        written_over = _under(boxes, centers[0].p) if len(forms) > 1 else None
        if written_over is not None and _under(boxes, centers[1].p) != written_over:
            forms, centers = forms[:1], centers[:1]
            unmirrored = (
                f" (its mirror image for the {arm} arm lands off the object it is written over, "
                "so it is not tried)"
            )

        # A tilted symmetric object's own-axis turns are a second search, run only where the
        # whole search at its turn as measured finds nothing.
        phases = [turns[:1], turns[1:]] if found_at.fallback else [turns]
        # a spec naming no arm: its whole search as written, then its mirror image's
        last_resort = found_at.last_resort and len(forms) > 1
        groups_of = [[0], [1]] if last_resort else [list(range(len(forms)))]
        searches = [(n, ks) for ks in groups_of for n in range(len(phases))]
        held_turns = found_at.held_turns
        if held_turns:  # a pinned frame's equivalent turns: after everything as written
            searches.append((len(phases), list(range(len(forms)))))
            phases.append(held_turns)

        def options(search: int | None = None) -> Iterator[tuple[Pose, float, int]]:
            for m, (n, ks) in enumerate(searches):
                if search is not None and m != search:
                    continue
                for k in ks:
                    form, center = forms[k], centers[k]
                    each = (
                        (Pose(p=(float(p[0]), float(p[1]), float(p[2])), q=center.q), turn, k)
                        for p in region_points(center.p, form.region_m)
                        for turn in phases[n]
                    )
                    yield from islice(each, MAX_PLACE_OPTIONS)  # lazily: many region points

        searched = place.region_m > 0 or len(turns) > 1 or len(forms) > 1
        searched = searched or place.receiver is not None or bool(unmirrored)
        searched = searched or check
        tried, found = 0, next(options())
        if not searched and held_turns:
            # as written first, where it plans
            goal, turn, k = found
            fine = (await (await self.plannable(arm))([poses(goal, turn, k, approach_m)]))[0]
            if fine:
                at = await (await self.into_other(arm))([poses(goal, turn, k, stop_back_m)])
                fine = at[0] is not None
            searched = not fine
        if (
            searched
            and last_resort
            and place.region_m == 0
            and (len(turns) == 1 or found_at.fallback)
        ):
            # as written first, where it plans
            goal, turn, k = found
            fine = (await (await self.plannable(arm))([poses(goal, turn, k, approach_m)]))[0]
            if fine and place.receiver is None:
                at = await (await self.into_other(arm))([poses(goal, turn, k, stop_back_m)])
                fine = at[0] is not None
            searched = not (fine and place.receiver is None)
        if searched and found_at.fallback and place.region_m == 0 and len(forms) == 1:
            # a tilted symmetric object's own-axis turns stand in only where its turn as
            # measured does not plan
            goal, turn, k = found
            fine = (await (await self.plannable(arm))([poses(goal, turn, k, approach_m)]))[0]
            if fine and place.receiver is None:
                at = await (await self.into_other(arm))([poses(goal, turn, k, stop_back_m)])
                fine = at[0] is not None
            searched = not (fine and place.receiver is None)
        in_way, not_taken, taken_into, planned = 0.0, False, 0.0, 0
        if searched:
            plannable = await self.plannable(arm)
            into_other = await self.into_other(arm)
            receives = None
            if place.receiver is not None:
                receives = await self.receives(arm, obj, place.receiver)
            chosen: int | None = None
            checked = 0
            groups = [options(m) for m in range(len(searches))]
            if option is not None:
                groups = [islice(options(), option, option + 1)]
                tried = option
            for group in groups:
                blocked = unreceived = best = None  # best: the least-overlapping meeting
                for batch in _batches(group, BATCH):

                    def pre_places(b: list[tuple[Pose, float, int]] = batch) -> list[Pose]:
                        return [poses(goal, turn, k, approach_m) for goal, turn, k in b]

                    def places(b: list[tuple[Pose, float, int]] = batch) -> list[Pose]:
                        return [poses(goal, turn, k, stop_back_m) for goal, turn, k in b]

                    pre = await anyio.to_thread.run_sync(pre_places)
                    ok = await plannable(pre)
                    planned += sum(ok)
                    depths: list[float | None] = [None] * len(batch)
                    if any(ok):  # and the place itself within reach, clear of the other arm
                        at = await anyio.to_thread.run_sync(places)
                        good = [i for i, fine in enumerate(ok) if fine]
                        both = await into_other([pose for i in good for pose in (pre[i], at[i])])
                        for n, i in enumerate(good):
                            pair = both[2 * n : 2 * n + 2]
                            if pair[0] is not None and pair[1] is not None:
                                depths[i] = max(pair[0], pair[1])
                    clear = [d is not None and d == 0 for d in depths]
                    if receives is not None:  # and the other arm takes it there, clear of this
                        for i in [i for i, fine in enumerate(clear) if fine]:
                            into = await receives(at[i]) if checked < MAX_HANDOVER_CHECKS else None
                            checked += 1
                            if into == 0:
                                break
                            if into is not None and (best is None or into < best[2]):
                                best = (tried + i, batch[i], into)
                            if unreceived is None:
                                unreceived = (tried + i, batch[i])
                            clear[i] = False
                    if any(clear):
                        chosen = tried + clear.index(True)
                        found = batch[clear.index(True)]
                        break
                    first = next((i for i, d in enumerate(depths) if d is not None), None)
                    if first is not None and blocked is None:  # if no option is clear of it
                        blocked = (tried + first, batch[first], depths[first])
                    tried += len(batch)
                    if checked >= MAX_HANDOVER_CHECKS:
                        break
                if chosen is None and best is not None:  # the meeting pose both arms share best
                    chosen, found, taken_into = best
                elif chosen is None and unreceived is not None:  # the other arm takes it nowhere
                    chosen, found = unreceived
                    not_taken = True
                elif chosen is None and blocked is not None:
                    chosen, found, depth = blocked
                    in_way = depth or 0.0
                if chosen is not None:
                    break
            if chosen is None:
                region = f" in the {place.region_m * 100:.0f} cm region" if place.region_m else ""
                at = (
                    f" turned up to {place.free_turn_deg:.0f} deg either way"
                    if place.free_turn_deg > 0
                    else f" at any of the object's {len(turns)} equivalent turns"
                    if len(turns) > 1
                    else f" as held or at any of its shape's {len(held_turns)} other equivalent "
                    "turns"
                    if held_turns
                    else ""
                )
                mirrored = (
                    f", as written for the {place.arm or 'either'} arm or at its mirror image"
                    if len(forms) > 1
                    else ""
                )
                first = f" (the first {tried} options)" if tried >= MAX_PLACE_OPTIONS else ""
                note = (
                    f"no placement{region}{at}{mirrored} is within reach{first}: {planned} of "
                    f"{tried} pre-places planned, none with the place reached by IK"
                    if planned
                    else f"no placement{region}{at}{mirrored} is within reach{first}: none of "
                    f"{tried} pre-places planned"
                )
                note += unmirrored
                refused = await self._refusal(ReceiptReason.plan_failed, arm, note)
                return Built(arm, refused=refused, call=call)
            tried = chosen
        goal, turn, k = found
        segments: list[Segment] = [
            Move(poses(goal, turn, k, approach_m)),
            Move(
                poses(goal, turn, k, stop_back_m),
                straight=approach_m != stop_back_m,
                contact=True,
                past_m=max(0.0, -stop_back_m),
                free_fallback=True,
            ),
        ]
        if is_open:
            segments.append(Grip(1.0))
        if not searched:
            return Built(arm, tuple(segments), call=call, placed_from=held_from)
        away = float(np.linalg.norm(np.subtract(goal.p, centers[k].p))) * 100
        note = (
            f"placement option {tried + 1} ({away:.0f} cm from the center, turned "
            f"{np.degrees(turn):.0f} deg)"
        )
        if k == 1:
            written = f"the {place.arm} arm" if place.arm is not None else "either arm, as written"
            note += (
                f" at the mirror image of the placement written for {written} "
                "(as written, it is out of reach or the other arm stands in its way)"
            )
        if turn in held_turns and turn not in turns:
            note += (
                "; its shape looks the same turned so, and the frame its grasp pinned is out of "
                "reach as written"
            )
        if in_way:
            note += f"; the other arm stands in its way ({in_way * 100:.0f} cm)"
        if not_taken:
            note += "; the other arm takes it with its receiving grasp at no option tried"
        if taken_into:
            note += (
                "; the other arm takes it with its receiving grasp, clear of this arm at no "
                f"option tried (it runs {taken_into * 100:.0f} cm into it at this one, the least)"
            )
        return Built(
            arm, tuple(segments), label=note, call=call, option=tried, placed_from=held_from
        )

    async def move_by(
        self,
        arm: Arm,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        quat: Sequence[float] | None = None,
        move_axis: Literal["world", "arm"] = "world",
    ) -> Built:
        """A displacement (`displaced`), a straight move the executor clamps to
        the per-action limits, from where the arm is, or from where it is driven to when it
        stands within `LAG_M` of that (it lags its drive under a load: from where it is, a move
        smaller than the lag would land where it is already driven and make none, and any other
        would move it by the lag too). Lowering the TCP by 1 cm or more is meant to touch."""
        if move_axis == "arm" and (x or y):
            raise ValueError(ARM_AXIS_TAKES_Z)
        state = await self.executor.env.robot(self.executor.episode_id)
        arm_state = state.arms[arm]
        ee_to_tcp = self.profile.arms[arm].ee_to_tcp
        start = await self.executor.drive_ee(arm, arm_state)
        if float(np.linalg.norm(np.subtract(start.p, arm_state.ee.p))) > LAG_M:
            start = arm_state.ee  # stopped against something: from where it stopped
        target = displaced(start, (x, y, z), quat, move_axis, ee_to_tcp)
        contact = self._lowers(arm, arm_state.tcp, target)
        move = Move(target, straight=True, contact=contact, bounded=True)
        call = _call("move_by", arm, x, y, z, quat, move_axis)
        shift = np.subtract(target.p, start.p)
        far = float(np.linalg.norm(shift))
        if arm not in self.executor.pulling or quat is not None or far < 1e-6:
            return Built(arm, (move,), call=call)
        far = min(far, self.executor.config.max_translation_m)
        tcp = np.asarray(arm_state.tcp.p, dtype=float)
        line = Joint("prismatic", tcp, shift / np.linalg.norm(shift), 0.0, "guess")
        return Built(arm, (move,), call=call, follow=Follow(line, far))

    async def move_to_pose(self, arm: Arm, pose: Pose) -> Built:
        """The ee to `pose` (7-D, not the TCP) along a planned path."""
        state = await self.executor.env.robot(self.executor.episode_id)
        move = Move(pose, contact=self._lowers(arm, state.arms[arm].tcp, pose))
        return Built(arm, (move,), call=_call("move_to_pose", arm, pose.as_list()))

    async def rotate_about(
        self,
        arm: Arm,
        hinge: tuple[ArrayLike, ArrayLike],
        angle_deg: float,
        steps: int | None = None,
    ) -> Built:
        """Turn the gripper, and what it holds, about a hinge (a point on its axis and the axis'
        direction, world; right-handed) by `angle_deg`, in `steps` straight moves along the arc
        from the measured ee pose (None: one per `ARC_STEP_DEG`): opening a lid or a door held
        at its edge or handle. The turn stops at the last pose along the arc that the arm
        reaches (the label says how far)."""
        if steps is None:
            steps = arc_steps(angle_deg)
        state = (await self.executor.env.robot(self.executor.episode_id)).arms[arm]
        start = state.ee
        poses = arc_poses(hinge, start, angle_deg, steps)
        ok = await (await self.reachable(arm))(poses)
        reached = next((i for i, fine in enumerate(ok) if not fine), len(poses))
        call = _call("rotate_about", arm, angle_deg, steps)
        if reached == 0:
            note = f"no pose along the {angle_deg:.0f} deg turn is within the arm's reach"
            refused = await self._refusal(ReceiptReason.unreachable, arm, note)
            return Built(arm, refused=refused, call=call)
        moves = [Move(pose, straight=True, contact=True) for pose in poses[:reached]]
        closed = state.gripper_cmd < self.profile.arms[arm].gripper.open
        if closed and angle_deg:
            sign = 1.0 if angle_deg > 0 else -1.0
            point = np.asarray(hinge[0], dtype=float)
            direction = sign * np.asarray(hinge[1], dtype=float)
            guess = Joint("revolute", point, direction / np.linalg.norm(direction), 0.0, "guess")
            return Built(arm, tuple(moves), call=call, follow=Follow(guess, abs(angle_deg)))
        turned = angle_deg * reached / steps
        label = f"turned {turned:.0f} deg in {reached} steps"
        if reached < steps:
            label += f" (of {angle_deg:.0f}: the rest is out of reach)"
        return Built(arm, tuple(moves), label=label, call=call)

    def open_gripper(self, arm: Arm, pos: float = 1.0) -> Built:
        return Built(arm, (Grip(pos),), call=_call("open_gripper", arm, pos))

    def close_gripper(self, arm: Arm, pos: float = 0.0) -> Built:
        """A close that expects no hold: closing on nothing is no failure (to push, say)."""
        return Built(arm, (Grip(pos),), call=_call("close_gripper", arm, pos))

    async def return_home(self, arm: Arm) -> Built:
        """The ee back to the arm's home pose (`Executor.home_pose`) along a planned path."""
        home = await self.executor.home_pose(arm)
        return Built(arm, (Move(home),), call=_call("return_home", arm))

    def _releases_a_place(self, arm: Arm, segments: Sequence[Segment]) -> bool:
        """Whether the action ends by opening the gripper right after a move meant to touch (a
        placement's release): that release is held back until the place is seen to arrive."""
        if len(segments) < 2:
            return False
        last, place = segments[-1], segments[-2]
        opens = isinstance(last, Grip) and last.value >= self.profile.arms[arm].gripper.open
        return opens and isinstance(place, Move) and place.contact

    async def _release(
        self, done: ActionReceipt, releases: Mapping[Arm, Sequence[Segment]]
    ) -> ActionReceipt:
        """The held-back releases of `run`, each only where its arm got to its place: the batch
        runs open-loop, and a place that stopped short would let go of the object far from where
        it was to go. An arm that did not get there keeps holding, and the note says so."""
        state = await self.executor.env.robot(self.executor.episode_id)
        go: dict[Arm, tuple[Segment, ...]] = {}
        notes = []
        for arm, segments in releases.items():
            place = segments[-2]
            assert isinstance(place, Move)
            target = self.profile.tcp_pose(arm, place.target)
            off = float(np.linalg.norm(np.subtract(state.arms[arm].tcp.p, target.p)))
            threshold = self.profile.arms[arm].gripper.holding_threshold
            limit = self.executor.config.contact_off_path_m + place.past_m
            if off > limit and holding(state.arms[arm], threshold):
                notes.append(
                    f"{arm} kept holding: it stopped {off * 100:.1f} cm from the placement, so "
                    "it did not let go there"
                )
            else:
                go[arm] = (segments[-1],)
        if not go:
            return done.model_copy(update={"note": "; ".join(filter(None, [done.note, *notes]))})
        opened = await self.executor.run(go.get("left", ()), go.get("right", ()))
        return chained(done, opened, notes, moved_last=True)

    async def run(self, *actions: Built) -> ActionReceipt:
        """Execute one action, or two of different arms at once (`together`); a refused one
        returns its refusal and nothing moves. With `reject_repeats` an immediate repeat of the call
        that just
        failed is refused (`rejected_repeat`)."""
        arms = [action.arm for action in actions]
        if not 1 <= len(actions) <= 2 or len(set(arms)) != len(arms):
            raise ValueError(f"run takes one action per arm, got arms {arms}")
        if (rejected := await self._rejected(actions)) is not None:
            return rejected
        refused = next((action.refused for action in actions if action.refused), None)
        if refused is not None:
            self._failed = actions
            return refused
        withdrawn = [w for action in actions if (w := await self._withdraw(action)) is not None]
        cleared = await self._clear_other(actions[0]) if len(actions) == 1 else None
        if cleared is not None and cleared.reason != ReceiptReason.ok:
            self._failed = actions
            return cleared
        follow = actions[0].follow if len(actions) == 1 else None
        if follow is not None:
            done = await self._follow(actions[0].arm, follow)
        else:
            by_arm = {action.arm: action.segments for action in actions}
            releases = {arm: s for arm, s in by_arm.items() if self._releases_a_place(arm, s)}
            by_arm = {arm: s[:-1] if arm in releases else s for arm, s in by_arm.items()}
            done = await self.executor.run(by_arm.get("left", ()), by_arm.get("right", ()))
            if releases and not done.hard_failure:
                done = await self._release(done, releases)
        self._failed = actions if done.reason in _FAILED else None
        if self.executor.config.withdraw_any_move:
            # a release is withdrawn from by the arm's next action that moves it (`_withdraw`)
            self._released.difference_update(
                a.arm for a in actions if a.follow or any(isinstance(s, Move) for s in a.segments)
            )
        else:
            self._released.difference_update(arms)
        self._released.update(g.arm for g in done.grips if not g.closed)
        if cleared is not None:
            done = done.model_copy(
                update={
                    "steps_used": cleared.steps_used + done.steps_used,
                    "note": f"{done.note}; first {cleared.note}",
                }
            )
        for first in withdrawn:  # its path and steps come first in the receipt
            done = chained(first, done, [f"first {first.note}"])
        notes = [action.label for action in actions if action.label]
        notes += [n for n in (self._unpressed(actions, done), self._unchanged(actions, done)) if n]
        if not notes:
            return done
        grasp = next((action.grasp for action in actions if action.grasp is not None), None)
        return done.model_copy(update={"grasp": grasp, "note": "; ".join([done.note, *notes])})

    async def _withdraw(self, action: Built) -> ActionReceipt | None:
        """After a release, the TCP straight back along its pointing axis before the arm's next
        move (`ExecutorConfig.release_withdraw_m`); None when none is due or it cannot be made
        (the action goes on from where the arm is). With `ExecutorConfig.withdraw_any_move`,
        whatever the action, and only a straight first move back along the approach is the
        withdraw itself: a free one is planned in joint space and may sweep the open fingers
        through what they let go of (a hand that gave an object away can go home through it and
        drop it); without, an action whose first segment is a move, any such move back along the
        approach being the withdraw."""
        config = self.executor.config
        if config.withdraw_any_move:
            first = next((s for s in action.segments if isinstance(s, Move)), None)
        else:
            first = next((s for s in action.segments[:1] if isinstance(s, Move)), None)
        if action.arm not in self._released or first is None:
            return None
        if config.release_withdraw_m <= 0:
            return None
        state = await self.executor.env.robot(self.executor.episode_id)
        arm = state.arms[action.arm]
        back = -arm.tcp.matrix()[:3, 0]  # the TCP's x points from the wrist to the fingertips
        tcp = np.asarray(arm.tcp.p, dtype=float)
        boxes = [_grown_box(b, config.obstacle_margin_m) for b in self.executor.obstacles]
        out = max([_exit(b, tcp, back) for b in boxes], default=0.0)
        distance = min(max(out, config.release_withdraw_m), config.release_withdraw_max_m)
        ahead = np.subtract(self.profile.tcp_pose(action.arm, first.target).p, tcp)
        along = float(ahead @ back)
        cone = math.tan(math.radians(WITHDRAW_CONE_DEG))
        aside = float(np.linalg.norm(ahead - along * back))
        itself = first.straight or not config.withdraw_any_move
        if itself and along >= distance and aside <= along * cone:
            return None  # the move itself starts (straight) back along the approach
        ee = arm.ee
        p = np.asarray(ee.p, dtype=float) + distance * back
        target = Pose(p=(float(p[0]), float(p[1]), float(p[2])), q=ee.q)
        by_arm = {action.arm: (Move(target, straight=True),)}
        done = await self.executor.run(by_arm.get("left", ()), by_arm.get("right", ()))
        if done.reason != ReceiptReason.ok:
            return None
        note = f"withdrew {100 * distance:.0f} cm back along the approach after the release"
        return done.model_copy(update={"note": note})

    async def _follow(self, arm: Arm, follow: Follow) -> ActionReceipt:
        """Follow a pull on the part's joint (`manipulation.articulation`): straight contact
        moves of at most `follow_step_m` (a turn at most `ARC_STEP_DEG`) along the current
        estimate's arc or line from the measured ee, the joint refitted after each to the start
        and the achieved TCP positions (`estimate`, of the guess's kind: points gathered while
        turning can be the grip sliding straight along the part, so the other kind is taken
        only once a step stalls and the points fit it), until what is left of the travel asked
        (and of `follow.past` beyond it) moves the TCP less than `follow_tolerance_m`. A first
        step that makes no headway (`follow_stall`) is tried the other way once if
        `follow.turn_back` (one that slips after headway is not: the way was right); a lost hold
        (also one the executor reports `lost_hold`, if `follow_regrasp_reported`) is taken again
        where the part was last held (`follow_regrasps`), and a fit the hold slipped on is
        dropped for the guess for the rest of the pull (the points it fitted were the grip
        sliding along the part); a step the arm cannot reach ends the pull where it is. Past the
        travel asked, a stall or a slip ends the pull done (the part went as far as it goes, or
        as far as asked). The receipt reports the joint fitted (`joint`), its travel against
        the travel asked."""
        config = self.executor.config
        tolerance = config.follow_tolerance_m
        threshold = self.profile.arms[arm].gripper.holding_threshold
        state = (await self.executor.env.robot(self.executor.episode_id)).arms[arm]
        close = state.gripper_cmd
        held = holding(state, threshold)
        guess = follow.guess
        start = np.asarray(state.tcp.p, dtype=float)
        r0 = guess.radius(start) if guess.kind == "revolute" else 0.0
        total = follow.target + follow.past

        def metres(travel: float) -> float:  # the guess's travel as the TCP's distance
            return math.radians(travel) * r0 if guess.kind == "revolute" else travel

        want_m, total_m = metres(follow.target), metres(total)
        points, joint = [start], guess
        last_held = state.ee
        receipt: ActionReceipt | None = None
        notes: list[str] = []
        regrasps, reversed_, stalled, lost = config.follow_regrasps, False, False, False
        beyond, ended = False, ""  # past the travel asked; how the pull past it ended
        fits, switched = True, False  # fits are dropped once the hold slips on one
        limit = 2 * math.ceil(total_m / config.follow_step_m) + 6
        for _ in range(limit):
            now = (await self.executor.env.robot(self.executor.episode_id)).arms[arm]
            tcp = np.asarray(now.tcp.p, dtype=float)
            beyond = beyond or _left_m(joint, guess, follow.target, want_m, start, tcp) <= tolerance
            left_m = _left_m(joint, guess, total, total_m, start, tcp)
            if left_m <= tolerance:
                ended = "as far as it may go" if follow.past else ""
                break
            step_m = min(config.follow_step_m, left_m)
            if joint.kind == "revolute":
                radius = max(joint.radius(tcp), 1e-3)
                deg = min(ARC_STEP_DEG, math.degrees(step_m / radius))
                step_m = math.radians(deg) * radius
                pose = arc_poses((joint.point, joint.direction), now.ee, deg, 1)[0]
            else:
                p = np.add(now.ee.p, joint.direction * step_m).tolist()
                pose = Pose(p=(p[0], p[1], p[2]), q=now.ee.q)
            move = (Move(pose, straight=True, contact=True),)
            ran = await self.executor.run(*((move, ()) if arm == "left" else ((), move)))
            if ran.reason in (ReceiptReason.unreachable, ReceiptReason.plan_failed):
                if receipt is None:
                    return ran
                if beyond:
                    ended = f"until the arm reached no farther ({ran.reason})"
                else:
                    notes.append(f"the rest of the pull is out of reach ({ran.reason}: {ran.note})")
                    stalled = True
                break
            receipt = ran if receipt is None else chained(receipt, ran)
            retaken = config.follow_regrasp_reported and ran.reason == ReceiptReason.lost_hold
            if ran.reason in _FAILED and not retaken:
                break
            after = (await self.executor.env.robot(self.executor.episode_id)).arms[arm]
            reached = np.asarray(after.tcp.p, dtype=float)
            gained = _gained_m(joint, tcp, reached)
            slipped = held and not holding(after, threshold)
            if not slipped and ran.reason == ReceiptReason.lost_hold:
                break  # a hold the executor lost that the state does not show: as reported
            first = len(points) == 1
            if slipped and beyond:
                ended = "until the hold slipped"
                break
            if slipped:
                if regrasps == 0:
                    lost = True
                    break
                regrasps -= 1
                again = await self._regrasp(arm, last_held, close)
                receipt = chained(receipt, again)
                notes.append("the hold slipped and was taken again where it last held")
                if joint.source != "guess":  # the points were the grip sliding, not the part
                    fits = False
                    notes.append(f"the {joint.kind} fit it slipped on was dropped for the guess")
                if again.reason in _FAILED:
                    break
                after = (await self.executor.env.robot(self.executor.episode_id)).arms[arm]
                if not holding(after, threshold):
                    lost = True
                    break
                reached = np.asarray(after.tcp.p, dtype=float)
            if gained < config.follow_stall * step_m:  # headway says the way is right
                if first and not reversed_ and follow.turn_back:
                    reversed_ = True
                    guess = Joint(guess.kind, guess.point, -guess.direction, 0.0, guess.source)
                    start, points, joint = reached, [reached], guess
                    notes.append("the first step made no headway: the pull went the other way")
                    continue
                if not slipped and beyond:
                    ended = "until the part stopped following"
                    break
                if not slipped:
                    other = estimate(points, guess) if fits and not switched else joint
                    if other.kind == joint.kind:
                        stalled = True
                        break
                    switched, joint = True, other
                    notes.append(
                        f"the part stopped following: its motion fits a {other.kind} joint"
                    )
                    continue
            points.append(reached)
            last_held = after.ee
            joint = estimate(points, guess, switched) if fits else estimate(points[:2], guess)
        if receipt is None:  # nothing was left to pull
            return await self.executor.run()
        final = (await self.executor.env.robot(self.executor.episode_id)).arms[arm]
        slid = lost or ended == "until the hold slipped"
        tcp = points[-1] if slid else np.asarray(final.tcp.p, dtype=float)  # where it was held
        left_m = _left_m(joint, guess, follow.target, want_m, start, tcp)
        report = _report(joint, guess, follow.target, want_m, start, tcp, len(points))
        unit = "deg" if report.kind == "revolute" else "m"
        digits = 0 if unit == "deg" else 3
        fitted = (
            f"followed the pull on a {report.kind} joint ({report.source}"
            + (f", radius {report.radius_m * 1000:.0f} mm" if report.radius_m else "")
            + f", residual {report.residual_m * 1000:.1f} mm over {report.points} points): "
            f"{report.travel:.{digits}f} of {report.target:.{digits}f} {unit}"
        )
        if beyond and follow.past:
            fitted += f" (on past the {report.target:.{digits}f} {unit} asked {ended})"
        reason = receipt.reason
        if lost:
            reason = ReceiptReason.lost_hold
            notes.append(f"the hold was lost (opening {final.gripper or 0.0:.3f})")
        elif beyond and reason == ReceiptReason.lost_hold:  # it slipped past the travel asked
            reason = ReceiptReason.ok
        elif receipt.ok and (stalled or left_m > tolerance):
            reason = ReceiptReason.stopped_short
            if stalled and not notes:
                notes.append("the part stopped following the gripper")
        note = "; ".join(filter(None, [receipt.note, fitted, *notes]))
        return receipt.model_copy(update={"reason": reason, "joint": report, "note": note})

    def _unpressed(self, actions: tuple[Built, ...], done: ActionReceipt) -> str:
        """A press's grasp whose push nothing stopped: the fingers reached the target, so
        nothing there gave way under them."""
        pressed = {p.arm for p in done.presses}
        idle = [a.arm for a in actions if a.press and a.refused is None and a.arm not in pressed]
        if not idle or done.reason not in (ReceiptReason.ok, ReceiptReason.clamped):
            return ""
        short = self.executor.config.press_m * 100
        return (
            f"{' and '.join(idle)}: the closed fingers reached the press target without being "
            f"stopped, so nothing was pressed there (a press is a push that what it meets stops "
            f"at least {short:.0f} cm short of a target past the surface): aim deeper or "
            "elsewhere on the part"
        )

    def _unchanged(self, actions: tuple[Built, ...], done: ActionReceipt) -> str:
        """For one arm's move meant to touch that ended holding and pressing nothing: how many
        times before such a move ended the same way, with the same target and the TCP where it
        is now (within the repeat tolerances), said from the third on."""
        if len(actions) != 1 or done.commanded is None or done.achieved is None:
            return ""
        action, config = actions[0], self.executor.config
        touches = any(isinstance(s, Move) and s.contact for s in action.segments)
        holding = done.gripper is not None and done.gripper.holding
        if not touches or holding or done.presses or action.arm != done.arm:
            return ""
        ended = (
            action.arm,
            done.commanded.tcp,
            done.achieved.tcp,
            ReceiptReason(done.reason).value,
        )

        def alike(a: Pose, b: Pose) -> bool:
            far = float(np.linalg.norm(np.subtract(a.p, b.p)))
            return far <= config.repeat_tolerance_m * 2 and (
                float(quat_angle_deg(a.q, b.q)) <= config.repeat_tolerance_deg
            )

        same = sum(
            arm == ended[0]
            and reason == ended[3]
            and alike(aimed, ended[1])
            and alike(at, ended[2])
            for arm, aimed, at, reason in self._touches
        )
        self._touches.append(ended)
        if same < 2:
            return ""
        x, y, z = ended[2].p
        return (
            f"this move has now ended the same way {same + 1} times (the {action.arm} TCP at "
            f"({x:.3f}, {y:.3f}, {z:.3f}), {ended[3]}, holding and pressing nothing): repeating "
            "it changes nothing; change where or how it touches, or use another tool"
        )

    async def _rejected(self, actions: tuple[Built, ...]) -> ActionReceipt | None:
        """With `reject_repeats`, the refusal of an immediate repeat of the call that just failed;
        else None."""
        config = self.executor.config
        if not config.reject_repeats or self._failed is None or not self._repeats(actions):
            return None
        state = await self.executor.env.robot(self.executor.episode_id)
        arms = [action.arm for action in actions]
        return ActionReceipt(
            reason=ReceiptReason.rejected_repeat,
            arm=arms[0] if len(arms) == 1 else None,
            steps_left=steps_left(state),
            note=(
                "rejected_repeat: this action repeats the one that just failed (the same "
                f"call, or targets within {config.repeat_tolerance_m * 1000:.0f} mm and "
                f"{config.repeat_tolerance_deg:.0f} deg); change it"
            ),
        )

    async def path_moves(
        self, arm: Arm, frames: Sequence[Array], path: PathSpec
    ) -> tuple[Segment, ...]:
        """The straight contact moves `follow_path` sends one arm's ee along to carry what it
        holds along `path` (every waypoint's legs, from where the arm is now; no joint tail is
        followed): what `check_reach` plans to see whether the carry is in reach."""
        state = await self.executor.env.robot(self.executor.episode_id)
        tcp, ee = state.arms[arm].tcp, state.arms[arm].ee.matrix()
        if path.arm not in (None, arm):  # fitted at the other arm: its mirror image
            frame, path, _, _ = choose_mirrored(frames, path, [tcp], self.lateral)
        else:
            frame, _, _ = choose_frame(frames, path, [tcp])
        worlds = motions(frame, path)
        return tuple(
            Move(Pose.from_matrix(m @ ee), straight=True, contact=True)
            for moves in legs(worlds, [tcp.matrix()])
            for m in moves
        )

    async def follow_path(
        self,
        arms: Sequence[Arm],
        frames: Sequence[Array],
        path: PathSpec,
        articulated: bool = False,
    ) -> ActionReceipt:
        """Carry what `arms` hold (one arm, or both holding one object) along a held path
        (`manipulation.paths`): `frames` the body frames (world) of what they hold, as measured
        now; the one the path's start fits the holding TCPs best is taken. Every gripper moves by
        the path's motion from where it holds, together, keeping its hold (no arm retreats along
        its grasp); each waypoint's leg is straight contact moves along the screw between
        waypoints, planned before it moves (the executor's fallbacks: IK along the line, an
        orientation-locked plan, a replan). A leg that cannot be planned ends the path there
        (`stopped_short`; the first: its refusal, nothing moved); a hold lost ends it
        (`lost_hold`, unless it slips at the last waypoint from a hold that may: a pull's).

        One arm's path whose end moves on one joint (`joint_tail`; `follow_joint_paths`) is a
        held part's pull there: that end is followed on the joint from where the gripper is
        (`_follow`: the joint refitted to the achieved TCP positions, a slipped hold taken again)
        instead of replayed from the path's start, where a frame a few millimeters off makes the
        arm fight the hinge. On a part known to move so (a turn; a slide when `articulated`, as
        the caller's scene knows it, or held by a pull's close) it goes on past the last waypoint
        until the part stops following, by at most `follow_past` of the end's travel.

        The receipt's `path` says how many waypoints were reached, each arm's deviation from each
        and whether, and where, a hold was lost; a followed end's `joint` the joint and its
        travel against the end's (degrees or meters).

        One arm follows a path fitted at the other arm (`PathSpec.arm`) at its mirror image
        (`choose_mirrored`); a path naming no arm as written, else, when its first leg is out of
        reach (refused, nothing moved), at its mirror image."""
        if not 1 <= len(arms) <= 2 or len(set(arms)) != len(arms):
            raise ValueError(f"follow_path takes one arm or two different ones, got {list(arms)}")
        mirror = len(arms) == 1 and path.arm not in (None, arms[0])
        receipt = await self._follow_path(arms, frames, path, articulated, mirror)
        refused = (
            receipt.reason in _FAILED
            and receipt.steps_used == 0
            and receipt.path is not None
            and receipt.path.reached == 0
        )
        if not (refused and len(arms) == 1 and path.arm is None):
            return receipt
        failed, self._failed = self._failed, None
        other = await self._follow_path(arms, frames, path, articulated, True)
        if other.reason in _FAILED and other.steps_used == 0:
            self._failed = failed
            note = f"{receipt.note}; its mirror image (as the other arm's) is out of reach too"
            return receipt.model_copy(update={"note": note})
        note = (
            f"as written its first leg is out of reach ({receipt.reason}): followed its mirror "
            f"image instead (the path as the other arm's); {other.note}"
        )
        return other.model_copy(update={"note": note})

    async def _follow_path(
        self,
        arms: Sequence[Arm],
        frames: Sequence[Array],
        path: PathSpec,
        articulated: bool,
        mirror: bool,
    ) -> ActionReceipt:
        """`follow_path` with the path as written, or at its mirror image (`mirror`)."""
        built = tuple(Built(arm, call=_call("follow_path", arm, path.poses)) for arm in arms)
        if (rejected := await self._rejected(built)) is not None:
            return rejected
        config = self.executor.config
        state = await self.executor.env.robot(self.executor.episode_id)
        threshold = {arm: self.profile.arms[arm].gripper.holding_threshold for arm in arms}
        empty = [arm for arm in arms if not holding(state.arms[arm], threshold[arm])]
        if empty:
            arm = empty[0]
            opening = state.arms[arm].gripper
            shown = "not measured" if opening is None else f"{opening:.3f}"
            note = (
                f"the {arm} gripper holds nothing (opening {shown}): a path carries what the "
                "arm holds; grasp it first"
            )
            self._failed = built
            return ActionReceipt(
                reason=ReceiptReason.invalid_args, arm=arm, steps_left=steps_left(state), note=note
            )
        tcps = {arm: state.arms[arm].tcp.matrix() for arm in arms}
        ees = {arm: state.arms[arm].ee.matrix() for arm in arms}
        holding_tcps = [state.arms[a].tcp for a in arms]
        if mirror:
            frame, path, start_m, start_deg = choose_mirrored(
                frames, path, holding_tcps, self.lateral
            )
        else:
            frame, start_m, start_deg = choose_frame(frames, path, holding_tcps)
        worlds = motions(frame, path)
        steps = legs(worlds, list(tcps.values()))
        tail = None
        if len(arms) == 1 and config.follow_joint_paths:
            tail = joint_tail(worlds, tcps[arms[0]])
        end = len(worlds) if tail is None else tail.start
        deviation: dict[Arm, list[float]] = {arm: [] for arm in arms}
        turn: dict[Arm, list[float]] = {arm: [] for arm in arms}
        holds = {arm: PathHold(kept=True) for arm in arms}
        receipt: ActionReceipt | None = None
        notes: list[str] = []
        reached, stopped = 0, False
        prefix = zip(worlds[:end], steps[:end], strict=True)
        for index, (world, moves) in enumerate(prefix, start=1):
            segments = {
                arm: tuple(
                    Move(Pose.from_matrix(m @ ees[arm]), straight=True, contact=True) for m in moves
                )
                for arm in arms
            }
            ran = await self.executor.run(segments.get("left", ()), segments.get("right", ()))
            if ran.steps_used == 0 and ran.reason in _FAILED:  # refused: nothing moved
                if receipt is None:
                    return self._refused_path(built, ran, len(worlds), holds)
                notes.append(
                    f"waypoint {index} of {len(worlds)} could not be planned ({ran.reason}: "
                    f"{ran.note}): the path ends at waypoint {reached}"
                )
                stopped = True
                break
            receipt = ran if receipt is None else chained(receipt, ran)
            after = await self.executor.env.robot(self.executor.episode_id)
            for arm in arms:
                target = world @ tcps[arm]
                measured = after.arms[arm].tcp
                deviation[arm].append(float(np.linalg.norm(np.subtract(measured.p, target[:3, 3]))))
                turn[arm].append(_angle_deg(measured.matrix()[:3, :3], target[:3, :3]))
                if holds[arm].kept and not holding(after.arms[arm], threshold[arm]):
                    p = measured.p
                    holds[arm] = PathHold(kept=False, lost_at=index, tcp=(p[0], p[1], p[2]))
            slipped = [arm for arm, hold in holds.items() if not hold.kept]
            if slipped or ran.reason in _FAILED:
                ending = index == len(worlds) and ran.reason not in _FAILED
                if ending and all(arm in self.executor.lenient for arm in slipped):
                    reached = index  # a pull's hold may slip at its end
                break
            reached = index
        followed = False
        ran_ok = receipt is None or receipt.reason not in _FAILED
        if tail is not None and not stopped and reached == end and ran_ok:
            arm = arms[0]
            known = tail.joint.kind == "revolute" or articulated or arm in self.executor.pulling
            past = config.follow_past * tail.target if known else 0.0
            pull = Follow(tail.joint, tail.target, turn_back=False, past=past)
            pulled = await self._follow(arm, pull)
            if pulled.steps_used == 0 and pulled.reason in _FAILED:  # refused: nothing moved
                if receipt is None:
                    return self._refused_path(built, pulled, len(worlds), holds)
                notes.append(
                    f"waypoint {end + 1} of {len(worlds)} could not be planned ({pulled.reason}: "
                    f"{pulled.note}): the path ends at waypoint {reached}"
                )
                stopped = True
            else:
                followed = True
                receipt = pulled if receipt is None else chained(receipt, pulled)
                targets = [worlds[i] @ tcps[arm] for i in range(end, len(worlds))]
                done = _tail_reached(tail, pulled, config.follow_tolerance_m)
                went = [Pose.from_matrix(tcps[arm] if end == 0 else worlds[end - 1] @ tcps[arm])]
                went += pulled.tcp_path.get(arm, [])
                for target in targets[:done]:
                    off, near = _nearest(went, target[:3, 3])
                    deviation[arm].append(off)
                    turn[arm].append(_angle_deg(near.matrix()[:3, :3], target[:3, :3]))
                reached = end + done
                now = (await self.executor.env.robot(self.executor.episode_id)).arms[arm]
                if not pulled.ok and not holding(now, threshold[arm]):
                    p, at = now.tcp.p, min(reached + 1, len(worlds))
                    holds[arm] = PathHold(kept=False, lost_at=at, tcp=(p[0], p[1], p[2]))
        assert receipt is not None  # the first leg either ran or returned its refusal
        reason = receipt.reason
        lost = [arm for arm, hold in holds.items() if not hold.kept]
        for arm in lost:
            hold = holds[arm]
            assert hold.lost_at is not None and hold.tcp is not None
            at = _xyz(hold.tcp)
            last = hold.lost_at == len(worlds) and arm in self.executor.lenient and not followed
            if last:
                notes.append(f"the {arm} hold slipped at the last waypoint at {at} (a pull's)")
            else:
                reason = ReceiptReason.lost_hold
                notes.append(f"the {arm} arm lost its hold after waypoint {hold.lost_at} at {at}")
        if not lost and receipt.ok:
            if stopped:
                reason = ReceiptReason.stopped_short
            elif not followed:  # a followed end says how far it went in its joint
                off = max(deviation[arm][-1] for arm in arms)
                if off > config.stop_tolerance_m:
                    reason = ReceiptReason.stopped_short
                    notes.append(f"it ended {off * 1000:.0f} mm from the path's last waypoint")
        report = PathReport(
            waypoints=len(worlds),
            reached=reached,
            deviation_m=deviation,
            angle_deg=turn,
            holds=holds,
            start_m=start_m,
            start_deg=start_deg,
        )
        summary = f"followed the held path to waypoint {reached} of {len(worlds)}"
        if mirror:
            summary += f" at its mirror image (fitted at the {_OTHER[arms[0]]} arm)"
        if followed and tail is not None:
            summary += f", from waypoint {end} on its {tail.joint.kind} joint"
        note = "; ".join(filter(None, [summary, receipt.note, *notes]))
        self._failed = built if reason in _FAILED else None
        return receipt.model_copy(update={"reason": reason, "path": report, "note": note})

    def _refused_path(
        self,
        built: tuple[Built, ...],
        refused: ActionReceipt,
        waypoints: int,
        holds: dict[Arm, PathHold],
    ) -> ActionReceipt:
        """A path whose first leg was refused: the refusal, nothing moved, no waypoint reached."""
        self._failed = built
        report = PathReport(waypoints=waypoints, reached=0, holds=holds)
        return refused.model_copy(update={"path": report})

    async def _regrasp(self, arm: Arm, ee: Pose, close: float) -> ActionReceipt:
        """Open, go back to the ee pose `ee` where the part was last held and close there as a
        pull (closing on nothing is `empty_grasp`)."""
        segments = (
            Grip(1.0),
            Move(ee, straight=True, contact=True),
            Grip(close, grasp=True, pull=True),
        )
        return await self.executor.run(*((segments, ()) if arm == "left" else ((), segments)))

    def rehearsal(self, state: RobotState) -> "ActionApi":
        """This API over the robot as `state` has it, for building and `check_reach` only: the
        same planner, profile, configuration, table and obstacles, so what it builds and plans
        is what a call would once the robot is so (a grasp, then a carry, then a placement,
        each from where the last leaves it: `Executor.after`); it executes nothing."""
        ex = self.executor
        executor = Executor(
            _Rehearsed(state), ex.episode_id, ex.planner, ex.profile, ex.config,
            table_top_z=ex.table_top_z,
        )  # fmt: skip
        executor.obstacles = list(ex.obstacles)
        executor.held = {arm: list(boxes) for arm, boxes in ex.held.items()}
        executor.lenient, executor.pulling = set(ex.lenient), set(ex.pulling)
        executor.placed = set(ex.placed)
        rehearsed = ActionApi(executor)
        rehearsed.deadline, rehearsed.planning = self.deadline, self.planning  # the call's
        return rehearsed

    def forget(self) -> None:
        """The next run is not taken for a repeat of the failed one (a caller's own try of
        another candidate within one call)."""
        self._failed = None

    async def _clear_other(self, action: Built) -> ActionReceipt | None:
        """Move the other arm home first when `action`'s arm would run into it at one of its
        targets and home clears the way; not when it holds something or offers it (the gripper
        closed, as in a handover), is already home or has no home. Returns that move's receipt,
        its note saying why, or None."""
        targets = [s.target for s in action.segments if isinstance(s, Move)]
        if not targets:
            return None
        state = await self.executor.env.robot(self.executor.episode_id)
        other = next(a for a in state.arms if a != action.arm)
        spec = self.profile.arms[other]
        if spec.home is None or state.arms[other].gripper_cmd < spec.gripper.open:
            return None
        home = await self.executor.home_pose(other)
        if float(np.linalg.norm(np.subtract(state.arms[other].ee.p, home.p))) <= HOME_M:
            return None
        depths = [d for d in await (await self.into_other(action.arm))(targets) if d is not None]
        if not depths or max(depths) == 0:
            return None
        homed = await self.into_other(action.arm, other_joints=spec.home)
        if any(d for d in await homed(targets) if d is not None):
            return None
        done = await self.executor.run(
            *(((Move(home),), ()) if other == "left" else ((), (Move(home),)))
        )
        why = (
            f"moved the {other} arm home: the {action.arm} arm would run "
            f"{max(depths) * 100:.0f} cm into it ({done.note})"
        )
        return done.model_copy(update={"note": why})

    async def hold(self, steps: int) -> ActionReceipt:
        """`steps` actions keeping every drive target and gripper command (it ends a repeat)."""
        self._failed = None
        return await self.executor.hold(steps)

    def _repeats(self, actions: tuple[Built, ...]) -> bool:
        """Whether `actions` repeat the last failed ones: arm by arm the same call, or segments
        matching within the repeat tolerances."""
        failed, config = self._failed, self.executor.config
        assert failed is not None
        if sorted(a.arm for a in failed) != sorted(a.arm for a in actions):
            return False
        before = {action.arm: action for action in failed}
        for action in actions:
            old = before[action.arm]
            if action.call and action.call == old.call:
                continue
            if action.refused or old.refused or len(action.segments) != len(old.segments):
                return False
            for new, was in zip(action.segments, old.segments, strict=True):
                if isinstance(new, Grip) and isinstance(was, Grip):
                    if abs(new.value - was.value) > 1e-6:
                        return False
                elif isinstance(new, Move) and isinstance(was, Move):
                    far = float(np.linalg.norm(np.subtract(new.target.p, was.target.p)))
                    turn = float(quat_angle_deg(new.target.q, was.target.q))
                    if far > config.repeat_tolerance_m or turn > config.repeat_tolerance_deg:
                        return False
                else:
                    return False
        return True

    async def check_reach(self, *actions: Built) -> Reach:
        """Plan the actions without executing (nothing moves, no step is used): whether they
        would run, and the reason their receipt would give."""
        refused = next((action.refused for action in actions if action.refused), None)
        if refused is not None:
            return Reach.of(refused)
        by_arm = {action.arm: action.segments for action in actions}
        return reach(await self.executor.plan(by_arm.get("left", ()), by_arm.get("right", ())))

    @property
    def base(self) -> Pose:
        """The robot base frame (`robot_base`)."""
        return robot_base(self.profile)

    @property
    def lateral(self) -> tuple[float, float, float]:
        """The robot's lateral axis in the world (the base frame's y, as its x points ahead):
        the normal of the sagittal plane that grasps and placements are mirrored across."""
        y = self.base.matrix()[:3, 1]
        return (float(y[0]), float(y[1]), float(y[2]))

    async def plannable(self, arm: Arm) -> Plannable:
        """Which ee poses `arm` can plan to from its drive target, in batches of `BATCH`: those
        with a path the executor would take, no joint travelling farther than its
        `max_joint_travel_rad` (a wrap) or past its stop."""
        start = (await self.executor.env.robot(self.executor.episode_id)).arms[arm].joints_cmd
        base = self.executor.bases[arm]
        planner = self.executor.planner
        limit = self.executor.config.max_joint_travel_rad
        spec = self.profile.arms[arm]

        def taken(path: list[list[float]] | None) -> bool:
            if path is None:
                return False
            joints = np.asarray([start, *path[1:]], dtype=float)
            travel = float(np.abs(np.diff(joints, axis=0)).sum(axis=0).max())
            return travel <= limit and spec.past_stops(joints) is None

        # A search asks again for goals it planned before (a grasp's candidates once per
        # candidate passed over): from the same start the answer is the same, so it is kept.
        known: dict[tuple[float, ...], bool] = {}
        log = self.planning

        async def plannable(poses: list[Pose]) -> list[bool]:
            keys = [_pose_key(pose) for pose in poses]
            asked = [pose for pose, key in zip(poses, keys, strict=True) if key not in known]
            log.cached += len(poses) - len(asked)
            chunks = iter(asked)
            while chunk := list(islice(chunks, BATCH)):
                self._in_time()
                request = BatchPlanRequest(arm=arm, base=base, start=start, goals=chunk)
                began = time.monotonic()
                results = await planner.plan_batch(request)
                log.seconds += time.monotonic() - began
                log.plans += len(chunk)
                for pose, result in zip(chunk, results, strict=True):
                    known[_pose_key(pose)] = result.reason == "ok" and taken(result.path)
            return [known[key] for key in keys]

        return plannable

    async def reachable(self, arm: Arm) -> Plannable:
        """Which ee poses some IK solution of `arm` within its stops reaches (seeded with its
        drive target)."""
        seed = (await self.executor.env.robot(self.executor.episode_id)).arms[arm].joints_cmd
        base = self.executor.bases[arm]
        spec = self.profile.arms[arm]

        async def reachable(poses: list[Pose]) -> list[bool]:
            ok: list[bool] = []
            chunks = iter(poses)
            while chunk := list(islice(chunks, MAX_GOALS)):
                request = IKRequest(arm=arm, base=base, goals=chunk, seed=seed)
                ok += [
                    solution.joints is not None and spec.past_stops([solution.joints]) is None
                    for solution in await self._ik(request)
                ]
            return ok

        return reachable

    async def receives(
        self, arm: Arm, obj: ObjectProxy, grasp: GraspSpec, approach_m: float = 0.1
    ) -> Callable[[Pose], Awaitable[float | None]]:
        """With `arm` (holding `obj`) moved to each ee pose, how far the other arm taking the
        object with `grasp` as `grasp_object` would runs into `arm` standing there (m, `into_other`;
        its grasp and pre-grasp, the least over the candidates clear of this gripper with a
        plannable pre-grasp at `approach_m`, `grasp_object`'s default): 0 clear, None when no
        candidate is plannable or `arm` reaches no such pose. Two grippers meeting stand close, so
        a meeting pose is chosen by the least, not by the first clear one."""
        state = await self.executor.env.robot(self.executor.episode_id)
        other = next(a for a in state.arms if a != arm)
        ee, tcp = state.arms[arm].ee.matrix(), state.arms[arm].tcp.matrix()
        preference = self.profile.arms[other].grasp
        if preference is None:
            raise ValueError(f"the {self.profile.name} profile has no grasp preference for {other}")
        offset = self.profile.arms[other].ee_to_tcp[0]
        plannable = await self.plannable(other)
        base = self.executor.bases[arm]
        seed = state.arms[arm].joints_cmd
        spec = self.profile.arms[arm]

        async def receives(pose: Pose) -> float | None:
            moved = pose.matrix() @ np.linalg.inv(ee)
            there = obj.at(moved @ obj.pose)
            ours = Pose.from_matrix(moved @ tcp)
            request = IKRequest(arm=arm, base=base, goals=[pose], seed=seed)
            joints = (await self._ik(request))[0].joints
            if joints is None or spec.past_stops([joints]) is not None:
                return None
            candidates = grasp_candidates(
                grasp,
                there,
                other,
                other_tcp=ours.p,
                lateral=self.lateral,
                span_m=self.profile.arms[other].gripper.span_m,
            )
            proxy = there.with_frames({CONTACT: [c.matrix for c in candidates]})
            ids = [
                i
                for i, c in enumerate(candidates)
                if not c.air and clearance(there.pose @ c.matrix, ours, approach_m) >= CLEAR_M
            ]
            into = await self.into_other(other, joints)
            least: float | None = None
            while ids:
                chosen = await choose_grasp(
                    proxy,
                    other,
                    approach_m,
                    0.0,
                    ids,
                    preference,
                    offset,
                    plannable,
                    self.profile.poses,
                )
                if chosen is None:
                    break
                depths = await into([chosen.pre, chosen.grasp])
                if None not in depths:
                    depth = max(d for d in depths if d is not None)
                    least = depth if least is None else min(least, depth)
                    if depth == 0:
                        break
                ids = [i for i in ids if i != chosen.frame]
            return least

        return receives

    async def into_other(
        self, arm: Arm, other_joints: Sequence[float] | None = None
    ) -> Callable[[list[Pose]], Awaitable[list[float | None]]]:
        """How far `arm` at each ee pose (an IK solution seeded with its drive target) would run
        into the other arm where that stands (its drive target, or `other_joints`): the depth
        (m) by which their collision spheres come closer than `ARM_GAP_M`, 0 when clear, None
        when no IK solution within the arm's stops reaches the pose."""
        state = await self.executor.env.robot(self.executor.episode_id)
        other = next(a for a in state.arms if a != arm)
        planner, bases = self.executor.planner, self.executor.bases
        joints = list(other_joints) if other_joints is not None else state.arms[other].joints_cmd
        request = FKRequest(arm=other, base=bases[other], joints=[joints], spheres=True)
        theirs = np.asarray((await planner.fk(request)).spheres[0], dtype=float).reshape(-1, 4)
        seed = state.arms[arm].joints_cmd
        spec = self.profile.arms[arm]

        async def into_other(poses: list[Pose]) -> list[float | None]:
            depths: list[float | None] = []
            chunks = iter(poses)
            while chunk := list(islice(chunks, MAX_GOALS)):
                request = IKRequest(arm=arm, base=bases[arm], goals=chunk, seed=seed)
                solved = [
                    q if q is not None and spec.past_stops([q]) is None else None
                    for q in (solution.joints for solution in await self._ik(request))
                ]
                found = [q for q in solved if q is not None]
                spheres = iter([])
                if found:
                    fk = FKRequest(arm=arm, base=bases[arm], joints=found, spheres=True)
                    spheres = iter((await planner.fk(fk)).spheres)
                for q in solved:
                    depths.append(None if q is None else _overlap(next(spheres), theirs))
            return depths

        return into_other

    async def _ik(self, request: IKRequest) -> list[IKSolution]:
        """The planner's IK for a search: counted in `planning`, stopped past the deadline."""
        self._in_time()
        began = time.monotonic()
        solutions = await self.executor.planner.ik(request)
        self.planning.seconds += time.monotonic() - began
        self.planning.iks += len(request.goals)
        return solutions

    def _lowers(self, arm: Arm, tcp: Pose, target: Pose) -> bool:
        return self.profile.tcp_pose(arm, target).p[2] <= tcp.p[2] - PRESS_M

    async def _refusal(self, reason: ReceiptReason, arm: Arm, note: str) -> ActionReceipt:
        state = await self.executor.env.robot(self.executor.episode_id)
        return ActionReceipt(reason=reason, arm=arm, steps_left=steps_left(state), note=note)


def _left_m(
    joint: Joint, guess: Joint, target: float, want_m: float, start: Array, tcp: Array
) -> float:
    """What is left of a followed pull, as TCP distance along `joint`: the asked travel in the
    joint's units (degrees of the guess's turn, or the guess's arc length `want_m` on a line),
    less the travel from `start` to `tcp`."""
    done = joint.travel(start, tcp)
    if joint.kind == "revolute":
        radius = joint.radius(tcp)
        want = target if guess.kind == "revolute" else math.degrees(want_m / max(radius, 1e-3))
        return math.radians(want - done) * radius
    return want_m - done


def _gained_m(joint: Joint, before: Array, after: Array) -> float:
    """How far the TCP went along `joint` from `before` to `after` (the arc length about a
    revolute joint)."""
    travel = joint.travel(before, after)
    return math.radians(travel) * joint.radius(after) if joint.kind == "revolute" else travel


def _report(
    joint: Joint,
    guess: Joint,
    target: float,
    want_m: float,
    start: Array,
    tcp: Array,
    points: int,
) -> JointReport:
    travel = joint.travel(start, tcp)
    radius = joint.radius(tcp) if joint.kind == "revolute" else None
    if joint.kind == guess.kind:
        want = target
    elif joint.kind == "revolute":
        assert radius is not None
        want = math.degrees(want_m / max(radius, 1e-3))
    else:
        want = want_m
    x, y, z = (float(v) for v in joint.point)
    u, v, w = (float(c) for c in joint.direction)
    return JointReport(
        kind=joint.kind,
        point=(x, y, z),
        direction=(u, v, w),
        radius_m=radius,
        residual_m=joint.residual_m,
        source=joint.source,
        travel=travel,
        target=want,
        points=points,
    )


def _nearest(poses: Sequence[Pose], point: Array) -> tuple[float, Pose]:
    """How far `point` lies from the polyline through `poses`' positions, and the pose nearest
    it."""
    at = np.array([p.p for p in poses])
    best = float(np.linalg.norm(at[0] - point))
    for a, b in pairwise(at):
        d = b - a
        s = float(np.clip((point - a) @ d / max(float(d @ d), 1e-12), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(a + s * d - point)))
    near = min(poses, key=lambda p: float(np.linalg.norm(np.subtract(p.p, point))))
    return best, near


def _tail_reached(tail: JointTail, pulled: ActionReceipt, tolerance_m: float) -> int:
    """How many of a joint end's waypoints a followed pull reached: all if it went the whole
    way, else those whose travel its own travel covers (as a share of the travel asked, within
    the tolerance)."""
    if pulled.ok:
        return len(tail.travels)
    report = pulled.joint
    if report is None or not report.target:
        return 0
    covered = report.travel / report.target * tail.target
    if tail.joint.kind == "revolute":
        slack = math.degrees(tolerance_m / max(report.radius_m or 0.1, 1e-3))
    else:
        slack = tolerance_m
    return sum(1 for t in tail.travels if t <= covered + slack)


def arc_steps(angle_deg: float) -> int:
    """How many straight moves a hinge turn takes by default: one per `ARC_STEP_DEG`."""
    return max(1, math.ceil(round(abs(angle_deg) / ARC_STEP_DEG, 6)))


def _reach_m(built: Built, home: Array) -> float:
    """How far a built grasp's last move goes from the arm's home ee position (inf: refused)."""
    targets = [s.target for s in built.segments if isinstance(s, Move)]
    if built.refused is not None or not targets:
        return math.inf
    return float(np.linalg.norm(np.asarray(targets[-1].p) - home))


def _angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """The angle (deg) of the rotation between two rotation matrices."""
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@dataclass(frozen=True)
class Placement:
    """What `ActionApi.place_object` computes for a placement before it plans anything: the held
    object (`obj`, faced as the spec's `x_toward` says), the spec's forms for the arm (as
    written, then its mirror image where it may fall back to it), each form's center (the
    `target_pose` of the placement point) and the turns tried about it; `ee` gives the
    placement's ee pose for an option (`goal` a center or a region point, a turn, the
    form's index) at `distance` back along its approach (`approach_m` or `stop_back_m`)."""

    obj: ObjectProxy
    forms: list[PlaceSpec]
    centers: list[Pose]
    turns: list[float]
    ee: Callable[[Pose, float, int, float], Pose]
    fallback: bool = False
    """The turns are a tilted symmetric object's about its own axis (`tilted_turns`): the first
    (as measured) is searched first; the others only where that search finds nothing."""
    last_resort: bool = False
    """The second form is the mirror image of a spec that names no arm: searched only where the
    whole search as written finds nothing."""
    held_turns: list[float] = field(default_factory=list)
    """The turns a held object's shape looks the same under whose frame a choice pinned
    (`ObjectProxy.chosen`): searched only where everything above finds nothing."""


def placement(
    obj: ObjectProxy,
    place: PlaceSpec,
    target: ObjectProxy | Pose | None,
    arm: Arm,
    ee: Pose,
    base: Pose,
    lateral: ArrayLike,
    *,
    constrain: Constrain = "auto",
    align_axis: ArrayLike | None = None,
    object_axis: ArrayLike = (1.0, 0.0, 0.0),
    object_axis_type: Literal["object", "world"] = "object",
    approach_axis: Literal["grasp", "target_z"] | Sequence[float] = "grasp",
    honor_front: bool = True,
    tilted_turns: bool = True,
    conventions: PoseConventions,
) -> Placement:
    """The placement `place_object` makes of the held `obj` (where its grasp holds it, the arm's
    ee at `ee`) with these call arguments (see `ActionApi.place_object`): one function, so that a
    check outside the API (a dry run, `demo_relation`) computes what a call would command.
    `honor_front` (`ExecutorConfig.honor_front`): a front the spec names that the object
    resolves makes it look different turned, so the spec's turn is taken as written;
    `tilted_turns` (`ExecutorConfig.tilted_turns`): a symmetric object placed tilted
    (`relative`) is offered at its equivalent turns about its own axis. `conventions`: the
    embodiment profile's (`EmbodimentProfile.poses`)."""
    if place.x_toward is not None:
        faced = obj.facing(place.x_toward)
        if honor_front and faced is not obj:  # `facing` gives the object itself on no front
            faced = replace(faced, symmetry="none")
        obj = faced
    point = placement_point_pose(
        obj, place.placement_point, place.placement_point_at, conventions=conventions
    )
    held = obj.with_frames({PLACEMENT: [point]})
    frame = target_frame(place, target, base)
    forms = [for_arm(place, arm, frame, lateral)]
    last_resort = False
    if place.arm not in (None, arm) and not place.mirror:  # its mirror image if out of reach
        forms.append(for_arm(place.model_copy(update={"mirror": True}), arm, frame, lateral))
    elif place.arm is None and place.target.kind == "object" and place.orient == "relative":
        # A tilted spec that names no arm was still measured at one arm's pose: its orientation
        # is that arm's wrist, so the other arm may reach only its mirror image (onto the same
        # object only: `ActionApi.place_object`'s `_under` check), tried last.
        written = place.model_copy(update={"arm": _OTHER[arm], "mirror": True})
        forms.append(for_arm(written, arm, frame, lateral))
        last_resort = True
    centers = [target_pose(f, frame, obj.pose @ point, obj.symmetry, point) for f in forms]
    # A directed placement moves along its approach; it may turn about it (else about the
    # vertical) by its free turn, or a symmetric object by its equivalent turns: about the
    # vertical when upright, about its own axis as placed when tilted (`relative`: a round
    # vessel looks the same spun about its axis, but the gripper does not; its frame's measured
    # yaw is arbitrary, and the turn that leaves the wrist reachable is found by planning).
    axes = [
        None if f.approach is None else frame[:3, :3] @ np.asarray(f.approach, dtype=float)
        for f in forms
    ]
    onto = [
        approach_axis if a is None else tuple(c.matrix()[:3, :3].T @ a)
        for a, c in zip(axes, centers, strict=True)
    ]
    about: list[NDArray[np.float64] | None]
    if place.free_turn_deg > 0:
        turns = free_turns(place.free_turn_deg)
        about = [
            a if a is not None else _approach_axis(approach_axis, c)
            for a, c in zip(axes, centers, strict=True)
        ]
    elif place.orient in ("upright", "yaw"):
        turns = equivalent_turns(obj.symmetry)
        about = [None] * len(forms)
    elif place.orient == "relative" and tilted_turns:  # about its own axis as placed
        turns = equivalent_turns(obj.symmetry)
        about = [(c.matrix() @ np.linalg.inv(point))[:3, 2] for c in centers]
    else:  # `keep` holds the object's turn
        turns, about = [0.0], [None] * len(forms)
    fallback = place.orient == "relative" and len(turns) > 1 and place.free_turn_deg == 0
    # A held object whose frame a choice pinned (a pose grasp's, `SceneRegistry.pin`) is placed
    # as written; where nothing as written is within reach, the turns its shape looks the same
    # under stand in, last (about the vertical when upright, about its own axis when tilted).
    held_turns: list[float] = []
    pinned = obj.chosen and obj.symmetry == "none" and obj.shape_symmetry in ("pi", "half_pi")
    if pinned and place.x_toward is None and place.free_turn_deg == 0:
        assert obj.shape_symmetry is not None
        if place.orient in ("upright", "yaw") or (place.orient == "relative" and tilted_turns):
            held_turns = equivalent_turns(obj.shape_symmetry)[1:]  # `about` as above

    def poses(goal: Pose, turn: float, k: int, distance: float) -> Pose:
        placed = place_ee_pose(
            held,
            arm,
            goal,
            ee,
            conventions,
            constrain=constrain,
            align_axis=align_axis,
            object_axis=object_axis,
            object_axis_type=object_axis_type,
            placement_frame=0,
            approach_m=distance,
            approach_axis=onto[k],
        )
        return turned(placed, goal.p, turn, about[k])

    return Placement(obj, forms, centers, list(turns), poses, fallback, last_resort, held_turns)


WITHDRAW_CONE_DEG: Final = 30.0
"""A move after a release that starts within this of straight back along the TCP's pointing
axis is itself the withdraw (`ExecutorConfig.release_withdraw_m`)."""


def _grown_box(box: OrientedBox, margin: float) -> OrientedBox:
    size = tuple(float(v) + 2 * margin for v in box.size)
    return box.model_copy(update={"size": size})


@dataclass(frozen=True)
class _Rehearsed:
    """An env plane that reports `state` and executes nothing (`ActionApi.rehearsal`)."""

    state: RobotState

    async def robot(self, episode_id: EpisodeId) -> RobotState:
        return self.state

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation:
        raise RuntimeError("a rehearsal observes nothing")

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        raise RuntimeError("a rehearsal executes nothing")


LANDS_BY_M: Final = 0.05
"""A placement lands by an object whose footprint it lies within this of, horizontally: a target
only partly in view has a perceived box that ends short of its true extent."""


def _under(boxes: Sequence[OrientedBox], point: Sequence[float]) -> int | None:
    """The box a placement at `point` lands over, in or right by (the highest whose footprint,
    grown by `LANDS_BY_M`, holds it and whose bottom lies below it; a box is taken as upright,
    as perceived ones are), by index; None: bare table."""
    p = np.asarray(point, dtype=float)
    found, top = None, -math.inf
    for n, box in enumerate(boxes):
        rotation = quat_to_matrix(box.q)
        local = (p - np.asarray(box.center)) @ rotation
        half = np.asarray(box.size) / 2
        reach_z = float(np.abs(rotation[2]) @ half)  # the box's half height in the world
        low, high = box.center[2] - reach_z, box.center[2] + reach_z
        if np.all(np.abs(local[:2]) <= half[:2] + LANDS_BY_M) and low < p[2] and high > top:
            found, top = n, high
    return found


def _exit(box: OrientedBox, point: Array, direction: Array) -> float:
    """How far from `point` along the unit `direction` the box ends (0 when the point lies
    outside it)."""
    rotation = quat_to_matrix(box.q)
    local = (point - np.asarray(box.center)) @ rotation
    half = np.asarray(box.size) / 2
    if np.any(np.abs(local) > half):
        return 0.0
    d = direction @ rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(np.abs(d) > 1e-9, (np.sign(d) * half - local) / d, np.inf)
    return float(np.min(exits))


def _frame_turn(grasp: GraspSpec, label: str) -> float | None:
    """The turn of the whole object's frame a pose grasp's candidate (`pose/<turn>`) was made
    in; None for any other grasp, or a pose aimed by `x_toward` (its frame is not the
    registry's)."""
    if grasp.anchor != "pose" or grasp.x_toward is not None or not label.startswith("pose/"):
        return None
    return float(label.removeprefix("pose/").split()[0])


def _pose_turns_note(grasp: GraspSpec, candidates: Sequence[Candidate], obj: ObjectProxy) -> str:
    """For an unreachable pose grasp tried in one turn of the object's frame only, why: no turn
    of the frame was open to it (its shape taken to look different turned), so the pose's own
    side of the object was all there was; its offset or approach distance did not decide that."""
    turns = {found[1] for c in candidates if (found := POSE_LABEL.fullmatch(c.label))}
    if grasp.anchor != "pose" or len(turns) != 1:
        return ""
    return (
        f"; the pose was tried in the {obj.name} frame as registered only (pose/{turns.pop()}), "
        "approaching from that one side: its shape is measured as looking "
        "different turned, so no turned frame was open to it"
    )


def hold_turn(candidates: Sequence[Candidate], chosen: int) -> float:
    """The turn (deg) of the object's frame a pose grasp's chosen candidate holds it in: that of
    the unrolled candidate (`pose/<turn>`) nearest it, position plus `FRAME_M_PER_DEG` per degree
    of orientation (`manipulation.paths.choose_frame`'s measure). A candidate rolled half a turn
    about a vertical approach lies in orientation on the frame turned half a turn, where the
    demonstration's grasp, which fixed the spec's orientation, had it: a perceived frame turned
    from the fit's is read back, not taken as the object turned."""
    at = candidates[chosen].matrix
    best: tuple[float, float] | None = None
    for candidate in candidates:
        found = POSE_LABEL.fullmatch(candidate.label)
        if found is None:
            continue
        m = candidate.matrix
        cost = float(np.linalg.norm(m[:3, 3] - at[:3, 3])) + FRAME_M_PER_DEG * _angle_deg(
            m[:3, :3], at[:3, :3]
        )
        if best is None or cost < best[0]:
            best = (cost, float(found[1]))
    if best is None:
        raise ValueError(f"no unrolled pose candidate among {[c.label for c in candidates]}")
    return best[1]


POSE_LABEL: Final = re.compile(r"pose/(-?\d+(?:\.\d+)?)")
"""An unrolled pose grasp candidate's label, with its frame turn."""


def robot_base(profile: EmbodimentProfile) -> Pose:
    """The robot base frame: the root the arms' planners hang off (the world without), what a
    placement's `robot_frame` target means."""
    for spec in profile.arms.values():
        if spec.planner is not None:
            return spec.planner.base
    return Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))


def _batches(items: Iterator[tuple[Pose, float]], size: int) -> Iterator[list[tuple[Pose, float]]]:
    while batch := list(islice(items, size)):
        yield batch


def _approach_axis(
    approach_axis: Literal["grasp", "target_z"] | Sequence[float], goal: Pose
) -> NDArray[np.float64] | None:
    """The world direction a placement at `goal` moves in by its `approach_axis` (a vector in the
    goal's frame, `fp`: its z); None (the vertical) for `grasp`."""
    if isinstance(approach_axis, str):
        return goal.matrix()[:3, 2] if approach_axis == "target_z" else None
    rotated: NDArray[np.float64] = goal.matrix()[:3, :3] @ np.asarray(approach_axis, dtype=float)
    return rotated


def _call(name: str, arm: Arm, *args: object) -> str:
    """A builder call as text, objects by name and pose (to the millimeter and the degree)."""
    words = [name, arm]
    for arg in args:
        if isinstance(arg, ObjectProxy):
            p = arg.pose[:3, 3]
            words.append(f"{arg.name}@{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}")
        elif isinstance(arg, GraspSpec | PlaceSpec):
            words.append(arg.model_dump_json())
        else:
            words.append(repr(arg))
    return " ".join(words)


def clearance(contact: ArrayLike, other_tcp: Pose, approach_m: float) -> float:
    """How far a gripper grasping at `contact` (world 4x4, contact frame: y against the
    approach), fingers and approach swept from `approach_m` back, passes from the fingers of
    the other gripper at `other_tcp` (both finger spans `FINGER_M` back from the TCP along the
    pointing axis)."""
    c = np.asarray(contact, dtype=float)
    approach = -c[:3, 1]
    ours = _samples(c[:3, 3] - approach * (approach_m + FINGER_M), c[:3, 3])
    pointing = other_tcp.matrix()[:3, 0]
    theirs = _samples(np.asarray(other_tcp.p) - pointing * FINGER_M, np.asarray(other_tcp.p))
    return float(np.linalg.norm(ours[:, None] - theirs[None], axis=2).min())


def _overlap(ours: Sequence[Sequence[float]], theirs: NDArray[np.float64]) -> float:
    """By how much two sets of collision spheres come closer than `ARM_GAP_M` (m; 0 if not)."""
    mine = np.asarray(ours, dtype=float).reshape(-1, 4)
    if not len(mine) or not len(theirs):
        return 0.0
    gaps = np.linalg.norm(mine[:, None, :3] - theirs[None, :, :3], axis=2)
    gaps -= mine[:, None, 3] + theirs[None, :, 3]
    return max(0.0, ARM_GAP_M - float(gaps.min()))


def _samples(a: ArrayLike, b: ArrayLike, n: int = 21) -> NDArray[np.float64]:
    t = np.linspace(0.0, 1.0, n)[:, None]
    result: NDArray[np.float64] = np.asarray(a) + t * (np.asarray(b) - np.asarray(a))
    return result


def displaced(
    ee: Pose,
    shift: Sequence[float],
    quat: Sequence[float] | None,
    move_axis: Literal["world", "arm"],
    ee_to_tcp: Sequence[float],
) -> Pose:
    """A displacement's target from the measured ee pose `ee`: shifted by (x, y, z) along the world
    axes, or with `move_axis` arm by -z
    along the gripper's pointing axis (the direction of the ee-to-TCP offset); `quat`, if given,
    replaces the orientation."""
    if move_axis == "world":
        offset = np.asarray(shift, dtype=float)
    else:
        pointing = np.asarray(ee_to_tcp, dtype=float) / np.linalg.norm(ee_to_tcp)
        offset = -float(shift[2]) * quat_rotate(ee.q, pointing)
    x, y, z = np.add(ee.p, offset).tolist()
    q = ee.q if quat is None else tuple(float(v) for v in np.asarray(quat) / np.linalg.norm(quat))
    return Pose(p=(x, y, z), q=q)  # type: ignore[arg-type]


def _pose_key(pose: Pose) -> tuple[float, ...]:
    """A pose as a cache key: to 0.1 mm and 1e-6 of a quaternion component."""
    return (*(round(v, 4) for v in pose.p), *(round(v, 6) for v in pose.q))


def _plural(word: str, n: int) -> str:
    return word if n == 1 else f"{word}s"


def _nearest_material(candidates: Sequence[Candidate]) -> str:
    """Where the measured surface nearest an air candidate's contact lies (the nearest of all),
    in the spec's part frame, so that a fix moves the offset there rather than guessing."""
    misses = [c.miss for c in candidates if c.miss is not None]
    if not misses:
        return ""
    x, y, z = min(misses, key=lambda m: math.hypot(*m))
    return (
        f"; the nearest measured surface lies {100 * math.hypot(x, y, z):.1f} cm from the "
        f"contact, at ({100 * x:+.1f}, {100 * y:+.1f}, {100 * z:+.1f}) cm in the spec's part frame"
    )


def _xyz(p: Sequence[float]) -> str:
    return "(" + ", ".join(f"{float(v):.3f}" for v in p) + ")"
