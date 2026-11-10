"""The qpos-waypoint executor (docs/concepts/embodied-core.md): per-arm target ee poses and
gripper commands become one `ActionBatch` of `take_action(qpos)` steps, executed through `EnvAgent`,
and an `ActionReceipt` says what happened. It never touches a simulator object.

Every move is planned from the arm's drive target, where `take_action` starts. A free move follows
the planner's path; the planner avoids only the robot itself and the `obstacles` it is given, so
clearance comes from sending the path's waypoints closely (`waypoints.select`) and from the scene's
boxes: a caller sets `Executor.obstacles` (the objects as perceived, none it holds), and each free
move is first planned around them, grown by `obstacle_margin_m`, leaving out those its start or goal
TCP lies in (an object just let go, or reached into); where no such plan is found, it is planned
without them, and noted. A free move with the gripper empty that starts lower than an obstacle's top
near its way first rises straight up (`lift_clearance_m`): a look pose or a release leaves the TCP
beside tall objects; a carry's first free move rises until what it holds clears its support (the
table) by as much. A straight move (a grasp's approach) comes from IK along the line; where no IK
line keeps to it (a branch change, e.g. where cuRobo's collision spheres rule out the seeded
branch), from an orientation-locked plan, which is not trusted (its lock is only a soft cost) but
taken only if it keeps near the line. Everything is planned and checked before anything moves, so a
refused action costs no steps: every move gets `1 + replans` attempts (fresh planner seeds each),
then is reported (`unreachable` when no IK solution reaches the target, else `plan_failed`); a plan
whose joints travel too far (a wrap) or past a joint's stop (`ArmProfile.stops`: the arm halts
there, off its path, though the planner's model allows more) counts as failed; a displacement
is clamped (`clamped`); a target TCP below the table is `invalid_args`. A close is followed by
holds until the measured opening settles (`ActionStep.until_gripper_stable`), opening a closed
gripper by the holds the fingers take to open. Where one step moves a joint only so far (an env
whose `take_action` runs a fixed time: `max_action_joint_step_rad`), a waypoint
farther than that from the step before is reached in even steps along the joint line to it.

Three recoveries: IK solutions along a straight line are moved by
whole turns to the equivalent nearest the previous solution (a solver's 2 pi jumps), and a free
plan whose joints travel too far is requested once more; a place (a contact move made while the
gripper holds something) is sent in short steps, slowing it (`place_step_m`), and the release
after it waits `release_holds` holds; a straight move that ends the action `deviated` is planned
once more from where the arm stopped (its measured joints) to the same target, and the two
receipts are joined.

Pre-checks, before any planning or with the plans but before anything moves: a target TCP far
outside the arm's probed workspace (`workspace`, opt-in) is `unreachable` once one IK call
confirms it; a two-arm action (`together`) whose TCPs would come closer than `min_arm_gap_m`
along their planned paths (`tcp_gap`: an arm without a move holds its place) is `invalid_args`,
unless they start closer and come at most `APPROACH_M` closer (leaving a shared hold).
(`rejected_repeat`
is the action API's: it knows the calls.)
"""

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import EnvAgent
from roboscript.contracts.errors import InfraError
from roboscript.contracts.executor import ExecutorConfig
from roboscript.contracts.geometry import OrientedBox, Pose
from roboscript.contracts.ids import BatchId, EpisodeId, new_id
from roboscript.contracts.motion import MAX_GOALS, FKRequest, IKRequest, Planner, PlanRequest
from roboscript.contracts.receipt import (
    ActionReceipt,
    CommandedPoses,
    ReceiptReason,
)
from roboscript.contracts.robot import Arm, ArmState, RobotState
from roboscript.embodiments import EmbodimentProfile
from roboscript.manipulation.plan import Grip, GripCheck, Move, MoveCheck, Plan, Segment, tcp_gap
from roboscript.manipulation.receipts import receipt, steps_left
from roboscript.manipulation.waypoints import TcpFK, line, polyline_distance, select
from roboscript.manipulation.workspace import load_workspace, reach_gap
from roboscript.spatial.pose import pose_compose
from roboscript.spatial.rotation import quat_angle_deg, quat_slerp, quat_to_matrix

_DENSITIES: Final = (1, 2, 4)  # a straight line's samples: as configured, twice, four times
_AT_TARGET_M: Final = 0.001
_AT_TARGET_DEG: Final = 0.5
_ORIGIN: Final = Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))
REACHED_M: Final = 0.01  # a target point this near the reach probed toward it is in reach
_WAY_STEP_M: Final = 0.01  # where a free move's way (seen from above) is sampled for obstacles
_ENTRY_CLEAR_M: Final = 0.005  # a way in starts this far outside the held object's grown box
_WRAP_RAD: Final = 1.5 * math.pi
"""A joint changing this much or more between neighboring IK solutions jumped by whole turns (a
half turn about the gripper's axis is a change of branch, which it keeps)."""
_CLOSED_BELOW: Final = 0.5
APPROACH_M: Final = 0.005
"""Arms whose TCPs start closer than `min_arm_gap_m` may still move together if they come at most
this much closer than they start (leaving a shared hold, or moving side by side)."""


class Executor:
    """Executes actions in one env episode through its agent plane. `table_top_z`, the table's
    height, refuses targets below it (None skips the check). A profile without planner assets
    (MockEnv's) plans in the world frame. `succeeded` turns true once the task's success check
    has latched; `lenient` holds the arms whose hold may slip without failing: a pull's (it may
    slip at the end), or one that a press or a plain close took; `pulling` those whose last close
    was a pull's (a part held that way moves on its joint: `ActionApi` follows it). `obstacles`
    are world boxes free moves keep clear of (the caller's to keep current; empty: none)."""

    def __init__(
        self,
        env: EnvAgent,
        episode_id: EpisodeId,
        planner: Planner,
        profile: EmbodimentProfile,
        config: ExecutorConfig,
        *,
        table_top_z: float | None = None,
    ) -> None:
        self.env, self.episode_id, self.planner, self.profile = env, episode_id, planner, profile
        self.config = config
        self.table_top_z = table_top_z
        self.obstacles: list[OrientedBox] = []
        self.held: dict[Arm, list[OrientedBox]] = {}
        """What each arm holds, where it holds it: the other arm's free moves keep clear of it
        while the holder stays still, and never plan through it (`_Planning._firm`)."""
        self.workspace = (
            load_workspace(self.config.workspace) if self.config.workspace is not None else None
        )
        if self.workspace is not None and self.workspace.embodiment != profile.name:
            raise ValueError(
                f"the workspace {self.config.workspace} is {self.workspace.embodiment}'s, "
                f"not {profile.name}'s"
            )
        self.bases = {
            arm: spec.planner.base if spec.planner is not None else _ORIGIN
            for arm, spec in profile.arms.items()
        }
        self.succeeded = False
        self.lenient: set[Arm] = set()
        self.pulling: set[Arm] = set()
        self.placed: set[Arm] = set()  # arms whose last move placed what they hold

    async def run(
        self, left: Sequence[Segment] = (), right: Sequence[Segment] = ()
    ) -> ActionReceipt:
        """Plan the action, execute it unless it is refused, and report; a straight move that
        ends it `deviated` is planned once more from where the arm stopped."""
        plan = await self.plan(left, right)
        done = await self.execute(plan)
        move = self._deviated(plan, done)
        if move is None:
            return done
        now = await self.env.robot(self.episode_id)
        state = now.arms[move.arm]
        if state.joints is None:  # nothing measured to start from
            return done
        stopped = state.model_copy(update={"joints_cmd": state.joints})
        start = now.model_copy(update={"arms": {**now.arms, move.arm: stopped}})
        segments = [  # the same move again: a place's contact, push past and fallback kept
            Move(
                move.target,
                straight=True,
                contact=move.contact,
                past_m=move.past_m,
                free_fallback=move.free_fallback,
            )
        ]
        again = await self.plan(
            segments if move.arm == "left" else (),
            segments if move.arm == "right" else (),
            before=start,
        )
        return _joined(done, await self.execute(again))

    async def plan(
        self,
        left: Sequence[Segment] = (),
        right: Sequence[Segment] = (),
        before: RobotState | None = None,
    ) -> Plan:
        """The action planned from the drive targets in `before` (the robot now when None),
        without executing it."""
        if before is None:
            before = await self.env.robot(self.episode_id)
        arms: tuple[tuple[Arm, Sequence[Segment]], ...] = (("left", left), ("right", right))
        lists: dict[Arm, list[Segment]] = {arm: list(s) for arm, s in arms if s}
        return await _Planning(self, before, lists).run()

    async def after(self, plan: Plan, holding: Mapping[Arm, float] | None = None) -> RobotState:
        """The robot as a planned action leaves it, nothing executed: each arm's drive targets
        and gripper commands those of the plan's last step, its ee and TCP their FK (measured as
        commanded), the openings unchanged unless `holding` gives an arm's (a close expected to
        hold settles above the holding threshold, at the opening given)."""
        if plan.refused is not None or not plan.steps:
            return plan.before
        qpos = plan.steps[-1].qpos
        arms = dict(plan.before.arms)
        for arm, state in plan.before.arms.items():
            spec = self.profile.arms[arm]
            start, stop = spec.action
            joints = list(qpos[start:stop])
            if joints == state.joints_cmd and qpos[spec.gripper_index] == state.gripper_cmd:
                continue
            request = FKRequest(arm=arm, base=self.bases[arm], joints=[joints])
            found = await self.planner.fk(request)
            opening = (holding or {}).get(arm, state.gripper)
            arms[arm] = state.model_copy(
                update={
                    "ee": found.ee[0],
                    "tcp": found.tcp[0],
                    "joints": joints,
                    "joints_cmd": joints,
                    "gripper": opening,
                    "gripper_cmd": qpos[spec.gripper_index],
                }
            )
        step = plan.before.step + len(plan.steps)
        return plan.before.model_copy(update={"arms": arms, "step": step, "qpos_cmd": qpos})

    async def hold(self, steps: int) -> ActionReceipt:
        """`steps` actions keeping every drive target and gripper command."""
        before = await self.env.robot(self.episode_id)
        holds = [ActionStep(qpos=before.qpos_cmd)] * steps
        return await self.execute(Plan(before, holds, placed=frozenset(self.placed)))

    async def home_pose(self, arm: Arm) -> Pose:
        """The ee pose of the arm's home joints (`ArmProfile.home`), where `return_home` goes:
        the ee pose once the arm has settled at its home state."""
        home = self.profile.arms[arm].home
        if home is None:
            raise ValueError(f"the {self.profile.name} profile gives the {arm} arm no home joints")
        return await self._fk_ee(arm, np.asarray(home, dtype=float))

    async def execute(self, plan: Plan) -> ActionReceipt:
        """Execute a planned action and report; a refused one returns its refusal, and nothing
        moves."""
        if plan.refused is not None:
            return plan.refused
        if not plan.steps:  # every target is where the arm already is
            return receipt(plan, [], "completed", self.profile, self.config)
        batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=plan.steps)
        result = await self.env.execute(self.episode_id, batch)
        if result.stopped == "error":
            raise InfraError("sim_fault", result.error or "the simulator failed in the action")
        self.succeeded = self.succeeded or result.stopped == "success"
        executed = {state.index for state in result.states}
        self.placed = set(plan.placed)
        for grip in plan.grips:
            if grip.end not in executed:
                continue
            closes = grip.value < self.profile.arms[grip.arm].gripper.open
            if closes and (grip.pull or not grip.grasp):
                self.lenient.add(grip.arm)
            elif grip.grasp or not closes:
                self.lenient.discard(grip.arm)
            if closes and grip.pull:
                self.pulling.add(grip.arm)
            else:
                self.pulling.discard(grip.arm)
        return receipt(plan, result.states, result.stopped, self.profile, self.config)

    def _deviated(self, plan: Plan, done: ActionReceipt) -> MoveCheck | None:
        """The straight move that ended the action `deviated`, if one did: the last segment of
        its arm, the task not yet done (`retry_deviated`)."""
        if not self.config.retry_deviated or self.succeeded:
            return None
        if done.reason != ReceiptReason.deviated or done.commanded is None:
            return None
        found = [m for m in plan.moves if m.arm == done.arm and m.target == done.commanded.ee]
        if not found or not found[-1].straight:
            return None
        move = found[-1]
        later = [m.end for m in plan.moves if m.arm == move.arm]
        later += [g.end for g in plan.grips if g.arm == move.arm]
        return move if max(later) == move.end else None

    async def drive_ee(self, arm: Arm, state: ArmState) -> Pose:
        """The ee pose the arm is driven to (FK of its commanded joints)."""
        return await self._fk_ee(arm, np.asarray(state.joints_cmd, dtype=float))

    async def reach_note(self, state: RobotState, arm: Arm, goal: Sequence[float]) -> str:
        """What an `unreachable` refusal of `arm` for the TCP point `goal` says about reach: how
        far beyond each arm's reach it lies, from where its TCP is in `state` (`reach_gap`),
        the axis it lies farthest beyond along, or that the point is reached at another
        orientation; so the Actor can change the approach or arm, or see that neither gets
        there."""
        parts = []
        for a in sorted(state.arms, key=lambda a: a != arm):
            now = state.arms[a]
            gap = await reach_gap(
                self.planner, self.profile, a, self.bases[a], now.joints_cmd, now.tcp.p, goal
            )
            size = float(np.linalg.norm(gap))
            if size < REACHED_M:
                other = ", not at this one: approach it another way" if a == arm else ""
                parts.append(f"the {a} arm reaches that point at some orientation{other}")
                continue
            axis = "xyz"[int(np.abs(gap).argmax())]
            parts.append(
                f"the {a} arm's reach ends about {size * 100:.0f} cm short of it (x "
                f"{gap[0]:+.3f}, y {gap[1]:+.3f}, z {gap[2]:+.3f} m beyond it, most in {axis})"
            )
        return "reach: " + "; ".join(parts)

    async def _fk_ee(self, arm: Arm, joints: NDArray[np.float64]) -> Pose:
        request = FKRequest(arm=arm, base=self.bases[arm], joints=[joints.tolist()])
        return (await self.planner.fk(request)).ee[0]

    def _tcp_fk(self, arm: Arm) -> TcpFK:
        async def fk(joints: NDArray[np.float64]) -> NDArray[np.float64]:
            request = FKRequest(arm=arm, base=self.bases[arm], joints=joints.tolist())
            return np.array([pose.p for pose in (await self.planner.fk(request)).tcp])

        return fk

    def _tcp_positions(self, arm: Arm, ees: Sequence[Pose]) -> NDArray[np.float64]:
        offset = [*self.profile.arms[arm].ee_to_tcp, 1.0, 0.0, 0.0, 0.0]
        return pose_compose(np.array([pose.as_list() for pose in ees]), offset)[:, :3]


@dataclass(frozen=True)
class _Path:
    joints: NDArray[np.float64]  # (n, dof), from the start
    tcp: NDArray[np.float64]  # (n, 3) planned TCP positions
    end: Pose  # the ee pose it ends at
    every: bool = False  # sent point by point (a place), not as the fewest waypoints


@dataclass(frozen=True)
class _Refusal:
    reason: ReceiptReason
    detail: str
    planner: str | None = None  # the planner's own reason (`PlanReason`), where it refused


@dataclass
class _Planning:
    """One action's planning, pair by pair, each arm from where its previous segment left it."""

    ex: Executor
    before: RobotState
    lists: dict[Arm, list[Segment]]
    steps: list[ActionStep] = field(default_factory=list)
    moves: list[MoveCheck] = field(default_factory=list)
    grips: list[GripCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    clamped: bool = False
    moving: set[Arm] = field(default_factory=set)  # the arms the pair being planned moves

    def __post_init__(self) -> None:
        self.config, self.profile = self.ex.config, self.ex.profile
        self.joints = {
            a: np.asarray(s.joints_cmd, dtype=float) for a, s in self.before.arms.items()
        }
        self.gripper = {a: s.gripper_cmd for a, s in self.before.arms.items()}
        self.ee: dict[Arm, Pose] = {}  # FK of the drive target, then where each move ends
        threshold = {a: spec.gripper.holding_threshold for a, spec in self.profile.arms.items()}
        self.holding = {  # closed on something: when the action starts, or by a grasp in it
            a: s.gripper_cmd < _CLOSED_BELOW and s.gripper is not None and s.gripper >= threshold[a]
            for a, s in self.before.arms.items()
        }
        self.placed = set(self.ex.placed)

    async def run(self) -> Plan:
        arms = sorted(self.lists)
        for k in range(max((len(s) for s in self.lists.values()), default=0)):
            pair = {arm: segments[k] for arm, segments in self.lists.items() if k < len(segments)}
            refused = await self._pair(pair)
            if refused is not None:
                return Plan(self.before, arms=arms, refused=refused)
        # A plain close's hold may slip without failing while it stands, but not while it is
        # carried (a free move or a displacement). A pull's hold stays lenient.
        carried = {
            arm
            for arm, segments in self.lists.items()
            for segment in segments
            if isinstance(segment, Move) and (not segment.straight or segment.bounded)
        }
        plan = Plan(
            self.before,
            self.steps,
            self.moves,
            self.grips,
            arms,
            self.notes,
            self.clamped,
            lenient=frozenset(self.ex.lenient - (carried - self.ex.pulling)),
            placed=frozenset(self.placed),
        )
        if len(self.lists) < 2:  # one arm at a time is not checked: a hand-over meets
            return plan
        start, closest = tcp_gap(plan)
        gap = self.config.min_arm_gap_m
        if closest >= gap or closest >= start - APPROACH_M:
            return plan
        note = (
            f"the TCPs would come within {closest * 100:.1f} cm of each other (from "
            f"{start * 100:.1f} cm); moving together keeps them {gap * 100:.0f} cm apart"
        )
        refused = ActionReceipt(
            reason=ReceiptReason.invalid_args, steps_left=steps_left(self.before), note=note
        )
        return replace(plan, refused=refused)  # its moves say where the arms were headed

    async def _pair(self, pair: dict[Arm, Segment]) -> ActionReceipt | None:
        paths: dict[Arm, tuple[_Path, Move, Pose]] = {}
        places: set[Arm] = set()
        self.moving = {arm for arm, segment in pair.items() if isinstance(segment, Move)}
        for arm, segment in pair.items():
            if not isinstance(segment, Move):
                continue
            if arm not in self.ee:
                self.ee[arm] = await self.ex._fk_ee(arm, self.joints[arm])
            target = self._clamp(arm, segment.target) if segment.bounded else segment.target
            tcp = self.profile.tcp_pose(arm, target)
            table = self.ex.table_top_z
            if table is not None and tcp.p[2] < table:
                below = (table - tcp.p[2]) * 100
                note = (
                    f"the {arm} target TCP is below the table top: raise it at least {below:.1f} cm"
                )
                return self._refuse(ReceiptReason.invalid_args, arm, target, note)
            if _near(self.ee[arm], target):
                continue
            if (outside := await self._outside(arm, target, tcp)) is not None:
                reach = await self.ex.reach_note(self.before, arm, tcp.p)
                return self._refuse(ReceiptReason.unreachable, arm, target, f"{outside}; {reach}")
            path = await self._plan_move(arm, target, segment.straight)
            if isinstance(path, _Refusal) and segment.straight and segment.free_fallback:
                free = await self._plan_move(arm, target, False)
                if isinstance(free, _Path):
                    self.notes.append(f"{arm}: no straight way in ({path.detail}); planned freely")
                    segment = replace(segment, straight=False)  # no straight retry of a free plan
                    path = free
                else:  # both said: the straight way's reason first, an unreachable one kept
                    either = {path.reason, free.reason}
                    unreachable = ReceiptReason.unreachable in either
                    path = _Refusal(
                        ReceiptReason.unreachable if unreachable else free.reason,
                        f"no straight way in ({path.detail}); no free plan either ({free.detail})",
                        free.planner,
                    )
            if isinstance(path, _Refusal):
                note = f"{arm}: {path.detail}"
                if path.reason == ReceiptReason.unreachable:
                    note += "; " + await self.ex.reach_note(self.before, arm, tcp.p)
                return self._refuse(path.reason, arm, target, note)
            if segment.contact and self.holding[arm]:  # a place
                places.add(arm)
                if (step := self.config.place_step_m) is not None:
                    path = _slowed(path, step)
            paths[arm] = (path, segment, target)
        end = await self._emit(pair, {arm: path for arm, (path, _, _) in paths.items()})
        for arm, (path, segment, target) in paths.items():
            tcp = self.profile.tcp_pose(arm, target)
            check = MoveCheck(
                arm,
                end,
                target,
                tcp,
                path.tcp,
                segment.contact,
                segment.straight,
                segment.past_m,
                segment.free_fallback,
            )
            self.moves.append(check)
            self.ee[arm] = path.end
            if arm in places:
                self.placed.add(arm)
            else:
                self.placed.discard(arm)
        return None

    async def _emit(self, pair: dict[Arm, Segment], paths: dict[Arm, _Path]) -> int:
        """Append the pair's steps: the moving arms' waypoints at shared path fractions (both
        arms move in the same calls), the new gripper commands from the first step on, then the
        holds of a settling close or of an opening. Returns the moves' last step."""
        tolerance = self.config.waypoint_tolerance_m
        fractions: set[float] = set()
        for arm, path in paths.items():
            if path.every:
                chosen = list(range(1, len(path.joints)))
            else:
                chosen = await select(path.joints, path.tcp, self.ex._tcp_fk(arm), tolerance)
            fractions |= {round(i / (len(path.joints) - 1), 9) for i in chosen}
        grips = {  # a command the gripper already has is no action
            arm: segment
            for arm, segment in pair.items()
            if isinstance(segment, Grip) and segment.value != self.gripper[arm]
        }
        if not paths and not grips:  # the pair's arms are already where its segments go
            return len(self.steps) - 1
        previous = dict(self.gripper)
        releases = [
            arm
            for arm, grip in grips.items()
            if arm in self.placed and grip.value >= _CLOSED_BELOW > previous[arm]
        ]
        if releases:  # what was just placed settles before the fingers let go
            self.steps += [ActionStep(qpos=self._qpos())] * self.config.release_holds
            self.placed -= set(releases)
        self.gripper.update({arm: grip.value for arm, grip in grips.items()})
        for arm, grip in grips.items():
            self.holding[arm] = grip.value < _CLOSED_BELOW and (grip.grasp or self.holding[arm])
        cap = self.config.max_action_joint_step_rad
        for fraction in sorted(fractions) or [1.0]:
            start = {arm: self.joints[arm] for arm in paths}
            goal = {
                arm: path.joints[round(fraction * (len(path.joints) - 1))]
                for arm, path in paths.items()
            }
            n = splits(start, goal, cap)
            for k in range(1, n + 1):
                for arm in paths:  # the waypoint itself last, as computed
                    self.joints[arm] = (
                        goal[arm] if k == n else start[arm] + (goal[arm] - start[arm]) * k / n
                    )
                self.steps.append(ActionStep(qpos=self._qpos()))
        end = len(self.steps) - 1
        opened = {arm: self.profile.arms[arm].gripper.open for arm in grips}
        if any(grip.value < opened[arm] for arm, grip in grips.items()):
            self.steps[end] = self.steps[end].model_copy(
                update={"until_gripper_stable": True, "max_repeats": self.config.settle_holds}
            )
        if any(grip.value >= opened[arm] > previous[arm] for arm, grip in grips.items()):
            self.steps += [ActionStep(qpos=self.steps[end].qpos)] * self.config.open_holds
        for arm, grip in grips.items():
            last = end if grip.value < opened[arm] else len(self.steps) - 1
            self.grips.append(GripCheck(arm, last, grip.value, grip.grasp, grip.pull))
        return end

    async def _outside(self, arm: Arm, target: Pose, tcp: Pose) -> str | None:
        """Why the target lies outside the arm's probed workspace, if it does: farther than the
        margin from every reachable cell, and no IK solution reaches it (the table is coarse)."""
        workspace = self.ex.workspace
        if workspace is None:
            return None
        away = workspace.distance(arm, tcp.p)
        if away <= self.config.workspace_margin_m:
            return None
        request = IKRequest(
            arm=arm, base=self.ex.bases[arm], goals=[target], seed=self.joints[arm].tolist()
        )
        if (await self.ex.planner.ik(request))[0].joints is not None:
            return None
        return f"the {arm} target TCP is {away * 100:.0f} cm outside the arm's workspace"

    async def _plan_move(
        self,
        arm: Arm,
        target: Pose,
        straight: bool,
        held: list[OrientedBox] | None = None,
    ) -> _Path | _Refusal:
        """Up to `1 + replans` attempts (cuRobo's refusals are not deterministic, and each call
        advances its seeds): a free move is planned again; a straight one is IK along the line,
        then orientation-locked plans that keep to it. A free move whose goal TCP lies in the
        grown box of what another arm holds (`_firm` lets that box go) first comes in straight
        along the pointing axis from outside it (`_entered`); `held`: the firm boxes to keep
        clear of instead of `_firm`'s (the way in's own transit)."""
        if not straight and held is None:
            entered = await self._entered(arm, target)
            if entered is not None:
                return entered
        refusals: list[_Refusal] = []
        attempts = 1 + self.config.replans
        firm = [] if straight else self._firm(arm, target) if held is None else held
        lift = None if straight else await self._lift(arm, target)
        if not straight and (self.ex.obstacles or firm):
            boxes = self._obstacles(arm, target)
            if boxes or firm:
                found = await self._clear(arm, target, lift, [*boxes, *firm])
                if isinstance(found, _Path):
                    return found
                if found.planner == "invalid_start" and self.config.drop_start_colliders:
                    start = self.joints[arm] if lift is None else lift.joints[-1]
                    rest = await self._clear_of_start(arm, start, boxes)
                    firm = await self._clear_of_start(arm, start, firm)  # cannot be cleared
                    if (rest or firm) and len(rest) < len(boxes):
                        kept = await self._clear(arm, target, lift, [*rest, *firm])
                        if isinstance(kept, _Path):
                            self.notes.append(
                                f"{arm}: kept clear of {len(rest)} of the {len(boxes)} perceived "
                                "objects, not of those its start already touches"
                            )
                            return kept
                        found = kept
                # The hand at either end may touch a grown box beside it (a pre-grasp's fingers
                # the box of what they will grasp): first keep clear of those as they are, not
                # grown, then of the rest only (a transit swinging round the object to grasp it
                # from the side would go through it when let go).
                far = self._far(arm, target, boxes)
                margin = self.config.obstacle_margin_m
                near = [_grown(box, -margin) for box in boxes if box not in far]
                if near and self.config.near_bare and found.planner != "invalid_start":
                    kept = await self._clear(arm, target, lift, [*far, *near, *firm])
                    if isinstance(kept, _Path):
                        self.notes.append(
                            f"{arm}: kept clear of the {len(boxes)} perceived objects, of those "
                            "within a hand's length of its start or goal by less than the margin"
                        )
                        return kept
                if (far or firm) and len(far) < len(boxes):
                    kept = await self._clear(arm, target, lift, [*far, *firm])
                    if isinstance(kept, _Path):
                        self.notes.append(
                            f"{arm}: kept clear of {len(far)} of the {len(boxes)} perceived "
                            "objects, not of those within a hand's length of its start or goal"
                        )
                        return kept
                if firm:  # what the other arm holds is never planned through: it would drop
                    if boxes:
                        kept = await self._clear(arm, target, lift, firm)
                        if isinstance(kept, _Path):
                            self.notes.append(
                                f"{arm}: kept clear of what the other arm holds, not of the "
                                f"{len(boxes)} perceived objects ({found.detail})"
                            )
                            return kept
                        found = kept
                    detail = (
                        f"no plan keeps clear of what the other arm holds ({found.detail}): "
                        "passing through it would knock it from that grip; move that arm and "
                        "what it holds clear (or lift it) first"
                    )
                    return _Refusal(ReceiptReason.plan_failed, detail, found.planner)
                self.notes.append(
                    f"{arm}: no plan kept clear of the {len(boxes)} perceived objects "
                    f"({found.detail}); planned without them"
                )
            for _ in range(attempts):
                found = await self._planned(arm, target, locked=False, after=lift)
                if isinstance(found, _Path):
                    return found
                refusals.append(found)
            details = "; ".join(dict.fromkeys(refusal.detail for refusal in refusals))
            return _Refusal(refusals[-1].reason, f"{details} (tried {attempts} times)")
        for attempt in range(attempts):
            if not straight:
                found = await self._planned(arm, target, locked=False, after=lift)
            elif attempt == 0:
                found = await self._straight(arm, target)
            else:
                found = await self._planned(arm, target, locked=True)
            if isinstance(found, _Path):
                return found
            refusals.append(found)
        details = "; ".join(dict.fromkeys(refusal.detail for refusal in refusals))
        reasons = {refusal.reason for refusal in refusals}
        # An IK line that finds no solution for the target says more than a failed locked plan.
        unreachable = straight and ReceiptReason.unreachable in reasons
        reason = ReceiptReason.unreachable if unreachable else refusals[-1].reason
        return _Refusal(reason, f"{details} (tried {attempts} times)")

    async def _planned(
        self,
        arm: Arm,
        target: Pose,
        *,
        locked: bool,
        after: _Path | None = None,
        obstacles: Sequence[OrientedBox] = (),
    ) -> _Path | _Refusal:
        """The planner's path, refused when a joint travels far (a wrap) or past its stop
        (`ArmProfile.stops`, which the planner's model may lack) or, for a locked plan
        (whose lock is only a soft cost), when it strays from the straight line or turns; a free
        one from the end of `after` (a lift), which it then follows."""
        start = self.joints[arm] if after is None else after.joints[-1]
        request = PlanRequest(
            arm=arm,
            base=self.ex.bases[arm],
            start=start.tolist(),
            goal=target,
            constraint="orientation" if locked else "free",
            ee_path=True,
            obstacles=list(obstacles),
        )
        for again in (False, True):  # a plan travelling too far is requested once more
            result = await self.ex.planner.plan(request)
            if result.path is None or result.ee_path is None:
                detail = f"no plan to the target ({result.reason}{': ' if result.detail else ''}"
                detail += f"{result.detail})"
                unreachable = result.reason == "unreachable"
                return _Refusal(
                    ReceiptReason.unreachable if unreachable else ReceiptReason.plan_failed,
                    detail,
                    result.reason,
                )
            joints = np.asarray(result.path, dtype=float)
            joints[0] = start
            travel = np.abs(np.diff(joints, axis=0)).sum(axis=0)
            if travel.max() <= self.config.max_joint_travel_rad:
                break
            if again:
                detail = f"the plan turns joint {int(travel.argmax())} by {travel.max():.1f} rad"
                return _Refusal(ReceiptReason.plan_failed, detail)
        assert result.ee_path is not None
        past = self.profile.arms[arm].past_stops(joints)
        if past is not None:
            return _Refusal(ReceiptReason.plan_failed, f"the plan's {past}")
        tcp = self.ex._tcp_positions(arm, result.ee_path)
        table = self.ex.table_top_z
        if table is not None and tcp[:, 2].min() < table:
            below = (table - tcp[:, 2].min()) * 100
            detail = f"the plan's TCP dips below the table top, at least {below:.1f} cm"
            return _Refusal(ReceiptReason.plan_failed, detail)
        if locked:
            ends = self.ex._tcp_positions(arm, [self.ee[arm], target])
            off = float(polyline_distance(tcp, ends).max())
            turn = max(float(quat_angle_deg(pose.q, target.q)) for pose in result.ee_path)
            if off > self.config.lock_tolerance_m or turn > self.config.lock_tolerance_deg:
                detail = f"the locked plan strays {off * 100:.1f} cm and {turn:.0f} deg"
                return _Refusal(ReceiptReason.plan_failed, detail)
        if after is not None:
            joints, tcp = np.vstack([after.joints, joints[1:]]), np.vstack([after.tcp, tcp[1:]])
        return _Path(joints, tcp, result.ee_path[-1])

    async def _clear(
        self, arm: Arm, target: Pose, lift: _Path | None, boxes: Sequence[OrientedBox]
    ) -> _Path | _Refusal:
        """A free plan keeping clear of `boxes`, tried `1 + replans` times as a plan without them
        is (cuRobo's refusals are not deterministic; one dipping under the table top, say, is
        followed by one that does not); a start in collision is not tried again."""
        found: _Path | _Refusal = _Refusal(ReceiptReason.plan_failed, "not planned")
        for _ in range(1 + self.config.replans):
            found = await self._planned(arm, target, locked=False, after=lift, obstacles=boxes)
            if isinstance(found, _Path) or found.planner == "invalid_start":
                return found
        return found

    def _firm(self, arm: Arm, target: Pose) -> list[OrientedBox]:
        """What the other arms hold (`Executor.held`), grown by the margin, while they stay
        still: not moving in this pair nor earlier in this action (where they held it then is
        not where it is now), still holding, and not holding the arm's start or goal TCP (a
        hand-over's receiving grasp ends in it)."""
        moved = self.moving | {move.arm for move in self.moves}
        ends = self.ex._tcp_positions(arm, [self.ee[arm], target])
        margin = self.config.obstacle_margin_m
        return [
            box
            for other, held in self.ex.held.items()
            if other != arm and other not in moved and self.holding.get(other, False)
            for box in (_grown(b, margin) for b in held)
            if not _inside(box, ends).any()
        ]

    async def _entered(self, arm: Arm, target: Pose) -> _Path | None:
        """A free move into the grown box of what a still arm holds (a hand-over's pre-grasp
        before the held object's face; the start outside it): planned to the goal backed off
        along the gripper's pointing axis until its TCP is `_ENTRY_CLEAR_M` outside that box,
        clear of the object (its bare box: the open fingers there are within the margin of it),
        then straight in. Otherwise `_firm` lets the box go and the free plan may sweep the
        fingers through what the other arm holds, and knock it from the other grip.
        None where there is no such box or no such way in (then planned the usual way)."""
        moved = self.moving | {move.arm for move in self.moves}
        start, goal = self.ex._tcp_positions(arm, [self.ee[arm], target])
        margin = self.config.obstacle_margin_m
        boxes = [
            (bare, _grown(bare, margin))
            for other, held in self.ex.held.items()
            if other != arm and other not in moved and self.holding.get(other, False)
            for bare in held
        ]
        boxes = [(b, g) for b, g in boxes if not _inside(g, start).any()]
        entered = [g for _, g in boxes if _inside(g, goal).any()]
        if not entered:
            return None
        back = np.asarray(target.p, dtype=float) - goal
        back /= np.linalg.norm(back)  # from the fingertips back to the wrist
        out = max(_exit(box, goal, back) for box in entered) + _ENTRY_CLEAR_M
        held = [b if g in entered else g for b, g in boxes]
        for distance in (out, out + margin):  # farther back where the arm cannot stand there
            p = np.asarray(target.p, dtype=float) + distance * back
            standoff = Pose(p=(float(p[0]), float(p[1]), float(p[2])), q=target.q)
            transit = await self._plan_move(arm, standoff, False, held=held)
            if isinstance(transit, _Refusal):
                continue
            ee, joints = self.ee[arm], self.joints[arm]
            self.ee[arm], self.joints[arm] = transit.end, transit.joints[-1]
            try:
                line = await self._straight(arm, target)
            finally:
                self.ee[arm], self.joints[arm] = ee, joints
            if isinstance(line, _Refusal):
                continue
            self.notes.append(
                f"{arm}: came in straight the last {distance * 100:.0f} cm, clear of what the "
                "other arm holds"
            )
            joined = np.vstack([transit.joints, line.joints[1:]])
            return _Path(joined, np.vstack([transit.tcp, line.tcp[1:]]), line.end)
        return None

    def _obstacles(self, arm: Arm, target: Pose) -> list[OrientedBox]:
        """The executor's obstacles grown by the margin, but those holding the arm's start or goal
        TCP."""
        ends = self.ex._tcp_positions(arm, [self.ee[arm], target])
        grown = [_grown(box, self.config.obstacle_margin_m) for box in self.ex.obstacles]
        return [box for box in grown if not _inside(box, ends).any()]

    async def _clear_of_start(
        self, arm: Arm, start: NDArray[np.float64], boxes: Sequence[OrientedBox]
    ) -> list[OrientedBox]:
        """The boxes none of the arm's collision spheres (the planner's, `FKRequest.spheres`) at
        the joints `start` reaches into: a start the planner calls colliding is in those it
        reaches into, and only they need be left out."""
        request = FKRequest(arm=arm, base=self.ex.bases[arm], joints=[start.tolist()], spheres=True)
        found = (await self.ex.planner.fk(request)).spheres
        if not found:
            return list(boxes)
        spheres = np.asarray(found[0], dtype=float).reshape(-1, 4)
        return [box for box in boxes if not _touches(box, spheres)]

    def _far(self, arm: Arm, target: Pose, boxes: Sequence[OrientedBox]) -> list[OrientedBox]:
        """The boxes farther than the hand's length (ee to TCP) from the start and goal TCP."""
        reach = float(np.linalg.norm(self.profile.arms[arm].ee_to_tcp))
        ends = self.ex._tcp_positions(arm, [self.ee[arm], target])
        return [box for box in boxes if _gap(box, ends) > reach]

    async def _lift(self, arm: Arm, target: Pose) -> _Path | None:
        """A straight rise of the TCP to `lift_clearance_m` over the tops of the obstacles near
        the move's way seen from above (those it starts in too, not one it ends in), when it
        starts lower and the gripper is empty, or a carry's rise off its support
        (`_carry_lift`); None when there is nothing to rise over or no IK line rises."""
        if self.holding[arm]:
            return await self._carry_lift(arm)
        start, goal = self.ex._tcp_positions(arm, [self.ee[arm], target])
        clearance = self.config.lift_clearance_m
        margin = self.config.obstacle_margin_m
        grown = (_grown(box, margin) for box in self.ex.obstacles)
        near = [
            box
            for box in grown
            if not _inside(box, goal).any() and _xy_gap(box, start, goal) <= clearance
        ]
        if not near:
            return None
        rise = min(max(_top(box) for box in near) + clearance - start[2], self.config.max_lift_m)
        if rise <= 0:
            return None
        return await self._rise(arm, rise)

    async def _carry_lift(self, arm: Arm) -> _Path | None:
        """A straight rise of the TCP until what the arm holds (its boxes in `Executor.held`,
        where the action starts) is `lift_clearance_m` over its support, the table top or the
        top of a perceived object under it, by at most `max_lift_m`: a free plan from a grasp
        may otherwise keep it on the table, even push it down, and drag it out of the fingers. Not
        for a pull's hold (on a fixture), nor once this action has moved the arm (the box is not
        there)."""
        held = self.ex.held.get(arm)
        if not held or arm in self.ex.pulling or any(m.arm == arm for m in self.moves):
            return None
        bottom = min(_bottom(box) for box in held)
        clearance = self.config.lift_clearance_m
        center = np.asarray(held[0].center, dtype=float)
        supports = [] if self.ex.table_top_z is None else [self.ex.table_top_z]
        supports += [
            _top(box)
            for box in self.ex.obstacles
            if _top(box) <= bottom + clearance and _xy_gap(box, center, center) == 0
        ]
        if not supports:
            return None
        rise = min(max(supports) + clearance - bottom, self.config.max_lift_m)
        if rise <= 0:
            return None
        return await self._rise(arm, rise)

    async def _rise(self, arm: Arm, rise: float) -> _Path | None:
        ee = self.ee[arm]
        up = Pose(p=(ee.p[0], ee.p[1], ee.p[2] + rise), q=ee.q)
        found = await self._straight(arm, up)
        return found if isinstance(found, _Path) else None

    async def _straight(self, arm: Arm, target: Pose) -> _Path | _Refusal:
        """IK along the line from the arm's ee to `target`, seeded with the start. The joint
        lines between neighboring solutions must keep the TCP on the line (within the waypoint
        tolerance) and turn no joint by more than `max_joint_step_rad` (a wrist spin the TCP does
        not show). Near a wrist singularity joints turn fast yet continuously, so the line is
        sampled twice and four times as densely before that counts as a change of IK branch."""
        start, base, config = self.ee[arm], self.ex.bases[arm], self.config
        jump = off = 0.0
        for density in _DENSITIES:
            step_m, step_deg = config.straight_step_m / density, config.straight_step_deg / density
            poses = line(start, target, step_m, step_deg, MAX_GOALS)
            request = IKRequest(arm=arm, base=base, goals=poses, seed=self.joints[arm].tolist())
            solved = [solution.joints for solution in await self.ex.planner.ik(request)]
            if solved[-1] is None:
                return _Refusal(ReceiptReason.unreachable, "no IK solution reaches the target")
            missing = next((i for i, q in enumerate(solved) if q is None), None)
            if missing is not None:
                along = float(np.linalg.norm(np.subtract(poses[missing].p, start.p))) * 100
                detail = f"no IK solution {along:.0f} cm along the line"
                return _Refusal(ReceiptReason.plan_failed, detail)
            joints = _unwrapped(self.joints[arm], np.asarray(solved, dtype=float))
            travel = np.abs(np.diff(joints, axis=0)).sum(axis=0)
            if travel.max() > config.max_joint_travel_rad:
                detail = f"the IK line turns joint {int(travel.argmax())} by {travel.max():.1f} rad"
                return _Refusal(ReceiptReason.plan_failed, detail)
            past = self.profile.arms[arm].past_stops(joints)
            if past is not None:
                return _Refusal(ReceiptReason.plan_failed, f"along the IK line {past}")
            tcp = self.ex._tcp_positions(arm, [start, *poses])
            jump = float(np.abs(np.diff(joints, axis=0)).max())
            off = await self._off_line(arm, joints, tcp)
            if jump <= config.max_joint_step_rad and off <= config.waypoint_tolerance_m:
                return _Path(joints, tcp, poses[-1])
        detail = (
            f"between IK solutions along the line a joint turns {jump:.2f} rad and the TCP "
            f"leaves the line by {off * 1000:.1f} mm"
        )
        return _Refusal(ReceiptReason.plan_failed, detail)

    async def _off_line(
        self, arm: Arm, joints: NDArray[np.float64], tcp: NDArray[np.float64]
    ) -> float:
        """How far the TCP strays from the line `tcp` on the joint lines between neighboring
        solutions (three points inside each)."""
        inside = np.array([0.25, 0.5, 0.75])[:, None]
        between = np.concatenate([a + inside * (b - a) for a, b in itertools.pairwise(joints)])
        return float(polyline_distance(await self.ex._tcp_fk(arm)(between), tcp).max())

    def _clamp(self, arm: Arm, target: Pose) -> Pose:
        """`target` as a displacement from the arm's ee, bounded in translation and rotation."""
        start = self.ee[arm]
        offset = np.subtract(target.p, start.p)
        distance, angle = float(np.linalg.norm(offset)), float(quat_angle_deg(start.q, target.q))
        scale = min(1.0, self.config.max_translation_m / distance) if distance else 1.0
        turn = min(1.0, self.config.max_rotation_deg / angle) if angle else 1.0
        if scale == 1.0 and turn == 1.0:
            return target
        self.clamped = True
        self.notes.append(
            f"{arm}: the displacement ({distance:.2f} m, {angle:.0f} deg) was clamped to "
            f"{distance * scale:.2f} m and {angle * turn:.0f} deg"
        )
        position = np.add(start.p, offset * scale).tolist()
        turned = quat_slerp(start.q, target.q, turn).tolist()
        return Pose(p=(position[0], position[1], position[2]), q=(*turned,))

    def _refuse(self, reason: ReceiptReason, arm: Arm, target: Pose, note: str) -> ActionReceipt:
        return ActionReceipt(
            reason=reason,
            arm=arm,
            commanded=CommandedPoses(ee=target, tcp=self.profile.tcp_pose(arm, target)),
            steps_left=steps_left(self.before),
            note="; ".join([note, *self.notes]),
        )

    def _qpos(self) -> list[float]:
        qpos = [0.0] * self.profile.action_dim
        for arm, spec in self.profile.arms.items():
            start, stop = spec.action
            qpos[start:stop] = self.joints[arm].tolist()
            qpos[spec.gripper_index] = self.gripper[arm]
        return qpos


def splits(
    start: Mapping[Arm, NDArray[np.float64]],
    goal: Mapping[Arm, NDArray[np.float64]],
    cap: float | None,
) -> int:
    """The steps that take the arms from `start` to `goal` along the joint-space line, moving no
    joint by more than `cap` in one (`ExecutorConfig.max_action_joint_step_rad`): one without a
    cap or a moving arm."""
    if cap is None or not start:
        return 1
    jump = max(float(np.abs(goal[arm] - start[arm]).max()) for arm in start)
    return max(1, math.ceil(jump / cap - 1e-9))


def _unwrapped(start: NDArray[np.float64], solved: NDArray[np.float64]) -> NDArray[np.float64]:
    """`start` and the IK solutions after it, each joint that jumped `_WRAP_RAD` or more from the
    previous one moved by whole turns to the nearest equivalent (the same pose), where that
    stays within the larger of the two magnitudes (inside the arm's limits as far as either
    is)."""
    rows = [start]
    for q in solved:
        change = q - rows[-1]
        turns = np.where(np.abs(change) >= _WRAP_RAD, np.round(change / (2 * math.pi)), 0.0)
        moved = q - 2 * math.pi * turns
        inside = np.abs(moved) <= np.maximum(np.abs(q), np.abs(rows[-1]))
        rows.append(np.where(inside, moved, q))
    return np.vstack(rows)


def _slowed(path: _Path, step_m: float) -> _Path:
    """`path` resampled so the TCP moves at most `step_m` between its points (joints and TCP
    interpolated by the distance along it), each sent: a place, slowed."""
    lengths = np.linalg.norm(np.diff(path.tcp, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(lengths)])
    if along[-1] <= step_m:  # one step anyway
        return path
    at = np.linspace(0.0, along[-1], math.ceil(along[-1] / step_m - 1e-9) + 1)
    joints = np.stack([np.interp(at, along, column) for column in path.joints.T], axis=1)
    tcp = np.stack([np.interp(at, along, column) for column in path.tcp.T], axis=1)
    return _Path(joints, tcp, path.end, every=True)


def _joined(first: ActionReceipt, second: ActionReceipt) -> ActionReceipt:
    """One receipt for an action and its retry from where it deviated (a refused retry leaves
    the first, noted)."""
    if second.steps_used == 0:
        return first.model_copy(update={"note": f"{first.note}; retrying it: {second.note}"})
    offset = first.steps_used
    arms = [*first.tcp_path, *(a for a in second.tcp_path if a not in first.tcp_path)]
    return second.model_copy(
        update={
            "steps_used": offset + second.steps_used,
            "tcp_path": {
                a: [*first.tcp_path.get(a, []), *second.tcp_path.get(a, [])] for a in arms
            },
            "grips": [
                *first.grips,
                *(g.model_copy(update={"step": g.step + offset}) for g in second.grips),
            ],
            "presses": [
                *first.presses,
                *(p.model_copy(update={"step": p.step + offset}) for p in second.presses),
            ],
            "note": f"{second.note}; planned again from where it stopped: {first.note}",
        }
    )


def _grown(box: OrientedBox, margin: float) -> OrientedBox:
    size = tuple(float(v) + 2 * margin for v in box.size)
    return box.model_copy(update={"size": size})


def _local(box: OrientedBox, points: NDArray[np.float64]) -> NDArray[np.float64]:
    return (np.atleast_2d(points) - np.asarray(box.center)) @ quat_to_matrix(box.q)


def _inside(box: OrientedBox, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    return np.all(np.abs(_local(box, points)) <= np.asarray(box.size) / 2, axis=1)


def _exit(box: OrientedBox, point: NDArray[np.float64], direction: NDArray[np.float64]) -> float:
    """How far from `point`, inside the box, along the unit `direction` the box ends."""
    local, d = _local(box, point)[0], direction @ quat_to_matrix(box.q)
    half = np.asarray(box.size) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(np.abs(d) > 1e-9, (np.sign(d) * half - local) / d, np.inf)
    return float(np.min(exits))


def _gap(box: OrientedBox, points: NDArray[np.float64]) -> float:
    """The distance from the box to the nearest of `points` (0 inside)."""
    outside = np.maximum(np.abs(_local(box, points)) - np.asarray(box.size) / 2, 0.0)
    return float(np.linalg.norm(outside, axis=1).min())


def _touches(box: OrientedBox, spheres: NDArray[np.float64]) -> bool:
    """Whether any sphere (rows x, y, z, radius) reaches into the box."""
    outside = np.maximum(np.abs(_local(box, spheres[:, :3])) - np.asarray(box.size) / 2, 0.0)
    return bool(np.any(np.linalg.norm(outside, axis=1) < spheres[:, 3]))


def _top(box: OrientedBox) -> float:
    """The height of the box's highest corner."""
    return float(_corner_heights(box).max())


def _bottom(box: OrientedBox) -> float:
    """The height of the box's lowest corner."""
    return float(_corner_heights(box).min())


def _corner_heights(box: OrientedBox) -> NDArray[np.float64]:
    corners = np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * np.asarray(box.size)
    return np.asarray((corners @ quat_to_matrix(box.q).T)[:, 2] + box.center[2], dtype=float)


def _xy_gap(box: OrientedBox, start: NDArray[np.float64], goal: NDArray[np.float64]) -> float:
    """How far the line from `start` to `goal`, seen from above, passes from the box: sampled at
    the box's mid height."""
    n = max(2, math.ceil(float(np.linalg.norm((goal - start)[:2])) / _WAY_STEP_M) + 1)
    way = start + np.linspace(0.0, 1.0, n)[:, None] * (goal - start)
    way[:, 2] = box.center[2]
    local = _local(box, way)
    outside = np.maximum(np.abs(local) - np.asarray(box.size) / 2, 0.0)
    return float(np.linalg.norm(outside, axis=1).min())


def _near(a: Pose, b: Pose) -> bool:
    close = float(np.linalg.norm(np.subtract(a.p, b.p))) <= _AT_TARGET_M
    return close and float(quat_angle_deg(a.q, b.q)) <= _AT_TARGET_DEG
