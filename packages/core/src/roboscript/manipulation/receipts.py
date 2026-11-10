"""Receipts (docs/concepts/embodied-core.md): what one action did, from its plan and the robot
state the env returned after every executed step.

The checks, each where its step executed: a move not meant to touch that ends farther than the
tolerance from its target `stopped_short` (on its planned path: blocked on the way) or `deviated`
(off it); a move meant to touch only `deviated`, when it ended far off both its path and its target
(`contact_off_path_m`); a grasp whose settled opening is below the profile's `holding_threshold` is
`empty_grasp`; an arm carrying something (closed, opening at or above the threshold) whose opening
falls below it has `lost_hold`. The reason is the first hard failure, else `budget_exhausted` at
the step limit, else the first stop or deviation, else `clamped` or `ok`.
`grips` lists the gripper changes the scene registry's grasp relations follow (`grip_events`);
`presses` the moves meant to touch, with nothing held, that stopped at least `press_m` short of
their target on their path (what the fingers met stopped them: an actuation, not a failure;
`press_events`). A receipt shows only what a real robot measures.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from roboscript.contracts.action import StepState, Stopped
from roboscript.contracts.executor import ExecutorConfig
from roboscript.contracts.geometry import Pose
from roboscript.contracts.receipt import (
    HARD_FAILURES,
    AchievedPose,
    ActionReceipt,
    CommandedPoses,
    GripEvent,
    GripperReport,
    Press,
    ReceiptReason,
)
from roboscript.contracts.robot import Arm, ArmState, RobotState
from roboscript.embodiments import EmbodimentProfile
from roboscript.manipulation.plan import MoveCheck, Plan
from roboscript.manipulation.waypoints import polyline_distance
from roboscript.spatial.rotation import quat_angle_deg

_CLOSED_BELOW: Final = 0.5
"""A gripper command below this is closed (open or closed events use the command's 0.5)."""
SETTLED: Final = 1e-3
"""A gripper whose measured opening changes by less than this in a step has settled."""


@dataclass(frozen=True)
class _Issue:
    step: int  # the batch step where it showed
    reason: ReceiptReason | None  # None: only noted (a pull's hold slipping)
    arm: Arm
    note: str
    move: MoveCheck | None = None
    achieved: Pose | None = None  # the TCP at the end of `move`


def steps_left(state: RobotState) -> int | None:
    return None if state.step_limit is None else max(0, state.step_limit - state.step)


def receipt(
    plan: Plan,
    states: list[StepState],
    stopped: Stopped,
    profile: EmbodimentProfile,
    config: ExecutorConfig,
) -> ActionReceipt:
    """The receipt of `plan` executed up to `states` (one per executed step, repeats included),
    which stopped as `stopped`."""
    last = {state.index: state.robot for state in states}  # the last state of each step
    final = states[-1].robot if states else plan.before
    threshold = {arm: spec.gripper.holding_threshold for arm, spec in profile.arms.items()}
    issues = sorted(
        [
            *_misses(plan, last, config, threshold),
            *_empty_grasps(plan, last, threshold),
            *_lost_holds(plan, states, profile, threshold),
        ],
        key=lambda issue: issue.step,
    )
    hard = [issue for issue in issues if issue.reason in HARD_FAILURES]
    focus = hard[0] if hard else None
    reported = [issue for issue in issues if issue.reason is not None]
    out_of_actions = stopped in ("step_limit", "ended")  # nothing counts after either
    if focus is None and not out_of_actions and reported:
        focus = reported[0]
    if focus is not None:
        assert focus.reason is not None
        reason = focus.reason
    elif out_of_actions:
        reason = ReceiptReason.budget_exhausted
    else:
        reason = ReceiptReason.clamped if plan.clamped else ReceiptReason.ok
    arm = focus.arm if focus is not None else (plan.arms[0] if len(plan.arms) == 1 else None)
    commanded = achieved = deviation = gripper = None
    if arm is not None:
        move = focus.move if focus is not None and focus.move else _last_move(plan, arm, last)
        tcp = focus.achieved if focus is not None and focus.achieved else final.arms[arm].tcp
        if move is not None:
            commanded = CommandedPoses(ee=move.target, tcp=move.tcp)
            achieved = AchievedPose(tcp=tcp, orient_err_deg=_angle(tcp, move.tcp))
            deviation = _distance(tcp, move.tcp)
        gripper = _gripper(final.arms[arm], threshold[arm])
    notes = [issue.note for issue in issues if issue is not focus]  # in step order
    presses = press_events(plan, states, threshold, config)
    notes.insert(0, focus.note if focus is not None else _summary(plan, last, final, threshold))
    if reason == ReceiptReason.budget_exhausted and stopped == "ended":
        notes.insert(0, "the benchmark ended the episode: no action counts any more")
    elif reason == ReceiptReason.budget_exhausted:
        notes.insert(0, f"the step limit ({final.step_limit}) was reached")
    return ActionReceipt(
        reason=reason,
        arm=arm,
        commanded=commanded,
        achieved=achieved,
        deviation_m=deviation,
        steps_used=len(states),
        steps_left=steps_left(final),
        gripper=gripper,
        tcp_path={a: [state.robot.arms[a].tcp for state in states] for a in plan.arms},
        grips=grip_events(plan.before, states, profile),
        presses=presses,
        note="; ".join([*notes, *(_pressed(press) for press in presses), *plan.notes]),
    )


def press_events(
    plan: Plan, states: Sequence[StepState], threshold: dict[Arm, float], config: ExecutorConfig
) -> list[Press]:
    """The moves meant to touch whose arm held nothing (open, or closed on nothing: the measured
    opening below the holding threshold) as they ended at least `press_m` short of the target,
    on the planned path: stopped by what the fingers met."""
    at = {state.index: n for n, state in enumerate(states)}  # a step's last executed state
    found = []
    for move in plan.moves:
        n = at.get(move.end)
        if n is None or not move.contact:
            continue
        arm = states[n].robot.arms[move.arm]
        if arm.gripper is None or holding(arm, threshold[move.arm]):
            continue  # unmeasured, or what it holds met something (a placement)
        short = _distance(arm.tcp, move.tcp)
        on_path = polyline_distance(np.array([arm.tcp.p]), move.path)[0] <= config.stop_tolerance_m
        if short >= config.press_m and on_path:
            closed = arm.gripper_cmd < _CLOSED_BELOW
            found.append(Press(arm=move.arm, step=n, tcp=arm.tcp.p, short_m=short, closed=closed))
    return found


def _pressed(press: Press) -> str:
    fingers = "closed" if press.closed else "open"
    return (
        f"{press.arm} was stopped {press.short_m * 100:.1f} cm short of its target with its "
        f"{fingers} fingers: it pressed on what it met"
    )


def chained(
    first: ActionReceipt,
    second: ActionReceipt,
    notes: Sequence[str] = (),
    *,
    moved_last: bool = False,
) -> ActionReceipt:
    """One receipt for an action run in two parts (a coarse move, then the fine rest): the
    second's outcome, the steps of both, each arm's TCP path and gripper changes in order (the
    second's steps counted after the first's), and `notes` added to its note. `moved_last`: where
    the second part moved no arm (its targets were the drive targets the first left: a placement
    whose pre-place is its place, the first already stopped there short or not), the outcome is
    the first's, the last move made (`ExecutorConfig.coarse_stops`); a second part that failed
    hard keeps its own."""
    paths = {
        arm: [*first.tcp_path.get(arm, []), *second.tcp_path.get(arm, [])]
        for arm in {*first.tcp_path, *second.tcp_path}
    }
    later = [g.model_copy(update={"step": g.step + first.steps_used}) for g in second.grips]
    pressed = [p.model_copy(update={"step": p.step + first.steps_used}) for p in second.presses]
    update = {
        "steps_used": first.steps_used + second.steps_used,
        "look_steps": first.look_steps + second.look_steps,
        "tcp_path": paths,
        "grips": [*first.grips, *later],
        "presses": [*first.presses, *pressed],
        "note": "; ".join(filter(None, [second.note, *notes])),
    }
    unmoved = second.commanded is None and first.commanded is not None
    if moved_last and unmoved and not second.hard_failure:
        update |= {
            "reason": first.reason,
            "arm": first.arm,
            "commanded": first.commanded,
            "achieved": first.achieved,
            "deviation_m": first.deviation_m,
            "note": "; ".join(filter(None, [first.note, *notes])),
        }
    return second.model_copy(update=update)


def grip_events(
    before: RobotState, states: Sequence[StepState], profile: EmbodimentProfile
) -> list[GripEvent]:
    """The gripper changes of an action that started as `before` and executed `states` (one per
    executed step, repeats included), in step order: per arm, a close once its opening settled
    (the last state of a step repeated until the fingers stopped, or one whose opening changed by
    less than `SETTLED`), an open where it was first commanded, and a lost hold where a holding
    gripper's opening first falls below the holding threshold (where the receipt says
    `lost_hold`). What an action tool hands the scene registry's grasp relations
    (`roboscript.toolsets.scene.after_action`)."""
    events: list[GripEvent] = []
    for arm, spec in profile.arms.items():
        threshold = spec.gripper.holding_threshold
        previous = before.arms[arm]
        closed = previous.gripper_cmd < _CLOSED_BELOW
        held = holding(previous, threshold)
        for n, state in enumerate(states):
            now = state.robot.arms[arm]
            last = n + 1 == len(states) or states[n + 1].index != state.index
            repeated = n > 0 and states[n - 1].index == state.index
            if now.gripper_cmd >= _CLOSED_BELOW:
                if closed:
                    events.append(GripEvent(arm=arm, step=n, closed=False, opening=now.gripper))
                closed = held = False
            elif held and now.gripper is not None and now.gripper < threshold:
                events.append(GripEvent(arm=arm, step=n, closed=True, opening=now.gripper))
                held = False
            elif last and (repeated or _still(previous, now)):
                holds = now.gripper is not None and now.gripper >= threshold
                if not closed or holds != held:
                    events.append(GripEvent(arm=arm, step=n, closed=True, opening=now.gripper))
                closed, held = True, holds
            previous = now
    return sorted(events, key=lambda event: event.step)


def _still(a: ArmState, b: ArmState) -> bool:
    return a.gripper is not None and b.gripper is not None and abs(a.gripper - b.gripper) < SETTLED


def _misses(
    plan: Plan, last: dict[int, RobotState], config: ExecutorConfig, threshold: dict[Arm, float]
) -> Iterator[_Issue]:
    """Moves that ended off their target: on their path, short of it (`stopped_short`), else off
    it (`deviated`). A move meant to touch may stop short on its path (what it met: a press or a
    place, or be pushed aside a little); one that ended far off its path and target
    (`contact_off_path_m`) never got where it was going (a grasp that closes there closes away
    from its object), and is `deviated` too. A place (a move meant to touch made holding
    something) that ended farther than `contact_off_path_m` from its target, beyond what it was
    to push past (`MoveCheck.past_m`), did not put what it holds where it was to go, on its path
    or not: a place that stops short and then lets go must not read `ok`."""
    for move in plan.moves:
        state = last.get(move.end)
        if state is None:
            continue
        tcp = state.arms[move.arm].tcp
        off, turn = _distance(tcp, move.tcp), _angle(tcp, move.tcp)
        if off <= config.stop_tolerance_m and turn <= config.stop_tolerance_deg:
            continue
        away = float(polyline_distance(np.array([tcp.p]), move.path)[0])
        on_path = away <= config.stop_tolerance_m
        placing = move.contact and holding(state.arms[move.arm], threshold[move.arm])
        short = placing and off > config.contact_off_path_m + move.past_m
        if move.contact and min(away, off) <= config.contact_off_path_m and not short:
            continue
        if short:
            note = (
                f"{move.arm} stopped {off * 100:.1f} cm from where it was to place what it holds"
                f" ({'on' if on_path else 'off'} its way there): it is not placed there"
            )
            reason = ReceiptReason.stopped_short if on_path else ReceiptReason.deviated
            yield _Issue(move.end, reason, move.arm, note, move, tcp)
        elif off > config.stop_tolerance_m and on_path:
            note = f"{move.arm} stopped {off * 100:.1f} cm short of its target"
            yield _Issue(move.end, ReceiptReason.stopped_short, move.arm, note, move, tcp)
        else:
            note = f"{move.arm} ended {off * 100:.1f} cm and {turn:.0f} deg away from its target"
            yield _Issue(move.end, ReceiptReason.deviated, move.arm, note, move, tcp)


def _empty_grasps(
    plan: Plan, last: dict[int, RobotState], threshold: dict[Arm, float]
) -> Iterator[_Issue]:
    for grip in plan.grips:
        state = last.get(grip.end)
        if not grip.grasp or state is None:
            continue
        opening = state.arms[grip.arm].gripper
        if opening is not None and opening < threshold[grip.arm]:
            note = f"{grip.arm} closed on nothing: the fingers settled at {opening:.3f}"
            yield _Issue(grip.end, ReceiptReason.empty_grasp, grip.arm, note)


def _lost_holds(
    plan: Plan, states: list[StepState], profile: EmbodimentProfile, threshold: dict[Arm, float]
) -> Iterator[_Issue]:
    """An arm carries after a close settles on something (or when the action starts so); it
    stops carrying when commanded open, or when its opening falls below the threshold: a lost
    hold, only noted for a lenient hold (a pull's, which may slip at the end of the pull, or one
    a close that is no grasp took, a press pinching something)."""
    carrying = {arm: holding(plan.before.arms[arm], threshold[arm]) for arm in profile.arms}
    lenient = set(plan.lenient)
    settles = {(grip.end, grip.arm): grip for grip in plan.grips if grip.value < _CLOSED_BELOW}
    for n, state in enumerate(states):
        qpos = plan.steps[state.index].qpos
        settled = n + 1 == len(states) or states[n + 1].index != state.index
        for arm, spec in profile.arms.items():
            opening = state.robot.arms[arm].gripper
            if qpos[spec.gripper_index] >= _CLOSED_BELOW:
                carrying[arm] = False
            elif carrying[arm] and opening is not None and opening < threshold[arm]:
                carrying[arm] = False
                if arm in lenient:
                    note = f"{arm} let go of what its last close held (opening {opening:.3f})"
                    yield _Issue(state.index, None, arm, note)
                else:
                    note = f"{arm} lost its hold: the opening fell to {opening:.3f}"
                    yield _Issue(state.index, ReceiptReason.lost_hold, arm, note)
            if settled and (grip := settles.get((state.index, arm))) is not None:
                carrying[arm] = opening is not None and opening >= threshold[arm]
                if grip.pull or not grip.grasp:
                    lenient.add(arm)
                else:
                    lenient.discard(arm)


def _last_move(plan: Plan, arm: Arm, last: dict[int, RobotState]) -> MoveCheck | None:
    executed = [move for move in plan.moves if move.arm == arm and move.end in last]
    return executed[-1] if executed else None


def _summary(
    plan: Plan, last: dict[int, RobotState], final: RobotState, threshold: dict[Arm, float]
) -> str:
    """One clause per arm: how far from its target, what its gripper holds."""
    parts = []
    for arm in plan.arms:
        words = []
        move = _last_move(plan, arm, last)
        if move is not None:
            off = _distance(final.arms[arm].tcp, move.tcp)
            words.append(f"{off * 1000:.0f} mm from its target")
        if any(grip.arm == arm and grip.end in last for grip in plan.grips):
            words.append(_grip_words(final.arms[arm], threshold[arm]))
        parts.append(f"{arm} " + " and ".join(words or ["did not move"]))
    return ", ".join(parts) or "held still"


def _grip_words(state: ArmState, threshold: float) -> str:
    if state.gripper is None:
        return f"gripper commanded {state.gripper_cmd:.2f}"
    if state.gripper_cmd >= _CLOSED_BELOW:
        return f"gripper open ({state.gripper:.3f})"
    holding = state.gripper >= threshold
    return f"{'holding' if holding else 'closed on nothing'} (opening {state.gripper:.3f})"


def _gripper(state: ArmState, threshold: float) -> GripperReport:
    return GripperReport(
        commanded=state.gripper_cmd,
        measured=state.gripper,
        holding=None if state.gripper is None else holding(state, threshold),
    )


def holding(state: ArmState, threshold: float) -> bool:
    """The gripper is commanded closed and its measured opening says it holds something."""
    closed = state.gripper_cmd < _CLOSED_BELOW
    return closed and state.gripper is not None and state.gripper >= threshold


def _distance(a: Pose, b: Pose) -> float:
    return float(np.linalg.norm(np.subtract(a.p, b.p)))


def _angle(a: Pose, b: Pose) -> float:
    return float(quat_angle_deg(a.q, b.q))
