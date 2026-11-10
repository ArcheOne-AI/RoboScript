"""The fitted calls in the demonstration's scene, through the core's API on the env planes (the
entry check's episodes, `core_entry.check.Check`), judged by the control plane's truth.

**Explaining a segment:** a fresh episode replays the demonstration's commands up to the
segment's start (a place's or a carry's from where its grasp settled, the gripper then closed as
the API closes it; a release's up to its open, the other arm then holding), runs the fitted call
(a place with its fitted `constrain` and approach), and judges where it ended: a grasp by the
object rising with a lift (`Criteria.held_m`, the lift kept within the benchmark's end rules); a
place by the object resting at the demonstration's resting point (`placed_m` across,
`resting_m` up), a carry the same at the recording's end, the object kept; a touch (a press reused
for it) by the object where the demonstration's hold ends, as the demonstration left it
across, up and turned (the move tolerance's angle: a nut it turned is not pressed still); a
release by the object staying up (it drops less than `held_m`: the other arm has it); a move by
the TCP ending where the demonstration's did (the entry check's move tolerance); a home by the
arm back at its start pose: `exact` at the move tolerance, its gripper open unless the
demonstration's was closed (where a demonstration may end an arm), else within the benchmark's own
back-to-origin check (`origin_tolerance`, its official value, given by the glue). A call whose
receipt is not ok is an entry that fails; one that reported success and ended off the segment's end
is a fit out of tolerance.

**The ceiling** (privileged, scripted: truth gives the objects, only the skillset's calls act)
runs every planned call from the reset, mid-task homes included, then sends each arm home, and
records whether the official success latched, with the benchmark's success checks at the end
and their margins (`diagnose`'s `conditions`). A segment no skill fits is where the ceiling fails
(recorded, `no_skill`): the plan cannot do the task. The calls run in the order their segments
end; a segment of the other arm that moves while this one does (`active`, from the recorded
commands) runs at once with it, as the demonstration ran them (`ActionApi.run` with one action
per arm), where both are the core's single actions (a written skill's motions run on their own:
`sequential_overlaps` counts those). Where the demonstration's arms both stood still before a
call (its arm starts moving after every segment run so far had ended: a support arm's turn, a
wait for something to settle), the robot holds still as long first (`wait`, demonstration
frames; the demonstration's clock starts at the actions the episode counted before the first
call, `opened_at`, and a touch's call holds to its open, `performed`). Each segment may use at
most `step_budget` actions (the executor's batches are cut there: `budget_exhausted`), so no one
call eats the episode's steps; a grasp whose fingers closed farther than `GRASP_SHORT_M` from its
grasp pose is reported not ok (`honest`), whatever the executor's contact rules made of it.

**The chained explanation** (`chained`, behind the glue's `cover_demos.py --chained`): one rollout
per task from the reset, each segment run from the state the calls before it left and judged as
`explain` judges it, plus the benchmark's own stages (`GtSnapshot.progress`: left before and after
each segment, where the env tracks them); the first segment that is not explained is the task's
gap (its receipt's reason as the mechanism), the ones after it not reached. A move or a home
through which the demonstration carried an object the cut read no grasp of (`Segment.carries`)
is explained only where that object ended within `placed_m` of where the demonstration had it
at the segment's end (`left_behind`, mechanism "left behind"): a carry the plan misses breaks
the chain where it happens, not where its object is next needed. Given the demonstration's stages
(`demo_stages`), a segment whose receipts all reached but that ended off the demonstration's end
counts explained where it did the benchmark's own work instead: it completed a stage and left no
more stages to do than the demonstration had left after it (`stage_met`, `explained_by` "stages").
The demonstration's end may lie only just across the official boundary, so a call that ends inside
the success region but away from it is not held to it; a segment that completes no stage is judged
by its criteria.

**A task plan's transitions** (`Call.before`): calls a task plan puts before a segment's
call, run from where the scene is (`transitions`), in the chain and the ceiling: a written skill's
`act` fitted on that segment, or a core entry that takes the arm only; one whose receipt is not
ok is an entry of the segment that fails.

**The chained reuse** (`chain_reuse`, the glue's `cover_demos.py --chain`):
where the chain breaks, that segment's alternatives are tried in rollouts of their own from the
reset (the calls before it run first); the first that explains it, with the calls before it
explained and no fewer segments explained in all, replaces the plan's call and the chain is judged
on. A chain that explains every segment without the official success breaks where its stages fell
behind the demonstration's for good (`lagging`, the demonstration replayed for its stages:
`demo_stages`). A session's case on such a gap (`chained_case`) runs its call in place of the plan's
the same way.

**A written skill's call** (`../builder/README.md`) runs through its `act` where the core
entry's motion would run, after the same pre-steps and judged by the same criteria: `act` gets
the API, the segment's arm, its object and its target from truth now (None where the segment has
none or truth has no pose for it) and the call's `params`; the motions it returns run in order
until one's receipt is not ok. A skill whose `act` raises (anything: an infrastructure error it
let through too), runs past `ACT_S`, returns nothing to run or more than `MOTIONS` motions, or one
of whose motions fails to run for a reason not the infrastructure's (`InfraError`, `OSError`), is
an entry that fails (`act_error`), never an infrastructure error. Its `act` runs between two
`ACT_MARK` lines on stderr: a job stopped between them (a loop that never awaits) was stopped in
the skill.

**Explaining from another start** (the gate's transfer cases): `explain` with `start_m` first
checks that the replay put the segment's objects where the recording had them at its start
(within `start_m`); a replay that did not is `start missed`: that demonstration cannot tell, the
same every time, so the caller treats it as not applicable rather than as infrastructure.
"""

import asyncio
import math
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from roboscript.contracts.action import ActionBatch, ActionStep, ExecuteResult
from roboscript.contracts.errors import InfraError
from roboscript.contracts.ids import BatchId, new_id
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.robot import Arm
from roboscript.demos import gt_scene
from roboscript.demos.gt_scene import obstacles
from roboscript.manipulation import Built
from roboscript.manipulation.paths import hold_frames
from roboscript.manipulation.proxy import ObjectProxy
from roboscript.spatial.rotation import quat_angle_deg
from roboscript_methods.skill_builder.core_entry.check import (
    LIFT_M,
    RULE_MARGIN_M,
    Check,
    Criteria,
    Episode,
    binding_rule,
    move_tolerance,
    receipt_row,
    tcp_error,
)
from roboscript_methods.skill_builder.demo_coverage.fitting import ACT_MARK, TRANSITION_CORE, Call
from roboscript_methods.skill_builder.demo_coverage.segments import Recording, Segment

if TYPE_CHECKING:
    from roboscript_methods.skill_builder.builder.plugins import WrittenSkill

REPLAY_CHUNK: Final = 100
"""Commands per batch of a replay."""
GRIP_CLOSED_BELOW: Final = 0.5
"""A gripper command below this is closed (the executor's reading of a command)."""
ACT_S: Final = 900.0
"""A written skill's `act` that has not returned by then has failed (its planning calls are
bounded each; a case run is stopped at 1800 s)."""
MOTIONS: Final = 32
"""Motions a written skill's `act` may return for one segment, at most."""
STEP_FACTOR: Final = 3.0
STEP_MARGIN: Final = 60
"""A segment's actions in the ceiling, at most: `STEP_FACTOR` times its frames plus
`STEP_MARGIN`."""
GRASP_SHORT_M: Final = 0.02
"""A grasp whose fingers closed farther than this from its grasp pose did not grasp where it
was made to, whatever the executor's contact rules call it."""
ACTIVE_RAD: Final = 0.02
ACTIVE_GRIP: Final = 0.05
"""An arm moves in a segment from the first frame its commanded joints differ from the
segment's start by more than `ACTIVE_RAD` (or its gripper by `ACTIVE_GRIP`): a segment starts at
the arm's last event, and an arm waiting for the other is not moving."""
SETTLE_FRAMES: Final = 10
"""Frames after a close the replay of a place runs on, for the grasp to settle (the executor
binding's `settle_holds`)."""
REPLAY_LATCHED: Final = (
    "the demonstration's replay latched the benchmark's success before the segment starts, and "
    "no action runs after the latch: the segment cannot be judged"
)
"""Why a segment whose replayed prefix latched the success is not judged (`replay ended`)."""


async def replay(check: Check, ep: Episode, commands: np.ndarray, upto: int, since: int = 0) -> str:
    """Send the recorded commands from frame `since` up to frame `upto`; how the last batch
    stopped. Each batch stops at the benchmark's success latch, as a benchmark may take no
    action after it: a replay that latches stops there (`success`)."""
    stopped = "completed"
    for begin in range(since, upto, REPLAY_CHUNK):
        steps = [
            ActionStep(qpos=[float(v) for v in q])
            for q in commands[begin : min(upto, begin + REPLAY_CHUNK)]
        ]
        batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps)
        stopped = (await ep.agent.execute(ep.opened.episode_id, batch)).stopped
        if stopped != "completed":
            break
    ep.ended = ep.ended or (await check.control.judge(ep.opened.episode_id)).ended
    return stopped


async def close_fully(check: Check, ep: Episode, arm: Arm) -> None:
    """One hold with `arm`'s gripper commanded closed, so the executor reads the hold (an
    operator's partial close may command more than the API's closed)."""
    state = await check.robot(ep)
    spec = check.profile.arms[arm].gripper
    qpos = list(state.qpos_cmd)
    qpos[check.profile.arms[arm].gripper_index] = spec.closed
    batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=[ActionStep(qpos=qpos)])
    await ep.agent.execute(ep.opened.episode_id, batch)


@dataclass
class Ran:
    """A call's motion: its receipts, its rows for the report, and whether it failed before any
    receipt could say so (a written skill that raised or built nothing)."""

    receipts: list[ActionReceipt]
    row: dict[str, Any] = field(default_factory=dict)
    failed: bool = False


def written(skills: Sequence["WrittenSkill"], call: Call) -> "WrittenSkill | None":
    """The written skill `call` names, if it names one of `skills`."""
    return next((k for k in skills if k.name == call.skill), None)


async def act_written(
    check: Check,
    ep: Episode,
    skill: "WrittenSkill",
    call: Call,
    truth: gt_scene.Scene,
    honest_grasps: bool = False,
) -> Ran:
    """`skill.act` for `call`'s segment on truth now (bounded, `ACT_S`), its motions run in order
    until one is not ok (or the benchmark ends the episode; `honest_grasps`: a grasp closed short
    of its pose is not ok)."""
    s = call.segment
    obj = truth.proxy(s.obj) if s.obj in truth.poses else None
    target = truth.proxy(s.target) if s.target in truth.poses else None

    def failed(why: str, receipts: Sequence[ActionReceipt] = ()) -> Ran:
        return Ran(list(receipts), {"motions": [receipt_row(r) for r in receipts],
                                    "act_error": why}, True)  # fmt: skip

    async def contained() -> Any:
        """`act`'s result, or what it raised (caught here: a task lets `SystemExit` out)."""
        try:
            return await skill.act(ep.api, s.arm, obj, target, dict(call.params))
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # the Agent's code: whatever it raises is its failure (an
            return error  # infrastructure error it raised could otherwise hide a loss)

    print(f"{ACT_MARK} begin {skill.name}", file=sys.stderr, flush=True)
    try:
        built = await asyncio.wait_for(contained(), ACT_S)
    except asyncio.TimeoutError:
        return failed(f"act did not return within {ACT_S:.0f} s")
    finally:
        print(f"{ACT_MARK} end", file=sys.stderr, flush=True)
    if isinstance(built, BaseException):
        return failed(f"{type(built).__name__}: {built}")
    if not isinstance(built, list) or not all(isinstance(b, Built) for b in built):
        return failed(f"TypeError: act returned {type(built).__name__}, not a list of Built")
    if len(built) > MOTIONS:
        return failed(f"act built {len(built)} motions, more than {MOTIONS}")
    receipts: list[ActionReceipt] = []
    for k, motion in enumerate(built):
        try:
            done = await check.run(ep, motion)
            receipts.append(honest(done, motion) if honest_grasps else done)
        except Exception as error:  # a motion the skill built that does not run is its failure
            if infrastructure(error):
                raise
            return failed(f"its motion {k} did not run: {type(error).__name__}: {error}", receipts)
        if not receipts[-1].ok or ep.ended:
            break
    row: dict[str, Any] = {"motions": [receipt_row(r) for r in receipts]}
    if not built:
        row["act_error"] = "act built no motion"
    return Ran(receipts, row, not built)


async def transitions(
    check: Check, ep: Episode, call: Call, skills: Sequence["WrittenSkill"]
) -> tuple[bool, list[dict[str, Any]]]:
    """`call`'s transitions (`Call.before`), in order, each from where the scene is: a written
    skill's `act` (with its params fitted on `call`'s segment), or a core entry of
    `TRANSITION_CORE` on the segment's arm. Whether every receipt was ok, and a row per
    transition; stops at the first that was not (or at the benchmark's end)."""
    rows: list[dict[str, Any]] = []
    for t in call.before:
        skill = written(skills, t)
        if skill is not None:
            ran = await act_written(check, ep, skill, t, await check.truth(ep), honest_grasps=True)
            ok = not ran.failed and all(r.ok for r in ran.receipts)
            rows.append({"transition": t.skill, "ok": ok, **ran.row})
        else:
            assert t.skill in TRANSITION_CORE, t.skill  # a plan's transitions are checked built
            arm = t.segment.arm
            built = (await ep.api.return_home(arm) if t.skill == "return_home"
                     else ep.api.open_gripper(arm))  # fmt: skip
            receipt = await check.run(ep, built)
            ok = receipt.ok
            rows.append({"transition": t.skill, **receipt_row(receipt)})
        if not ok or ep.ended:
            return False, rows
    return True, rows


def stage_met(row: dict[str, Any], demo: dict[str, int] | None) -> bool:
    """Whether a chained segment that every receipt reached but that ended off the
    demonstration's end (`fit out of tolerance`) did the benchmark's work there instead: it
    completed a stage (fewer left after it than before) and left no more than the demonstration
    had left after the same segment (`demo`, `demo_stages`)."""
    stages = row.get("stages")
    then = (demo or {}).get(row["segment"])
    return bool(
        not row.get("explained")
        and row.get("cause") == "fit out of tolerance"
        and stages is not None
        and then is not None
        and stages["after"] < stages["before"]
        and stages["after"] <= then
    )


def infrastructure(error: BaseException) -> bool:
    """An error that is the infrastructure's (a service's, the network's), not the code's that
    raised it through."""
    return isinstance(error, (InfraError, OSError))


def entry_failed(*receipts: ActionReceipt) -> bool:
    """Whether the entry did not do what it was told: a receipt that is not ok (refused, failed
    hard, stopped short or deviated) is the entry's failure, not the fit's."""
    return not all(receipt.ok for receipt in receipts)


async def explain(
    check: Check,
    recording: Recording,
    call: Call,
    criteria: Criteria,
    origin_tolerance: tuple[float, float],
    skills: Sequence["WrittenSkill"] = (),
    start_m: float | None = None,
    final: bool = False,
) -> dict[str, Any]:
    """Run `call` from its segment's start in a fresh episode: the row says whether it explains
    the segment, and if not why (`entry fails`, `fit out of tolerance`). A call naming one of
    `skills` runs through its `act`. With `start_m`, a replay that left the segment's objects
    further than that from the recording's poses at the segment's start is `start missed`.
    `final`: a home among the cut's closing ones (`closing`), judged as the episode's end is."""
    s = call.segment
    started = time.monotonic()
    row: dict[str, Any] = {"segment": s.name, "skill": call.skill}
    settled = min(s.start + SETTLE_FRAMES, s.end)
    upto = {"place": settled, "carry": settled, "release": s.end}.get(s.kind, s.start)
    async with check.episode() as ep:
        stopped = await replay(check, ep, recording.commands, upto)
        row["replayed"] = {"frames": upto, "stopped": stopped}
        if stopped == "success":  # nothing runs after the latch: the segment cannot be judged
            return row | {"explained": False, "cause": "replay ended", "ended": ep.ended,
                          "why": REPLAY_LATCHED}  # fmt: skip
        if ep.ended or stopped != "completed":
            return row | {"explained": False, "cause": "replay ended", "ended": ep.ended}
        truth = await check.truth(ep)
        if start_m is not None:
            off = start_error(check, recording, s, truth, upto)
            row["start_off_m"] = round(off, 4)
            if off > start_m:
                return row | {"explained": False, "cause": "start missed",
                              "why": "the replay did not reach the segment's start"}  # fmt: skip
        found = await perform(check, ep, recording, call, criteria, origin_tolerance, skills,
                              replayed=True, final=final)  # fmt: skip
        verdict = {k: found.pop(k) for k in ("explained", "cause", "ended") if k in found}
        row |= found
        row["seconds"] = round(time.monotonic() - started, 1)
        return row | verdict


async def perform(
    check: Check,
    ep: Episode,
    recording: Recording,
    call: Call,
    criteria: Criteria,
    origin_tolerance: tuple[float, float],
    skills: Sequence["WrittenSkill"] = (),
    replayed: bool = False,
    honest_grasps: bool = False,
    final: bool = False,
) -> dict[str, Any]:
    """`call` run in `ep` from where it is and judged (`explain`'s criteria): the row's motions,
    measures, `explained` and `cause`. `replayed`: the episode was replayed to the segment's
    start (a place first closes the gripper as the API closes it: an operator's partial close);
    `honest_grasps`: a grasp closed short of its pose is not ok (`honest`); `final`: a home
    among the closing ones (`closing`, `home_judged`)."""
    s = call.segment
    skill = written(skills, call)
    row: dict[str, Any] = {}
    truth = await check.truth(ep)
    if s.kind == "grasp":
        assert s.obj is not None
        before = truth.poses[s.obj].p[2]
        if skill is None:
            assert call.grasp is not None
            made = await ep.api.grasp_object(s.arm, truth.proxy(s.obj), call.grasp)
            grasp = await check.run(ep, made)
            grasp = honest(grasp, made) if honest_grasps else grasp
            ran = Ran([grasp], {"grasp": receipt_row(grasp)})
        else:
            ran = await act_written(check, ep, skill, call, truth, honest_grasps)
        rule = binding_rule(ep.first.end_rules, s.obj)
        lift_m = LIFT_M if rule is None else min(LIFT_M, rule.limit_m - RULE_MARGIN_M)
        lift = await check.run(ep, await ep.api.move_by(s.arm, z=lift_m))
        rose = float((await check.truth(ep)).poses[s.obj].p[2] - before)
        done = rose >= criteria.held_m
        row |= ran.row | {"lift": receipt_row(lift), "rose_m": round(rose, 4)}
        failed = ran.failed or entry_failed(*ran.receipts, lift)
    elif s.kind in ("place", "carry"):
        assert s.obj is not None and s.rest is not None
        if replayed:
            await close_fully(check, ep, s.arm)
            truth = await check.truth(ep)
        if skill is None:
            assert call.place is not None
            target = truth.proxy(s.target) if s.target in truth.poses else None
            built = await ep.api.place_object(
                s.arm,
                truth.proxy(s.obj),
                call.place,
                target,
                is_open=s.kind == "place",
                constrain=call.constrain,
                approach_m=call.approach_m,
                approach_axis=call.approach_axis,
            )
            place = await check.run(ep, built)
            ran = Ran([place], {"place": receipt_row(place)})
        else:
            ran = await act_written(check, ep, skill, call, truth, honest_grasps)
        off, dz = rest_error(check, recording, s, await check.truth(ep), s.rest)
        done = off <= criteria.placed_m and abs(dz) <= criteria.resting_m
        row |= ran.row | {"off_m": round(off, 4), "dz_m": round(dz, 4)}
        failed = ran.failed or entry_failed(*ran.receipts)
    elif s.kind == "touch":  # a press or a held path reused for it
        assert s.obj is not None and s.rest is not None
        if skill is None:
            assert call.grasp is not None
            built = await ep.api.grasp_object(s.arm, truth.proxy(s.obj), call.grasp)
            pressed = await check.run(ep, built)
            ran = Ran([pressed])
            if call.path is not None and pressed.ok:
                ran.receipts.append(await follow_hold(check, ep, call))
                ran.row["path"] = receipt_row(ran.receipts[-1])
                await check.run(ep, ep.api.open_gripper(s.arm))
            ran.row["grasp"] = receipt_row(pressed)
        else:
            ran = await act_written(check, ep, skill, call, truth, honest_grasps)
        now = await check.truth(ep)
        off, dz = rest_error(check, recording, s, now, s.rest)
        then = recording.poses(s.rest)[s.obj]
        turn = float(quat_angle_deg(now.poses[s.obj].q, then.q))
        done = (
            off <= criteria.placed_m
            and abs(dz) <= criteria.resting_m
            and turn <= move_tolerance(criteria, check.config)[1]
        )
        row |= ran.row | {"off_m": round(off, 4), "dz_m": round(dz, 4)}
        row["turn_deg"] = round(turn, 2)
        failed = ran.failed or entry_failed(*ran.receipts)
    elif s.kind == "move":
        await keep_grip(check, ep, s)
        if skill is None:
            ee = check.profile.ee_pose(s.arm, s.tcp)
            moved = await check.run(ep, await ep.api.move_to_pose(s.arm, ee))
            ran = Ran([moved], {"move": receipt_row(moved)})
        else:
            ran = await act_written(check, ep, skill, call, await check.truth(ep), honest_grasps)
        error = tcp_error((await check.robot(ep)).arms[s.arm].tcp, s.tcp)
        done = check.reached(error)
        row |= ran.row | error
        failed = ran.failed or entry_failed(*ran.receipts)
    elif s.kind == "release":
        assert s.obj is not None
        before = truth.poses[s.obj].p[2]
        if skill is None:
            opened = await check.run(ep, ep.api.open_gripper(s.arm))
            ran = Ran([opened], {"release": receipt_row(opened)})
        else:
            ran = await act_written(check, ep, skill, call, truth, honest_grasps)
        dropped = float(before - (await check.truth(ep)).poses[s.obj].p[2])
        done = dropped < criteria.held_m
        row |= ran.row | {"dropped_m": round(dropped, 4)}
        failed = ran.failed or entry_failed(*ran.receipts)
    else:  # home
        if skill is None:
            home = await check.run(ep, await ep.api.return_home(s.arm))
            ran = Ran([home], {"home": receipt_row(home)})
        else:
            ran = await act_written(check, ep, skill, call, truth, honest_grasps)
        state = await check.robot(ep)
        now, then = state.arms[s.arm].ee, ep.start.arms[s.arm].ee
        axis = float(np.abs(np.subtract(now.p, then.p)).max())
        angle = float(quat_angle_deg(now.q, then.q))
        opened = s.closed or state.arms[s.arm].gripper_cmd >= GRIP_CLOSED_BELOW
        done, exact = home_judged(axis, angle, opened, move_tolerance(criteria, check.config),
                                  origin_tolerance, final)  # fmt: skip
        row |= ran.row | {"axis_m": round(axis, 4), "deg": round(angle, 2), "exact": exact}
        row["gripper_cmd"] = round(state.arms[s.arm].gripper_cmd, 3)
        failed = ran.failed or entry_failed(*ran.receipts)
    if ep.ended:
        return row | {
            "explained": False,
            "cause": "the benchmark ended the episode",
            "ended": ep.ended,
        }
    cause = None if done else "entry fails" if failed else "fit out of tolerance"
    return row | {"explained": done, "cause": cause}


def home_judged(
    axis: float,
    angle: float,
    opened: bool,
    tight: tuple[float, float],
    origin_tolerance: tuple[float, float],
    final: bool,
) -> tuple[bool, bool]:
    """A home's end (per-axis metres, degrees off the arm's start pose, its gripper as the
    demonstration's end has it): whether it is back, and whether exactly. Its target is the start
    pose itself (a demonstration may end an arm exactly there, gripper open): exact at the move
    tolerance (`tight`). Short of that, a `final` home (only homes after it, `closing`: what the
    benchmark's own end check, `origin_tolerance`, then judges) back within that check still counts,
    so it does not stop a chain (`return_home` may leave an arm already within that check where
    it is, and open no gripper; judged at the move tolerance alone such homes would
    break chains the benchmark counts as done). A home with other work after it is judged exactly:
    an arm left short could be in the way of what follows, whose segment would then take the
    blame."""
    exact = (axis <= min(tight[0], origin_tolerance[0])
             and angle <= min(tight[1], origin_tolerance[1]) and opened)  # fmt: skip
    back = final and axis <= origin_tolerance[0] and angle <= origin_tolerance[1]
    return exact or back, exact


def closing(segments: Sequence[Segment], s: Segment) -> bool:
    """Whether `s` is a home with only homes after it in `segments` (the cut, in its order):
    one of the plan's closing homes, which the episode's end check judges."""
    k = [x.name for x in segments].index(s.name)
    return s.kind == "home" and all(x.kind == "home" for x in segments[k:])


async def keep_grip(check: Check, ep: Episode, s: Segment) -> None:
    """Before a move, the gripper as the demonstration had it through it: closed for a press with
    the fingertips, open otherwise."""
    commanded = (await check.robot(ep)).arms[s.arm].gripper_cmd
    if s.closed and commanded >= GRIP_CLOSED_BELOW:
        await check.run(ep, ep.api.close_gripper(s.arm))
    elif not s.closed and commanded < GRIP_CLOSED_BELOW:
        await check.run(ep, ep.api.open_gripper(s.arm))


async def follow_hold(check: Check, ep: Episode, call: Call) -> ActionReceipt:
    """A touch's held path (`call.path`) followed from where its grasp holds, the scene's objects
    as obstacles, in the held object's frames as measured now."""
    s = call.segment
    assert call.path is not None and s.obj is not None
    truth = await check.truth(ep)
    ep.api.executor.obstacles = obstacles(truth, await check.robot(ep), 0.0)
    frames = hold_frames(truth.proxy(s.obj), call.path.x_toward)
    receipt = await ep.api.follow_path([s.arm], frames, call.path)
    ep.ended = ep.ended or (await check.control.judge(ep.opened.episode_id)).ended
    return receipt


def start_error(
    check: Check, recording: Recording, s: Segment, now: gt_scene.Scene, frame: int
) -> float:
    """How far the segment's object and target are, at most, from where the recording had them
    at `frame` (their poses' positions)."""
    then = recording.poses(frame)
    names = [n for n in (s.obj, s.target) if n is not None and n in then and n in now.poses]
    return max(
        (float(np.linalg.norm(np.subtract(now.poses[n].p, then[n].p))) for n in names), default=0.0
    )


def rest_error(
    check: Check, recording: Recording, s: Segment, now: gt_scene.Scene, frame: int
) -> tuple[float, float]:
    """How far the segment's object's bottom centre ended from where the demonstration's was at
    `frame`: across and up."""
    assert check.measured is not None and s.obj is not None
    then = check.measured.at(recording.poses(frame), {}).proxy(s.obj)
    found = now.proxy(s.obj)
    a, b = bottom(found), bottom(then)
    return float(np.linalg.norm(a[:2] - b[:2])), float(a[2] - b[2])


def bottom(proxy: ObjectProxy) -> np.ndarray:
    box = proxy.pose @ proxy.whole.frame
    return np.asarray(box[:3, 3] - box[:3, 2] * proxy.whole.size[2] / 2)


def step_budget(s: Segment) -> int:
    """The actions `s`'s call may use in the ceiling (`STEP_FACTOR`, `STEP_MARGIN`)."""
    return math.ceil(STEP_FACTOR * max(s.end - s.start, 0)) + STEP_MARGIN


class Budgeted:
    """The env agent an episode's executor sends its batches to, each batch cut at `left`
    actions (None: no cut): a cut batch stops `step_limit`, which its receipt reports as
    `budget_exhausted`; `used` counts the actions sent since `left` was set."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.left: int | None = None
        self.used = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def allow(self, steps: int | None) -> None:
        self.left, self.used = steps, 0

    async def execute(self, episode_id: Any, batch: ActionBatch) -> ExecuteResult:
        if self.left is None or len(batch.steps) <= self.left:
            result: ExecuteResult = await self.agent.execute(episode_id, batch)
            cut = False
        elif self.left <= 0:
            return ExecuteResult(batch_id=batch.batch_id, states=[], executed=0,
                                 stopped="step_limit")  # fmt: skip
        else:
            part = batch.model_copy(update={"steps": batch.steps[: self.left]})
            result = await self.agent.execute(episode_id, part)
            cut = result.stopped == "completed"
        if self.left is not None:
            self.left -= result.executed
            self.used += result.executed
        return result.model_copy(update={"stopped": "step_limit"}) if cut else result


def honest(receipt: ActionReceipt, built: Built) -> ActionReceipt:
    """A grasp's receipt (not a press's) that says ok although the fingers closed farther than
    `GRASP_SHORT_M` from its grasp pose: `stopped_short`, said so."""
    if built.grasp is None or built.press or not receipt.ok or receipt.arm != built.arm:
        return receipt
    closed = receipt.gripper is not None and receipt.gripper.commanded < GRIP_CLOSED_BELOW
    short = receipt.deviation_m
    if not closed or short is None or short <= GRASP_SHORT_M:
        return receipt
    note = (f"the fingers closed {1000 * short:.0f} mm from the grasp pose (more than "
            f"{1000 * GRASP_SHORT_M:.0f} mm): not the grasp made; {receipt.note}")  # fmt: skip
    return receipt.model_copy(update={"reason": ReceiptReason.stopped_short, "note": note})


def active(recording: Recording, check: Check, s: Segment) -> tuple[int, int]:
    """The frames `s`'s arm moves in (`ACTIVE_RAD`, `ACTIVE_GRIP`): from its first move to the
    segment's end; (end, end) when it never moves."""
    commands = recording.commands
    end = min(s.end, len(commands) - 1)
    if s.start >= end:
        return end, end
    spec = check.profile.arms[s.arm]
    joints = commands[s.start : end + 1, spec.action[0] : spec.action[1]]
    grip = commands[s.start : end + 1, spec.gripper_index]
    moved = (np.abs(joints - joints[0]).max(axis=1) > ACTIVE_RAD) | (
        np.abs(grip - grip[0]) > ACTIVE_GRIP
    )
    first = int(np.argmax(moved)) if moved.any() else end - s.start
    return s.start + first, end


async def wait(check: Check, ep: Episode, frames: int) -> str:
    """The robot held where it is commanded for `frames` actions (a demonstration's pause: a
    support arm plays its turn meanwhile); how the last batch stopped."""
    if frames <= 0:
        return "completed"
    qpos = np.asarray((await check.robot(ep)).qpos_cmd, dtype=float)
    return await replay(check, ep, np.tile(qpos, (frames, 1)), frames)


def opened_at(ep: Episode) -> int:
    """The demonstration frames already past when the first call runs: the actions the episode
    counted before it (an env that opens an episode once a support arm's demonstration is over
    counts the holds meanwhile; one frame is one action, as `replay` sends them). Counted from 0,
    the wait before the first grasp would count the support arm's actions twice and could reach
    the step limit."""
    return ep.start.step


def performed(s: Segment) -> int:
    """The last demonstration frame `s`'s call performs: a touch's hold to its open (`rest`: the
    call presses, holds or turns, and lets go), else its event. Counted from the close, the hold
    would be waited out again before the arm's next call."""
    return s.rest if s.kind == "touch" and s.rest is not None else s.end


def paused(recording: Recording | None, check: Check, calls: list[Call], clock: int) -> int:
    """The demonstration frames both arms stood still before `calls`: from `clock` (the latest
    end of a segment run so far) to the first frame one of their arms moves."""
    if recording is None:
        return 0
    first = min(active(recording, check, c.segment)[0] for c in calls)
    return max(first - clock, 0)


def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@dataclass
class Step:
    """What the ceiling runs at once: one call, or two of different arms that moved together."""

    calls: list[Call]


def steps(
    check: Check, calls: list[Call], recording: Recording | None, skills: Sequence["WrittenSkill"]
) -> tuple[list[Step], int]:
    """The plan's calls in the order their segments end, each joined by the other arm's next
    call where both arms moved at once in the demonstration and both are the core's single
    actions (not a move that brings a hand-over's object: the other arm takes it where that
    ends); how many such overlaps a written skill made run one after the other."""
    order, out, apart = list(calls), [], 0
    taken: set[int] = set()

    def single(call: Call) -> bool:
        return call.skill is not None and written(skills, call) is None and not call.before

    def brings(call: Call) -> bool:  # to the other arm, which takes it where it ends: first
        return call.segment.kind == "move" and call.segment.handover

    for i, call in enumerate(order):
        if i in taken:
            continue
        taken.add(i)
        partner = None
        if recording is not None and call.skill is not None:
            j = next((k for k in range(len(order)) if k not in taken
                      and order[k].segment.arm != call.segment.arm), None)  # fmt: skip
            if (
                j is not None
                and order[j].skill is not None
                and not brings(call)
                and not brings(order[j])
                and overlap(
                    active(recording, check, call.segment),
                    active(recording, check, order[j].segment),
                )
            ):
                if single(call) and single(order[j]):
                    partner = j
                else:
                    apart += 1
        if partner is not None:
            taken.add(partner)
            out.append(Step([call, order[partner]]))
        else:
            out.append(Step([call]))
    return out, apart


async def core_action(check: Check, ep: Episode, call: Call, truth: gt_scene.Scene) -> Built:
    """The core's single action for `call` (a move's grip set first, as the demonstration had
    it)."""
    s = call.segment
    if s.kind in ("grasp", "touch"):
        assert call.grasp is not None and s.obj is not None
        return await ep.api.grasp_object(s.arm, truth.proxy(s.obj), call.grasp)
    if s.kind == "release":
        return ep.api.open_gripper(s.arm)
    if s.kind == "home":
        return await ep.api.return_home(s.arm)
    if s.kind == "move":
        await keep_grip(check, ep, s)
        return await ep.api.move_to_pose(s.arm, check.profile.ee_pose(s.arm, s.tcp))
    assert call.place is not None and s.obj is not None
    target = truth.proxy(s.target) if s.target in truth.poses else None
    return await ep.api.place_object(
        s.arm,
        truth.proxy(s.obj),
        call.place,
        target,
        is_open=s.kind == "place",
        constrain=call.constrain,
        approach_m=call.approach_m,
        approach_axis=call.approach_axis,
    )


async def ceiling(
    check: Check,
    calls: list[Call],
    skills: Sequence["WrittenSkill"] = (),
    recording: Recording | None = None,
) -> dict[str, Any]:
    """Every planned call from the reset, mid-task homes included (the module's docstring: the
    other arm's overlapping call at once where `recording` shows them moving together, a step
    budget per segment, honest grasps), then each arm home; whether the official success
    latched. A segment no skill fits ends it (`failed_at`). A call naming one of `skills` runs
    through its `act`."""
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    plan, apart = steps(check, calls, recording, skills)
    failed_at: dict[str, Any] | None = None
    together = 0
    async with check.episode() as ep:
        clock = opened_at(ep)
        budget = Budgeted(ep.api.executor.env)
        ep.api.executor.env = budget
        for step in plan:
            missing = next((c for c in step.calls if c.skill is None), None)
            if missing is not None:
                s = missing.segment
                rows.append({"segment": s.name, "reason": "no_skill", "ok": False, "steps": 0,
                             "note": f"no skill fits it: {missing.why}"[:300]})  # fmt: skip
                failed_at = {"segment": s.name, "reason": "no_skill"}
                break
            if (held := paused(recording, check, step.calls, clock)) > 0:
                await wait(check, ep, held)
                rows.append({"segment": "wait", "frames": held, "ok": True})
                if ep.ended:
                    break
            clock = max(clock, *(performed(c.segment) for c in step.calls))
            budget.allow(max(step_budget(c.segment) for c in step.calls))
            names = [c.segment.name for c in step.calls]
            if step.calls[0].before:  # a task plan's transitions (a call with them runs alone)
                moved, made_rows = await transitions(check, ep, step.calls[0], skills)
                rows += [{"segment": names[0], **r} for r in made_rows]
                if not moved and failed_at is None:
                    failed_at = {"segment": names[0], "reason": "transition"}
                if ep.ended:
                    break
            truth = await check.truth(ep)
            if (skill := written(skills, step.calls[0])) is not None:
                call = step.calls[0]
                if call.segment.kind == "move":
                    await keep_grip(check, ep, call.segment)
                ran = await act_written(check, ep, skill, call, truth, honest_grasps=True)
                found = [{"segment": names[0], **m} for m in ran.row["motions"]]
                if "act_error" in ran.row:
                    found.append({"segment": names[0], "act_error": ran.row["act_error"],
                                  "ok": False})  # fmt: skip
                ok = not ran.failed and all(r.ok for r in ran.receipts)
            else:
                made = [await core_action(check, ep, c, truth) for c in step.calls]
                receipt = await check.run(ep, *made)
                for m in made:
                    receipt = honest(receipt, m)
                row = receipt_row(receipt)
                found = [{"segment": n, **row} for n in names]
                if len(names) == 2:
                    together += 1
                    found = [r | {"together": names[1 - k]} for k, r in enumerate(found)]
                ok = receipt.ok
                for call in step.calls:  # a touch's held path, after its press
                    if call.segment.kind == "touch" and call.path is not None and receipt.ok:
                        followed = await follow_hold(check, ep, call)
                        found.append({"segment": f"{call.segment.name} path",
                                      **receipt_row(followed)})  # fmt: skip
                        await check.run(ep, ep.api.open_gripper(call.segment.arm))
                        ok = ok and followed.ok
            for r in found:
                r["budget"] = budget.used if r is found[0] else 0
            found[0]["budget_of"] = max(step_budget(c.segment) for c in step.calls)
            rows += found
            if failed_at is None and not ok:
                failed_at = {"segment": names[0], "reason": found[-1].get("reason")
                             or "act_error"}  # fmt: skip
            judged = await check.control.judge(ep.opened.episode_id)
            if judged.success or ep.ended:
                break
        budget.allow(None)
        judged = await check.control.judge(ep.opened.episode_id)
        if not judged.success and not ep.ended:
            for arm in ("left", "right"):
                receipt = await check.run(ep, await ep.api.return_home(arm))
                rows.append({"segment": f"{arm} home", **receipt_row(receipt)})
            judged = await check.control.judge(ep.opened.episode_id)
        progress = (await check.control.gt_snapshot(ep.opened.episode_id)).progress
        stages = None if progress is None else {"left": progress.left, "of": progress.stages}
        # the benchmark's checks now, with their margins: read once the result is judged
        diagnosis = await check.control.diagnose(ep.opened.episode_id)
        return {
            "success": judged.success,
            "stages": stages,
            "conditions": [c.model_dump(mode="json") for c in diagnosis.conditions],
            "success_step": judged.success_step,
            "steps_used": judged.steps_used,
            "ended": judged.ended,
            "calls": rows,
            "first_failure": failed_at,
            "together": together,
            "sequential_overlaps": apart,
            "seconds": round(time.monotonic() - started, 1),
        }


async def chained(
    check: Check,
    recording: Recording,
    calls: list[Call],
    criteria: Criteria,
    origin_tolerance: tuple[float, float],
    skills: Sequence["WrittenSkill"] = (),
    demo: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """One rollout from the reset: each call in the order its segment ends, run from the state
    the calls before it left and judged as `explain` judges it (`perform`), with the benchmark's
    stages left before and after it (`GtSnapshot.progress`, where the env tracks them; `stages`
    says whether it moved the task on); the first segment not explained ends the rollout, its
    receipt's reason its `mechanism`; those after it are not reached (`cause` "not reached").
    Each call may use `step_budget` actions, as in the ceiling (`budget`); a call whose segment
    overlaps the other arm's still runs on its own, one after the other. A call's transitions
    (`Call.before`) run first, within its budget (`transitions`). With the demonstration's
    stages (`demo`), a segment off the demonstration's end that did the benchmark's work there
    counts explained (`stage_met`, `explained_by` "stages").
    A segment no skill fits is not explained there (`cause` "no skill"). One row per call."""
    rows: list[dict[str, Any]] = []
    async with check.episode() as ep:
        clock = opened_at(ep)
        executor = getattr(getattr(ep, "api", None), "executor", None)
        budget = Budgeted(executor.env) if executor is not None else None
        if executor is not None:
            executor.env = budget  # a segment's steps, as the ceiling bounds them
        stopped: str | None = None
        for call in calls:
            s = call.segment
            row: dict[str, Any] = {"segment": s.name, "skill": call.skill, "chained": True}
            if stopped is not None:
                rows.append(row | {"explained": False, "cause": "not reached",
                                   "applicable": False, "after": stopped})  # fmt: skip
                continue
            if call.skill is None:
                rows.append(row | {"explained": False, "cause": "no skill"})
                stopped = s.name
                continue
            if (held := paused(recording, check, [call], clock)) > 0:
                await wait(check, ep, held)
                row["waited"] = held
            clock = max(clock, performed(s))
            before = (await check.control.gt_snapshot(ep.opened.episode_id)).progress
            started = time.monotonic()
            if budget is not None:
                budget.allow(step_budget(s))
            moved = True
            if call.before:
                moved, row["transitions"] = await transitions(check, ep, call, skills)
            if moved and not ep.ended:
                final = closing([c.segment for c in calls], s)
                row |= await perform(check, ep, recording, call, criteria, origin_tolerance,
                                     skills, honest_grasps=True, final=final)  # fmt: skip
            else:
                cause = "the benchmark ended the episode" if ep.ended else "entry fails"
                row |= {"explained": False, "cause": cause, "mechanism": "transition"}
            if budget is not None:
                row["budget"], row["budget_of"] = budget.used, step_budget(s)
                budget.allow(None)
            after = (await check.control.gt_snapshot(ep.opened.episode_id)).progress
            if before is not None and after is not None:
                row["stages"] = {"before": before.left, "after": after.left,
                                 "of": after.stages}  # fmt: skip
            if s.carries and row.get("explained"):
                row["carried_off_m"] = await left_behind(check, ep, recording, s)
                if max(row["carried_off_m"].values()) > criteria.placed_m:
                    row |= {"explained": False, "cause": "fit out of tolerance",
                            "mechanism": "left behind"}  # fmt: skip
            row["seconds"] = round(time.monotonic() - started, 1)
            if stage_met(row, demo):
                row |= {"explained": True, "cause": None, "explained_by": "stages"}
            if not row.get("explained"):
                row["mechanism"] = row.get("mechanism") or mechanism(row)
                stopped = s.name
            rows.append(row)
            judged = await check.control.judge(ep.opened.episode_id)
            if judged.success:
                row["success"] = True
                stopped = stopped or "success"
    return rows


async def left_behind(
    check: Check, ep: Episode, recording: Recording, s: Segment
) -> dict[str, float]:
    """How far each object the demonstration carried through a move or a home (`Segment.carries`)
    ended from where the demonstration had it at the segment's end."""
    now, then = (await check.truth(ep)).poses, recording.poses(s.end)
    return {label: round(float(np.linalg.norm(np.subtract(now[label].p, then[label].p))), 4)
            for label in s.carries if label in now and label in then}  # fmt: skip


def mechanism(row: dict[str, Any]) -> str:
    """Why a chained segment was not explained, in one word: the first motion's receipt reason
    that was not ok, else its cause (`fit out of tolerance`, the benchmark's end)."""
    motions = [m for m in row.get("motions", []) if isinstance(m, dict)]
    motions += [row[k] for k in ("grasp", "place", "move", "release", "home", "lift", "path")
                if isinstance(row.get(k), dict)]  # fmt: skip
    bad = next((m for m in motions if m.get("ok") is False), None)
    if bad is not None:
        return str(bad.get("reason"))
    return "act_error" if row.get("act_error") else str(row.get("cause"))


CHAIN_INFRA: Final = ("not judged", "replay ended")
"""A chain's stop that is the infrastructure's: no alternative is tried there. The benchmark
ending the episode is not among them in a chain: nothing is replayed there, so the plan's own
calls fired its failure query (a gap, `clusters.chain_gap`'s class `ended`)."""
BREAK_TRIES: Final = 6
"""Alternatives the chained reuse tries at one break, at most, besides those written for it
(`chain_reuse`: a skill whose `cases.yaml` names the break's gap is always tried, first)."""
AGAIN: Final = 2
"""Times the chained reuse runs the plan unchanged at a break before it tries alternatives (a
chain fix can break in a rollout or two on a moving scene, and an alternative judged against
one of them would replace it)."""
CHAIN_ROLLOUTS: Final = 40
"""Rollouts the chained reuse spends on one task, at most, whatever is tried (a sweep's job is
bounded; chains take seconds to minutes a rollout)."""
CEILING_ROLLOUTS: Final = 1.5
"""A ceiling's time in the task's longest chained rollouts: what a chained step keeps for its
ceilings (`chain_reuse`'s `after`)."""


def depth(rows: list[dict[str, Any]]) -> int:
    """A chain's segments explained."""
    return sum(bool(r.get("explained")) for r in rows)


def chain_stop(rows: list[dict[str, Any]]) -> int | None:
    """The index of a chain's first segment not explained (where it broke), None when none."""
    return next((k for k, r in enumerate(rows)
                 if not r.get("explained") and r.get("applicable") is not False), None)  # fmt: skip


def lagging(rows: list[dict[str, Any]], demo: dict[str, int] | None) -> int | None:
    """A chain whose every segment was explained but whose benchmark stages lag: the first
    segment after which its stages left never again matched the demonstration's (`demo`: left
    after each segment, `demo_stages`; a transient lag, an overlapping segment's stage counted
    early in the demonstration, recovers); None when they match at the end or either is
    unknown."""
    if not demo or not rows:
        return None
    lags = []
    for r in rows:
        stages = r.get("stages")
        if stages is None or r["segment"] not in demo:
            return None
        lags.append(int(stages["after"]) - demo[r["segment"]])
    if lags[-1] <= 0:
        return None
    matched = [k for k, lag in enumerate(lags) if lag <= 0]
    return matched[-1] + 1 if matched else 0


def took(
    rows: list[dict[str, Any]], k: int, mode: str, demo: dict[str, int] | None
) -> tuple[bool | None, str]:
    """Whether a rollout explained its `k`th segment (`mode` "stages": with the stages left
    after it no more than the demonstration's); None when the calls before it did not all
    explain (the rollout cannot tell); and why, in a line."""
    prefix = next((r for r in rows[:k] if not r.get("explained")), None)
    if prefix is not None:
        return None, f"the call before it did not chain: {prefix['segment']} ({mechanism(prefix)})"
    row = rows[k]
    if not row.get("explained"):
        return False, mechanism(row)
    if mode == "stages" and demo is not None:
        left, then = (row.get("stages") or {}).get("after"), demo.get(row["segment"])
        if left is None or then is None or left > then:
            return False, f"stages left {left} after it, the demonstration's {then}"
    return True, "explained"


async def demo_stages(
    check: Check, recording: Recording, calls: list[Call]
) -> dict[str, int] | None:
    """The benchmark's stages left after each call's segment in the demonstration: its recorded
    commands replayed from the reset in one episode, read at each segment's end (a segment past
    where the replay stopped: as it stopped). None where the env tracks no stages."""
    found: dict[str, int] = {}
    async with check.episode() as ep:
        at, going = 0, True
        for call in sorted(calls, key=lambda c: c.segment.end):
            end = min(call.segment.end, len(recording.commands))
            if going and end > at:
                stopped = await replay(check, ep, recording.commands, end, since=at)
                at, going = end, stopped == "completed" and not ep.ended
            progress = (await check.control.gt_snapshot(ep.opened.episode_id)).progress
            if progress is None:
                return None
            found[call.segment.name] = progress.left
    return found


async def chain_reuse(
    check: Check,
    recording: Recording,
    calls: list[Call],
    options: Callable[[int, str], tuple[list[Call], list[Call]]],
    criteria: Criteria,
    origin_tolerance: tuple[float, float],
    skills: Sequence["WrittenSkill"] = (),
    tries: int = BREAK_TRIES,
    demo: Callable[[list[Call]], Awaitable[dict[str, int] | None]] | None = None,
    rollouts: int = CHAIN_ROLLOUTS,
    fixed: Sequence[dict[str, Any]] = (),
    earlier: dict[str, Any] | None = None,
    until: float | None = None,
    after: float = 0.0,
) -> dict[str, Any]:
    """The chained reuse: the plan's chain (`chained`; `calls` may hold the fixes an earlier
    chain made, `fixed`, kept unless they break again); where it breaks, the same plan run again
    first, `AGAIN` times at most (the planner and a moving scene flip outcomes run to run): the
    first rollout that explains the break, the calls before it too and no fewer segments in all,
    is the chain. Then that segment's alternatives (`options(k, mode)`: those written for its gap,
    newest first, then the rest; the plan's own call left out) each tried in a rollout of its own
    from the reset, in place of the plan's call: the first whose rollout explains it, on the same
    terms, replaces it and its rollout is the chain, which is judged on to where it breaks next.
    A chain that explains every segment without the official success breaks where its stages
    fell behind the demonstration's for good (`lagging`, `demo` giving the demonstration's: asked
    first, every rollout judging by them, `stage_met`), and an alternative there must also catch
    the stages up. Each break gets `tries` of the rest besides every one written for it; `rollouts`
    in all besides the first. A stop the infrastructure's (`CHAIN_INFRA`) is not tried; a try the
    infrastructure broke is recorded and spent. `earlier`: the best earlier chain on these same
    `calls` (its report: `rows`, `gap`, `explained`); a chain that ends explaining fewer is noise
    against it, not a fix: the earlier chain is kept (its rows, `fixed`, `calls`) and this one
    recorded in `noisy`, with where the kept plan's own first rollout broke (`broke`: a flaky
    segment, `clusters.chain_gap`). `until` (`time.monotonic()`): no rollout starts that would leave
    less than `after` of the longest rollout so far before it (what the step runs next, its
    ceilings), `out_of_time` then (the ceilings must fit within the job's bound)."""
    calls, start = list(calls), list(calls)
    stages = None if demo is None else await demo(calls)  # every rollout judges by them
    longest, out_of_time = 0.0, False

    async def rollout(trial: list[Call]) -> list[dict[str, Any]]:
        nonlocal longest
        began = time.monotonic()
        try:
            return await chained(check, recording, trial, criteria, origin_tolerance, skills,
                                 stages)  # fmt: skip
        finally:
            longest = max(longest, time.monotonic() - began)

    def in_time() -> bool:
        nonlocal out_of_time
        if until is not None and time.monotonic() + longest * (1.0 + after) > until:
            out_of_time = True
        return not out_of_time

    rows = await rollout(calls)
    attempts: list[dict[str, Any]] = []
    closed: dict[str, dict[str, Any]] = {f["segment"]: dict(f) for f in fixed}
    tried: set[tuple[int, str | None, str | None]] = set()
    spent, gap = 0, None
    broke: dict[str, Any] | None = None  # where the plan's own first rollout broke, its row

    async def attempt(
        k: int, mode: str, trial: list[Call], record: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """`trial` rolled out and judged at its `k`th call as a fix of the chain `rows` there
        (recorded in `attempts`, its row's data with it): its rows where that holds."""
        nonlocal spent
        spent += 1
        try:
            again = await rollout(trial)
        except Exception as error:
            if not infrastructure(error):
                raise
            lost = {"explained": False, "cause": "not judged"}
            attempts.append(record | lost | {"error": f"{type(error).__name__}: {error}"})
            return None
        ok, why = took(again, k, mode, stages)
        if ok and depth(again) < depth(rows):
            ok, why = False, f"explained, but the chain explained {depth(again)} < {depth(rows)}"
        # not ok with no cause: explained, its stages behind the demonstration's
        cause = (None if ok else "not judged" if ok is None
                 else again[k].get("cause") or "stages behind")  # fmt: skip
        ran = {n: v for n, v in again[k].items() if n not in ("segment", "skill", "chained")}
        attempts.append(record | ran | {"explained": bool(ok), "cause": cause, "why": why})
        return again if ok else None

    while True:
        k, mode = chain_stop(rows), "chained"
        if k is None:
            if any(r.get("success") for r in rows) or demo is None:
                break
            k, mode = lagging(rows, stages), "stages"
            if k is None:
                break
        gap = {"segment": calls[k].segment.name, "mode": mode, "index": k}
        broke = broke or gap | {"row": rows[k]}
        if mode == "chained" and rows[k].get("cause") in CHAIN_INFRA:
            break
        current, found = calls[k], None
        if (k, mode, "again") not in tried:
            tried.add((k, mode, "again"))
            for n in range(1, AGAIN + 1):
                if spent >= rollouts or not in_time():
                    break
                record = {"segment": current.segment.name, "mode": mode, "form": current.form,
                          "skill": current.skill, "again": n}  # fmt: skip
                if (again := await attempt(k, mode, calls, record)) is not None:
                    found = (current, again)
                    break
        if found is None:
            named, rest = options(k, mode)
            others = 0
            for option, written_for in [(o, True) for o in named] + [(o, False) for o in rest]:
                key = (k, option.form, option.skill)
                if spent >= rollouts or option.skill is None or not in_time():
                    continue
                if not written_for and others >= tries:
                    continue
                if (option.form, option.skill) == (current.form, current.skill) or key in tried:
                    continue
                tried.add(key)
                others += not written_for
                trial = [*calls[:k], option, *calls[k + 1 :]]
                record = {"segment": current.segment.name, "mode": mode, "form": option.form,
                          "skill": option.skill, "written_for": written_for}  # fmt: skip
                if (again := await attempt(k, mode, trial, record)) is not None:
                    found = (option, again)
                    break
        if found is None:
            break
        calls[k], rows = found
        c = calls[k]
        if c is not current:
            closed[c.segment.name] = {"segment": c.segment.name, "mode": mode, "form": c.form,
                                      "skill": c.skill, "call": c.data(skills)}  # fmt: skip
        gap = None
    noisy = None
    if earlier is not None and depth(rows) < int(earlier["explained"]):
        noisy = {"explained": depth(rows), "of": int(earlier["explained"]), "gap": gap,
                 "broke": broke, "rows": rows, "closed": list(closed.values()),
                 "calls": [c.record() for c in calls]}  # fmt: skip
        rows, gap, calls = earlier["rows"], earlier.get("gap"), start
        closed = {f["segment"]: dict(f) for f in fixed}
    return {
        "rows": rows,
        "tries": attempts,
        "closed": list(closed.values()),
        "calls": [c.record() for c in calls],
        "gap": gap,
        "demo_stages": stages,
        "explained": depth(rows),
        "to_end": chain_stop(rows) is None,
        "success": any(r.get("success") for r in rows),
        "noisy": noisy,
        "out_of_time": out_of_time,
        "_calls": calls,
    }


async def chained_case(
    check: Check,
    recording: Recording,
    plan: list[Call],
    k: int,
    call: Call,
    criteria: Criteria,
    origin_tolerance: tuple[float, float],
    skills: Sequence["WrittenSkill"] = (),
    mode: str = "chained",
    demo: dict[str, int] | None = None,
) -> dict[str, Any]:
    """A case on a chained gap (`mode` "chained") or a stage gap ("stages"): `call` in place of
    the plan's `k`th call, in one rollout from the reset (`chained`, the plan's calls before it
    run first), judged there as the chain judges (a stage gap: also its stages left after it,
    no more than the demonstration's, `demo`). The row is the segment's, with how far the
    rollout went (`chain`: segments explained, of, the official success, stages left at the
    end). A rollout whose calls before it did not all explain cannot tell (`not judged`)."""
    trial = [*plan[:k], call, *plan[k + 1 :]]
    rows = await chained(check, recording, trial, criteria, origin_tolerance, skills, demo)
    ok, why = took(rows, k, mode, demo)
    s = call.segment
    if ok is None:
        return {"segment": s.name, "skill": call.skill, "chained": True, "explained": False,
                "cause": "not judged", "why": why}  # fmt: skip
    row = dict(rows[k])
    reached = [r for r in rows if r.get("applicable") is not False]
    last = next((r["stages"] for r in reversed(reached) if r.get("stages")), None)
    row["chain"] = {"explained": depth(rows), "of": len(rows),
                    "success": any(r.get("success") for r in rows),
                    "stages_left": None if last is None else last["after"]}  # fmt: skip
    if mode == "stages":
        row["demo_stages_left"] = (demo or {}).get(s.name)
    if not ok:
        cause = row.get("cause") if not row.get("explained") else "stages behind"
        return row | {"explained": False, "cause": cause, "why": why}
    return row | {"explained": True, "cause": None}
