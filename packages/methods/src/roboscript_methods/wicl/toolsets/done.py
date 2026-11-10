"""wicl's `done` toolset (docs/concepts/agent-runtime.md): the done tools, on the
episode's scene registry that the core `worldstate` toolset keeps (both work on one `SceneConfig`).

`check_done` measures the Memory's done criteria (the `DONE` fact, `perception.done`) and
`read_relations` (system only: S1's record at a try's end) every metric among the registered
roles, both on a new observation: first each object an arm has let go of is measured again there,
since the registry only predicts where a release left it (one not seen again fails its
criteria). The Hub refuses the model's `finish` until a `check_done` passed after the last action
(`PhaseRules.done_check`). With goal views in the fact (the demonstration's last keyframe, head
and wrist cameras), `check_done` also compares every one of those cameras' current view with them
(a VLM); it passes only when every criterion holds and the comparison says the end state is
reached: an undecided or failed comparison is not done. `read_end_targets` (system only) reads
the demonstration's end as targets.
"""

import dataclasses
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Final, Literal, get_args

import anyio
import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import InfraError, ToolError
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.hub import DoneReport
from roboscript.contracts.ids import FrameId
from roboscript.contracts.robot import Arm, RobotState
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.demos import KeypointSet, frame_alias
from roboscript.manipulation import robot_base
from roboscript.memory.objects import directed_arm
from roboscript.perception.criteria import (
    METRICS,
    TABLE,
    UNARY,
    UNITS,
    DoneCheckConfig,
    DoneSpec,
    EndTarget,
    for_arm,
)
from roboscript.perception.fit import vec3
from roboscript.perception.frames import read_rgb
from roboscript.perception.measure import measure as remeasure
from roboscript.perception.predicates import press_offset
from roboscript.perception.registry import (
    ObjectGeometry,
    ObjectRecord,
    SceneRegistry,
    object_geometry,
    start_geometry,
)
from roboscript.perception.table import Table
from roboscript.perception.views import View
from roboscript.perception.vlm import GoalMatch, goal_match
from roboscript.spatial.pose import pose_from_matrix, pose_to_matrix
from roboscript.spatial.rotation import quat_to_matrix
from roboscript.tools import (
    ARM_SIDES,
    DEMO,
    DONE,
    MEMORY_SPECS,
    ToolContext,
    Toolset,
    tool,
)
from roboscript.tools.context import ToolsetRuntime
from roboscript.toolsets.manipulation.arms import named_arms
from roboscript.toolsets.scene import (
    SceneAccess,
    SceneConfig,
    embodiment,
)
from roboscript_methods.wicl.perception.done import (
    ANY_LIFT,
    POSITIONAL,
    DoneValue,
    capped,
    in_the_air,
    lifts,
    measure,
    surface_change,
    surface_points,
    verdict,
)
from roboscript_methods.wicl.perception.endstate import (
    EndLine,
    end_lines,
    end_targets,
    end_text,
    lift_excess,
)
from roboscript_methods.wicl.perception.endstate import roles_of as end_roles
from roboscript_methods.wicl.perception.pairfix import SEARCH_M, PairFix, pair_fixes
from roboscript_methods.wicl.relation import replay_grips, replay_presses
from roboscript_methods.wicl.releases import demo_releases

_FAILED = "done_failed"  # per criterion label: failing check_done calls in a row


_GAPS = "done_gaps"  # per criterion label: (signed gap, measured) at the last check (`overshoots`)


_EDGE = "done_edge"  # per line label: failing checks in a row near passing (`edges`)


_SPACING = "done_spacing"  # per pair (a, b): its spacing at the last check (`_overshot_pairs`)


class DoneGoal(BaseModel):
    """What done looks like: the demonstration's final-state views (camera -> RGB blob, its last
    keyframe) and the episode's live instruction."""

    views: dict[str, BlobRef]
    instruction: str


class DoneFact(BaseModel):
    """The `DONE` fact: the Memory's done criteria (maybe none), the goal views `check_done`
    also compares the scene with (None: no demonstration package), and `press`: the
    demonstration pushed something with empty fingers (press, click, toggle: an end state the
    views may barely show), so the episode must have its own press as well. `reference`: the
    demonstration's final frame (its `Observation`), whose own table a lift is measured against
    in the robot's base frame (`DoneCheckConfig.relative_height`; None: over this scene's table
    alone). `latched`: the roles whose lift latched the demonstration's success (held by an
    arm still rising into its last keyframe, the success frame its recording stops at:
    `goal.latched_lifts`; `ANY_LIFT`: every role), so their lifts are minimums while
    they are in the air
    (`DoneCheckConfig.latch_margin_m`); a carry into the latch rides on the `end` targets
    themselves (`EndTarget.motion`). `criteria_read` false: the episode reads none of the
    Memory's criteria (S1 is gated by the demonstration's end alone), so an object no end
    target places is not one a criterion could fix (`uncovered_text`). `latch`: the
    demonstration's arms still moving what they held into its success latch, each with that
    motion (world, m: `goal.latch_motion`), named to an Actor that makes no progress
    (`Stuck.latch`)."""

    criteria: list[DoneSpec] = []
    goal: DoneGoal | None = None
    press: bool = False
    end: list[EndTarget] = []
    reference: BlobRef | None = None
    latched: list[str] = []
    criteria_read: bool = True
    latch: dict[Arm, Vec3] = {}


class DoneLine(BaseModel):
    """A criterion, what was measured (None: undecidable, `note` says why), whether it holds;
    `failed`: how many check_done calls in a row it has failed, this one included. A lift
    (`perception.done.lifts`, the object in the air) held to the robot's base too: its bottom's
    height over the base (`base`) and the demonstration's (`base_need`, m); the band then
    applies to the smaller of the two excesses (`perception.done.verdict`)."""

    spec: DoneSpec
    measured: float | None
    passed: bool
    note: str = ""
    failed: int = 0
    base: float | None = None
    base_need: float | None = None
    minimum: float | None = None


class LatchMove(BaseModel):
    """The demonstration's motion into its success latch (`DoneFact.latch`) as an arm here makes
    it: `arm`, `motion` (world, m), and `mirrored` when another arm holds what the
    demonstration's moving arm held, so its mirror image across the robot's sagittal plane."""

    arm: Arm
    motion: Vec3
    mirrored: bool = False


class Stuck(BaseModel):
    """`check_done` making no progress (`DoneCheckConfig.stuck_after`): `checks`, the failing
    calls in a row since the best one (the fewest failing lines since the arms last took or let
    go of something) with no fewer failing lines; `failing`, each line's label and how many of
    those calls it failed; `moved_m`, the most an arm's TCP moved between two of them; `unseen`,
    the roles measured again and not found that many calls in a row, with `near` the arm whose TCP
    is nearest each (horizontally, m); `latch`, the demonstration's motion into its latch for
    the arms here; `give_up_after`, the config's; `identical`, the failing calls in a row
    with the same failing lines since the arms last took or let go of something, and
    `give_up_identical`, the config's."""

    checks: int
    failing: dict[str, int] = {}
    moved_m: float = 0.0
    unseen: dict[str, int] = {}
    near: dict[str, tuple[Arm, float]] = {}
    latch: list[LatchMove] = []
    give_up_after: int = 0
    identical: int = 0
    give_up_identical: int = 0


class Overshoot(BaseModel):
    """A criterion whose line failed on one side of its band at the last check_done and fails on
    the other side now: `before` and `now`, its signed gaps to the band's nearest edge (+: above,
    -: below), and `step`, how much its measurement changed between them, in `unit`."""

    label: str
    unit: str
    before: float
    now: float
    step: float


class Edge(BaseModel):
    """A line that has failed within `NEAR_EDGE_M` of passing `checks` calls in a row: `off`, how
    far now (m); for a criterion, its band's middle `centre` and the `change` to it."""

    label: str
    checks: int
    off: float
    centre: float | None = None
    change: float | None = None


class CheckDoneReport(DoneReport):
    """The done criteria on a new observation `frame`; `remeasured`: the roles measured again
    there first, with the measurement's reason (`ok`, `not_visible`, `low_confidence`); `goal`:
    the VLM's comparison with the goal views, when the fact has some; `pressed`, under a press
    goal: the registered objects the episode's empty fingers were stopped pushing on (the
    executor's receipts, `ActionReceipt.presses`); `uncovered`: the objects the episode took
    that no criterion or end target places (`DoneCheckConfig.coverage`); `wrong_arm`: the
    roles with an `arm side:` (`ARM_SIDES`) held by the other arm than the one the side their x
    points to gives (`wrong_arms`). `passed` needs every
    criterion, with goal views the comparison's confident yes (`reached` true; not when a role
    is `CONTAINED`, which the views cannot show), under a press
    goal a press, and nothing uncovered. `stuck`: how it is making no progress, once that is
    `DoneCheckConfig.stuck_after` calls (`gave_up` once it is `give_up_after`). `pairs`: one
    correction for the lines on two roles' spacing with a failing one (`perception.pairfix`;
    its moves at most half the spacing's last change when that went past it, `last_change`);
    `overshot`: the criteria whose last move took them across their band; `edge`: the lines
    failing just short of passing `EDGE_CHECKS` or more calls in a row."""

    frame: FrameId
    lines: list[DoneLine]
    remeasured: dict[str, str]
    goal: GoalMatch | None = None
    pressed: list[str] | None = None
    end: list[EndLine] = Field(default_factory=list)
    uncovered: list[str] = Field(default_factory=list)
    wrong_arm: list[str] = Field(default_factory=list)
    criteria_read: bool = True
    stuck: Stuck | None = None
    pairs: list[PairFix] = Field(default_factory=list)
    overshot: list[Overshoot] = Field(default_factory=list)
    edge: list[Edge] = Field(default_factory=list)


class EndTargetsReport(BaseModel):
    """The demonstration's end as targets (`perception.endstate`), read at its last keyframe."""

    targets: list[EndTarget]


class RelationsReport(BaseModel):
    """Every metric among the registered roles on a new observation `frame` (see
    `CheckDoneReport` for `remeasured`), and `start`: each at the episode's start, the objects
    where they were first measured (no surface change, nothing held); `sides` (each role's
    smaller footprint side, m) and `done` (the check's config) let a reader judge criteria with
    the tolerances `check_done` caps them to (`perception.done.capped`)."""

    frame: FrameId
    values: list[DoneValue]
    remeasured: dict[str, str]
    start: list[DoneValue] = []
    sides: dict[str, float] = {}
    done: DoneCheckConfig | None = None


STUCK: Final = 3
"""After this many failing check_done calls in a row, a criterion's line says to change course."""


CHECK_TIMEOUT_S: Final = 300.0
"""check_done observes, measures again each object an arm let go of (up to 120 s each elsewhere)
and may ask the VLM, whose retries and the gateway's queue take minutes."""


READ_TIMEOUT_S: Final = 600.0
"""read_relations measures every registered role again, at the demonstration's end as well."""


def done_line(line: DoneLine) -> str:
    spec = line.spec
    unit = UNITS[spec.metric][0]
    want = f"needs {spec.value:g} ± {spec.tol:g} {unit}"
    if line.minimum is not None:
        want = f"needs at least {spec.value + line.minimum:.3f} {unit} (higher passes)"
    if line.measured is None:
        text = f"{spec.label()}: undecided, {line.note}; {want}"
    elif line.minimum is not None:
        off = line.measured - spec.value
        text = f"{spec.label()}: {line.measured:.3f} {unit}, {want}"
        if line.base is not None and line.base_need is not None:
            text += (
                f"; height over robot base {line.base:.3f} {unit}, needs at least "
                f"{line.base_need + line.minimum:.3f} {unit}"
            )
            off = min(off, line.base - line.base_need)
        if not line.passed:
            text += f" ({line.minimum - off:.4f} {unit} too low)"
        text += (
            " (the demonstration stops the moment the task's success latched, so its height is "
            "the least that passes)"
        )
    elif line.base is not None and line.base_need is not None:
        text = (
            f"{spec.label()}: gap over table {line.measured:.3f} {unit}, {want}; height over "
            f"robot base {line.base:.3f} {unit}, needs {line.base_need:.3f} ± {spec.tol:g} {unit}"
        )
        off = min(line.measured - spec.value, line.base - line.base_need)
        if not line.passed:
            text += f" ({abs(off) - spec.tol:.4f} {unit} too {'high' if off > 0 else 'low'})"
    else:
        text = f"{spec.label()}: {line.measured:.3f} {unit}, {want}"
        off = line.measured - spec.value
        if not line.passed:
            # 4 places: a value just outside the band must not read as "0.000 too high".
            text += f" ({abs(off) - spec.tol:.4f} {unit} too {'high' if off > 0 else 'low'})"
            text += f"; to the band's middle: change it by {-off:+.3f} {unit}"
    if line.failed >= STUCK:
        text += (
            f"; failed {line.failed} checks in a row: what was done since has not moved it into "
            "range, so change how you act on it rather than repeat"
        )
    return text


def render_done(result: CheckDoneReport) -> list[ContentPart]:
    rows = [("pass " if line.passed else "FAIL ") + done_line(line) for line in result.lines]
    again = ", ".join(
        f"{role} ({CONTAINED_TEXT if reason == CONTAINED else reason})"
        for role, reason in result.remeasured.items()
    )
    head = "done: " + ("yes" if result.passed else "not yet")
    text = "\n".join([head + (f"; measured again: {again}" if again else ""), *rows])
    if result.goal is not None:
        reached = {True: "yes", False: "no", None: "cannot tell"}[result.goal.reached]
        differs = f"; differs: {result.goal.differs}" if result.goal.differs else ""
        text += f"\nlooks like the demonstration's final state: {reached}{differs}"
        if result.goal.reached is not True and CONTAINED in result.remeasured.values():
            text += f" (not deciding: {HIDDEN_GOAL})"
    if result.pressed is not None:
        text += "\npressed: " + (", ".join(result.pressed) or "nothing yet")
    for line in result.end:
        text += "\n" + ("pass " if line.passed else "FAIL ") + end_text(line)
    criteria = [line.spec for line in result.lines]
    for role in result.uncovered:
        text += "\nFAIL " + uncovered_text(role, criteria, result.criteria_read)
    for failing in result.failing:
        if any(failing.startswith(f"{role} is held by the ") for role in result.wrong_arm):
            text += "\nFAIL " + failing
    for overshot in result.overshot:
        text += "\n" + overshoot_text(overshot)
    for edge in result.edge:
        text += "\n" + edge_text(edge, result.pairs)
    for fix in result.pairs:
        text += "\n" + pair_text(fix)
    if result.stuck is not None:
        text += "\n" + stuck_text(result.stuck, result.gave_up)
    return [TextPart(text=text)]


def signed_gap(line: DoneLine) -> float | None:
    """How far a two-sided criterion's measurement is outside its band, to the nearest edge (+:
    above it, -: below it, 0: inside); None for a lift held to the base or to a minimum, or an
    undecided line."""
    if line.measured is None or line.minimum is not None or line.base is not None:
        return None
    off, tol = line.measured - line.spec.value, line.spec.tol
    return off - tol if off > tol else off + tol if off < -tol else 0.0


def overshoots(state: dict[str, object], lines: Sequence[DoneLine]) -> list[Overshoot]:
    """The criteria failing on the other side of their band than at the last check_done of the
    episode (`_GAPS`, which this call then holds); a line between them inside its band, or
    undecided, is no overshoot."""
    last = state.get(_GAPS, {})
    assert isinstance(last, dict)
    now: dict[str, tuple[float, float]] = {}
    found = []
    for line in lines:
        gap = signed_gap(line)
        if gap is None:
            continue
        assert line.measured is not None
        label = line.spec.label()
        now[label] = (gap, line.measured)
        before = last.get(label)
        if before is not None and before[0] * gap < 0.0:
            unit = UNITS[line.spec.metric][0]
            step = line.measured - before[1]
            found.append(Overshoot(label=label, unit=unit, before=before[0], now=gap, step=step))
    state[_GAPS] = now
    return found


NEAR_EDGE_M: Final = 0.005
"""A failing line this close to passing (m) is at its band's edge (`edges`)."""


EDGE_CHECKS: Final = 3
"""After this many such checks in a row, check_done says to aim for the band's middle."""


def edges(
    state: dict[str, object], lines: Sequence[DoneLine], ends: Sequence[EndLine]
) -> list[Edge]:
    """The lines failing within `NEAR_EDGE_M` of passing `EDGE_CHECKS` or more check_done calls in
    a row (`_EDGE`, which this call then holds): a criterion in metres, and a pair, base or
    directed place end target (whose `measured` is how far outside its band it is)."""
    last = state.get(_EDGE, {})
    assert isinstance(last, dict)
    near: dict[str, tuple[float, float | None, float | None]] = {}
    for line in lines:
        gap = signed_gap(line)
        if gap is None or line.passed or UNITS[line.spec.metric][0] != "m":
            continue
        assert line.measured is not None
        if abs(gap) <= NEAR_EDGE_M:
            near[line.spec.label()] = (abs(gap), line.spec.value, line.spec.value - line.measured)
    for end in ends:
        outside = end.target.kind in ("pair", "base") or end.along is not None
        close = end.measured is not None and end.measured <= NEAR_EDGE_M
        if outside and not end.passed and close:
            assert end.measured is not None
            near[end.target.label()] = (end.measured, None, None)
    now = {label: last.get(label, 0) + 1 for label in near}
    state[_EDGE] = now
    return [
        Edge(label=label, checks=now[label], off=off, centre=centre, change=change)
        for label, (off, centre, change) in near.items()
        if now[label] >= EDGE_CHECKS
    ]


def edge_text(edge: Edge, pairs: Sequence[PairFix]) -> str:
    text = (
        f"AT THE EDGE: {edge.label} has failed within {1000 * NEAR_EDGE_M:.0f} mm of passing "
        f"{edge.checks} checks in a row ({1000 * edge.off:.1f} mm now): small corrections to "
        "its edge leave the next measurement on either side of it, so aim for the middle of its "
        "band, not the edge"
    )
    if edge.centre is not None and edge.change is not None:
        text += f" (its middle is {edge.centre:g} m: change it by {edge.change:+.3f} m)"
    elif not any(edge.label in fix.labels for fix in pairs):
        text += " (well past a minimum, not just over it)"
    if any(edge.label in fix.labels for fix in pairs):
        text += "; the TOGETHER move aims there"
    return text + "."


PAIR_MOVED_M: Final = 0.01
"""A spacing that changed less than this between checks was not moved past its target, whatever
perception's noise says."""


def _overshot_pairs(state: dict[str, object], fixes: Sequence[PairFix]) -> list[PairFix]:
    """`fixes` with the moves of a pair whose spacing went past its target since the last
    check_done cut to half that change: it changed by `PAIR_MOVED_M` or more toward the change
    the last check asked for, and the change asked for now points back (`_SPACING`: each pair's
    spacing and asked change at the last check, which this call then holds)."""
    last = state.get(_SPACING, {})
    assert isinstance(last, dict)
    asked = {
        (f.a, f.b): (f.spacing, np.subtract(f.target, f.spacing))
        for f in fixes
        if f.target is not None
    }
    state[_SPACING] = asked
    out = []
    for fix in fixes:
        before, now = last.get((fix.a, fix.b)), asked.get((fix.a, fix.b))
        if before is None or now is None:
            out.append(fix)
            continue
        change = np.subtract(fix.spacing, before[0])
        step, size = float(np.linalg.norm(change)), float(np.linalg.norm(now[1]))
        toward = float(change @ before[1]) > 0.0
        if step < PAIR_MOVED_M or not toward or float(before[1] @ now[1]) >= 0.0:
            out.append(fix)
            continue
        scale = min(1.0, step / 2 / max(size, 1e-9))
        moves = {r: (round(m[0] * scale, 4), round(m[1] * scale, 4)) for r, m in fix.moves.items()}
        last_change = (round(float(change[0]), 4), round(float(change[1]), 4))
        out.append(fix.model_copy(update={"moves": moves, "last_change": last_change}))
    return out


def overshoot_text(o: Overshoot) -> str:
    side = {True: "high", False: "low"}
    return (
        f"OVERSHOT: {o.label} was {abs(o.before):.4f} {o.unit} too {side[o.before > 0]} at the "
        f"last check and is {abs(o.now):.4f} {o.unit} too {side[o.now > 0]} now (it changed by "
        f"{o.step:+.3f} {o.unit}): the last move went past its band, so change it back by at most "
        f"half that step ({abs(o.step) / 2:.3f} {o.unit}), not a full reversal."
    )


def pair_text(fix: PairFix) -> str:
    """One correction for the lines on two roles' spacing (`perception.pairfix`)."""
    sx, sy = fix.spacing
    text = (
        f"TOGETHER: {fix.a} and {fix.b}: {' and '.join(fix.labels)} "
        f"{'fails' if len(fix.labels) == 1 else 'fail'}, and every line on their "
        f"spacing reads one thing, {fix.a}'s center less {fix.b}'s (now x {sx:+.3f}, y {sy:+.3f} "
        f"m, {math.hypot(sx, sy):.3f} m apart), so correcting one line at a time breaks another."
    )
    if fix.target is None:
        return text + (
            f" No spacing within {SEARCH_M:g} m of it passes them all: redo the step that sets "
            "their end instead."
        )
    tx, ty = fix.target
    moves = []
    for role, (dx, dy) in fix.moves.items():
        arm = fix.arms.get(role)
        by = f" (the {arm} arm holds it)" if arm is not None else ""
        moves.append(f"{role} by x {dx:+.3f}, y {dy:+.3f} m{by}")
    both = " together, keeping their midpoint" if len(fix.moves) > 1 else ""
    text += (
        f" A spacing {'in the middle of all their bands' if fix.middle else 'that passes them all'}"
        f" is x {tx:+.3f}, y {ty:+.3f} m "
        f"({math.hypot(tx, ty):.3f} m apart): move {' and '.join(moves)}{both}, sideways only, "
        "then check again; this replaces those lines' own corrections."
    )
    if fix.last_change is not None:
        cx, cy = fix.last_change
        text += (
            f" Their spacing changed by x {cx:+.3f}, y {cy:+.3f} m since the last check and went "
            "past it, so these moves are at most half that step."
        )
    return text


SMALL_MOVE_M: Final = 0.03
"""Arms that moved at most this between checks made small corrections (`stuck_text`)."""


NEAR_ARM_M: Final = 0.15
"""An arm's TCP this near an object, horizontally, may hide it from the cameras."""


def stuck_text(stuck: Stuck, gave_up: bool) -> str:
    """What `check_done` tells an Actor making no progress: that what it does is not closing the
    gap, why a role stays undecided, what else to try (the demonstration's motion into its latch,
    else the step that sets the end), and, once it gave up, that `finish` ends the episode."""
    parts = []
    if stuck.checks:
        counts = sorted(stuck.failing.items(), key=lambda kv: (-kv[1], kv[0]))
        named = ", ".join(f"{label} {n}x" for label, n in counts)
        small = (
            f"; the arms moved at most {1000 * stuck.moved_m:.0f} mm between them"
            if stuck.moved_m <= SMALL_MOVE_M
            else ""
        )
        parts.append(
            f"NO PROGRESS: {stuck.checks} checks in a row have failed with no fewer failing "
            f"lines than the best before them (failing among them: {named}){small}. What was done "
            "between them is not closing the gap (closing one line opens another, or nothing "
            "changes), so stop repeating it."
        )
    for role, n in sorted(stuck.unseen.items()):
        near = stuck.near.get(role)
        if near is not None and near[1] <= NEAR_ARM_M:
            why = (
                f"the {near[0]} arm's TCP is {near[1]:.2f} m from it and may hide it: move that "
                "arm well away from it, out of the cameras' view, then check again"
            )
        else:
            why = (
                "no arm is near it, so moving the arms will not change that: what the cameras see "
                "there does not match it as it was set down (it may have tipped, turned, slid, or "
                "moved with what it is in or on), and checking again from here will not decide it"
            )
        parts.append(
            f"{role} has not been measured again with confidence in {n} checks in a row; {why}."
        )
    if stuck.checks and stuck.latch:
        moves = []
        for move in stuck.latch:
            dx, dy, dz = move.motion
            size = math.sqrt(dx * dx + dy * dy + dz * dz)
            image = ", mirrored for the arm that holds it here" if move.mirrored else ""
            moves.append(
                f"the {move.arm} arm by x {dx:+.3f}, y {dy:+.3f}, z {dz:+.3f} m "
                f"({size:.3f} m{image})"
            )
        parts.append(
            "The demonstration did not stop at its end pose: when its success latched, its arm "
            "was still moving what it held. The success may need that motion itself (a strike, a "
            "push, a press: a contact its end pose cannot show), so make it once, as one move of "
            "about that size (move_by of that arm), then check again: " + "; ".join(moves) + "."
        )
    elif stuck.checks:
        parts.append(
            "Redo the step that sets this end instead (grasp or place again from where it is "
            "seen), not another small shift."
        )
    if gave_up:
        same = stuck.give_up_identical and stuck.identical >= stuck.give_up_identical
        how = (
            f"has failed on the same lines {stuck.identical} times in a row"
            if same
            else f"has made no progress in {stuck.give_up_after} calls"
        )
        parts.append(
            f"check_done {how}, so it no longer "
            "holds finish. The attempt still goes on within the step limit, and the "
            "demonstration's motion into its latch or redoing the step may still make the "
            "official check latch; a finish ends the episode and is never credited."
        )
    return " ".join(parts)


NO_PRESS: Final = (
    "no press yet: the demonstration pushed with empty fingers, and no action of this episode "
    "has had a closed, empty gripper stopped pushing on a registered object"
)


def _goal_failing(goal: GoalMatch) -> str:
    """The comparison's failing line (the refused `finish` repeats it)."""
    said = "does not look like" if goal.reached is False else "cannot be confirmed to look like"
    differs = f": {goal.differs}" if goal.differs else ""
    return f"the scene {said} the demonstration's final state{differs}"


def render_relations(result: RelationsReport) -> list[ContentPart]:
    rows = []
    for i, v in enumerate(result.values):
        name = " ".join(
            x for x in (v.metric, v.a if v.part is None else f"{v.a}.{v.part}", v.b) if x
        )
        now = f"{v.measured:.4f}" if v.measured is not None else f"undecided ({v.reason})"
        start = result.start[i].measured if i < len(result.start) else None
        rows.append(f"{name}: {now}" + (f" (start {start:.4f})" if start is not None else ""))
    return [TextPart(text="\n".join(rows))]


_PROGRESS = "done_progress"  # check_done's progress across calls (`_progress`)


@dataclass
class _Progress:
    """Failing check_done calls since the best one (`best` failing lines, `holds` what the arms
    held then: (object, arm) pairs), their failing labels, the most an arm moved between two of
    them, the TCPs at the last one, and each role undecided in calls in a row."""

    holds: tuple[tuple[str, str], ...]
    best: int
    checks: int = 0
    failing: dict[str, int] = dataclasses.field(default_factory=dict)
    moved_m: float = 0.0
    tcps: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    unseen: dict[str, int] = dataclasses.field(default_factory=dict)
    labels: tuple[str, ...] = ()  # the last call's failing labels
    identical: int = 0  # failing calls in a row with those same labels


def _progress(
    state: dict[str, object],
    labels: Sequence[str],
    read: "_Read",
    registry: SceneRegistry,
    latch: Mapping[Arm, Vec3],
    base: np.ndarray | None,
    config: DoneCheckConfig,
) -> Stuck | None:
    """This call's place in check_done's progress (`DoneCheckConfig.stuck_after`): `labels`, its
    failing lines (none: it passed, and the progress starts over). A call with fewer failing
    lines than the best since the arms last took or let go of something is progress; so is such
    a take or release (a new attempt). The `Stuck` once `stuck_after` calls in a row made none,
    or a role stayed undecided that long; None before, or with `stuck_after` 0."""
    if not labels:
        state.pop(_PROGRESS, None)
        return None
    holds = tuple(sorted((r.name, arm) for r in registry.objects.values() for arm in r.grasps))
    tcps = (
        {} if read.robot is None else {a: np.asarray(s.tcp.p) for a, s in read.robot.arms.items()}
    )
    last = state.get(_PROGRESS)
    assert last is None or isinstance(last, _Progress)
    same = tuple(sorted(labels))
    fresh = last is None or last.holds != holds or last.labels != same
    identical = 1 if fresh or last is None else last.identical + 1
    if last is None or last.holds != holds or len(labels) < last.best:
        now = _Progress(holds=holds, best=len(labels), unseen=last.unseen if last else {})
    else:
        now = last
        now.checks += 1
        moves = [float(np.linalg.norm(p - now.tcps[a])) for a, p in tcps.items() if a in now.tcps]
        now.moved_m = max([now.moved_m, *moves])
        for label in labels:
            now.failing[label] = now.failing.get(label, 0) + 1
    now.tcps, now.labels, now.identical = tcps, same, identical
    now.unseen = {
        role: now.unseen.get(role, 0) + 1
        for role, why in read.remeasured.items()
        if why not in SEEN
    }
    state[_PROGRESS] = now
    worst = max(now.unseen.values(), default=0)
    repeated = bool(config.give_up_identical) and identical >= config.give_up_identical
    if not config.stuck_after or (max(now.checks, worst) < config.stuck_after and not repeated):
        return None
    unseen = {r: n for r, n in now.unseen.items() if n >= config.stuck_after}
    near = {}
    for role in unseen:
        g = read.geometry.get(role)
        if g is None or not tcps:
            continue
        gaps = {a: float(np.linalg.norm(p[:2] - np.asarray(g.center[:2]))) for a, p in tcps.items()}
        arm = min(gaps, key=gaps.__getitem__)
        near[role] = (arm, round(gaps[arm], 3))
    held = {arm for _, arm in holds}
    return Stuck(
        checks=now.checks if now.checks >= config.stuck_after else 0,
        failing=dict(now.failing) if now.checks >= config.stuck_after else {},
        moved_m=round(now.moved_m, 4),
        unseen=unseen,
        near=near,
        latch=latch_moves(latch, held, base),
        give_up_after=config.give_up_after,
        identical=identical,
        give_up_identical=config.give_up_identical,
    )


def _gives_up(stuck: Stuck, config: DoneCheckConfig, stage: str) -> bool:
    """Whether check_done stops holding finish (`DoneCheckConfig.give_up_after`,
    `give_up_identical`): in P3 only (the Hub honors it under `PhaseRules.recover`), on calls
    without progress or failing the same lines, not on a role left undecided."""
    cap, same = config.give_up_after, config.give_up_identical
    stopped = (bool(cap) and stuck.checks >= cap) or (bool(same) and stuck.identical >= same)
    return stage == "p3" and stopped


def latch_moves(
    latch: Mapping[Arm, Vec3], held: Collection[str], base: np.ndarray | None
) -> list[LatchMove]:
    """The demonstration's motions into its latch (`DoneFact.latch`) for the arms here: as
    recorded for an arm that holds something now (or with nothing held), mirrored across the
    robot's sagittal plane (the base frame's xz plane, `base` 4x4 world) for the other arm when
    only that one holds something (the scene mirrored, the other arm does the task)."""
    moves = []
    for arm, motion in sorted(latch.items()):
        other = next(a for a in get_args(Arm) if a != arm)
        if arm in held or not held or base is None or other not in held:
            moves.append(LatchMove(arm=arm, motion=motion))
            continue
        lateral = base[:3, 1] / max(float(np.linalg.norm(base[:3, 1])), 1e-9)
        m = np.asarray(motion, dtype=float)
        image = m - 2 * float(m @ lateral) * lateral
        moves.append(LatchMove(arm=other, motion=vec3(image), mirrored=True))
    return moves


def _line(
    spec: DoneSpec,
    found: DoneValue,
    held: DoneSpec | None = None,
    lift: tuple[float, float, float] | None = None,
    minimum: float | None = None,
) -> DoneLine:
    """A criterion's line, judged by `held` (the criterion with its tolerance capped,
    `DoneCheckConfig.tol_cap`) when it differs, which the line then shows with the cap noted;
    a lift held to the robot's base too by `lift` (`perception.done.lift_excess`); with
    `minimum`, a lift of a demonstration cut at its success latch, at least that much above
    (`perception.done.verdict`)."""
    judged = spec if held is None else held
    note = found.reason
    if judged.tol < spec.tol:
        cap = f"tolerance capped from {spec.tol:g} by the objects' size"
        note = f"{note}; {cap}" if note else cap
    elif judged.tol > spec.tol:
        floor = f"tolerance {spec.tol:g} read as {judged.tol:g}, perception's error"
        note = f"{note}; {floor}" if note else floor
    if minimum is not None:
        latch = (
            f"needs at least {minimum:g} m above it, higher passes too: the demonstration stops "
            "the moment the task's success latched"
        )
        note = f"{note}; {latch}" if note else latch
    excess, base, need = (None, None, None) if lift is None else lift
    return DoneLine(
        spec=judged,
        measured=found.measured,
        passed=verdict(judged, found.measured, excess, minimum),
        note=note,
        base=base,
        base_need=need,
        minimum=minimum,
    )


UNCOVERED: Final = (
    "no done criterion or end target says where {role} should end, and this episode moved it: "
    "the check cannot tell a misplaced {role} from a done task"
)


def uncovered_text(role: str, criteria: Sequence[DoneSpec], read: bool) -> str:
    """The failing line for a role the episode took that nothing places (`covers`), saying what
    would: with the Memory's criteria read, the done_spec it lacks (in the edit's terms: metric
    and roles), naming the criteria on the role that do not place it; without (`read` false: S1,
    gated by the demonstration's end alone), that the end targets read on this seed miss it, so
    a criterion added to the Memory would not change this check."""
    head = UNCOVERED.format(role=role)
    if not read:
        return (
            f"{head}. This attempt is checked against the demonstration's end alone, and the end "
            f"read on this seed gives no target for {role}; the Memory's done criteria are not "
            "read here, so a done_spec added for it does not change this check"
        )
    on = [c.label() for c in criteria if role in (c.a, c.b)]
    said = f"; its criteria here ({', '.join(on)}) do not say where it is" if on else ""
    return (
        f"{head}{said}. The Memory needs a done_spec on {role} that places it: gap_z {role} "
        f"table (its height over the table), or distance_xy, offset_x, offset_y or gap_z between "
        f"{role} and the object it is placed by or moved with (overlap, inside and press place it "
        "too; held and yaw_diff do not)"
    )


WRONG_ARM: Final = (
    "{role} is held by the {arm} arm, but its x points the way its arm side ({rule!r}) gives the "
    "other arm: the demonstration's end mirrored holds only with the arm and the object's "
    "direction mirrored together, so it is carried the wrong way round. Set it down and take it "
    "with the other arm"
)


def wrong_arms(
    registry: SceneRegistry,
    sides: Mapping[str, str],
    lateral: ArrayLike,
    named: Sequence[Arm] = (),
    staged: Collection[str] = (),
) -> dict[str, Arm]:
    """The roles with an `arm side:` rule held by one arm alone, their frame's x decided and
    pointing to a side, where that arm is not the one the rule gives (role -> the arm holding
    it). The other arm may carry it to the mirror image of the demonstration's end, which looks
    right; only the object's own direction tells it was carried the wrong way round. The rule
    relates the arm to where x pointed when it took the object, so x is read at its start
    (its first measurement, as `plan_arms` reads it before the grasp), not where the carry
    turned it. An arm the instruction gives wins over the rule, as in `plan_arms`: a role the
    identification read the instruction's arm for (`ObjectRecord.arm`), or one held by an arm
    the instruction names (`named`, `toolsets.manipulation.arms.named_arms`: which stage that
    arm does is the workflow's), is left out, as is a role the workflow grasps with both arms in
    turn (`staged`, the `stage_arms` of the `MEMORY_SPECS` fact): each stage's holder is the
    stage's arm."""
    out: dict[str, Arm] = {}
    for record in registry.objects.values():
        rule = sides.get(record.name)
        if rule is None or len(record.grasps) != 1 or not record.x_known:
            continue
        if record.arm is not None or set(record.grasps) & set(named) or record.name in staged:
            continue
        start = record.history[0].pose if record.history else record.pose
        x_axis = pose_to_matrix(start.as_list())[:3, 0]
        want = directed_arm(rule, tuple(x_axis), tuple(np.asarray(lateral, dtype=float)))
        (arm,) = record.grasps
        if want is not None and want != arm:
            out[record.name] = arm
    return out


def covers(criteria: Sequence[DoneSpec], end: Sequence[EndTarget], role: str) -> bool:
    """Whether a criterion (`POSITIONAL`) or an end target (place, pair, base, lift, act) says
    where `role` ends (`DoneCheckConfig.coverage`)."""
    if any(c.metric in POSITIONAL and role in (c.a, c.b) for c in criteria):
        return True
    kinds = ("place", "pair", "base", "lift", "overlap", "act")
    return any(t.kind in kinds and role in (t.a, t.b, *t.peers) for t in end)


INSTANCE: Final = "#"  # a role's key for its n-th object: "<role>#<n>"


def every_instance(
    criteria: Sequence[DoneSpec], plural: Mapping[str, Sequence[str]]
) -> tuple[list[DoneSpec], dict[str, str]]:
    """`criteria` with each one on a role of several objects (`plural`: role -> ids) repeated
    for its second and later objects, keyed "<role>#<n>", and those keys' ids: a done criterion
    on such a role holds for every object of it."""
    out: list[DoneSpec] = []
    ids: dict[str, str] = {}
    for spec in criteria:
        out.append(spec)
        for field in ("a", "b"):
            role = getattr(spec, field)
            for n, obj in enumerate(plural.get(role or "", [])[1:], start=2):
                key = f"{role}{INSTANCE}{n}"
                ids[key] = obj
                out.append(spec.model_copy(update={field: key}))
    return out, ids


@dataclass(frozen=True)
class _Read:
    """A new observation's registry state for measuring metrics (`WorldState._read`)."""

    frame: FrameId
    records: dict[str, ObjectRecord]
    geometry: dict[str, ObjectGeometry]
    table: Table
    before: dict[FrameId, dict[str, View]]
    after: dict[str, View]
    remeasured: dict[str, str]
    robot: RobotState | None = None  # the frame's, for the geometry of what an arm holds

    def seen(self) -> dict[str, ObjectGeometry]:
        """The roles' geometry, without those measured again here and not found."""
        return {r: g for r, g in self.geometry.items() if self.remeasured.get(r, "ok") in SEEN}

    def why_unseen(self) -> dict[str, str]:
        """Why each role measured again here and not found is undecidable, and what helps."""
        return {r: _unseen(r, why) for r, why in self.remeasured.items() if why not in SEEN}

    def capped(self, spec: DoneSpec, config: DoneCheckConfig) -> DoneSpec:
        """`spec` with its tolerance capped by its objects' size now (`perception.done.capped`)."""
        sides = self.sides()
        return capped(spec, sides.get(spec.a), sides.get(spec.b or ""), config)

    def sides(self) -> dict[str, float]:
        """Each role's smaller footprint side (m), what tolerances are capped by."""
        return {r: float(min(g.size[:2])) for r, g in self.geometry.items()}

    def value(self, spec: DoneSpec | DoneValue) -> DoneValue:
        """`spec`'s metric measured, or why it is undecidable."""
        found = DoneValue(metric=spec.metric, a=spec.a, b=spec.b, part=spec.part, measured=None)
        roles = [r for r in (spec.a, spec.b) if r is not None and r != TABLE]
        missing = [r for r in roles if r not in self.records]
        if missing:
            return found.model_copy(update={"reason": f"{missing[0]} is not registered"})
        unseen = [r for r in roles if self.remeasured.get(r, "ok") not in SEEN]
        if unseen:
            role = unseen[0]
            return found.model_copy(update={"reason": _unseen(role, self.remeasured[role])})
        record = self.records[spec.a]
        if spec.metric == "press":
            measured, reason = press_offset(record, self.geometry[spec.a], spec.part)
            return found.model_copy(update={"measured": measured, "reason": reason})
        if spec.metric == "surface_change":
            if record.moved or record.taken or spec.a in self.remeasured:
                return found.model_copy(update={"reason": "an arm has carried it"})
            if spec.part is not None and spec.part not in record.parts:
                return found.model_copy(update={"reason": f"it has no part {spec.part}"})
            change = surface_change(
                surface_points(record, spec.part), self.before[_first_frame(record)], self.after
            )
            reason = "" if change is not None else "too little of it shows in the fixed cameras"
            return found.model_copy(update={"measured": change, "reason": reason})
        b = None if spec.b in (None, TABLE) else self.geometry[spec.b]
        measured, reason = measure(spec.metric, self.geometry[spec.a], b, self.table)
        return found.model_copy(update={"measured": measured, "reason": reason})

    def start(self, value: DoneValue) -> DoneValue:
        """`value`'s metric at the episode's start: each object at its first measured pose, held
        by no arm, no surface changed, nothing pressed (None where the value itself is undecided
        for a missing role)."""
        found = value.model_copy(update={"measured": None, "reason": ""})
        roles = [r for r in (value.a, value.b) if r is not None and r != TABLE]
        if any(r not in self.records for r in roles) or value.metric == "press":
            return found
        if value.metric in UNARY:
            return found.model_copy(update={"measured": 0.0})
        a = _at_start(self.records[value.a])
        b = None if value.b in (None, TABLE) else _at_start(self.records[value.b])
        measured, reason = measure(value.metric, a, b, self.table)
        return found.model_copy(update={"measured": measured, "reason": reason})


CONTAINED: Final = "contained"
"""An object an arm let go of into another (a basket, a bin) whose walls now hide it from every
camera: not seen again, so judged from where it was released (`contained`)."""


CONTAINED_TEXT: Final = "not visible; judged from where it was released, inside another object"


HIDDEN_GOAL: Final = "an object is hidden inside another, which the views cannot show"


RELEASED: Final = "released"
"""At the demonstration's end, an object its arm let go of that the last keyframe did not measure
again with confidence: where the replayed release left it (`demo_end_reason`)."""


SEEN: Final = frozenset({"ok", "relocated", RELEASED, CONTAINED})


def contained(
    registry: SceneRegistry, record: ObjectRecord, robot: RobotState | None
) -> ObjectRecord | None:
    """`record`, released and not seen again, as it stands inside the registered object its
    center was let go of in (that object's box where it stood unheld, the walls hiding it: a
    can in a basket); moved along with that object since, when an arm holds it now (a basket
    lifted).
    None when it was let go of inside none."""
    at = object_geometry(record, robot).center
    for other in registry.objects.values():
        if other.id == record.id or other.part_grasps:
            continue
        then = object_geometry(other.model_copy(update={"grasps": {}}), robot)
        local = (at - np.asarray(then.obb.center)) @ quat_to_matrix(then.obb.q)
        if not np.all(np.abs(local) <= np.asarray(then.obb.size) / 2):
            continue
        if not other.grasps:
            return record
        now = object_geometry(other, robot)
        moved = pose_to_matrix(now.pose) @ np.linalg.inv(pose_to_matrix(then.pose))
        pose = pose_from_matrix(moved @ pose_to_matrix(record.pose.as_list()))
        return record.model_copy(update={"pose": Pose.from_list(pose)})
    return None


def demo_end_reason(reason: str, fallback: bool) -> str:
    """How a role an arm let go of stands at the demonstration's end after measuring it again
    (`reason`): measured (`ok`), or, with `fallback` (`DoneCheckConfig.release_fallback`), where
    the replayed release put it (`RELEASED`): the recorded arm still hovers over what it just set
    down, so the last keyframe rarely matches it with confidence, while the release itself
    is the demonstration's proprioception on an object perception placed in its first frame."""
    return RELEASED if fallback and reason != "ok" else reason


def _unseen(role: str, why: str) -> str:
    # Only tools every role has: an Actor has no look_pose or refine (the Demo Reader's), and an
    # arm hanging over the object is the one cause an Actor can remove.
    if why == "not_visible":
        return f"{role} was not seen again where it was set down: move any arm over it away"
    return (
        f"{role} was seen but not matched with confidence where it was set down (an arm over "
        "it, or it lies otherwise than it was set down)"
    )


def _first_frame(record: ObjectRecord) -> FrameId:
    """The frame an object was first measured in: the baseline of its surface change (a later
    measurement, such as the one after each action, must not reset it)."""
    return record.history[0].frame if record.history else record.frame


def _at_start(record: ObjectRecord) -> ObjectGeometry:
    return start_geometry(record)


class Done(Toolset):
    """The done tools: check the Memory's done criteria and the demonstration's end, and read a
    try's end and the demonstration's (system only)."""

    name = "done"
    Config = SceneConfig
    config: SceneConfig

    async def start(self, runtime: ToolsetRuntime) -> None:
        self.scene = SceneAccess(self.config, runtime)

    async def stop(self) -> None:
        await self.scene.close()

    @tool(kind="perception", tier="D", timeout_s=CHECK_TIMEOUT_S, render=render_done)
    async def check_done(self, ctx: ToolContext) -> CheckDoneReport:
        """Whether the task looks done: each of the Memory's done criteria measured on a new
        observation (objects an arm let go of measured again there first), with the numbers; a
        criterion on a role with several objects holds for each of them (the second and later named
        `<role>#<n>`, in the role's order); a press criterion reads where an empty gripper last
        pushed on the object and was stopped (as an action's receipt reported it), since a press
        need not show afterwards. When the demonstration's final state is known, every camera's view
        (head and wrists) is also compared with it, and the check passes only if that comparison
        confirms it; when the
        demonstration pushed something with empty fingers, it also needs such a push of this
        episode stopped on a registered object. When the demonstration's end is known as
        targets, each object it moved is also measured against it (how many arms hold it, how
        high it is lifted, its yaw where it was turned, where it sits relative to the object
        under or beside it), each gap with the correction that closes it: a small shift
        of that object, not a new attempt from the start. A lift (an object held up in the air) is
        held to two heights, both as the demonstration's end had them: its gap over the table and
        its height over the robot's base (the table may be lower here). An object the
        demonstration was still carrying when its success latched must end past the
        demonstration's end along that carry (the line says which way and how far). A long
        object held at the end with its end on another object (a tool on its target) is held by
        that end, not by its whole: the end's center where the demonstration had it relative to
        the target, and no higher above it.
        finish needs a passing check_done after the last action; move an arm that hides an
        object out of the way before checking."""
        fact = ctx.frames.fact(DONE)
        if fact is None:
            # only without a demonstration (every Actor episode with one has the fact)
            raise ToolError(
                "nothing to check against: this episode has no demonstration and no done "
                "criteria, so the official check alone decides whether finish ends it"
            )
        done = DoneFact.model_validate(fact)
        config = self.scene.config.done
        kept = [t for t in done.end if config.end_pairs or t.kind not in ("pair", "base")]
        done = done.model_copy(update={"end": kept if config.end_targets else []})
        criteria, ids = done.criteria, {}
        plural = await self._plural(ctx)
        if plural:  # a criterion on a role with several objects holds for each of them
            criteria, ids = every_instance(criteria, plural)
        roles = {r for spec in criteria for r in (spec.a, spec.b) if r not in (None, TABLE)}
        read = await self._read(ctx, roles | end_roles(done.end), ids)
        base = self._base(ctx)
        reference = None
        if config.relative_height and done.reference is not None and base is not None:
            reference = await self.scene.reference_table(ctx, done.reference)
        seen = read.seen()
        # Stacked or set into something: tested against every registered object, not only
        # the roles the criteria name (the one under it may be named by none).
        registry = await ctx.frames.registry()
        scene = {r.name: object_geometry(r, read.robot) for r in registry.objects.values()}
        scene.update(seen)
        # lifts in the air latched (`done.latched`), and targets with a motion into the latch
        margin = config.latch_margin_m if config.latch_margin_m > 0 else None
        latched = set(done.latched)

        def latch_lift(role: str) -> bool:
            return ANY_LIFT in latched or role.split(INSTANCE)[0] in latched

        lines = []
        for spec in criteria:
            # a side the demonstration's arm gave holds for the arm that took the role here
            acting = seen.get(spec.a)
            spec = for_arm(spec, None if acting is None else acting.handled_by)
            found, held, lift = read.value(spec), read.capped(spec, config), None
            measured = found.measured is not None and spec.a in seen
            lifted = measured and lifts(spec) and in_the_air(spec.a, scene)
            if lifted and reference is not None and base is not None:
                lift = lift_excess(spec.value, seen[spec.a], reference, base)
            minimum = margin if lifted and latch_lift(spec.a) else None
            lines.append(_line(spec, found, held, lift, minimum))
        streaks: dict[str, int] = ctx.state.setdefault(_FAILED, {})
        for i, line in enumerate(lines):
            key = line.spec.label()
            streaks[key] = 0 if line.passed else streaks.get(key, 0) + 1
            lines[i] = line.model_copy(update={"failed": streaks[key]})
        goal = None if done.goal is None else await self._goal(ctx, read.frame, done.goal)
        failing = [done_line(line) for line in lines if not line.passed]
        labels = [line.spec.label() for line in lines if not line.passed]
        unseen = read.why_unseen()
        ends = (
            end_lines(
                done.end,
                seen,
                read.table,
                unseen,
                base,
                reference,
                margin,
                {r for r in seen if latch_lift(r) and in_the_air(r, scene)},
            )
            if done.end
            else []
        )
        failing += [end_text(line) for line in ends if not line.passed]
        labels += [line.target.label() for line in ends if not line.passed]
        uncovered = []
        if config.coverage:
            registry = await ctx.frames.registry()
            taken = sorted(r.name for r in registry.objects.values() if r.taken)
            uncovered = [r for r in taken if not covers(done.criteria, done.end, r)]
            failing += [uncovered_text(r, criteria, done.criteria_read) for r in uncovered]
            labels += [f"{r} placed by nothing" for r in uncovered]
        wrong = []
        sides = (ctx.frames.fact(ARM_SIDES) or {}).get("roles", {})
        if sides and base is not None:
            registry = await ctx.frames.registry()
            live = ctx.frames.live()
            instruction = None if live is None else (await ctx.frames.get(live.alias)).instruction
            staged = (ctx.frames.fact(MEMORY_SPECS) or {}).get("stage_arms") or {}
            named = named_arms(instruction)
            wrong = wrong_arms(registry, sides, base[:3, 1], named, staged)
            failing += [WRONG_ARM.format(role=r, arm=a, rule=sides[r]) for r, a in wrong.items()]
            labels += [f"{r} held by the wrong arm" for r in wrong]
        # The views cannot show what a container's walls hide: a basket that "appears empty" is
        # no evidence against the tracked release there.
        hidden = CONTAINED in read.remeasured.values()
        if goal is not None and goal.reached is not True and not hidden:
            failing.append(_goal_failing(goal))
            labels.append("the final-state views")
        pressed = None
        if done.press:
            registry = await ctx.frames.registry()
            pressed = sorted(r.name for r in registry.objects.values() if r.presses)
            if not pressed:
                failing.append(NO_PRESS)
                labels.append("press")
        registry = await ctx.frames.registry()
        stuck = _progress(ctx.state, labels, read, registry, done.latch, self._base(ctx), config)
        judged = [(line.spec, line.passed) for line in lines if line.measured is not None]
        edge = edges(ctx.state, lines, ends)
        pairs = _overshot_pairs(ctx.state, pair_fixes(judged, ends, seen, base, margin))
        return CheckDoneReport(
            passed=not failing,
            gave_up=stuck is not None and _gives_up(stuck, config, ctx.context.stage),
            stuck=stuck,
            failing=failing,
            frame=read.frame,
            lines=lines,
            remeasured=read.remeasured,
            goal=goal,
            pressed=pressed,
            end=ends,
            uncovered=uncovered,
            wrong_arm=sorted(wrong),
            criteria_read=done.criteria_read,
            pairs=pairs,
            overshot=overshoots(ctx.state, lines),
            edge=edge,
        )

    @tool(kind="perception", tier="D", timeout_s=READ_TIMEOUT_S, system_only=True, after_end=True)
    async def read_end_targets(self, ctx: ToolContext) -> EndTargetsReport:
        """The demonstration's end as targets a try's end is measured against
        (`perception.endstate`): the registered roles at its last keyframe (as `read_relations`
        reads them at `demo_end`) against where they were first measured; pair and base targets
        too under the `done` config's `end_pairs`."""
        read = await self._demo_end(ctx)
        start = {r: _at_start(rec) for r, rec in read.records.items()}
        pairs = self.scene.config.done.end_pairs
        base = self._base(ctx) if pairs else None
        targets = end_targets(read.seen(), start, read.table, pairs=pairs, base=base)
        return EndTargetsReport(targets=targets)

    def _base(self, ctx: ToolContext) -> np.ndarray | None:
        """The robot's base frame (world 4x4), None without an embodiment."""
        profile = self.scene.profile(ctx)
        return None if profile is None else robot_base(profile).matrix()

    async def _goal(self, ctx: ToolContext, frame: FrameId, goal: DoneGoal) -> GoalMatch:
        """The views of `frame` compared with the goal's, every goal camera paired
        (`vlm.goal_match`); a camera missing from the frame or a comparison that failed is
        undecided with why, which does not pass."""
        seen = await self.scene.frame(ctx, frame)
        cameras = list(goal.views)
        if missing := [c for c in cameras if c not in seen.cameras]:
            return GoalMatch(reached=None, differs=f"no current view of {', '.join(missing)}")
        now = {c: (await seen.image(c)).rgb for c in cameras}
        views = {c: await read_rgb(ctx.blobs, goal.views[c]) for c in cameras}
        client = self.scene.runtime.models(session_id=f"{ctx.episode_id}.perception")
        try:
            match, _ = await goal_match(
                client,
                now,
                views,
                goal.instruction,
                self.scene.config.vlm,
                ctx.blobs,
                self.scene.config.pipeline,
            )
        except (InfraError, ToolError) as error:
            ctx.emit("goal_match_failed", {"error": type(error).__name__, "detail": str(error)})
            return GoalMatch(reached=None, differs=f"the comparison failed: {error}")
        return match

    @tool(
        kind="perception",
        tier="D",
        timeout_s=READ_TIMEOUT_S,
        render=render_relations,
        system_only=True,
        after_end=True,
    )
    async def read_relations(
        self,
        ctx: ToolContext,
        at: Annotated[
            Literal["now", "demo_end"],
            Field(
                description="now: a new observation; demo_end: the demonstration's final frame "
                "(its keypoints and frames in this episode), the roles where its recorded "
                "grasps left them."
            ),
        ] = "now",
    ) -> RelationsReport:
        """Every done metric among the registered roles on a new observation (see check_done):
        the relations a try ended with; or at the demonstration's final state, what the
        criteria aim at."""
        registry = await ctx.frames.registry()
        roles = sorted(record.name for record in registry.objects.values())
        read = await (self._read(ctx, set(roles)) if at == "now" else self._demo_end(ctx))
        values = []
        for a in roles:
            parts = [None, *sorted(read.records[a].parts)]
            unary = [
                DoneValue(metric=metric, a=a, part=p, measured=None)
                for metric in ("surface_change", "press")
                for p in parts
            ]
            unary += [
                DoneValue(metric="held", a=a, measured=None),
                DoneValue(metric="gap_z", a=a, b=TABLE, measured=None),
            ]
            binary = [
                DoneValue(metric=metric, a=a, b=b, measured=None)
                for metric in METRICS
                if metric not in UNARY
                for b in roles
                if b != a
            ]
            values += [read.value(v) for v in unary + binary]
        start = [read.start(v) for v in values]
        return RelationsReport(
            frame=read.frame,
            values=values,
            remeasured=read.remeasured,
            start=start,
            sides=read.sides(),
            done=self.scene.config.done,
        )

    async def _plural(self, ctx: ToolContext) -> dict[str, list[str]]:
        """Each role with several registered objects: their ids, the identification's order
        first (`IdentifiedRole.instances`), then any added later."""
        done = self.scene.identified(ctx)
        if done is None or not any(r.instances for r in done.roles.values()):
            return {}
        registry = await ctx.frames.registry()
        plural = {}
        for name, role in done.roles.items():
            ids = [r.id for r in registry.objects.values() if r.name == name]
            if len(ids) > 1:
                known = [i for i in role.instances if i in ids]
                plural[name] = known + [i for i in ids if i not in known]
        return plural

    async def _read(
        self, ctx: ToolContext, roles: set[str], ids: Mapping[str, str] | None = None
    ) -> "_Read":
        """A new observation, with each of `roles` an arm has let go of measured again in it;
        `ids`: the registry id of a role key that names one object of a role with several."""
        seen = await self.scene.observe(ctx)
        after = await self.scene.geometry(ctx, seen)
        registry = await ctx.frames.registry()
        ids = {} if ids is None else ids

        def named(role: str) -> ObjectRecord | None:
            return registry.objects.get(ids[role]) if role in ids else registry.named(role)

        remeasured: dict[str, str] = {}
        for role in sorted(roles):
            record = named(role)
            if record is not None and record.carried and not record.grasps:
                found = await self.scene.remeasure(ctx, record.id, frame=seen.id)
                # relocated and stored is measured where it now is; relocated unsure is not
                sure = found.reason != "relocated" or found.stored
                remeasured[role] = found.reason if sure else "low_confidence"
        registry = await ctx.frames.registry()
        records = {r: rec for r in roles if (rec := named(r)) is not None}
        for role, why in remeasured.items():
            if why in ("not_visible", "low_confidence"):  # not seen elsewhere either
                inside = contained(registry, records[role], seen.robot)
                if inside is not None:
                    remeasured[role], records[role] = CONTAINED, inside
        befores = {}
        for record in records.values():
            first = _first_frame(record)
            if first not in befores:
                befores[first] = await self.scene.geometry(ctx, await self.scene.frame(ctx, first))
        return _Read(
            frame=seen.id,
            records=records,
            geometry={r: object_geometry(rec, seen.robot) for r, rec in records.items()},
            robot=seen.robot,
            table=await self.scene.table(ctx, seen.id),
            before={f: dict(g.views) for f, g in befores.items()},
            after=dict(after.views),
            remeasured=remeasured,
        )

    async def _demo_end(self, ctx: ToolContext) -> "_Read":
        """The registered roles at the demonstration's final state, its last frame (the success
        frame its recording stops at, the goal keyframe): a copy of the registry put back where
        the objects were first measured, the demonstration's gripper events replayed by its
        grasp relations (`demos.relation.replay_grips`; what it let go of where it was measured
        again after, `releases.demo_releases`) and its presses (`replay_presses`), then
        each object an arm let go of measured again in that frame. The demo seed's scene is this
        episode's, so the objects' first measurements are where the demonstration started;
        nothing is stored."""
        fact = ctx.frames.fact(DEMO)
        if fact is None:
            raise ToolError("this episode has no demonstration")
        keypoints = KeypointSet.model_validate(fact)
        final = keypoints.success_frame
        if final is None or ctx.context.embodiment is None:
            raise ToolError("the demonstration has no final frame, or the episode no embodiment")
        wanted = sorted({kp.frame for kps in keypoints.arms.values() for kp in kps} | {final})
        robots = {}
        for t in wanted:
            try:
                robots[t] = (await ctx.frames.get(frame_alias(t))).robot
            except LookupError:
                raise ToolError(f"the demonstration's {frame_alias(t)} is missing") from None
        seen = await self.scene.frame(ctx, frame_alias(final))
        after = await self.scene.geometry(ctx, seen)
        table = await self.scene.table(ctx, seen.id)
        # A copy (its own lock): replaying the grips moves the objects.
        registry = SceneRegistry.model_validate_json((await ctx.frames.registry()).to_json())
        for record in list(registry.objects.values()):
            registry.objects[record.id] = record.model_copy(
                update={"taken": False, "part_grasps": {}, "part_poses": {}, "presses": []}
            )
        profile = embodiment(ctx.context.embodiment)
        released = await demo_releases(self.scene, ctx, keypoints, registry, robots, profile, table)

        def replay() -> None:
            replay_grips(
                keypoints, registry, robots, profile, table, before=final + 1, released=released
            )
            replay_presses(keypoints, registry, robots, profile, final=final)

        await anyio.to_thread.run_sync(replay)
        remeasured = {}
        for record in list(registry.objects.values()):
            if record.carried and not record.grasps:
                run = partial(
                    remeasure, registry, after, record.id, robot=seen.robot,
                    pipeline=self.scene.config.pipeline,
                )  # fmt: skip
                found = await anyio.to_thread.run_sync(run)
                fallback = self.scene.config.done.release_fallback
                remeasured[record.name] = demo_end_reason(found.reason, fallback)
                if found.reason == "ok":
                    registry.objects[record.id] = found.record
        records = {rec.name: rec for rec in registry.objects.values()}
        befores = {}
        for record in records.values():
            first = _first_frame(record)
            if first not in befores:
                befores[first] = await self.scene.geometry(ctx, await self.scene.frame(ctx, first))
        return _Read(
            frame=seen.id,
            records=records,
            geometry={r: object_geometry(rec, seen.robot) for r, rec in records.items()},
            robot=seen.robot,
            table=table,
            before={f: dict(g.views) for f, g in befores.items()},
            after=dict(after.views),
            remeasured=remeasured,
        )
