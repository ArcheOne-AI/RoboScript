"""The end state of a demonstration as targets a try's end is measured against
(docs/concepts/learning.md): what the demonstration did to each object it moved, read by
perception alone from its last keyframe and its start, in terms that carry over to another
scene of the task.

For each role the demonstration moved (its center by over `MOVED_M`, its yaw by over
`TURNED_DEG`, or held at the end), `end_targets` keeps:

- `holds`: how many arms hold it at the end (0: it was let go of), unless it ends held over its
  anchor (the success check may hold before it is let go of: the hold is incidental there);
- `lift`: how high its bottom is above the table, when it ends up in the air (over `LIFTED_M`,
  over no other role): at least that high, less `lift_tol`;
- `yaw`: its yaw in the world, folded by its symmetry, when it ends set down (let go of, or held
  over its anchor; not held up in the air, where the turn is the grasp's): within `YAW_TOL_DEG`
  of where it ended, whether or not the demonstration turned it, since the end pose fixes it and a
  relation such as "on" says nothing of it (a round object has none);
- `place`: where its center ends relative to its anchor, the role it ends over (most of its
  footprint, `ANCHOR_OVERLAP`; a receiver an arm lifted counts) or else the nearest one within
  `NEAR_M` that no arm holds: within a tolerance from
  how much room the anchor's footprint leaves around it (`place_tol`); objects of about the
  same size that end on one anchor are interchangeable (`peers`): the try may put each in either
  one's spot;
- `overlap`: the share of its footprint over the anchor, when most of it ends over it: at
  least that share less `OVERLAP_TOL`;
- `pair` (with `pairs`): how far apart its center ends from another moved role's, for each two
  moved roles not both anchored, on each world axis along which the demonstration ends them
  apart (by at least the smaller footprint side, `tol`): the same order and at most twice the
  demonstration's spacing (within its spacing of it). Objects lifted or carried together have no
  anchor to place them by, yet the demonstration's end orders and spaces them; the band is wide
  because the spacing a task accepts is unknown;
- `base` (with `pairs` and the robot's base frame): where an unanchored one's center ends in
  the base frame (xy), for a role the Memory places at a robot-frame constant (the consumer
  decides: `tools.done.end_for`);
- `act`: a long object an arm holds at the end over no anchor, whose end nearest another role no
  arm holds reaches that role (within `ACT_NEAR_M` of its footprint, at most `ACT_ABOVE_M` above
  its top): a tool acting on a target. Its acting end (`acting_end`: the last `ACT_END` of its
  length at the end nearer the target) is held there, not the whole object: the end's center
  within `act_tol` of where the demonstration's end puts it relative to the target's center
  (xy). A whole-object relation cannot tell a tool whose middle overlaps the target from one
  whose working end is on it. Its height over the target is reported, not judged: a
  success and a false finish of a strike can end at the same measured height, so a ceiling
  would only ask to press down a tool already in contact.

A place target knows the arm that placed its role (`arm`, the arm that took it last at the
demonstration's end) and follows it (`mirror`, with the robot's base frame): on a scene where the
other arm takes the role, its offset from the anchor is mirrored across the robot's sagittal
plane, since a side the demonstration's arm gave it is not the task's (`place_offset`; the done
check keeps it fixed where the Memory's TASK names a side: `tools.done.end_for`).

Pair and base targets also hold in their mirror image across the robot's sagittal plane
(`mirror`): the robot is symmetric, and another scene may have the other arm do what the
demonstration's did. A base target's tolerance is half the smaller footprint side of its
object, within `PAIR_TOL_M`. A base target on an object an arm holds at the demonstration's end
knows that arm (`arm`, from the demonstration's own TCPs: `annotate_holds`) and whether that arm
carried it there (`carried`: moved it `CARRIED_M` or more sideways while closed on it): then the
image a try is held to follows the arm holding it now, as written for that arm and mirrored for
the other, never the other arm's side for the arm that holds it (two arms that each keep their
own side keep their own targets).

The thresholds are task-agnostic constants and the demonstration's own geometry; each is
tighter than the keypoint referee's (8 cm / 5 cm) so that a try which ends short of the
demonstration's end is told by how much. `end_lines` measures them on the scene now, each with
the correction that would close the gap (a world displacement, a turn about the vertical, a
lift). Targets only add conditions: they never pass anything the other checks refuse.
"""

import itertools
import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Final

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel

from roboscript.contracts.robot import Arm
from roboscript.perception.criteria import EndTarget
from roboscript.perception.registry import ObjectGeometry
from roboscript.perception.relations import overlap
from roboscript.perception.table import Table
from roboscript.spatial.rotation import quat_to_matrix

MOVED_M: Final = 0.02
TURNED_DEG: Final = 10.0
LIFTED_M: Final = 0.03
ANCHOR_OVERLAP: Final = 0.3
NEAR_M: Final = 0.25
YAW_TOL_DEG: Final = 6.0
OVERLAP_TOL: Final = 0.1
CORRECTION_MIN_M: Final = 0.003
"""A correction the text asks for is at least this (the done check itself is unchanged): a
smaller step falls within how far an arm under load settles from its command, and asking for a
millimeter has the Actor try moves the arm does not make."""
PLACE_TOL_M: Final = (0.02, 0.03)
"""A place target's tolerance: half the room the anchor's footprint leaves around the object,
within these bounds (m)."""
PAIR_TOL_M: Final = (0.02, 0.04)
"""A base target's tolerance: half the smaller footprint side, within these bounds (m)."""
HOLD_MATCH_M: Final = 0.12
"""A base target's object is the one an arm holds at the demonstration's end when that arm's
last TCP lies this near its center (horizontally), the nearest such."""
CARRIED_M: Final = 0.05
"""An arm carried what it holds when its TCP moved this far sideways (m) while closed on it."""
PEER_SIZE: Final = 0.15
"""Objects whose footprint sizes differ by at most this share are interchangeable on one anchor."""

ACT_END: Final = 0.25
"""An acting end: this share of a long object's length, at its end nearer the target."""
ACT_LONG: Final = 1.5
"""An object is long (has ends to act with) when its length is this many times its width."""
ACT_NEAR_M: Final = 0.05
"""An acting end reaches a target when its center lies this near the target's footprint (xy)."""
ACT_ABOVE_M: Final = 0.06
"""... and its bottom at most this far above the target's top."""
ACT_TOL_M: Final = (0.015, 0.025)
"""An act target's tolerance: half the target's smaller footprint side, within these bounds (m)."""


class EndLine(BaseModel):
    """A target measured now: `measured` (None: undecidable, `note` says why), whether it holds
    and the `correction` that would close the gap: `move` (world dx, dy m: of a's center),
    `turn` (deg about the vertical, counterclockwise seen from above), `raise_m`; a lift held
    to the robot's base too: a's bottom's height over the base (`base`) and the
    demonstration's (`base_need`, m); `margin`: a lift of a demonstration cut at its success
    latch, held at least this much above the demonstration's heights (None: within its
    tolerance below them). A target one-sided along the motion that carried its role into the
    latch: `along`, how far past the demonstration's end along it (m, signed), `across`, how
    far off it sideways (m), `toward`, the motion's world xy unit vector, and `margin`."""

    target: EndTarget
    measured: float | None
    passed: bool
    note: str = ""
    move: tuple[float, float] | None = None
    turn: float | None = None
    raise_m: float | None = None
    base: float | None = None
    base_need: float | None = None
    margin: float | None = None
    above: float | None = None
    """act: how far the acting end's bottom is above the target's top now (m)."""
    along: float | None = None
    across: float | None = None
    toward: tuple[float, float] | None = None


def _period(g: ObjectGeometry) -> float | None:
    return {"none": 360.0, "pi": 180.0, "half_pi": 90.0}.get(g.symmetry)


def _signed(delta: float, period: float) -> float:
    """`delta` folded into [-period / 2, period / 2)."""
    return (delta + period / 2) % period - period / 2


def _xy(g: ObjectGeometry) -> np.ndarray:
    return np.asarray(g.center[:2], dtype=float)


def _room(a: ObjectGeometry, b: ObjectGeometry) -> float:
    """How far a's center may move inside b's footprint and stay over it (m, may be < 0)."""
    return float(min(b.size[:2]) / 2 - max(a.size[:2]) / 2)


def place_tol(a: ObjectGeometry, b: ObjectGeometry) -> float:
    low, high = PLACE_TOL_M
    return float(np.clip(_room(a, b) / 2, low, high))


def lift_tol(gap: float) -> float:
    """At most 1 cm (perception's height error) short of the demonstration's own lift: never
    looser than the height its end reached."""
    return min(0.01, 0.3 * gap)


def _anchor(a: str, end: Mapping[str, ObjectGeometry]) -> tuple[str | None, float]:
    """The role `a` ends over (largest footprint share, at least `ANCHOR_OVERLAP`; one an arm
    holds counts, a receiver lifted with it in it, unless the same arm holds `a`), else the
    nearest within `NEAR_M` (xy) that no arm holds, and a's share over it."""
    ga = end[a]
    hands = set(ga.held_by)
    others = [b for b in end if b != a and not (hands & set(end[b].held_by))]
    shares = {b: overlap(ga.points, end[b]) for b in others}
    over = max(others, key=lambda b: shares[b], default=None)
    if over is not None and shares[over] >= ANCHOR_OVERLAP:
        return over, shares[over]
    free = [b for b in others if not end[b].held_by]
    near = min(free, key=lambda b: float(np.linalg.norm(_xy(ga) - _xy(end[b]))), default=None)
    if near is not None and np.linalg.norm(_xy(ga) - _xy(end[near])) <= NEAR_M:
        return near, shares[near]
    return None, 0.0


def _similar(a: ObjectGeometry, b: ObjectGeometry) -> bool:
    sa, sb = sorted(a.size[:2]), sorted(b.size[:2])
    return all(abs(x - y) <= PEER_SIZE * max(x, y) for x, y in zip(sa, sb, strict=True))


def _pair_tol(*objects: ObjectGeometry) -> float:
    low, high = PAIR_TOL_M
    return float(np.clip(_side(*objects) / 2, low, high))


def _side(*objects: ObjectGeometry) -> float:
    return float(min(min(g.size[:2]) for g in objects))


def _in_base(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    """A world point's xy in the base frame `base` (4x4, world)."""
    local = base[:3, :3].T @ (np.asarray(point, dtype=float) - base[:3, 3])
    return local[:2]


def over_base(base: np.ndarray, point: ArrayLike) -> float:
    """A world point's height in the robot's base frame `base` (4x4, world)."""
    return float((np.linalg.inv(base) @ np.array([*np.asarray(point, dtype=float)[:3], 1.0]))[2])


def lift_excess(
    value: float, a: ObjectGeometry, reference: Table, base: np.ndarray
) -> tuple[float, float, float]:
    """A lift of `value` m over the table measured against the robot's base: how far a's bottom
    is above the demonstration's height over the base (its `value` over the demonstration's
    table `reference`, a plane fixed in the base frame), a's bottom's height over the base now,
    and the demonstration's."""
    bottom = np.asarray(a.bottom_center, dtype=float)
    excess = float(reference.height(bottom[None])[0]) - value  # as the gap is measured
    now = over_base(base, bottom)
    return excess, now, now - excess


def end_targets(
    end: Mapping[str, ObjectGeometry],
    start: Mapping[str, ObjectGeometry],
    table: Table,
    *,
    pairs: bool = False,
    base: np.ndarray | None = None,
) -> list[EndTarget]:
    """The targets of the demonstration's end (`end`: each role at its last keyframe; `start`:
    where it was first measured), see the module; `pairs` adds pair targets, and base targets
    when the robot's base frame `base` (4x4, world) is given."""
    moved = []
    for a, ge in end.items():
        gs = start.get(a)
        if gs is None:
            continue
        shift = float(np.linalg.norm(np.asarray(ge.center) - np.asarray(gs.center)))
        period = _period(ge)
        turn = 0.0 if period is None else abs(_signed(ge.yaw_deg - gs.yaw_deg, period))
        if shift > MOVED_M or turn > TURNED_DEG or ge.held_by:
            moved.append((a, turn))
    targets: list[EndTarget] = []
    anchors: dict[str, tuple[str, float]] = {}
    for a, _ in moved:
        ge = end[a]
        b, share = _anchor(a, end)
        over = b is not None and share >= ANCHOR_OVERLAP
        if ge.held_by and not over and (act := _act(a, end)) is not None:
            targets.append(act)
        if not (ge.held_by and over):  # held over its anchor: still being put down, any hold
            targets.append(EndTarget(kind="holds", a=a, value=float(len(ge.held_by)), tol=0.5))
        if b is not None and (over or not ge.held_by):
            anchors[a] = (b, share)
        gap = float(table.height(np.asarray(ge.bottom_center)[None])[0])
        if gap > LIFTED_M and not over:
            targets.append(EndTarget(kind="lift", a=a, value=gap, tol=lift_tol(gap)))
        period = _period(ge)
        if period is not None and (not ge.held_by or over):
            value = float(_signed(ge.yaw_deg, period))
            targets.append(EndTarget(kind="yaw", a=a, value=value, tol=YAW_TOL_DEG, period=period))
    for a, (b, share) in anchors.items():
        ga, gb = end[a], end[b]
        peers = sorted(
            p for p, (pb, _) in anchors.items() if p != a and pb == b and _similar(end[p], ga)
        )
        offset = _xy(ga) - _xy(gb)
        targets.append(
            EndTarget(
                kind="place",
                a=a,
                b=b,
                offset=(float(offset[0]), float(offset[1])),
                tol=place_tol(ga, gb),
                peers=peers,
                arm=ga.handled_by,
                mirror=base is not None and ga.handled_by is not None,
            )
        )
        if share >= 0.5:
            targets.append(EndTarget(kind="overlap", a=a, b=b, value=share, tol=OVERLAP_TOL))
    if not pairs:
        return targets
    roles = sorted(a for a, _ in moved)
    for a, b in itertools.combinations(roles, 2):
        if a in anchors and b in anchors:
            continue  # each placed by its anchor: their spacing follows
        offset = _xy(end[a]) - _xy(end[b])
        side = _side(end[a], end[b])
        if not (np.abs(offset) >= side).any():
            continue  # side by side or stacked: no spacing to keep
        targets.append(
            EndTarget(
                kind="pair",
                a=a,
                b=b,
                offset=(float(offset[0]), float(offset[1])),
                tol=side,
                mirror=base is not None,
            )
        )
    if base is not None:
        for a in roles:
            if a in anchors:
                continue
            local = _in_base(base, end[a].center)
            targets.append(
                EndTarget(
                    kind="base",
                    a=a,
                    offset=(float(local[0]), float(local[1])),
                    tol=_pair_tol(end[a]),
                    mirror=True,
                )
            )
    return targets


def acting_end(g: ObjectGeometry, toward: ArrayLike) -> tuple[np.ndarray, float] | None:
    """A long object's acting end nearer the world point `toward` (xy): the center (xy) and the
    bottom (z) of its points in the last `ACT_END` of its length along its main horizontal axis
    at that end; None when it is not long (`ACT_LONG`) or has no points."""
    points = np.asarray(g.points, dtype=float)
    if len(points) < 3:
        return None
    xy = points[:, :2] - points[:, :2].mean(axis=0)
    _, _, axes = np.linalg.svd(xy, full_matrices=False)
    along, across = xy @ axes[0], xy @ axes[1]
    length, width = float(np.ptp(along)), float(np.ptp(across))
    if length < ACT_LONG * max(width, 1e-6):
        return None
    ends = [along >= along.max() - ACT_END * length, along <= along.min() + ACT_END * length]
    aim = np.asarray(toward, dtype=float)[:2]
    centers = [points[m, :2].mean(axis=0) for m in ends]
    i = int(np.argmin([np.linalg.norm(c - aim) for c in centers]))
    return centers[i], float(points[ends[i], 2].min())


def act_tol(b: ObjectGeometry) -> float:
    low, high = ACT_TOL_M
    return float(np.clip(min(b.size[:2]) / 2, low, high))


def _beside(point: np.ndarray, b: ObjectGeometry) -> float:
    """How far a world xy point lies outside b's footprint (0: over it)."""
    rotation = quat_to_matrix(b.obb.q)
    x = rotation[:2, 0] / max(float(np.linalg.norm(rotation[:2, 0])), 1e-9)
    local = (point - np.asarray(b.obb.center[:2])) @ np.stack([x, [-x[1], x[0]]]).T
    return float(np.linalg.norm(np.maximum(np.abs(local) - np.asarray(b.obb.size[:2]) / 2, 0.0)))


def _act(a: str, end: Mapping[str, ObjectGeometry]) -> EndTarget | None:
    """The act target of `a` (held, over no anchor) at the demonstration's end: on the role no
    arm holds that its acting end reaches (see the module), the nearest; None without one."""
    found = []
    for b, gb in end.items():
        if b == a or gb.held_by:
            continue
        tip = acting_end(end[a], gb.center)
        if tip is None:
            continue
        xy, bottom = tip
        above = bottom - float(gb.top_center[2])
        if _beside(xy, gb) <= ACT_NEAR_M and -float(gb.size[2]) <= above <= ACT_ABOVE_M:
            found.append((_beside(xy, gb), b, xy - _xy(gb), above))
    if not found:
        return None
    _, b, offset, above = min(found, key=lambda f: f[0])
    return EndTarget(
        kind="act",
        a=a,
        b=b,
        value=above,
        offset=(float(offset[0]), float(offset[1])),
        tol=act_tol(end[b]),
        mirror=True,
    )


def _acted(t: EndTarget, g: ObjectGeometry, gb: ObjectGeometry, base: np.ndarray | None) -> EndLine:
    """An act target measured now: a's acting end toward b, its center against the
    demonstration's offset from b's (or the offset's mirror image across the robot's sagittal
    plane, with `base`, whichever is nearer: the other arm may act from the other side) within
    `tol`; its bottom's height above b's top is reported (`above`)."""
    tip = acting_end(g, gb.center)
    if tip is None:
        return EndLine(target=t, measured=None, passed=False, note=f"{t.a} shows no long end now")
    assert t.offset is not None
    xy, bottom = tip
    aims = [np.asarray(t.offset, dtype=float)]
    if t.mirror and base is not None:
        lateral = base[:2, 1] / max(float(np.linalg.norm(base[:2, 1])), 1e-9)
        aims.append(aims[0] - 2 * float(aims[0] @ lateral) * lateral)
    now = xy - _xy(gb)
    aim = min(aims, key=lambda x: float(np.linalg.norm(now - x)))
    off = float(np.linalg.norm(now - aim))
    ok = off <= t.tol
    move = aim - now
    return EndLine(
        target=t,
        measured=off,
        passed=ok,
        move=None if ok else (float(move[0]), float(move[1])),
        above=bottom - float(gb.top_center[2]),
    )


def roles_of(targets: Sequence[EndTarget]) -> set[str]:
    return {r for t in targets for r in (t.a, t.b, *t.peers) if r is not None}


def end_lines(
    targets: Sequence[EndTarget],
    now: Mapping[str, ObjectGeometry],
    table: Table,
    unseen: Mapping[str, str] | None = None,
    base: np.ndarray | None = None,
    reference: Table | None = None,
    margin: float | None = None,
    aloft: AbstractSet[str] = frozenset(),
) -> list[EndLine]:
    """Every target measured on `now` (each role's geometry; a missing role is undecidable, for
    the reason `unseen` gives); pair and base targets use the robot's base frame `base` (4x4,
    world) for their mirror image and frame (a base target is undecidable without it). With the
    demonstration's table `reference` (and `base`), a lift target also holds the object's bottom
    at least as high over the robot's base as the demonstration's (`lift_excess`, less its
    tolerance): another scene's table may stand lower, and the correction raises it by the larger
    shortfall. With `margin`, a lift target on a role of `aloft` (whose lift latched the
    demonstration's success, in the air now) needs the bottom at least `margin` above both
    heights instead of within its tolerance below them; and a place, pair or base target with a
    `motion` of at least `MOVED_M` is one-sided along it (`_directed`)."""
    spots = _assign([t for t in targets if t.kind == "place"], now, base)
    lines = []
    for t in targets:
        missing = [r for r in (t.a, t.b) if r is not None and r not in now]
        if missing:
            note = (unseen or {}).get(missing[0], f"{missing[0]} is not registered")
            lines.append(EndLine(target=t, measured=None, passed=False, note=note))
            continue
        g = now[t.a]
        if t.kind == "holds":
            held = float(len(g.held_by))
            lines.append(EndLine(target=t, measured=held, passed=abs(held - t.value) <= t.tol))
        elif t.kind == "lift":
            gap = float(table.height(np.asarray(g.bottom_center)[None])[0])
            short = t.value - gap
            height = need = None
            if reference is not None and base is not None:
                excess, height, need = lift_excess(t.value, g, reference, base)
                short = max(short, -excess)
            latched = margin if t.a in aloft else None
            if latched is not None:
                short += latched  # how far short of the demonstration's height plus the margin
                ok = short <= 0.0
            else:
                ok = short <= t.tol
            lines.append(
                EndLine(
                    target=t,
                    measured=gap,
                    passed=ok,
                    raise_m=None if ok else short,
                    base=height,
                    base_need=need,
                    margin=latched,
                )
            )
        elif t.kind == "yaw":
            period = _period(g)
            if period is None:
                lines.append(
                    EndLine(target=t, measured=None, passed=False, note="it looks round now")
                )
                continue
            yaw = float(_signed(g.yaw_deg, t.period))
            turn = float(_signed(t.value - yaw, t.period))
            ok = abs(turn) <= t.tol
            lines.append(EndLine(target=t, measured=yaw, passed=ok, turn=None if ok else turn))
        elif t.kind == "overlap":
            share = overlap(g.points, now[t.b]) if t.b is not None else 0.0
            ok = share >= t.value - t.tol
            lines.append(EndLine(target=t, measured=share, passed=ok))
        elif t.kind in ("pair", "base"):
            lines.append(spaced_line(t, g, None if t.b is None else now[t.b], base, margin))
        elif t.kind == "act":
            assert t.b is not None
            lines.append(_acted(t, g, now[t.b], base))
        else:
            aim = spots[t.a]
            toward = _toward(t, margin)
            if toward is not None:
                assert margin is not None
                lines.append(_directed(t, _xy(g) - aim, toward, t.tol, margin))
                continue
            move = aim - _xy(g)
            off = float(np.linalg.norm(move))
            ok = off <= t.tol
            lines.append(
                EndLine(
                    target=t,
                    measured=off,
                    passed=ok,
                    move=None if ok else (float(move[0]), float(move[1])),
                )
            )
    return lines


def _toward(t: EndTarget, margin: float | None) -> np.ndarray | None:
    """The world xy unit vector of the motion that carried t's role into the success latch, when
    the target is one-sided along it: a margin, and a motion of at least `MOVED_M`."""
    if margin is None or t.motion is None:
        return None
    m = np.asarray(t.motion, dtype=float)
    size = float(np.linalg.norm(m))
    return m / size if size >= MOVED_M else None


def _directed(
    t: EndTarget, off: np.ndarray, toward: np.ndarray, tol: float, margin: float
) -> EndLine:
    """A target one-sided along `toward` (world xy unit): `off`, where the role is now less
    where the demonstration's end puts it (world xy); it passes when at least `margin` past it
    along `toward` (further passes too) and within `tol` of it across; `measured` is how far
    outside that it is (0: inside), the correction a world move of a."""
    along = float(off @ toward)
    side = off - along * toward
    across = float(np.linalg.norm(side))
    short = max(margin - along, 0.0)
    wide = max(across - tol, 0.0)
    ok = short == 0.0 and wide == 0.0
    move = short * toward - (side / across * wide if wide > 0.0 else 0.0)
    return EndLine(
        target=t,
        measured=float(np.hypot(short, wide)),
        passed=ok,
        move=None if ok else (float(move[0]), float(move[1])),
        margin=margin,
        along=along,
        across=across,
        toward=(float(toward[0]), float(toward[1])),
    )


def spaced_line(
    t: EndTarget,
    g: ObjectGeometry,
    other: ObjectGeometry | None,
    base: np.ndarray | None,
    margin: float | None = None,
) -> EndLine:
    """A pair target (a's center relative to `other`'s, world xy: on each axis its offset spans
    `tol` or more, within the offset's size of it) or a base target (a's center in the base
    frame, within `tol`), against its offset and, with `mirror`, the offset's mirror image across
    the sagittal plane (the base frame's xz plane), whichever is nearer; a base target that knows
    the demonstration's holding arm (`arm`) and is held by one arm now takes the image for that
    arm only (as written for the same arm, mirrored for the other); `measured` is how far
    outside its band it is (0: inside), and the correction a world move of a. With `margin`
    and a `motion` (`_toward`), a base target is one-sided along it (`_directed`, the motion
    mirrored with the image), and a pair target also needs a's offset from b at least `margin`
    past the demonstration's along it (its band unchanged)."""
    if t.kind == "base" and base is None:
        return EndLine(target=t, measured=None, passed=False, note="the robot's base is unknown")
    assert t.offset is not None
    aim = np.asarray(t.offset, dtype=float)
    toward = _toward(t, margin)
    # each aim with the motion into the latch in the target's frame, mirrored as its aim is
    unit = np.zeros(2) if toward is None else toward
    if t.kind == "pair":
        assert other is not None
        now = _xy(g) - _xy(other)
        aims = [(aim, unit)]
        if t.mirror and base is not None:
            lateral = base[:2, 1] / max(float(np.linalg.norm(base[:2, 1])), 1e-9)

            def reflect(x: np.ndarray) -> np.ndarray:
                return x - 2 * float(x @ lateral) * lateral

            aims.append((reflect(aim), reflect(unit)))
        to_world = np.eye(2)

        def outside(x: np.ndarray) -> np.ndarray:
            span = np.where(np.abs(x) >= t.tol, np.abs(x), np.inf)
            return np.maximum(np.abs(now - x) - span, 0.0)

    else:
        assert base is not None
        now = _in_base(base, g.center)
        to_world = base[:2, :2]
        flip = np.array([1.0, -1.0])
        local = to_world.T @ unit
        same, image = (aim, local), (aim * flip, local * flip)
        aims = [same, image] if t.mirror else [same]
        holder = held_alone(g)
        if t.arm is not None and holder is not None:
            aims = [same] if holder == t.arm else [image]

        def outside(x: np.ndarray) -> np.ndarray:
            return np.array([max(float(np.linalg.norm(now - x)) - t.tol, 0.0)])

    if toward is not None:
        assert margin is not None
        least = margin

        def short(x: np.ndarray, u: np.ndarray) -> float:
            return max(least - float((now - x) @ u), 0.0)

        if t.kind == "base":
            lines = [
                _directed(t, to_world @ (now - x), to_world @ u, t.tol, least) for x, u in aims
            ]
            return min(lines, key=lambda line: line.measured or 0.0)

        def cost(xu: tuple[np.ndarray, np.ndarray]) -> float:
            return float(np.hypot(np.linalg.norm(outside(xu[0])), short(*xu)))

        x, u = min(aims, key=cost)
        band = float(np.linalg.norm(outside(x)))
        gap = short(x, u)
        ok = band == 0.0 and gap == 0.0
        move = (x - now + least * u) if band > 0.0 else gap * u
        return EndLine(
            target=t,
            measured=float(np.hypot(band, gap)),
            passed=ok,
            move=None if ok else (float(move[0]), float(move[1])),
            margin=least,
            along=float((now - x) @ u),
            toward=(float(u[0]), float(u[1])),
        )
    best = min((x for x, _ in aims), key=lambda x: float(np.linalg.norm(outside(x))))
    off = float(np.linalg.norm(outside(best)))
    ok = off == 0.0
    move = to_world @ (best - now)
    return EndLine(
        target=t,
        measured=off,
        passed=ok,
        move=None if ok else (float(move[0]), float(move[1])),
    )


def held_alone(g: ObjectGeometry) -> Arm | None:
    """The arm holding an object, when one arm alone does."""
    return g.held_by[0] if len(g.held_by) == 1 else None


def annotate_holds(
    targets: Sequence[EndTarget],
    ends: Mapping[Arm, tuple[ArrayLike, ArrayLike]],
    base: np.ndarray,
) -> list[EndTarget]:
    """`targets` with each base target that does not know its arm given the arm that holds its
    object at the demonstration's end, and whether it carried it there (see the module): `ends`,
    each arm closed at the demonstration's end, its TCP (world) where it last closed and at its
    last keypoint; the arm is the one whose last TCP lies nearest the target's center (in the
    base frame, `base` 4x4 world, horizontally), within `HOLD_MATCH_M`. Others are unchanged."""
    tcps = {
        arm: (_in_base(base, np.asarray(close)), _in_base(base, np.asarray(last)))
        for arm, (close, last) in ends.items()
    }
    out = []
    for t in targets:
        if t.kind != "base" or t.arm is not None or t.offset is None:
            out.append(t)
            continue
        aim = np.asarray(t.offset, dtype=float)
        gaps = {arm: float(np.linalg.norm(last - aim)) for arm, (_, last) in tcps.items()}
        arm = min(gaps, key=gaps.__getitem__, default=None)
        if arm is None or gaps[arm] > HOLD_MATCH_M:
            out.append(t)
            continue
        close, last = tcps[arm]
        carried = float(np.linalg.norm(last - close)) >= CARRIED_M
        out.append(t.model_copy(update={"arm": arm, "carried": carried}))
    return out


def annotate_motion(
    targets: Sequence[EndTarget], motions: Mapping[Arm, ArrayLike]
) -> list[EndTarget]:
    """`targets` with each place, pair and base target of a role an arm was still carrying into
    the demonstration's success latch given that motion (`EndTarget.motion`): `motions`, each
    such arm's TCP displacement (world) between its last two keypoints; a role's arm is the one
    its base target knows (`annotate_holds`), a pair's motion a's less b's. A motion under
    `MOVED_M` sideways (a lift straight up) is none: the target keeps `motion` None, as read.
    Others unchanged."""
    moving = {
        t.a: np.asarray(motions[t.arm], dtype=float)[:2]
        for t in targets
        if t.kind == "base" and t.arm is not None and t.arm in motions
    }
    out = []
    for t in targets:
        m = None
        if t.kind in ("place", "base") and t.a in moving:
            m = moving[t.a]
        elif t.kind == "pair" and (t.a in moving or t.b in moving):
            m = moving.get(t.a, np.zeros(2)) - moving.get(t.b or "", np.zeros(2))
        if m is None or float(np.linalg.norm(m)) < MOVED_M:  # not carried sideways: no motion
            out.append(t)
            continue
        out.append(t.model_copy(update={"motion": (float(m[0]), float(m[1]))}))
    return out


def _assign(
    places: Sequence[EndTarget], now: Mapping[str, ObjectGeometry], base: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Where each place target's role should end (world xy): its own spot, or among peers on
    one anchor the assignment of the spots that leaves the largest miss smallest. A spot is the
    target's offset from its anchor, mirrored across the robot's sagittal plane (with `base`)
    when the target follows the arm that placed it there (`mirror`, `arm`) and the other arm
    took its role here (`place_offset`)."""
    spot = {
        t.a: _xy(now[t.b]) + place_offset(t, now[t.a], base)
        for t in places
        if t.b in now and t.a in now and t.offset is not None
    }
    aims = dict(spot)
    groups = {frozenset([t.a, *t.peers]) for t in places if t.peers}
    for group in groups:
        roles = sorted(r for r in group if r in spot and r in now)
        if len(roles) < 2:
            continue
        best = min(
            itertools.permutations(roles),
            key=lambda order: max(
                float(np.linalg.norm(spot[s] - _xy(now[r])))
                for r, s in zip(roles, order, strict=True)
            ),
        )
        for r, s in zip(roles, best, strict=True):
            aims[r] = spot[s]
    return aims


def place_offset(t: EndTarget, g: ObjectGeometry, base: np.ndarray | None) -> np.ndarray:
    """A place target's offset (world xy) as it holds for the arm that took its role last
    (`ObjectGeometry.handled_by`): as written for the target's own `arm`, mirrored across the
    robot's sagittal plane (the base frame's xz plane) for the other arm, when the target
    `mirror`s; as written otherwise (no base, no arm known)."""
    offset = np.asarray(t.offset, dtype=float)
    acting = g.handled_by
    if not t.mirror or base is None or t.arm is None or acting is None or acting == t.arm:
        return offset
    lateral = base[:2, 1] / max(float(np.linalg.norm(base[:2, 1])), 1e-9)
    return offset - 2 * float(offset @ lateral) * lateral


def end_text(line: EndLine) -> str:
    """A line as the model reads it: what was measured, what the demonstration's end has, and
    the correction."""
    t = line.target
    head = t.label()
    if line.measured is None:
        return f"{head}: undecided, {line.note}"
    if t.kind == "holds":
        text = (
            f"{head}: held by {line.measured:.0f} arm(s), the demonstration ends with {t.value:.0f}"
        )
    elif t.kind == "lift":
        slack = -t.tol if line.margin is None else line.margin
        text = (
            f"{head}: bottom {line.measured:.3f} m above the table, the demonstration's end needs "
            f"at least {t.value + slack:.3f} m"
        )
        if line.base is not None and line.base_need is not None:
            text += (
                f"; height over robot base {line.base:.3f} m, the demonstration's end needs at "
                f"least {line.base_need + slack:.3f} m"
            )
        if line.margin is not None:
            text += (
                f" (the demonstration stops the moment the task's success latched, so its height "
                f"is the least that passes: {line.margin:.3f} m more for perception's error; "
                "higher passes too)"
            )
    elif t.kind == "yaw":
        text = (
            f"{head}: yaw {line.measured:.1f} deg, the demonstration ends at {t.value:.1f} ± "
            f"{t.tol:g} deg (it looks the same turned by {t.period:g} deg)"
        )
    elif t.kind == "overlap":
        text = (
            f"{head}: {line.measured:.0%} of its footprint over {t.b}, the demonstration's end "
            f"needs at least {t.value - t.tol:.0%}"
        )
    elif t.kind == "act":
        assert t.offset is not None and line.above is not None
        text = (
            f"{head}: the end of {t.a} nearer {t.b} (its acting end) is {line.measured:.3f} m from "
            f"where the demonstration's end puts it relative to {t.b} (offset {t.offset[0]:+.3f}, "
            f"{t.offset[1]:+.3f} m), tolerance {t.tol:.3f} m (the whole of {t.a} over {t.b} "
            "does not pass: its acting end must be where the demonstration's was); its bottom is "
            f"{line.above:.3f} m above {t.b}'s top, the demonstration's end had {t.value:.3f} m"
        )
    elif line.along is not None and line.toward is not None and line.margin is not None:
        ux, uy = line.toward
        where = f"from {t.b}" if t.b is not None else "in the robot's base frame"
        text = (
            f"{head}: its center is {line.along:+.3f} m along the way the demonstration was still "
            f"moving it when the task's success latched (world direction x {ux:+.2f}, y "
            f"{uy:+.2f}) past where the demonstration's end puts it {where}; the demonstration "
            f"stops the moment the check latched, so its end is the least that passes: it needs at "
            f"least {line.margin:.3f} m past it that way, further passes too"
        )
        if line.across is not None:
            text += f", and at most {t.tol:.3f} m off sideways ({line.across:.3f} m now)"
        if t.kind == "pair" and t.offset is not None:
            text += (
                f", and its spacing within the demonstration's band (offset {t.offset[0]:+.3f}, "
                f"{t.offset[1]:+.3f} m in x, y)"
            )
        if not line.passed and line.along < line.margin:
            text += f"; move it {_least(line.margin - line.along):.3f} m further that way"
    elif t.kind == "pair":
        assert t.offset is not None
        mirrored = " or its mirror image" if t.mirror else ""
        text = (
            f"{head}: its spacing from {t.b} is {line.measured:.3f} m outside the band the "
            f"demonstration's end gives (offset {t.offset[0]:+.3f}, {t.offset[1]:+.3f} m in x, y"
            f"{mirrored}; on each axis where they end {t.tol:.3f} m or more apart, the same side "
            "and at most twice as far)"
        )
    elif t.kind == "base":
        assert t.offset is not None
        mirrored = " or its mirror image" if t.mirror else ""
        # `measured` is how far outside the tolerance it is (0: within it), not the distance
        off = f"{line.measured:.3f} m outside" if line.measured else "within"
        text = (
            f"{head}: its center is {off} the {t.tol:.3f} m tolerance around where the "
            f"demonstration's end puts it in the robot's base frame ({t.offset[0]:+.3f} ahead, "
            f"{t.offset[1]:+.3f} m left{mirrored})"
        )
    else:
        assert t.offset is not None
        text = (
            f"{head}: its center is {line.measured:.3f} m from where the demonstration's end puts "
            f"it relative to {t.b} (offset {t.offset[0]:+.3f}, {t.offset[1]:+.3f} m), tolerance "
            f"{t.tol:.3f} m"
        )
    fixes = []
    if line.move is not None:
        dx, dy = line.move
        scale = _least(math.hypot(dx, dy)) / max(math.hypot(dx, dy), 1e-9)
        fixes.append(f"move it by x {dx * scale:+.3f}, y {dy * scale:+.3f} m")
    if line.turn is not None:
        fixes.append(f"turn it by {line.turn:+.1f} deg about the vertical")
    if line.raise_m is not None:
        fixes.append(f"raise it by {_least(line.raise_m):.3f} m")
    return text + (f"; to correct: {', '.join(fixes)}" if fixes else "")


def _least(m: float) -> float:
    """A correction's size as the text asks for it: at least `CORRECTION_MIN_M`."""
    return max(m, CORRECTION_MIN_M)
