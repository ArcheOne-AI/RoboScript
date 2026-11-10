"""One correction for the done lines that share a pair of objects (docs/concepts/learning.md).

A `distance_xy`, `offset_x` or `offset_y` criterion between two roles and the demonstration's
`pair` end target on them all read one vector: a's center less b's, world xy (the spacing).
Corrected one line at a time (each line's own move of one object), fixing one breaks another.
`pair_fixes` searches for the spacing nearest the current one that every such line passes, each
judged by the code that judges it in `check_done` (`verdict` on the capped criterion,
`endstate.spaced_line` on the target), aiming at the middle of each band first (a correction to a
band's edge leaves the next measurement on either side of it), and splits the change between
the two objects: half each, opposite ways, when two arms hold them (their midpoint stays where
it is), all of it on the one an arm holds otherwise. Perception's measurements only.
"""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Final

import numpy as np
from pydantic import BaseModel

from roboscript.contracts.robot import Arm
from roboscript.perception.criteria import DoneSpec
from roboscript.perception.registry import ObjectGeometry
from roboscript_methods.wicl.perception.done import verdict
from roboscript_methods.wicl.perception.endstate import EndLine, spaced_line

PAIR_METRICS: Final = frozenset({"distance_xy", "offset_x", "offset_y"})
"""The criteria that read the spacing of two roles (xy, a's center less b's)."""
SEARCH_M: Final = 0.25
"""How far from the current spacing a passing one is looked for (m)."""
COARSE_M: Final = 0.004
FINE_M: Final = 0.001
INSETS: Final = ((0.2, 0.015), (0.5, 0.005), (1.0, 0.0))
"""Searched in turn, the first that some spacing passes: a criterion's band shrunk to the share of
its tolerance, a one-sided target held that much (m) past its margin. The middle of each band
first, so that perception's next measurement (millimetres off) still passes; the edges last."""


class PairFix(BaseModel):
    """The lines on roles `a` and `b`'s spacing, `labels` the failing ones; `spacing` now (a's
    center less b's, world xy, m); `target`, the nearest spacing every line passes (None: none
    within `SEARCH_M`); `moves`, role -> world xy move that makes it; `arms`, role -> the arm
    holding it; `middle`, `target` is in the middle of every band (the first of `INSETS`), not
    nearer an edge; `last_change`, set by the caller when the spacing's change since the last check
    went past `target` (then `moves` are at most half that change)."""

    a: str
    b: str
    labels: list[str]
    spacing: tuple[float, float]
    target: tuple[float, float] | None = None
    moves: dict[str, tuple[float, float]] = {}
    arms: dict[str, Arm] = {}
    middle: bool = False
    last_change: tuple[float, float] | None = None


def pair_fixes(
    criteria: Sequence[tuple[DoneSpec, bool]],
    ends: Sequence[EndLine],
    now: Mapping[str, ObjectGeometry],
    base: np.ndarray | None = None,
    margin: float | None = None,
) -> list[PairFix]:
    """A `PairFix` for each two roles with a failing line on their spacing: `criteria` the
    judged (capped) criteria with whether they passed, `ends` the end lines (`end_lines` with
    the same `base` and `margin`), `now` each role's geometry."""
    pairs: dict[tuple[str, str], tuple[list[DoneSpec], list[EndLine], list[str]]] = {}
    for spec, passed in criteria:
        if spec.metric not in PAIR_METRICS or spec.b is None:
            continue
        specs, _, failing = pairs.setdefault(_key(spec.a, spec.b), ([], [], []))
        specs.append(spec)
        if not passed:
            failing.append(spec.label())
    for line in ends:
        t = line.target
        if t.kind != "pair" or t.b is None or line.measured is None:
            continue
        _, lines, failing = pairs.setdefault(_key(t.a, t.b), ([], [], []))
        lines.append(line)
        if not line.passed:
            failing.append(t.label())
    fixes = []
    for (a, b), (specs, lines, failing) in sorted(pairs.items()):
        if not failing or a not in now or b not in now:
            continue
        ga, gb = now[a], now[b]
        spacing = np.asarray(ga.center[:2], dtype=float) - np.asarray(gb.center[:2], dtype=float)

        passes = partial(_passes, a=a, specs=specs, lines=lines, ga=ga, gb=gb, base=base)
        target, middle = _nearest(spacing, passes, margin)
        fix = PairFix(a=a, b=b, labels=failing, spacing=_pair(spacing))
        arms = {r: g.held_by[0] for r, g in ((a, ga), (b, gb)) if g.held_by}
        fix = fix.model_copy(update={"arms": arms})
        if target is None:
            fixes.append(fix)
            continue
        change = target - spacing
        if a in arms and b in arms and arms[a] != arms[b]:
            moves = {a: change / 2, b: -change / 2}
        elif b in arms and a not in arms:
            moves = {b: -change}
        else:
            moves = {a: change}
        fixes.append(
            fix.model_copy(
                update={
                    "target": _pair(target),
                    "moves": {r: _pair(m) for r, m in moves.items()},
                    "middle": middle,
                }
            )
        )
    return fixes


def _passes(
    d: np.ndarray,
    share: float,
    least: float | None,
    *,
    a: str,
    specs: Sequence[DoneSpec],
    lines: Sequence[EndLine],
    ga: ObjectGeometry,
    gb: ObjectGeometry,
    base: np.ndarray | None,
) -> bool:
    """Whether spacing `d` (a's center less b's) passes every criterion in `specs` with its band
    shrunk to `share` of its tolerance, and every pair line in `lines` with margin `least`."""
    for spec in specs:
        v = d if spec.a == a else -d
        measured = {
            "distance_xy": float(np.hypot(*v)),
            "offset_x": float(v[0]),
            "offset_y": float(v[1]),
        }[spec.metric]
        if not verdict(spec.model_copy(update={"tol": spec.tol * share}), measured):
            return False
    for line in lines:
        t = line.target
        ta, tb = (ga, gb) if t.a == a else (gb, ga)
        v = d if t.a == a else -d
        center = np.array([*(np.asarray(tb.center[:2]) + v), ta.center[2]])
        if not spaced_line(t, dataclasses.replace(ta, center=center), tb, base, least).passed:
            return False
    return True


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pair(v: np.ndarray) -> tuple[float, float]:
    return (round(float(v[0]), 4), round(float(v[1]), 4))


def _nearest(
    start: np.ndarray,
    passes: Callable[[np.ndarray, float, float | None], bool],
    margin: float | None,
) -> tuple[np.ndarray | None, bool]:
    """The spacing nearest `start` that `passes`, at the first of `INSETS` some spacing passes
    (and whether that is the first): the first on a coarse grid in order of distance, refined
    on a fine one around it."""
    n = int(SEARCH_M / COARSE_M)
    steps = np.arange(-n, n + 1) * COARSE_M
    grid = np.stack(np.meshgrid(steps, steps), -1).reshape(-1, 2)
    grid = grid[np.linalg.norm(grid, axis=1) <= SEARCH_M]
    grid = grid[np.argsort(np.linalg.norm(grid, axis=1), kind="stable")]
    k = int(COARSE_M / FINE_M)
    fine = np.arange(-k, k + 1) * FINE_M
    near = np.stack(np.meshgrid(fine, fine), -1).reshape(-1, 2)
    for i, (share, past) in enumerate(INSETS):
        least = None if margin is None else margin + past
        found = next((start + g for g in grid if passes(start + g, share, least)), None)
        if found is None:
            continue
        around = [found + f for f in near if passes(found + f, share, least)]
        return min(around, key=lambda d: float(np.linalg.norm(d - start))), i == 0
    return None, False
