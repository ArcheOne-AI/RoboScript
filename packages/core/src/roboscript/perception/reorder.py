"""The instruction's words checked against geometry (docs/concepts/perception.md): roles that
share a noun, each of whose words single its object out by one attribute (all by size, all by place
or all by color, each a different value), are re-ordered among the marks Set-of-Mark gave them by
what is measured (`perception.order`): volume (largest first), x (the robot's left first) or y
(nearest first) of their footprint centers, or the color nearest in CIE Lab to each named color
(a one-to-one assignment). A permutation of their own marks, never another mark: when
Set-of-Mark finds the right objects but binds them to the wrong words, this fixes it without a model
call. Size and place are measured, so the roles they order keep those marks (`identify` gives them
no vote); color words only re-order. Positions are as the robot sees them: x to its right, y ahead.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from roboscript.perception.order import Seen, by_colors, order_roles
from roboscript.perception.proposals import Proposal
from roboscript.perception.vlm import RoleWords

COLORS: Final = {
    "red": (200, 30, 30),
    "orange": (235, 130, 30),
    "yellow": (225, 210, 40),
    "green": (40, 160, 60),
    "cyan": (40, 190, 200),
    "blue": (40, 70, 200),
    "purple": (130, 50, 170),
    "pink": (235, 130, 180),
    "brown": (120, 75, 40),
    "beige": (215, 195, 160),
    "white": (238, 238, 238),
    "light gray": (185, 185, 185),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "dark gray": (70, 70, 70),
    "black": (25, 25, 25),
    "gold": (200, 160, 60),
    "golden": (200, 160, 60),
    "silver": (170, 175, 180),
}
"""Color words the instruction may use, as sRGB; any other word leaves the order alone."""
RANKS: Final = {
    "size": {"largest": 0, "middle": 1, "smallest": 2},
    "x": {"leftmost": 0, "middle": 1, "rightmost": 2},
    "y": {"nearest": 0, "middle": 1, "farthest": 2},
}


def reorder(
    words: Mapping[str, RoleWords],
    chosen: Mapping[str, int | None],
    proposals: Sequence[Proposal],
    rgb: NDArray[np.uint8],
    camera: str,
) -> tuple[dict[str, int | None], list[dict[str, Any]], set[str]]:
    """The marks after the check, the changes made (noun, by what, before, after), and the roles
    whose marks size or place words set (measured, whether they changed or not)."""
    by_id = {p.id: p for p in proposals}
    out = dict(chosen)
    changes: list[dict[str, Any]] = []
    measured: set[str] = set()
    groups: dict[str, list[str]] = {}
    for role, said in words.items():
        mark = chosen.get(role)
        if said.noun.strip() and mark is not None and mark in by_id:
            groups.setdefault(said.noun.strip().lower(), []).append(role)
    for noun, members in groups.items():
        marks = [by_id[m] for m in (chosen[r] for r in members) if m is not None]
        if len(members) < 2 or len({p.id for p in marks}) != len(members):
            continue
        said = [words[r] for r in members]
        new = _order(members, said, marks, rgb, camera)
        if new is None:
            continue
        if _distinct([w.size for w in said]) or _distinct([w.place for w in said]):
            measured.update(members)
        if any(new[r] != chosen[r] for r in members):
            out.update(new)
            before = {r: chosen[r] for r in members}
            changes.append({"noun": noun, "before": before, "after": new})
    return out, changes, measured


def _order(
    roles: list[str],
    said: list[RoleWords],
    marks: list[Proposal],
    rgb: NDArray[np.uint8],
    camera: str,
) -> dict[str, int] | None:
    """The roles' marks by the first attribute their words all give, each a different value
    (`order.order_roles`, `order.by_colors`)."""
    seen = [seen_of(p, rgb, camera) for p in marks]
    sizes = [w.size for w in said]
    if _distinct(sizes):
        ranks = [RANKS["size"][s] for s in sizes if s]
        found = order_roles(roles, seen, "size", ranks)
        return {role: marks[i].id for role, i in found.items()}
    places = [w.place for w in said]
    if _distinct(places):
        for axis, ranked in (("x", RANKS["x"]), ("y", RANKS["y"])):
            if all(p in ranked for p in places if p):
                ranks = [ranked[p] for p in places if p]
                found = order_roles(roles, seen, axis, ranks)  # type: ignore[arg-type]
                return {role: marks[i].id for role, i in found.items()}
        return None
    named = [None if w.color is None else COLORS.get(w.color.strip().lower()) for w in said]
    if all(c is not None for c in named) and len(set(named)) == len(named):
        colors = {r: c for r, c in zip(roles, named, strict=True) if c is not None}
        return {role: marks[i].id for role, i in by_colors(colors, seen).items()}
    return None


def seen_of(proposal: Proposal, rgb: NDArray[np.uint8], camera: str) -> Seen:
    """A proposal as `order` measures it: its footprint center, extents and pixels in `camera`."""
    summary = proposal.summary
    return Seen(
        center=tuple(float(v) for v in summary.center),  # type: ignore[arg-type]
        size=tuple(float(v) for v in summary.size),  # type: ignore[arg-type]
        pixels=rgb[proposal.masks[camera]],
    )


def _distinct(values: list[Any]) -> bool:
    return all(v is not None for v in values) and len(set(values)) == len(values)
