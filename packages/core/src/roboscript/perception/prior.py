"""Where a part lies on its object, relative to the object's box (docs/concepts/perception.md):
a `PartPrior` that `locate_part` reports for every part it registers and takes back to rank part
candidates and, when none lies where the prior says, to box the object's points there.

A prior is in fractions of the object's box along the box's axes (z up, x along the footprint's
long side): a hinged part's handle end at (0.4, -0.5, 0.1) of a boxy object, whatever its size
or where it stands. It holds no pose of one seed, so Memory keeps it as a parameter. Where the
object's x is not decided, the box turned half a turn about z is an equal reading, and both
turns are tried.
"""

from typing import Annotated, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import ConfigDict, Field, FiniteFloat

from roboscript.contracts.geometry import OrientedBox
from roboscript.contracts.tools import ToolModel
from roboscript.spatial.rotation import quat_to_matrix

SLACK: Final = 0.1
"""A candidate matches a prior when its center lies within the prior's box grown by this share
of the object's box on every side (a measured box moves by a centimeter or two)."""
SIZE_RATIO: Final = 2.0
"""A candidate matches a prior only when its box's extent and the prior's, each grown by `SLACK`,
differ by at most this factor along at least two axes and it is not larger by more along any: a
camera sees a part's surface, which may lack the part's depth, but not two of its extents (a
button's strip whose center lies on a lid is not the lid)."""
MIN_POINTS: Final = 3
"""A prior's box holding fewer of the object's points gives no part."""
_TURN: Final = np.array([-1.0, -1.0, 1.0])

_At = Annotated[FiniteFloat, Field(ge=-1.0, le=1.0)]
_Size = Annotated[FiniteFloat, Field(gt=0.0, le=2.0)]


class PartPrior(ToolModel):
    """A part's place on its object: the center of its points' box (`at`, from the object box's
    center; ±0.5 on the box's faces) and that box's extent (`size`), both in fractions of the
    object's box along its axes (z up, x along its long side)."""

    model_config = ConfigDict(frozen=True)

    at: tuple[_At, _At, _At]
    size: tuple[_Size, _Size, _Size]


def _local(box: OrientedBox, points: ArrayLike) -> NDArray[np.float64]:
    """World points (N, 3) in fractions of the box along its axes, from its center."""
    rotation = quat_to_matrix(box.q)
    offset = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(box.center)
    local: NDArray[np.float64] = offset @ rotation / np.asarray(box.size)
    return local


def part_prior(box: OrientedBox, points: ArrayLike) -> PartPrior:
    """The prior of a part whose points (world, at least one) lie on the object of `box`."""
    local = _local(box, points)
    lo, hi = local.min(axis=0), local.max(axis=0)
    at = np.clip((lo + hi) / 2, -1.0, 1.0)
    size = np.clip(hi - lo, 1e-3, 2.0)
    return PartPrior(
        at=tuple(round(float(v), 3) for v in at),  # type: ignore[arg-type]
        size=tuple(round(float(v), 3) for v in size),  # type: ignore[arg-type]
    )


def _turns(prior: PartPrior, both_ways: bool) -> list[NDArray[np.float64]]:
    at = np.asarray(prior.at)
    return [at, at * _TURN] if both_ways else [at]


def prior_distance(
    prior: PartPrior, box: OrientedBox, points: ArrayLike, both_ways: bool
) -> float | None:
    """How far a candidate (its points, world) lies from the prior: its center's largest offset
    along the box's axes from the prior's center, in halves of the prior's box grown by `SLACK`
    (at most 1 inside it), over the turns tried; None when it lies outside in every turn or its
    size does not agree with the prior's (`SIZE_RATIO`)."""
    local = _local(box, points)
    lo, hi = local.min(axis=0), local.max(axis=0)
    ratio = (hi - lo + SLACK) / (np.asarray(prior.size) + SLACK)
    if np.any(ratio > SIZE_RATIO) or np.sum(ratio < 1 / SIZE_RATIO) > 1:
        return None
    center = (lo + hi) / 2
    reach = np.asarray(prior.size) / 2 + SLACK
    found = min(float(np.max(np.abs(center - at) / reach)) for at in _turns(prior, both_ways))
    return found if found <= 1.0 else None


def prior_points(
    prior: PartPrior, box: OrientedBox, points: ArrayLike, both_ways: bool
) -> NDArray[np.float64]:
    """The object's points (world) inside the prior's box, in the turn that holds the most;
    none when fewer than `MIN_POINTS`."""
    world = np.asarray(points, dtype=float).reshape(-1, 3)
    local = _local(box, world)
    half = np.asarray(prior.size) / 2
    best = np.zeros(len(world), dtype=bool)
    for at in _turns(prior, both_ways):
        inside = np.all(np.abs(local - at) <= half, axis=1)
        if inside.sum() > best.sum():
            best = inside
    found: NDArray[np.float64] = world[best] if best.sum() >= MIN_POINTS else world[:0]
    return found
