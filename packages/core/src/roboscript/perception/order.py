"""Which of several look-alike objects goes first (docs/concepts/perception.md): the objects of
a group of roles of one kind, put in an order the scene measures, never a model's guess.

`order_roles` gives the roles, in rank order, the objects in a measured order: by size (volume,
largest first), by where they stand (`x`: the robot's left first; `y`: nearest first) or by the
fixed layout (`layout`: nearest first, then the robot's left), the order in which one
demonstration and every new scene bind look-alikes the same way. `by_colors` gives each role the
object whose color is nearest the one the instruction names for it (CIE Lab, one to one). `alike`
says when nothing the eye sees tells two objects apart: each extent within `OrderConfig.size_m`
or `size_frac` of the other's and their main colors within `color_de` in CIE Lab; then any words
that name them differently cannot be checked, and the layout decides. Positions are as the robot
sees them: x to its right, y ahead. Pure geometry and color, no model call.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from roboscript.perception.config import OrderConfig

By = Literal["size", "x", "y", "layout"]
SAMPLES = 2000
"""Pixels of an object compared with a color (a subsample of larger ones)."""


@dataclass(frozen=True)
class Seen:
    """One candidate object: its footprint center (world), its extents (long, short, height) and
    its pixels in one view (rgb, n x 3)."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    pixels: NDArray[np.uint8]


def order_roles(
    roles: Sequence[str], seen: Sequence[Seen], by: By, ranks: Sequence[int] | None = None
) -> dict[str, int]:
    """Each role -> the index (into `seen`) of its object: the roles in `ranks` order (their
    listed order without) take the objects in `by`'s order."""
    if len(roles) != len(seen):
        raise ValueError(f"{len(roles)} roles for {len(seen)} objects")
    if by == "size":
        key = [-float(np.prod(s.size)) for s in seen]
        measured = np.argsort(key, kind="stable")
    elif by == "layout":
        measured = np.lexsort(([s.center[0] for s in seen], [s.center[1] for s in seen]))
    else:
        axis = 0 if by == "x" else 1
        measured = np.argsort([s.center[axis] for s in seen], kind="stable")
    in_rank = list(roles) if ranks is None else [roles[i] for i in np.argsort(ranks, kind="stable")]
    return {role: int(i) for role, i in zip(in_rank, measured, strict=True)}


def by_colors(colors: Mapping[str, tuple[int, int, int]], seen: Sequence[Seen]) -> dict[str, int]:
    """Each role -> the index of the object whose main color is nearest the role's named color
    (sRGB), one to one."""
    roles = list(colors)
    cost = np.array([[distance(s.pixels, colors[r]) for s in seen] for r in roles])
    rows, cols = linear_sum_assignment(cost)
    return {roles[i]: int(j) for i, j in zip(rows, cols, strict=True)}


def alike(a: Seen, b: Seen, config: OrderConfig) -> bool:
    """Whether nothing the eye sees tells `a` from `b`: every extent (sorted footprint sides,
    then height) within `size_m` or `size_frac` of the other's, and main colors within
    `color_de` (CIE Lab)."""
    for x, y in zip(_extents(a), _extents(b), strict=True):
        if abs(x - y) > max(config.size_m, config.size_frac * max(x, y)):
            return False
    return bool(np.linalg.norm(_main(a.pixels) - _main(b.pixels)) <= config.color_de)


def distance(pixels: NDArray[np.uint8], color: tuple[int, int, int]) -> float:
    """How far an object's pixels are from a color: the 25th percentile of their Lab distances
    (its main color, not its shadows or labels)."""
    target = lab(np.array([color], dtype=float))[0]
    return float(np.percentile(np.linalg.norm(lab(_sample(pixels)) - target, axis=1), 25))


def lab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """sRGB (..., 3) in 0-255 as CIE Lab (D65)."""
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _extents(s: Seen) -> tuple[float, float, float]:
    long, short = sorted(s.size[:2], reverse=True)
    return float(long), float(short), float(s.size[2])


def _main(pixels: NDArray[np.uint8]) -> NDArray[np.float64]:
    """An object's main color in Lab: the per-channel median of its pixels."""
    return np.asarray(np.median(lab(_sample(pixels)), axis=0), dtype=float)


def _sample(pixels: NDArray[np.uint8]) -> NDArray[np.float64]:
    px = pixels.reshape(-1, 3).astype(float)
    return px[:: max(1, len(px) // SAMPLES)]
