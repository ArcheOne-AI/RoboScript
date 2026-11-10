"""Spatial relations between two objects, or an object and a point, as they are now.

`displacement` is a's center minus b's; `gap_z` is a's bottom minus b's top; `overlap` is the
share of a's footprint (its points seen from above) inside b's footprint (b's box seen from
above); `center_over`: a's center is over b's footprint. Directions are the robot's: left is -x,
in front is -y (nearer the robot).
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.geometry import Vec3
from roboscript.perception.fit import vec3
from roboscript.perception.registry import ObjectGeometry
from roboscript.spatial.rotation import quat_to_matrix

INSIDE_DEPTH = 0.005
"""An object is inside another when its bottom lies at least this far below the other's top."""


class Relation(BaseModel):
    model_config = ConfigDict(frozen=True)

    displacement: Vec3
    distance_xy: float
    gap_z: float
    overlap: float
    center_over: bool
    on: bool
    inside: bool
    above: bool
    left_of: bool
    in_front_of: bool


def relation(a: ObjectGeometry | Vec3, b: ObjectGeometry | Vec3, tol: float = 0.015) -> Relation:
    """How `a` sits relative to `b`; `tol` is the slack of every flag (meters)."""
    a_center, a_bottom, _, a_points = _extent(a)
    b_center, _, b_top, _ = _extent(b)
    d = a_center - b_center
    gap = float(a_bottom - b_top)
    share = overlap(a_points, b)
    center_over = overlap(a_center[None], b) == 1.0
    return Relation(
        displacement=vec3(d),
        distance_xy=float(np.hypot(d[0], d[1])),
        gap_z=gap,
        overlap=share,
        center_over=center_over,
        on=abs(gap) <= tol and share >= 0.5,
        inside=center_over and gap < -INSIDE_DEPTH,
        above=gap > -tol and share > 0.0,
        left_of=bool(d[0] < -tol),
        in_front_of=bool(d[1] < -tol),
    )


def overlap(points: NDArray[np.float64], b: ObjectGeometry | Vec3) -> float:
    """Share of `points` whose xy falls inside b's footprint (none for a point)."""
    if not isinstance(b, ObjectGeometry):
        return 0.0
    rotation = quat_to_matrix(b.obb.q)
    x = rotation[:2, 0] / max(float(np.linalg.norm(rotation[:2, 0])), 1e-9)
    local = (points[:, :2] - np.asarray(b.obb.center[:2])) @ np.stack([x, [-x[1], x[0]]]).T
    half = np.asarray(b.obb.size[:2]) / 2
    return float(np.mean(np.all(np.abs(local) <= half, axis=1))) if len(points) else 0.0


def _extent(
    x: ObjectGeometry | Vec3,
) -> tuple[NDArray[np.float64], float, float, NDArray[np.float64]]:
    """center, bottom z, top z, points."""
    if isinstance(x, ObjectGeometry):
        return x.center, float(x.bottom_center[2]), float(x.top_center[2]), x.points
    p = np.asarray(x, dtype=float)
    return p, float(p[2]), float(p[2]), p[None]
