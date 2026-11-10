"""Done criteria (docs/concepts/perception.md, docs/concepts/learning.md): what the scene
looks like when a task is done, as relations among registered objects measured by perception alone,
each with the value S1 measured at its successes and a tolerance.

A criterion is one metric of object role `a`, relative to role `b` where the metric is binary:
`distance_xy`, `offset_x`, `offset_y` (a's center minus b's, m, along the robot's x (right) and y
(back)), `gap_z` (a's bottom minus b's top, m; b `table`: a's bottom above the table), `overlap`
(share of a's footprint over b's), `inside` (1 when a's center is over b and its bottom below b's
top, else 0) and `yaw_diff` (deg between their yaws, folded by their symmetry; undecidable for a
round one); unary: `held` (1 when an arm holds a, else 0), `surface_change` (m: the mean
depth change of a's surface, or its part `part`'s, in the fixed cameras since a was last
measured; undecidable once an arm has carried a) and `press` (m: how far horizontally from a's
center, or its part `part`'s, an empty gripper was last stopped pushing on a, as the executor
reported it; undecidable until something pressed a; horizontal, since perception's box top may be
off by centimeters where a camera sees little of it). A press is an event the scene need not
show afterwards (a button springs back), so it is read from the actions, not the frames. The
metrics name no task: which ones matter, and their values, come from what S1 measured.
"""

from collections.abc import Mapping
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
)

from roboscript.perception.criteria import (
    FLOORS,
    METRICS,
    TABLE,
    DoneCheckConfig,
    DoneSpec,
    Metric,
    floored,
)
from roboscript.perception.registry import ObjectGeometry, ObjectRecord
from roboscript.perception.relations import overlap, relation
from roboscript.perception.table import Table
from roboscript.perception.views import View
from roboscript.spatial.camera import project
from roboscript.spatial.pose import pose_to_matrix
from roboscript_methods.wicl.perception.endstate import ANCHOR_OVERLAP, LIFTED_M

ANY_LIFT: Final = "*"
"""In a done fact's `latched`: the demonstration's lift latched its success, but no end target
says which role it lifted, so every lift in the air is a minimum."""

VISIBLE: Final = 0.01
"""A surface point shows at a pixel whose depth is not nearer than its own by more (m)."""
CLIP: Final = 0.05
"""A pixel's depth change counts at most this much (m): what moves in front of a surface is not
its change."""
MIN_PIXELS: Final = 30


class DoneValue(BaseModel):
    """A metric measured (None: undecidable, with the reason)."""

    model_config = ConfigDict(frozen=True)

    metric: Metric
    a: str
    b: str | None = None
    part: str | None = None
    measured: float | None
    reason: str = ""


def measure(
    metric: str, a: ObjectGeometry, b: ObjectGeometry | None, table: Table
) -> tuple[float | None, str]:
    """A metric other than `surface_change` of `a` (relative to `b`; None: the table for
    `gap_z`), with why it is undecidable."""
    if metric == "held":
        return float(bool(a.held_by)), ""
    if b is None:
        assert metric == "gap_z"  # DoneSpec allows the table for gap_z only
        return float(table.height(a.bottom_center)), ""
    if metric == "yaw_diff":
        if "continuous" in (a.symmetry, b.symmetry):
            return None, "a round object has no yaw"
        period = 90.0 if "half_pi" in (a.symmetry, b.symmetry) else 180.0
        return abs((a.yaw_deg - b.yaw_deg + period / 2) % period - period / 2), ""
    r = relation(a, b)
    values = {
        "distance_xy": r.distance_xy,
        "offset_x": r.displacement[0],
        "offset_y": r.displacement[1],
        "gap_z": r.gap_z,
        "overlap": r.overlap,
        "inside": float(r.inside),
    }
    return float(values[metric]), ""


def surface_change(
    points: NDArray[np.float64], before: Mapping[str, View], after: Mapping[str, View]
) -> float | None:
    """The mean depth change (m, each pixel's clipped at `CLIP`) at the pixels where `points`
    (world, as the initial frame `before` saw them) show, in the cameras that did not move
    between `before` and `after`, off the robot in both; None with fewer than `MIN_PIXELS`."""
    changes = []
    for camera, then in before.items():
        now = after.get(camera)
        if now is None or not np.allclose(then.image.cam2world, now.image.cam2world, atol=1e-6):
            continue
        uv, z, front = project(points, then.image.K, then.image.cam2world)
        ok = front & np.isfinite(uv).all(axis=1)
        pixel = np.rint(uv[ok]).astype(np.int64)
        h, w = then.depth.shape
        inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
        u, v, depth = pixel[inside, 0], pixel[inside, 1], z[ok][inside]
        seen = np.abs(then.depth[v, u] - depth) < VISIBLE
        u, v = u[seen], v[seen]
        unique = np.unique(np.stack([u, v], axis=1), axis=0)
        u, v = unique[:, 0], unique[:, 1]
        usable = then.usable[v, u] & now.usable[v, u]
        delta = np.abs(now.depth[v, u] - then.depth[v, u])[usable]
        changes.append(np.minimum(delta, CLIP))
    found = np.concatenate(changes) if changes else np.zeros(0)
    return float(np.mean(found)) if len(found) >= MIN_PIXELS else None


def surface_points(record: ObjectRecord, part: str | None) -> NDArray[np.float64]:
    """The world points of `record` (its part `part`: in the object's frame as measured) where
    the frame it was last measured in saw them."""
    if part is None:
        return np.asarray(record.points, dtype=np.float64)
    m = pose_to_matrix(record.pose.as_list())
    points = np.asarray(record.parts[part].points, dtype=np.float64)
    return points @ m[:3, :3].T + m[:3, 3]


POSITIONAL: Final = frozenset(METRICS) - {"held", "yaw_diff"}
"""The criteria that say where an object ends (or what happened to it: its surface, a press),
which `DoneCheckConfig.coverage` asks of every object an episode took."""
HORIZONTAL: Final = frozenset({"distance_xy", "offset_x", "offset_y"})


def capped(spec: DoneSpec, a: float | None, b: float | None, config: DoneCheckConfig) -> DoneSpec:
    """`spec` with its tolerance capped as `config.tol_cap` says, by the smaller footprint side
    (m) of its objects `a` and `b` (None: unknown; unchanged when it is off or `a`'s is), and
    never below perception's error (`floored`)."""
    spec = floored(spec)
    if not config.tol_cap:
        return spec
    if spec.metric in HORIZONTAL and a is not None:
        cap = config.cap_extent * float(min(a, a if b is None else b))
    elif spec.metric == "gap_z" and spec.b == TABLE:
        cap = config.cap_lift * abs(spec.value)
    else:
        return spec
    cap = max(cap, config.cap_floor_m, FLOORS.get(spec.metric, 0.0))
    return spec if spec.tol <= cap else spec.model_copy(update={"tol": cap})


def verdict(
    spec: DoneSpec,
    measured: float | None,
    base_excess: float | None = None,
    minimum: float | None = None,
) -> bool:
    """Whether `measured` is within `spec.tol` of its value; for a lift with `base_excess` (how
    far the bottom is above the demonstration's height over the robot's base, `lift_excess`),
    the smaller of the two excesses is. With `minimum` (a lift of a demonstration cut at its
    success latch, `DoneCheckConfig.latch_margin_m`), that excess must be at least `minimum`
    instead: one-sided, overshoot passes."""
    if measured is None:
        return False
    excess = measured - spec.value
    if base_excess is not None:
        excess = min(excess, base_excess)
    if minimum is not None:
        return excess >= minimum
    return abs(excess) <= spec.tol


def lifts(spec: DoneSpec) -> bool:
    """A `gap_z` over the table above `LIFTED_M`: a lift criterion."""
    return spec.metric == "gap_z" and spec.b == TABLE and spec.value > LIFTED_M


def in_the_air(a: str, geometry: Mapping[str, ObjectGeometry]) -> bool:
    """Whether role `a` stands over no other role (under `ANCHOR_OVERLAP` of its footprint over
    each, leaving out one the same arm holds): lifted, not stacked or set into something."""
    ga = geometry[a]
    hands = set(ga.held_by)
    return all(
        overlap(ga.points, gb) < ANCHOR_OVERLAP
        for b, gb in geometry.items()
        if b != a and not hands & set(gb.held_by)
    )
