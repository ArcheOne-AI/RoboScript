"""Which way an object's frame points (docs/concepts/perception.md): one convention for every
frame perception makes (a registered object, a part), so that the same object gets the same frame on
every seed and a spec fitted in it on one scene carries over to another.

The frame is z up with x along the footprint's long side (`perception.fit`); what the fit leaves
open is x's sign (and, for a near-square footprint, which side is long). `orient` decides it:

1. **Shape**: the object's height and width profiles along each candidate direction (the points
   in `BINS` slices across it: the highest point, the width across). A profile's lean is its
   centroid's offset from the middle, in half-lengths; the cue is the height's lean plus the
   width's, and x points toward the fuller, taller end (a heel, a lying vessel's base, a
   handle's side) when the cue is at least `FrameConfig.strength`. It decides x only from
   `FrameConfig.decisive` up: below that the same object's cue points x different ways on
   different seeds (the other turns stay candidates, which a Memory's `x toward:` words settle).
2. **Anchor**: otherwise x points away from the robot's base (`base`, world x, y) where that
   decides it (the direction at least `FrameConfig.ambiguous` along it), else toward world +y,
   else +x.

A part's x (`part_x`, where neither its candidate's shape nor `x_toward` decides it) points away
from the object's center, else by the anchor, and stays a candidate both ways; a moving part's
points away from its hinge (`toolsets.scene`). A single handle already decides a standing
cylinder's x (`fit`), a round one turns freely (its x
from the anchor). `Direction` records which rule decided and how strongly, and the turns (deg,
about z) that give equally good frames when the cue is weak: 180 for a long footprint, the
quarter turns for a near-square one (aspect under `FrameConfig.square`); consumers try them all
(candidate synthesis, the placement search) and keep the one a spec's reference orientation
prefers, or a VLM settles it from the Memory's `x toward:` words (`vlm.choose_axis`).
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.geometry import Pose
from roboscript.perception.config import DEFAULT_FRAMES, FrameConfig
from roboscript.perception.fit import Fit
from roboscript.spatial.rotation import quat_from_axis_angle, quat_to_matrix

Array = NDArray[np.float64]
Cue = Literal["shape", "handle", "anchor", "round", "continued", "chosen"]
"""What decided x: the shape's cue, a single handle, the anchor (weak), a round object's anchor,
the frame before (a re-measurement), a choice made later (the VLM's or the Actor's `orient`)."""
BINS = 8
MIN_BIN = 5
"""Points a slice needs to count in a profile."""


class Direction(BaseModel):
    """How a frame's x was decided: `cue`, its `strength` (the shape cue's size, 0 when it did
    not count), and the `turns` (deg, about z, 0 first) that give equally good frames."""

    model_config = ConfigDict(frozen=True)

    cue: Cue
    strength: float = 0.0
    turns: tuple[float, ...] = (0.0,)

    @property
    def weak(self) -> bool:
        return len(self.turns) > 1


def orient(
    fitted: Fit,
    points: ArrayLike,
    table_z: float | None = None,
    base: ArrayLike | None = None,
    config: FrameConfig = DEFAULT_FRAMES,
) -> tuple[Pose, Direction]:
    """The fit's frame turned about z by the convention (module docstring), and how."""
    pts = np.asarray(points, dtype=float)
    origin = np.asarray(fitted.frame.p, dtype=float)
    x0 = quat_to_matrix(fitted.frame.q)[:, 0]
    yaw0 = float(np.arctan2(x0[1], x0[0]))
    if fitted.symmetry == "continuous":
        yaw = _anchor_yaw(origin, [yaw0], base, config, free=True)
        return frame(origin, yaw), Direction(cue="round")
    if fitted.symmetry == "none" and fitted.shape == "cylinder":
        return fitted.frame, Direction(cue="handle", strength=1.0)
    size = np.asarray(fitted.obb.size, dtype=float)
    square = fitted.upright and size[0] < config.square * max(size[1], 1e-9)
    steps = (0.0, 90.0, 180.0, 270.0) if square else (0.0, 180.0)
    yaws = [yaw0 + np.radians(s) for s in steps]
    bottom = float(pts[:, 2].min()) if table_z is None else float(table_z)
    scores = [cue(pts, origin, y, bottom) for y in yaws]
    best = int(np.argmax(scores))
    if scores[best] >= config.decisive:
        return frame(origin, yaws[best]), Direction(cue="shape", strength=float(scores[best]))
    if scores[best] >= config.strength:  # points x, but the other turns stay candidates
        turns = tuple(sorted((s - steps[best]) % 360.0 for s in steps))
        return frame(origin, yaws[best]), Direction(
            cue="shape", strength=float(scores[best]), turns=turns
        )
    yaw = _anchor_yaw(origin, yaws, base, config)
    strength = float(max(scores[best], 0.0))
    return frame(origin, yaw), Direction(cue="anchor", strength=strength, turns=steps)


def cue(points: Array, origin: Array, yaw: float, bottom: float) -> float:
    """The shape cue along direction `yaw`: the height profile's lean plus the width's, in
    half-lengths (positive: fuller and taller toward +x)."""
    d = np.array([np.cos(yaw), np.sin(yaw)])
    rel = points[:, :2] - origin[:2]
    u, v = rel @ d, rel @ np.array([-d[1], d[0]])
    h = points[:, 2] - bottom
    lo, hi = np.percentile(u, [2, 98])
    if hi - lo < 1e-6:
        return 0.0
    t = np.clip((u - (lo + hi) / 2) / ((hi - lo) / 2), -1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, BINS + 1)
    which = np.clip(np.digitize(t, edges) - 1, 0, BINS - 1)
    mids, height, width = [], [], []
    for i in range(BINS):
        inside = which == i
        if inside.sum() < MIN_BIN:
            continue
        mids.append((edges[i] + edges[i + 1]) / 2)
        height.append(max(float(np.percentile(h[inside], 90)), 0.0))
        width.append(float(np.ptp(np.percentile(v[inside], [5, 95]))))
    if len(mids) < 3:
        return 0.0
    m = np.asarray(mids)
    return _lean(m, np.asarray(height)) + _lean(m, np.asarray(width))


def part_x(
    center: Array,
    x: Array,
    body: Array,
    base: ArrayLike | None = None,
    config: FrameConfig = DEFAULT_FRAMES,
) -> Array:
    """A part's x along `x` (horizontal) pointing away from the object's center `body` where that
    decides it, else by the anchor; either way stays a candidate (a part's points are too partial
    for the shape cue: its x is decided by a candidate's shape, `x_toward` or its hinge)."""
    yaw = float(np.arctan2(x[1], x[0]))
    out = np.asarray(center, dtype=float)[:2] - np.asarray(body, dtype=float)[:2]
    first = [out] if np.linalg.norm(out) > 1e-3 else []
    chosen = _anchor_yaw(
        np.asarray(center, dtype=float), [yaw, yaw + np.pi], base, config, first=first
    )
    return np.array([np.cos(chosen), np.sin(chosen), 0.0])


def reframe(fitted: Fit, pose: Pose) -> Fit:
    """The fit with its frame at `pose` (the same origin, turned about z) and its box turned
    along: x the frame's x (sides swapped for a quarter turn)."""
    x = quat_to_matrix(pose.q)[:, 0]
    bx = quat_to_matrix(fitted.obb.q)[:, 0]
    delta = float(np.arctan2(bx[0] * x[1] - bx[1] * x[0], bx @ x))
    size = fitted.obb.size
    if round(abs(delta) / (np.pi / 2)) % 2 == 1:
        size = (size[1], size[0], size[2])
    obb = fitted.obb.model_copy(update={"q": pose.q, "size": size})
    return fitted.model_copy(update={"frame": pose, "obb": obb})


def turned(pose: Pose, degrees: float) -> Pose:
    """`pose` turned about its own z by `degrees`."""
    x = quat_to_matrix(pose.q)[:, 0]
    return frame(pose.p, float(np.arctan2(x[1], x[0]) + np.radians(degrees)))


def frame(origin: ArrayLike, yaw: float) -> Pose:
    p = np.asarray(origin, dtype=float)
    q = quat_from_axis_angle([0.0, 0.0, 1.0], float(yaw))
    return Pose(p=(float(p[0]), float(p[1]), float(p[2])), q=tuple(float(c) for c in q))


def _lean(mids: Array, profile: Array) -> float:
    total = float(profile.sum())
    return 0.0 if total <= 1e-9 else float((mids * profile).sum() / total)


def _anchor_yaw(
    origin: Array,
    yaws: list[float],
    base: ArrayLike | None,
    config: FrameConfig,
    free: bool = False,
    first: list[Array] = [],  # noqa: B006 (read only)
) -> float:
    """The candidate yaw pointing most along `first` directions, then away from `base`, where
    that decides it, else most toward world +y, else +x (`free`: any yaw, exactly away)."""
    directions = [d / np.linalg.norm(d) for d in first]
    if base is not None:
        away = origin[:2] - np.asarray(base, dtype=float)[:2]
        if np.linalg.norm(away) > 1e-6:
            directions.append(away / np.linalg.norm(away))
    directions += [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    if free:
        d = directions[0]
        return float(np.arctan2(d[1], d[0]))
    for d in directions:
        along = [float(np.cos(y) * d[0] + np.sin(y) * d[1]) for y in yaws]
        best = int(np.argmax(along))
        second = max((a for i, a in enumerate(along) if i != best), default=-1.0)
        if along[best] - second >= 2 * config.ambiguous or d is directions[-1]:
            return yaws[best]
    return yaws[0]  # unreachable: the last direction always decides
