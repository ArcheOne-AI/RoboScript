"""A joint estimated from the robot's own motion (interactive kinematic estimation): while a
gripper holds a door's edge, a lid or a drawer's handle and moves it in short steps, the TCP
positions it achieves lie on the joint's path, an arc about a revolute joint's axis or a line
along a prismatic one. `estimate` fits both to the start point and the achieved points and keeps
the initial guess (vision's hinge, a displacement's line) until a fit explains them decisively
better: a circle (its plane's normal re-estimated only once the points leave a line by more than
the noise floor, before that the guess's axis direction) or a line, the other kind than the
guess's only on evidence of that kind (a line once the guessed arc would have bowed visibly,
a circle once it bows itself).

Pure numpy; `ActionApi.rotate_about`'s turn follows the estimate step by step.
"""

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

NOISE_M: Final = 0.005
"""Achieved TCP positions scatter this much about the joint's path (a held edge sags along the
axis and slides in the fingers by a few mm per step, besides tracking): a
model is taken over the one in use only if its RMS residual is lower by more than this, and a
circle's plane is fitted only once the points leave their line by more."""
MIN_RADIUS_M: Final = 0.02
MAX_RADIUS_M: Final = 2.0
"""A fitted circle outside these radii is no hinge a gripper turns (a line's noise, a pivot
at the fingers)."""


@dataclass(frozen=True)
class Joint:
    """A joint's path: `revolute` about the line through `point` along `direction` (unit,
    right-handed: positive turns go the way the points went), or `prismatic` along `direction`
    (the way the points went) from `point`; `residual_m` the RMS distance of the points to it,
    `source` where it came from (`guess`, `circle`, `line`)."""

    kind: Literal["revolute", "prismatic"]
    point: Array
    direction: Array
    residual_m: float
    source: Literal["guess", "circle", "line"]

    def radius(self, p: ArrayLike) -> float:
        rel = np.asarray(p, dtype=float) - self.point
        return float(np.linalg.norm(rel - (rel @ self.direction) * self.direction))

    def travel(self, start: ArrayLike, p: ArrayLike) -> float:
        """How far `p` is along the joint from `start`: degrees about a revolute joint (signed,
        right-handed about `direction`), meters along a prismatic one."""
        a = np.asarray(start, dtype=float)
        b = np.asarray(p, dtype=float)
        if self.kind == "prismatic":
            return float((b - a) @ self.direction)
        u, v = (self._radial(x) for x in (a, b))
        return math.degrees(math.atan2(float(np.cross(u, v) @ self.direction), float(u @ v)))

    def _radial(self, p: Array) -> Array:
        rel = p - self.point
        return rel - (rel @ self.direction) * self.direction


def circle_residual(points: Array, point: Array, direction: Array) -> float:
    """RMS distance of `points` to the circle about the line (`point`, unit `direction`)
    through the first one."""
    rel = points - point
    along = rel @ direction
    radius = np.linalg.norm(rel - np.outer(along, direction), axis=1)
    return float(np.sqrt(np.mean((radius - radius[0]) ** 2 + (along - along[0]) ** 2)))


def fit_line(points: ArrayLike) -> tuple[Array, Array, float]:
    """The least-squares line through `points` (n >= 2): a point on it, its unit direction
    (first to last) and the RMS distance of the points to it."""
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center)
    direction = vt[0] if (pts[-1] - pts[0]) @ vt[0] >= 0 else -vt[0]
    rel = pts - center
    off = rel - np.outer(rel @ direction, direction)
    return center, direction, float(np.sqrt(np.mean(np.sum(off**2, axis=1))))


def fit_circle(points: ArrayLike, axis: ArrayLike) -> tuple[Array, Array, float] | None:
    """The least-squares circle through `points` (n >= 3): its center, its unit axis (oriented
    so that the points turn positively about it) and the RMS residual. The plane is the points'
    own once they leave their line by more than `NOISE_M`, else the one normal to `axis` (the
    guess); None when the points fit no circle of a hinge's radius."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    normal = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    if fit_line(pts)[2] > NOISE_M:
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0))
        normal = vt[2] if vt[2] @ normal >= 0 else -vt[2]
    e1 = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    origin = pts[0]
    xy = np.column_stack([(pts - origin) @ e1, (pts - origin) @ e2])
    # Kasa: x^2 + y^2 = 2 a x + 2 b y + c, linear in (a, b, c)
    design = np.column_stack([2 * xy, np.ones(len(xy))])
    solution, *_ = np.linalg.lstsq(design, np.sum(xy**2, axis=1), rcond=None)
    a, b, c = solution
    radius_sq = c + a * a + b * b
    if not np.isfinite(radius_sq) or radius_sq <= 0:
        return None
    radius = math.sqrt(radius_sq)
    if not MIN_RADIUS_M <= radius <= MAX_RADIUS_M:
        return None
    center = origin + a * e1 + b * e2 + ((pts - origin) @ normal).mean() * normal
    turn = np.cross(pts[0] - center, pts[-1] - center) @ normal
    direction = normal if turn >= 0 else -normal
    return center, direction, circle_residual(pts, center, direction)


def sagitta(radius: float, chord: float) -> float:
    """How far an arc of `radius` bows from its `chord`."""
    half = min(chord / 2, radius)
    return radius - math.sqrt(radius * radius - half * half)


def estimate(points: ArrayLike, guess: Joint, switch: bool = True) -> Joint:
    """The joint the start point and achieved TCP positions `points` (in order) lie on, from
    three points on: the guess with its residual, unless a fit of its kind (a circle about a
    revolute guess, a line along a prismatic one) has a residual lower by more than `NOISE_M`.
    A fitted circle counts only once it bows from the chord travelled by more than that (a short
    stretch of any path is nearly straight, and a circle fitting it says nothing), and a line
    about a revolute guess only once the guessed hinge's own arc would have bowed so (then the
    lower of the two); a circle about a prismatic guess must also beat the line. Without
    `switch` only fits of the guess's kind are considered."""
    pts = np.asarray(points, dtype=float)
    if guess.kind == "revolute":
        own = circle_residual(pts, guess.point, guess.direction)
    else:
        rel = pts - pts[0]
        off = rel - np.outer(rel @ guess.direction, guess.direction)
        own = float(np.sqrt(np.mean(np.sum(off**2, axis=1))))
    best = Joint(guess.kind, guess.point, guess.direction, own, "guess")
    if len(pts) < 3:
        return best
    line = Joint("prismatic", *fit_line(pts), "line")
    circle = fit_circle(pts, guess.direction)
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    arc = None
    if circle is not None:
        arc = Joint("revolute", *circle, "circle")
        if sagitta(arc.radius(pts[0]), chord) <= NOISE_M:
            arc = None
    if not switch:
        same = arc if guess.kind == "revolute" else line
        return same if same is not None and same.residual_m < own - NOISE_M else best
    if guess.kind == "revolute":
        fits = [arc] if arc is not None else []
        if sagitta(guess.radius(pts[0]), chord) > NOISE_M:
            fits.append(line)
        fits = [f for f in fits if f.residual_m < own - NOISE_M]
        return min(fits, key=lambda f: f.residual_m) if fits else best
    if arc is not None and arc.residual_m < min(own, line.residual_m) - NOISE_M:
        return arc
    return line if line.residual_m < own - NOISE_M else best
