"""Geometric summaries and shape fits of an object's world points (meters, world frame, z up).

`summarize` is the one summary used for estimates and ground truth alike, so evaluations
compare the same quantities: the center is the footprint's minimum-area rectangle center, never the
points' centroid, which leans toward the camera. A standing round object whose far
half is hidden also gets `circle`, a circle fitted to one horizontal band of its visible arc.

`fit` adds a shape: `box` (upright, the footprint's minimum-area rectangle), `cylinder` (standing:
centered on `circle`, else on a circle through the footprint's outline; or lying along the
footprint's long side, centered on the footprint's rectangle, its axis's height and its radius
from a circle through the cross-section, which leans when fitted to the whole of a real object),
`plane` (RANSAC) or `line` (principal axis). `auto` takes a line for collinear points, a plane for
flat patches, a standing cylinder when a round band fits, a lying one when it explains the points
clearly better than the box, else the box. Every fit's frame and box are upright (z up). The
symmetry guess is from the shape alone, a geometric hint, not what the object is: a standing
cylinder is fully round only when its footprint is no longer than `ROUND` diameters; a longer
footprint has a handle (a vessel) or two (a pot), which its box keeps, turned along them, with a
two-fold symmetry for a pair on opposite sides and none for a single handle, which its frame's x
then points to.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict
from scipy import ndimage

from roboscript.contracts.geometry import OrientedBox, Pose, Vec3
from roboscript.spatial.cloud import oriented_box, ransac_plane
from roboscript.spatial.rotation import quat_from_axis_angle

TOP_BAND = 0.005
"""Points within this of the highest point form the top face."""
ON_TABLE = 0.015
"""A lowest point within this of the table: the object rests on it and its bottom is the table
(the underside is never visible)."""
FLAT = 0.003
"""`auto`: thinner point sets are planes (mats, cards)."""
SQUARE = 1.15
"""Footprints with a smaller aspect ratio are treated as square (four-fold symmetric)."""
CYLINDER_GAIN = 0.7
"""`auto` takes a lying cylinder when its residual is below this fraction of the box's."""
CELL = 0.0025
"""Footprint cells (`Summary.fill`, the outline of a standing cylinder)."""
BAND = 0.01
"""Height of the bands a round object's circle is fitted in."""
ROUND = 1.1
"""A standing cylinder is rotation-symmetric when its footprint's long side is at most this many
diameters."""
HANDLE_GAP = 0.005
"""A standing cylinder's points this far outside its circle are handles."""

Shape = Literal["box", "cylinder", "plane", "line"]
Symmetry = Literal["none", "pi", "half_pi", "continuous"]
"""Yaw symmetry: none, two-fold (180 deg), four-fold (90 deg), rotational."""


class Summary(BaseModel):
    """`center`: footprint (upright minimum-area rectangle) center, z halfway between bottom and
    top. `top_center`: centroid of the top band, at the top. `centroid`: the plain mean.
    `size`: long side, short side, height. `yaw_deg`: the long side's direction in [-90, 90).
    `fill`: share of the rectangle the footprint's cells cover. `circle`: xy center and radius
    of a round object (`round_center`), else None."""

    model_config = ConfigDict(frozen=True)

    n: int
    center: Vec3
    top_center: Vec3
    centroid: Vec3
    top_z: float
    bottom_z: float
    size: Vec3
    yaw_deg: float
    aspect: float
    fill: float
    circle: tuple[tuple[float, float], float] | None

    @property
    def shape_center(self) -> Vec3:
        """The circle's center for a round object, else the rectangle's."""
        if self.circle is None:
            return self.center
        (x, y), _ = self.circle
        return x, y, self.center[2]


class Fit(BaseModel):
    """A shape fitted to points. `upright`: the object stands (its frame's origin is the bottom
    center) rather than lies (origin at the center of its axis). `frame`: z up, x along
    `yaw_deg` (the footprint's long side; which way it points is not decided, except toward a
    single handle). `obb`: upright, turned by `yaw_deg`. `residual`: RMS distance of the points
    to the fitted surface."""

    model_config = ConfigDict(frozen=True)

    shape: Shape
    upright: bool
    n: int
    center: Vec3
    top_center: Vec3
    bottom_center: Vec3
    obb: OrientedBox
    axis: Vec3
    yaw_deg: float
    symmetry: Symmetry
    residual: float
    radius: float | None = None
    normal: Vec3 | None = None
    frame: Pose


def summarize(points: ArrayLike, table_z: float | None = None) -> Summary:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        raise ValueError("no points to summarize")
    z_hi, z_lo = float(pts[:, 2].max()), float(pts[:, 2].min())
    if table_z is not None and -0.005 < z_lo - table_z < ON_TABLE:
        z_lo = float(table_z)
    top = pts[pts[:, 2] >= z_hi - TOP_BAND]
    box = oriented_box(pts, "upright")
    long_side, short_side = float(box.size[0]), float(box.size[1])
    cells = len(np.unique(np.floor(pts[:, :2] / CELL).astype(np.int64), axis=0))
    return Summary(
        n=len(pts),
        center=(float(box.center[0]), float(box.center[1]), (z_lo + z_hi) / 2),
        top_center=(float(top[:, 0].mean()), float(top[:, 1].mean()), z_hi),
        centroid=vec3(pts.mean(axis=0)),
        top_z=z_hi,
        bottom_z=z_lo,
        size=(long_side, short_side, z_hi - z_lo),
        yaw_deg=float(np.degrees(np.arctan2(box.axes[0][1], box.axes[0][0]))),
        aspect=long_side / max(short_side, 1e-6),
        fill=cells * CELL**2 / max(long_side * short_side, 1e-9),
        circle=round_center(pts, long_side, short_side, box.center[:2]),
    )


def round_center(
    points: NDArray[np.float64],
    long_side: float,
    short_side: float,
    center: ArrayLike,
    min_points: int = 40,
    max_rms: float = 0.002,
) -> tuple[tuple[float, float], float] | None:
    """The xy center and radius of a round object from its visible arc: algebraic circle fits
    on `BAND`-high slices; of the tight ones (RMS < `max_rms`) whose diameter spans 70-120 % of
    the footprint's long side while the short side covers at least 90 % of the radius (round and
    seen at least half, not a box or a rod with a curved part), with the center inside the
    points' xy box and within half the short side of the rectangle `center`, the widest. A
    visible front arc suffices, so the hidden back half does not pull the center toward the
    camera. None when no slice qualifies."""
    lo, hi = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    z0 = points[:, 2].min()
    best: tuple[NDArray[np.float64], float] | None = None
    for k in range(int((points[:, 2].max() - z0) / BAND) + 1):
        band = points[(points[:, 2] >= z0 + k * BAND) & (points[:, 2] < z0 + (k + 1) * BAND), :2]
        if len(band) < min_points:
            continue
        c, r = _circle(band)
        rms = _rms(np.linalg.norm(band - c, axis=1) - r)
        spans = 0.35 * long_side <= r <= 0.6 * long_side and short_side >= 0.9 * r
        inside = bool(np.all(c >= lo) and np.all(c <= hi))
        near = float(np.linalg.norm(c - np.asarray(center))) <= 0.5 * short_side
        if rms < max_rms and spans and inside and near and (best is None or r > best[1]):
            best = (c, r)
    return None if best is None else ((float(best[0][0]), float(best[0][1])), best[1])


def fit(
    points: ArrayLike, shape: Shape | Literal["auto"] = "auto", table_z: float | None = None
) -> Fit:
    """Fit `shape` to world points (N, 3); `table_z` lets resting objects reach down to the
    table."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError(f"a fit needs at least 3 points, got {len(pts)}")
    if shape == "plane":
        return _plane(pts)
    if shape == "line":
        return _line(pts)
    s = summarize(pts, table_z)
    if shape == "box":
        return _box(pts, s)
    if shape == "cylinder":
        return _lying(pts, s) if _lies(s) else _standing(pts, s)
    variance = np.linalg.eigvalsh(np.cov(pts.T))
    if variance[1] < 0.01 * variance[2]:  # the second axis spreads < 10 % of the first
        return _line(pts)
    if s.size[2] < FLAT:
        return _plane(pts)
    if s.circle is not None:
        return _standing(pts, s)
    box = _box(pts, s)
    if _lies(s):
        lying = _lying(pts, s)
        return lying if lying.residual < CYLINDER_GAIN * box.residual else box
    return box


def _lies(s: Summary) -> bool:
    """Shaped like a cylinder lying along its footprint's long side."""
    return s.aspect >= 1.3 and s.size[2] <= 1.2 * s.size[1]


def _box(pts: NDArray[np.float64], s: Summary) -> Fit:
    center = np.array(s.center)
    yaw = np.radians(s.yaw_deg)
    local = _rotate_z(pts - center, -yaw)
    residual = _rms(_box_distance(local, np.array(s.size) / 2))
    long_side, _, height = s.size
    axis = (0.0, 0.0, 1.0) if height > long_side else (float(np.cos(yaw)), float(np.sin(yaw)), 0.0)
    return Fit(
        shape="box",
        upright=True,
        n=s.n,
        center=s.center,
        top_center=s.top_center,
        bottom_center=(s.center[0], s.center[1], s.bottom_z),
        obb=_obb(center, s.size, yaw),
        axis=axis,
        yaw_deg=s.yaw_deg,
        symmetry=_symmetry(s.aspect, residual, s.size[1]),
        residual=residual,
        frame=_frame((s.center[0], s.center[1], s.bottom_z), yaw),
    )


def _lying(pts: NDArray[np.float64], s: Summary) -> Fit:
    # The axis's height and the radius from a circle through the cross-section; its xy from the
    # footprint's rectangle. One circle through every cross-section of a real object (a
    # neck, handles, a flattened body) leans. A perfect half-seen cylinder's rectangle leans
    # toward the camera by a few millimeters instead.
    yaw = np.radians(s.yaw_deg)
    origin = np.array([s.center[0], s.center[1], 0.0])
    local = _rotate_z(pts - origin, -yaw)  # x along the axis
    (y0, z0), radius = _circle(local[:, 1:])
    lo, hi = float(local[:, 0].min()), float(local[:, 0].max())
    center = origin + _rotate_z(np.array([[(lo + hi) / 2, 0.0, z0]]), yaw)[0]
    rho = np.hypot(local[:, 1] - y0, local[:, 2] - z0)
    residual = _rms(_cylinder_distance(rho, local[:, 0] - (lo + hi) / 2, radius, (hi - lo) / 2))
    c = vec3(center)
    return Fit(
        shape="cylinder",
        upright=False,
        n=s.n,
        center=c,
        top_center=(c[0], c[1], c[2] + radius),
        bottom_center=(c[0], c[1], c[2] - radius),
        obb=_obb(center, (hi - lo, 2 * radius, 2 * radius), yaw),
        axis=(float(np.cos(yaw)), float(np.sin(yaw)), 0.0),
        yaw_deg=s.yaw_deg,
        symmetry="pi",
        residual=residual,
        radius=radius,
        frame=_frame(c, yaw),
    )


def _standing(pts: NDArray[np.float64], s: Summary) -> Fit:
    """Centered on the round band's circle, else on a circle through the footprint's outline. A
    footprint longer than `ROUND` diameters has handles: the points beyond the circle give their
    axis, which the frame's x and the box follow (x toward a single handle)."""
    (cx, cy), radius = s.circle if s.circle is not None else _circle(_outline(pts)[:, :2])
    height = s.size[2]
    center = np.array([cx, cy, (s.bottom_z + s.top_z) / 2])
    rho = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    residual = _rms(_cylinder_distance(rho, pts[:, 2] - center[2], radius, height / 2))
    yaw = np.radians(s.yaw_deg)
    symmetry: Symmetry = "continuous"
    obb = _obb(center, (2 * radius, 2 * radius, height), yaw)
    handles = pts[rho > radius + HANDLE_GAP, :2] - (cx, cy)
    if s.size[0] > ROUND * 2 * radius and len(handles) >= 10:
        axis = np.linalg.eigh(handles.T @ handles)[1][:, -1]
        along = handles @ axis
        symmetry = "pi" if min((along > 0).mean(), (along < 0).mean()) >= 0.25 else "none"
        if symmetry == "none" and along.sum() < 0:
            axis, along = -axis, -along  # x toward the handle
        # The middle of the angle the handles span, not their points' mean: a camera sees one
        # face of a handle more densely than the other.
        seen = handles[along > 0] if symmetry == "none" else handles * np.sign(along)[:, None]
        angle = np.arctan2(seen @ (-axis[1], axis[0]), seen @ axis)
        yaw = float(np.arctan2(axis[1], axis[0]) + (angle.min() + angle.max()) / 2)
        obb = _upright(pts, yaw)
    return Fit(
        shape="cylinder",
        upright=True,
        n=s.n,
        center=vec3(center),
        top_center=(float(cx), float(cy), s.top_z),
        bottom_center=(float(cx), float(cy), s.bottom_z),
        obb=obb,
        axis=(0.0, 0.0, 1.0),
        yaw_deg=float(np.degrees((yaw + np.pi / 2) % np.pi - np.pi / 2)),
        symmetry=symmetry,
        residual=residual,
        radius=radius,
        frame=_frame((float(cx), float(cy), s.bottom_z), yaw),
    )


def _plane(pts: NDArray[np.float64]) -> Fit:
    normal, offset, inliers = ransac_plane(pts, threshold=0.002)
    distance = pts @ normal + offset
    x = np.cross([0.0, 1.0, 0.0], normal) if abs(normal[2]) > 0.9 else np.cross(normal, [0, 0, 1])
    x /= np.linalg.norm(x)
    basis = np.stack([x, np.cross(normal, x), normal])  # rows: in-plane x, y, normal
    local = (pts[inliers] - pts[inliers].mean(axis=0)) @ basis.T
    flat = oriented_box(np.column_stack([local[:, :2], np.zeros(len(local))]), "upright")
    long_axis = basis.T @ _rz(float(np.arctan2(flat.axes[0][1], flat.axes[0][0])))[:, 0]
    yaw = float(np.arctan2(long_axis[1], long_axis[0]))
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    obb = _upright(pts[inliers], yaw)
    top = pts[pts[:, 2] >= pts[:, 2].max() - TOP_BAND]
    c = obb.center
    return Fit(
        shape="plane",
        upright=bool(abs(normal[2]) > 0.9),
        n=len(pts),
        center=c,
        top_center=vec3(top.mean(axis=0)),
        bottom_center=c,
        obb=obb,
        axis=vec3(normal),
        yaw_deg=float(np.degrees(yaw)),
        symmetry="half_pi" if flat.size[0] < SQUARE * flat.size[1] else "pi",
        residual=_rms(distance),
        normal=vec3(normal),
        frame=_frame(c, yaw),
    )


def _line(pts: NDArray[np.float64]) -> Fit:
    pca = oriented_box(pts, "pca")
    axis = pca.axes[0]
    offset = pts - pca.center
    residual = _rms(np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1))
    horizontal = np.hypot(axis[0], axis[1])
    yaw = float(np.arctan2(axis[1], axis[0])) if horizontal > 1e-9 else 0.0
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    obb = _upright(pts, yaw)
    return Fit(
        shape="line",
        upright=bool(abs(axis[2]) > horizontal),
        n=len(pts),
        center=obb.center,
        top_center=vec3(pts[np.argmax(pts[:, 2])]),
        bottom_center=vec3(pts[np.argmin(pts[:, 2])]),
        obb=obb,
        axis=vec3(axis),
        yaw_deg=float(np.degrees(yaw)),
        symmetry="pi",
        residual=residual,
        frame=_frame(obb.center, yaw),
    )


def _outline(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """The points in the boundary cells of the footprint (holes filled)."""
    ij = np.floor(pts[:, :2] / CELL).astype(np.int64)
    ij -= ij.min(axis=0) - 1  # one empty cell around the footprint
    grid = np.zeros(ij.max(axis=0) + 2, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    filled = ndimage.binary_fill_holes(grid)
    boundary = filled & ~ndimage.binary_erosion(filled, np.ones((3, 3), dtype=bool))
    return pts[boundary[ij[:, 0], ij[:, 1]]]


def _circle(xy: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Algebraic (Kasa) least-squares circle: center and radius."""
    a = np.column_stack([2 * xy, np.ones(len(xy))])
    solution = np.linalg.lstsq(a, (xy**2).sum(axis=1), rcond=None)[0]
    center = solution[:2]
    return center, float(np.sqrt(max(solution[2] + center @ center, 0.0)))  # 0: degenerate


def _box_distance(local: NDArray[np.float64], half: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed distance of box-frame points to the surface of the box with half extents `half`."""
    q = np.abs(local) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def _cylinder_distance(
    rho: NDArray[np.float64], along: NDArray[np.float64], radius: float, half: float
) -> NDArray[np.float64]:
    q = np.stack([rho - radius, np.abs(along) - half], axis=1)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def _symmetry(aspect: float, residual: float, short_side: float) -> Symmetry:
    if residual > max(0.003, 0.1 * short_side):
        return "none"  # the box explains the points poorly: an irregular object
    return "half_pi" if aspect < SQUARE else "pi"


def _upright(points: NDArray[np.float64], yaw: float) -> OrientedBox:
    """The upright box around points, turned by `yaw`."""
    local = _rotate_z(points, -yaw)
    lo, hi = local.min(axis=0), local.max(axis=0)
    return _obb(_rotate_z(((lo + hi) / 2)[None], yaw)[0], hi - lo, yaw)


def _obb(center: ArrayLike, size: ArrayLike, yaw: float) -> OrientedBox:
    q = quat_from_axis_angle([0.0, 0.0, 1.0], yaw)
    return OrientedBox(center=vec3(center), size=vec3(size), q=tuple(q.tolist()))


def _frame(origin: ArrayLike, yaw: float) -> Pose:
    return Pose(p=vec3(origin), q=tuple(quat_from_axis_angle([0.0, 0.0, 1.0], yaw).tolist()))


def _rz(yaw: float) -> NDArray[np.float64]:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotate_z(points: NDArray[np.float64], yaw: float) -> NDArray[np.float64]:
    return points @ _rz(yaw).T


def _rms(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def vec3(values: ArrayLike) -> Vec3:
    """A 3-vector as a tuple of floats (the wire type)."""
    x, y, z = np.asarray(values, dtype=float).tolist()
    return x, y, z
