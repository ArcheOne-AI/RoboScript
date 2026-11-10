"""Lifting 2D answers (masks, boxes, pixels) to an object's world points, across views.

Per view: the mask's usable pixels (with depth, not on the robot) minus the floor and the wall
behind the table, split by height with hysteresis: the object's core is what stands clearly above
the table (the proposals' `min_height`), and points above the table's own noise (`weak_height`:
`lift.support_height`, or `NOISE` times the table plane's RMS when that is larger, as on depth
stored in whole millimetres) join a core only within `close_cells` of it (a sole, a rim), so table
speckle inside a box never drags an object's footprint. The cores'
height-map components merge when the camera cannot see the table between them (a lying vessel's neck
hidden behind its body) or when the robot hides the gap between them (a held object split by the
fingers), and the largest is kept, so mask leakage onto the
table or onto an object behind falls away. A mask with (almost) no core is a flat patch (a 1-2 mm
mat): its points within `flat_band` of the table, the part above the table's noise when that is a
fifth of them (the mat, not the table around it in a box). Views merge anchored on the anchor
camera (else the view with most points): another view's points join when `view_share` of them
come within `view_gap` of the anchor's (a mask leaking onto a neighbor does not), and the merged
points are averaged per `voxel`, so a close wrist view, many times denser, does not outweigh the
head camera. An object whose mask touches the image border in every view it was lifted from is
`partially_visible`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict
from scipy import ndimage
from scipy.spatial import cKDTree

from roboscript.contracts.geometry import Vec3
from roboscript.perception.config import DEFAULTS, LiftConfig, PerceptionConfig
from roboscript.perception.fit import vec3
from roboscript.perception.table import Table
from roboscript.perception.views import FrameGeometry, View
from roboscript.spatial.camera import project
from roboscript.spatial.cloud import fit_plane, height_components, voxel_downsample

Kind = Literal["solid", "flat"]


BORDER = 2
"""A mask this close to the image border (pixels) is cut by it."""


@dataclass(frozen=True, eq=False)
class Lifted:
    """An object's world points (N, 3), the pixels kept in each view, whether it stands above
    the table (`solid`) or lies level with it (`flat`), and which views cut it at their border."""

    points: NDArray[np.float64]
    masks: dict[str, NDArray[np.bool_]]
    kind: Kind
    cut: dict[str, bool]

    @property
    def partially_visible(self) -> bool:
        return all(self.cut.values())


def touches_border(mask: NDArray[np.bool_], border: int = BORDER) -> bool:
    return bool(
        mask[:border].any()
        or mask[-border:].any()
        or mask[:, :border].any()
        or mask[:, -border:].any()
    )


class PixelMeasure(BaseModel):
    """The surface under one pixel: median point of its window, normal (toward the camera),
    camera depth and height above the table; None where the window has no usable depth."""

    model_config = ConfigDict(frozen=True)

    camera: str
    pixel: tuple[int, int]
    point: Vec3 | None
    normal: Vec3 | None
    depth: float | None
    height: float | None
    on_robot: bool


def lift(
    geometry: FrameGeometry,
    masks: Mapping[str, ArrayLike],
    config: PerceptionConfig = DEFAULTS,
) -> Lifted | None:
    """The object that `masks` (camera -> (H, W) bool) show, or None when no view keeps
    `lift.min_points` points."""
    parts: dict[str, tuple[NDArray[np.float64], NDArray[np.int64], Kind]] = {}
    for camera, mask in masks.items():
        found = lift_view(geometry.view(camera), mask, geometry.table, config)
        if found is not None:
            parts[camera] = found
    if not parts:
        return None
    anchor = config.anchor if config.anchor in parts else max(parts, key=lambda c: len(parts[c][0]))
    tree = cKDTree(parts[anchor][0])
    kept = {}
    for camera, (points, pixels, _) in parts.items():
        if camera != anchor:
            distance, _ = tree.query(points, distance_upper_bound=config.lift.view_gap)
            if np.isfinite(distance).mean() < config.lift.view_share:
                continue
        kept[camera] = (points, pixels)
    masks_out = {}
    for camera, (_, pixels) in kept.items():
        mask = np.zeros(geometry.view(camera).image.shape, dtype=bool)
        mask[pixels[:, 1], pixels[:, 0]] = True
        masks_out[camera] = mask
    points = np.concatenate([points for points, _ in kept.values()])
    return Lifted(
        points=voxel_downsample(points, config.lift.voxel) if len(kept) > 1 else points,
        masks=masks_out,
        kind=parts[anchor][2],
        cut={camera: touches_border(mask) for camera, mask in masks_out.items()},
    )


def lift_view(
    view: View, mask: ArrayLike, table: Table, config: PerceptionConfig = DEFAULTS
) -> tuple[NDArray[np.float64], NDArray[np.int64], Kind] | None:
    """One view's part of `lift`: (points, pixels (N, 2) as (u, v), kind) or None."""
    selected = np.asarray(mask, dtype=bool) & view.usable
    v, u = np.nonzero(selected)
    points, pixels = view.xyz[v, u], np.stack([u, v], axis=1)
    minimum = config.lift.min_points
    if len(points) < minimum:
        return None
    height = np.abs(table.height(points))
    around = ~on_background(points, table, config.lift)
    core = around & (height > config.proposals.min_height)
    above = around & (height > weak_height(table, config.lift))
    kind: Kind = "solid"
    if core.sum() < max(minimum, 0.1 * len(points)):
        kind = "flat"
        keep = around & (height < config.lift.flat_band)
        if (keep & above).sum() >= max(minimum, FLAT_SHARE * keep.sum()):
            keep &= above
        labels = _components(points[keep], config)
    else:
        keep = core | above
        labels = _grown(points, core, above & ~core, config)[keep]
    points, pixels = points[keep], pixels[keep]
    if len(points) < minimum or labels.max(initial=-1) < 0:
        return None
    parts = [np.flatnonzero(labels == k) for k in range(labels.max() + 1)]
    best = merge_fragments(points, [p for p in parts if len(p)], view, table.z, config)[0]
    return points[best], pixels[best], kind


NOISE = 4.0
"""Points closer to the table than this many times its plane's RMS are the table itself: depth
stored in whole millimetres scatters them; on noiseless depth `lift.support_height` rules."""
FLAT_SHARE = 0.2
"""A flat patch keeps its points above the table's noise when they are this share of its band."""


def weak_height(table: Table, config: LiftConfig = DEFAULTS.lift) -> float:
    """How far above the table a point must be not to be the table: `support_height`, or the
    table's own noise (`NOISE` times its plane's RMS) when that is larger."""
    return max(config.support_height, NOISE * table.rms)


def on_background(
    points: ArrayLike, table: Table, config: LiftConfig = DEFAULTS.lift
) -> NDArray[np.bool_]:
    """Points on the floor or on the wall behind the table."""
    p = np.asarray(points, dtype=float)
    below = p[:, 2] < config.floor_z
    if config.wall_margin is not None:
        below |= p[:, 1] > table.side("y1") + config.wall_margin
    return below


def _components(points: NDArray[np.float64], config: PerceptionConfig) -> NDArray[np.int64]:
    return height_components(points, config.cell, -np.inf, close=config.close_cells)


def _grown(
    points: NDArray[np.float64],
    core: NDArray[np.bool_],
    near: NDArray[np.bool_],
    config: PerceptionConfig,
) -> NDArray[np.int64]:
    """Height-map component labels (-1: none) of the `core` points, and of the `near` points
    within `close_cells` of a core cell (the label of the core there)."""
    labels = np.full(len(points), -1, dtype=np.int64)
    labels[core] = _components(points[core], config)
    if not near.any():
        return labels
    reach = config.close_cells
    ij = np.floor(points[:, :2] / config.cell).astype(np.int64)
    both = core | near
    origin = ij[both].min(axis=0) - reach
    grid = np.full(ij[both].max(axis=0) - origin + reach + 1, -1, dtype=np.int64)
    cells = ij[core] - origin
    grid[cells[:, 0], cells[:, 1]] = labels[core]
    # a near cell takes the largest label within reach; components rarely touch there
    grown = ndimage.grey_dilation(
        grid, size=(2 * reach + 1, 2 * reach + 1), mode="constant", cval=-1
    )
    cells = ij[near] - origin
    labels[near] = grown[cells[:, 0], cells[:, 1]]
    return labels


def merge_fragments(
    points: NDArray[np.float64],
    parts: Sequence[NDArray[np.int64]],
    view: View,
    table_z: float,
    config: PerceptionConfig = DEFAULTS,
) -> list[NDArray[np.int64]]:
    """Join fragments (index arrays into `points`) closer than `merge_gap` in xy whose table gap
    is hidden from the view, or closer than `occluded_gap` whose gap the robot hides; returns the
    merged index arrays, largest first."""
    parent = list(range(len(parts)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    flat = [cKDTree(points[p][:, :2]) for p in parts]
    solid = [cKDTree(points[p]) for p in parts] if view.robot.any() else []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if root(i) == root(j):
                continue
            a, b = points[parts[i]], points[parts[j]]
            distance, _ = flat[i].query(b[:, :2], distance_upper_bound=config.merge_gap)
            hidden = np.isfinite(distance).any() and not sees_table_between(a, b, view, table_z)
            covered = bool(solid) and robot_hides_gap(solid[i], a, b, view, config.occluded_gap)
            if hidden or covered:
                parent[root(j)] = root(i)
    groups: dict[int, list[NDArray[np.int64]]] = {}
    for i, p in enumerate(parts):
        groups.setdefault(root(i), []).append(p)
    return sorted((np.concatenate(g) for g in groups.values()), key=len, reverse=True)


def sees_table_between(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    view: View,
    table_z: float,
    margin: float = 0.005,
    pairs: int = 3,
    samples: int = 12,
) -> bool:
    """Is the table visible between point sets a and b (so they are separate objects)? Samples
    the table along the segments between their `pairs` closest xy point pairs; a sample is seen
    when the view's depth reaches it (within `margin`)."""
    distance, index = cKDTree(a[:, :2]).query(b[:, :2])
    closest = np.argsort(distance)[:pairs]
    start, end = a[index[closest], :2], b[closest, :2]
    t = np.linspace(0, 1, samples + 2)[1:-1, None, None]
    xy = (start + t * (end - start)).reshape(-1, 2)
    table = np.column_stack([xy, np.full(len(xy), table_z)])
    uv, z, front = project(table, view.image.K, view.image.cam2world)
    h, w = view.depth.shape
    ok = front & np.isfinite(uv).all(axis=1)
    u, v = np.rint(uv[ok]).astype(np.int64).T
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return bool(np.any(view.depth[v[inside], u[inside]] > z[ok][inside] - margin))


def robot_hides_gap(
    tree: cKDTree,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    view: View,
    reach: float,
    samples: int = 12,
    share: float = 0.8,
) -> bool:
    """Do the robot's pixels cover, in front of it, the segment between the closest points of
    `a` (whose 3D tree `tree` is) and `b`, when they are within `reach`? Then the gap may be the
    gripper, not free space."""
    distance, index = tree.query(b, distance_upper_bound=reach)
    if not np.isfinite(distance).any():
        return False
    j = int(np.argmin(distance))
    t = np.linspace(0, 1, samples + 2)[1:-1, None]
    segment = a[index[j]] + t * (b[j] - a[index[j]])
    uv, z, front = project(segment, view.image.K, view.image.cam2world)
    h, w = view.depth.shape
    ok = front & np.isfinite(uv).all(axis=1)
    u, v = np.rint(uv[ok]).astype(np.int64).T
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[ok][inside]
    hidden = view.robot[v, u] & (view.depth[v, u] < z)
    return int(hidden.sum()) >= share * samples


def box_mask(box: Sequence[float], shape: tuple[int, int]) -> NDArray[np.bool_]:
    """The pixels whose centers lie in a box (u0, v0, u1, v1) in continuous pixel coordinates
    (`roboscript.contracts.vision`: a mask's box (c0, r0, c1 + 1, r1 + 1) gives its columns c0
    to c1 and rows r0 to r1), clipped to the image."""
    u0, v0, u1, v1 = box
    if not (u0 < u1 and v0 < v1):
        raise ValueError(f"a box is (u0, v0, u1, v1) with u0 < u1 and v0 < v1, got {tuple(box)}")
    c0, r0, c1, r1 = (int(np.ceil(x - 0.5)) for x in box)  # first and past-the-last pixel
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, r0) : min(h, r1), max(0, c0) : min(w, c1)] = True
    return mask


def measure_pixel(
    geometry: FrameGeometry, camera: str, pixel: Sequence[float], window: int = 2
) -> PixelMeasure:
    """`lift` for the pixel a point (u, v) lies in (continuous pixel coordinates, as in
    `roboscript.contracts.vision`): the median point of its (2 * window + 1)^2 window and the
    normal of a plane through the surrounding 7 x 7 pixels, oriented toward the camera."""
    view = geometry.view(camera)
    h, w = view.image.shape
    u, v = (int(np.floor(x)) for x in pixel)
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError(f"pixel ({u}, {v}) lies outside the {w}x{h} image of {camera!r}")

    def patch(half: int) -> NDArray[np.float64]:
        rows, cols = slice(max(0, v - half), v + half + 1), slice(max(0, u - half), u + half + 1)
        return view.xyz[rows, cols][view.usable[rows, cols]]

    near = patch(window)
    empty = PixelMeasure(
        camera=camera,
        pixel=(u, v),
        point=None,
        normal=None,
        depth=None,
        height=None,
        on_robot=bool(view.robot[v, u]),
    )
    if not len(near):
        return empty
    point = np.median(near, axis=0)
    around = patch(max(window, 3))
    normal = None
    if len(around) >= 3:
        n, _ = fit_plane(around)
        eye = view.image.cam2world[:3, 3]
        normal = -n if n @ (eye - point) < 0 else n
    rows, cols = (
        slice(max(0, v - window), v + window + 1),
        slice(max(0, u - window), u + window + 1),
    )
    d = view.depth[rows, cols][view.usable[rows, cols]]
    return empty.model_copy(
        update={
            "point": vec3(point),
            "normal": None if normal is None else vec3(normal),
            "depth": float(np.median(d)),
            "height": float(geometry.table.height(point)),
        }
    )
