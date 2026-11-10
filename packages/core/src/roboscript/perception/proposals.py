"""Object proposals from the anchor camera: no identities, only geometry and color.

Height-map proposals: the usable points standing `min_height` above the table within its extent,
and (`beyond_table`) the points off the table within the table area, clear of the table plane
and above the floor (a big bin beside the table); labeled on the closed height map
(`PerceptionConfig.cell`, `close_cells`), fragments merged as in `lift`, specks under `min_cells`
cells dropped. Flat proposals (`color`), for mats a real depth sensor cannot resolve: patches of
the table plane (everything below the solid threshold, `color_plane`) that no height proposal
covers, nor holds in its footprint (a box's floor), and that are saturated, or, for
unsaturated mats and pads, whose color is `color_distance` away from a smooth model of the
table's own color (lighting) and that stand `rise` above the table on 5x5-median heights (a mat
does, a shadow does not), or, whatever their color, that stand `rise` above it; these two fill at
least `min_fill` of their rectangle (shadows and lighting rarely do) unless the image border cuts
them. On depth stored in whole millimetres a thin mat's heights spread, so a fixed band and color
alone miss some mats and add spurious ones. Each proposal also carries its mask
in the frame's other views, grown from the pixels near its points within its upright box widened by
`view_margin` (above the table; a flat one on it), so faces the anchor camera could not see (a box's
far side seen from a wrist) join; with `cut_grow_m`, a solid proposal the anchor image's border cuts
grows there without its top, so a view past the cut adds the rest of the object.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import cKDTree

from roboscript.perception.config import DEFAULTS, PerceptionConfig
from roboscript.perception.fit import Summary, summarize
from roboscript.perception.lift import Kind, merge_fragments, touches_border, weak_height
from roboscript.perception.table import Table
from roboscript.perception.views import FrameGeometry, View
from roboscript.spatial.cloud import height_components, oriented_box

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True, eq=False)
class Proposal:
    """A candidate object: `id` (1-based within the frame, in a total order that depends on the
    proposals alone: height proposals first, then by size (anchor pixels, most first), centroid
    (x, then y, to 0.1 mm) and anchor box), `kind`, the `camera` it was found in, its world
    points, their summary, and its mask in each view of the frame."""

    id: int
    kind: Kind
    camera: str
    points: NDArray[np.float64]
    summary: Summary
    masks: dict[str, NDArray[np.bool_]]


def propose(geometry: FrameGeometry, config: PerceptionConfig = DEFAULTS) -> list[Proposal]:
    view = geometry.view(geometry.anchor)
    table = geometry.table
    v, u = np.nonzero(view.usable)
    points, pixels = view.xyz[v, u], np.stack([u, v], axis=1)
    c = config.proposals
    inside = _within(points, table.extent)
    height = table.height(points)
    high = inside & (height > c.min_height)
    if c.beyond_table:
        around = ~inside & _within(points, config.table.area) & (np.abs(height) > c.min_height)
        around &= points[:, 2] > config.lift.floor_z + c.min_height
        if config.lift.wall_margin is not None:
            around &= points[:, 1] <= table.side("y1") + config.lift.wall_margin
        high |= around
    found: list[tuple[Kind, NDArray[np.float64], NDArray[np.int64]]] = []
    hp, hq = points[high], pixels[high]
    labels = height_components(
        hp, config.cell, -np.inf, min_cells=c.min_cells, close=config.close_cells
    )
    parts = [np.flatnonzero(labels == k) for k in range(labels.max(initial=-1) + 1)]
    for group in merge_fragments(hp, parts, view, table.z, config):
        found.append(("solid", hp[group], hq[group]))
    if c.color:
        plane = inside & (np.abs(height) < c.color_plane)
        taken = np.concatenate([q for _, _, q in found]) if found else np.zeros((0, 2), int)
        for sel in _flat_patches(view, pixels[plane], points[plane], taken, table, config):
            found.append(("flat", points[plane][sel], pixels[plane][sel]))
    proposals = []
    for i, (kind, pts, pix) in enumerate(sorted(found, key=_order), 1):
        summary = summarize(pts, table.z)
        masks = {view.name: np.zeros(view.image.shape, dtype=bool)}
        masks[view.name][pix[:, 1], pix[:, 0]] = True
        for name, other in geometry.views.items():
            if name != view.name:
                cut = touches_border(masks[view.name])
                masks[name] = view_mask(pts, summary, kind, other, table, config, cut)
        proposals.append(Proposal(i, kind, view.name, pts, summary, masks))
    return proposals


def _order(found: tuple[Kind, NDArray[np.float64], NDArray[np.int64]]) -> tuple[object, ...]:
    kind, points, pixels = found
    x, y = np.round(points[:, :2].mean(axis=0), 4)
    box = (*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist())
    return kind != "solid", -len(pixels), float(x), float(y), box


def _within(
    points: NDArray[np.float64], box: tuple[float, float, float, float]
) -> NDArray[np.bool_]:
    x0, x1, y0, y1 = box
    return (points[:, 0] > x0) & (points[:, 0] < x1) & (points[:, 1] > y0) & (points[:, 1] < y1)


def _flat_patches(
    view: View,
    plane: NDArray[np.int64],
    plane_points: NDArray[np.float64],
    taken: NDArray[np.int64],
    table: Table,
    config: PerceptionConfig,
) -> list[NDArray[np.int64]]:
    """Index arrays into `plane` (pixels on the table plane) of the flat proposals."""
    c = config.proposals
    shape = view.image.shape
    rgb = view.image.rgb.astype(np.float64)
    value = rgb.max(axis=2)
    saturated = (255 * (value - rgb.min(axis=2)) > c.min_saturation * np.maximum(value, 1)) & (
        value > c.min_value
    )
    rows, cols = plane[:, 1], plane[:, 0]
    occupied = np.zeros(shape, dtype=bool)
    occupied[taken[:, 1], taken[:, 0]] = True
    near = ndimage.binary_dilation(occupied, np.ones((c.exclusion_px, c.exclusion_px), bool))
    free = ~ndimage.binary_fill_holes(near)  # nor inside an object's footprint (a box's floor)
    index = np.full(shape, -1, dtype=np.int64)
    index[rows, cols] = np.arange(len(plane))
    passes: list[tuple[NDArray[np.bool_], float | None]] = [(saturated, None)]
    if len(plane):
        raised = None if c.rise is None else _heights(plane, plane_points, table, shape) > c.rise
        if c.color_distance is not None:
            distance = np.linalg.norm(rgb - table_color(rgb, plane), axis=2)
            differs = distance > c.color_distance
            passes.append((differs if raised is None else differs & raised, c.min_fill))
        if raised is not None:
            passes.append((raised, c.min_fill))
    patches: list[NDArray[np.int64]] = []
    claimed = np.zeros(shape, dtype=bool)
    for test, min_fill in passes:
        candidate = np.zeros(shape, dtype=bool)
        candidate[rows, cols] = test[rows, cols]
        candidate = ndimage.binary_opening(candidate & free & ~claimed, _EIGHT)
        labels, _ = ndimage.label(candidate, structure=_EIGHT)
        sizes = np.bincount(labels.ravel())
        for k in np.flatnonzero(sizes >= c.min_color_pixels):
            if not k:  # the background
                continue
            region = labels == k
            sel = index[region]
            sel = sel[sel >= 0]
            fill = min_fill is not None and not touches_border(region)  # a cut patch may fill less
            if fill and _filled_share(plane_points[sel], config.cell) < min_fill:
                continue  # not a rectangle: a shadow or lighting, not a mat
            patches.append(sel)
            claimed |= ndimage.binary_dilation(region, _EIGHT)
    return patches


def _heights(
    plane: NDArray[np.int64],
    plane_points: NDArray[np.float64],
    table: Table,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Each table-plane pixel's height above the table, the median of its 5 x 5 window (0 off
    the plane): a thin mat stands out of the quantized depth's scatter."""
    height = np.zeros(shape)
    height[plane[:, 1], plane[:, 0]] = table.height(plane_points)
    on = np.zeros(shape, dtype=bool)
    on[plane[:, 1], plane[:, 0]] = True
    return np.where(on, ndimage.median_filter(height, size=5), 0.0)


def table_color(rgb: NDArray[np.float64], plane: NDArray[np.int64]) -> NDArray[np.float64]:
    """A smooth model of the table's color over the image (H, W, 3): per channel a quadratic in
    the pixel coordinates, fitted to the table-plane pixels (u, v) and refitted without the ones
    that stray from it (mats, shadows) until it describes the lit table alone."""
    h, w = rgb.shape[:2]

    def basis(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = u / w - 0.5, v / h - 0.5
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    a = basis(plane[:, 0].astype(float), plane[:, 1].astype(float))
    colors = rgb[plane[:, 1], plane[:, 0]]
    keep = np.ones(len(plane), dtype=bool)
    for _ in range(4):
        coef = np.linalg.lstsq(a[keep], colors[keep], rcond=None)[0]
        residual = np.linalg.norm(colors - a @ coef, axis=1)
        spread = 1.4826 * float(np.median(residual[keep]))
        keep = residual < max(3 * spread, 4.0)
    v, u = np.mgrid[0:h, 0:w]
    return basis(u.astype(float), v.astype(float)) @ coef


def _filled_share(points: NDArray[np.float64], cell: float) -> float:
    """Share of its minimum-area rectangle that a footprint covers, holes (an object standing on
    the patch) filled."""
    ij = np.floor(points[:, :2] / cell).astype(np.int64)
    ij -= ij.min(axis=0)
    grid = np.zeros(ij.max(axis=0) + 1, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    covered = int(ndimage.binary_fill_holes(grid).sum()) * cell**2
    size = oriented_box(points, "upright").size
    return covered / max(float((size[0] + cell) * (size[1] + cell)), 1e-12)


def view_mask(
    points: NDArray[np.float64],
    summary: Summary,
    kind: Kind,
    view: View,
    table: Table,
    config: PerceptionConfig = DEFAULTS,
    cut: bool = False,
) -> NDArray[np.bool_]:
    """The pixels of `view` showing the object `points` (with their `summary`) came from; `cut`:
    the anchor image's border cut them (`ProposalConfig.cut_grow_m`)."""
    margin = config.proposals.view_margin
    beyond = config.proposals.cut_grow_m if cut and kind == "solid" else None
    yaw = np.radians(summary.yaw_deg)
    offset = view.xyz[..., :2] - np.asarray(summary.center[:2])
    along = offset[..., 0] * np.cos(yaw) + offset[..., 1] * np.sin(yaw)
    across = -offset[..., 0] * np.sin(yaw) + offset[..., 1] * np.cos(yaw)
    height = table.height(view.xyz)
    if kind == "solid":
        grow = margin if beyond is None else beyond
        level = height > weak_height(table, config.lift)
        if beyond is None:
            level &= view.xyz[..., 2] <= summary.top_z + margin
    else:  # a patch of the table plane: its own footprint only
        grow = 0.0
        level = np.abs(height) < config.proposals.color_plane
    inside = view.usable & level
    inside &= (np.abs(along) <= summary.size[0] / 2 + grow) & (
        np.abs(across) <= summary.size[1] / 2 + grow
    )
    v, u = np.nonzero(inside)
    if not len(u):
        return inside
    distance, _ = cKDTree(points).query(view.xyz[v, u], distance_upper_bound=margin)
    labels, _ = ndimage.label(inside, structure=_EIGHT)
    seeds = np.unique(labels[v[np.isfinite(distance)], u[np.isfinite(distance)]])
    return np.isin(labels, seeds[seeds > 0])
