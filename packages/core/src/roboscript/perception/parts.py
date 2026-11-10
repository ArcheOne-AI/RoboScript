"""Part candidates (docs/concepts/perception.md): where the parts of an object can be, from its
points alone, for `locate_part` to register one of and for grasp and place specs to aim at.
The points may be what cameras saw (its upper and
near side) or its whole surface (a mesh's samples, or its convex collision shapes sampled by
`sample_hulls`; hull vertices alone leave faces empty). Everything is measured in the object's frame
(z up; by default the footprint rectangle's, x along its long side), on a height map of `CELL`
cells: the highest point over each.

Kinds, each from one geometric rule that holds for any object:

- `end`: the outer `END` of a long object's length (aspect at least `LONG`) at each end; also a
  `neck` when its outer half is narrower and lower than the middle (`NARROW`: a lying vessel's neck,
  a rod's grip).
- `handle`: a group of points sticking out of the footprint's body (the footprint opened by a disc
  of a quarter of its short side, 1-3 cm), reaching at least `REACH` of the disc's radius past it
  (a corner the disc does not fit reaches 0.41 of it): a vessel's handle.
- `rocker` or `button` on a thin standing object (a plate: its short side under `THIN` and its
  height over twice that): what bulges `BULGE` past one of its faces (where most of the plate's
  length ends across it, `FACE`); a `rocker` when its surface tilts (`TILT`).
- `rim` and `opening`: a container, whose inside (the footprint `INSIDE` in from its edge) lies
  mostly well below its edge (`DEPTH`), or of which almost nothing shows (`SHOWN`: a camera
  looking in may not see the bottom; a closed top it would see): the opening is the footprint's
  cells that low (or unseen) reaching the inside (a box from its bottom, the tenth percentile of
  its heights, up to the rim), the rim the top band around it.
- `neck` on a standing object: the part above the height where its footprint area halves, when it
  stays under `NECK_AREA` of the body's area and is taller than `NECK_RATIO` times the width of
  its upper half.
- `lid`: the flat top (within `LID_BAND` of the most common top height over the body) when it
  covers `LID_AREA` of the body high on the object: a pot's lid, a box's top.
- `button` or `rocker` on top: each group of cells within `fit.TOP_BAND` of the top covering less
  than `BUTTON_AREA` of the footprint (a knob, a plunger); a `rocker` when its surface tilts.

A candidate lying mostly within another is one with it, with both kinds (a thin grip is a handle and
an end). A hinged part (a hinged lid) comes from the hinge estimate, not from here.

A candidate's frame: z up, x horizontal from the object's body toward the part when that decides
it (`decided`: handles, ends, bulges), else along the part's own long side (either way); the
origin at the center of the part's box in that frame, `extent` its size. `pivot` moves the origin
to the part's root end along a decided x (else its center) and to its bottom: `at(fraction)` is
the point `fraction` of the extent from the pivot, so a spec written in fractions carries over to
a larger or smaller instance (a handle's grip bar at (0.6, 0, 0.5); an opening's bottom center at
(0, 0, 0)). Where on a part a demonstration touched is a convention of the task that words do not
carry. The VLM may name a part's candidate differently from seed to seed, which keeping the part's
kind narrows.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from roboscript.perception.fit import TOP_BAND, summarize
from roboscript.perception.views import View
from roboscript.spatial.camera import project
from roboscript.spatial.cloud import oriented_box
from roboscript.spatial.pose import pose_to_matrix

PartKind = Literal["handle", "rim", "rocker", "button", "lid", "neck", "opening", "end"]
CELL: Final = 0.0025
"""The height map's cell (m)."""
LONG: Final = 1.8
"""An object with at least this aspect has two ends."""
END: Final = 0.2
"""An end: this share of the object's length at each end."""
NARROW: Final = 0.7
"""An end narrower and lower than this share of the middle's width and height is a neck."""
FAR_END_MIN_M: Final = 0.005
"""The far end decides a front only where the footprint's center lies this far off its box center
at least (else the object is as long either way and its two ends are noise apart)."""
FAR_END_BAND: Final = 0.9
"""The far end: the footprint cells at least this fraction of the farthest one's distance out."""
OPENING: Final = (0.25, 0.01, 0.03)
"""The disc that opens a footprint into its body: this share of the short side, within these
bounds (m)."""
REACH: Final = 0.7
"""A handle reaches at least this share of the disc's radius past the body."""
THIN: Final = 0.035
"""A standing object thinner than this (m) and over twice as tall is a plate."""
BULGE: Final = 0.003
"""A plate's bulge stands this far (m) past its face."""
FACE: Final = 20
"""A plate's face: the percentile of its extent across (cell by cell along it) past which this
share of the cells stand (a bulge may cover up to the rest)."""
TILT: Final = 10.0
"""A bulge or a top part whose surface tilts at least this much (deg) is a rocker."""
INSIDE: Final = (0.2, 0.01)
"""A container's inside: the footprint this share of its short side (at least this, m) in from
its edge."""
DEPTH: Final = (0.25, 0.01)
"""The inside lies this share of the height (at least this, m) below the edge's top."""
SHOWN: Final = 0.1
"""An inside of which less than this share shows is taken for hollow."""
RIM_BAND: Final = 0.01
"""A rim: the edge's points within this (m) of its top."""
NECK_AREA: Final = 0.3
NECK_RATIO: Final = 1.2
LID_BAND: Final = 0.006
LID_AREA: Final = 0.4
BUTTON_AREA: Final = 0.4
MIN_POINTS: Final = 20
"""A candidate with fewer points is dropped."""
SHARE: Final = 0.5
"""Two candidates are one when more than this share of the smaller one's points is in both."""
VISIBLE: Final = 0.01
"""A point shows in a view when the depth under it is within this of its own (m)."""
_EIGHT: Final = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True, eq=False)
class Part:
    """A part candidate: its kinds, its frame (4x4, origin at the center of its box, z up), the
    box's extent along the frame's axes, whether x points away from the body (`decided`), and its
    points; frame and points in the coordinates `candidates` reports in."""

    kinds: tuple[PartKind, ...]
    frame: NDArray[np.float64]
    extent: NDArray[np.float64]
    decided: bool
    points: NDArray[np.float64]

    @property
    def pivot(self) -> NDArray[np.float64]:
        """The frame moved to the part's root end along a decided x (else its center) and to its
        bottom."""
        shift = np.array([-self.extent[0] / 2 if self.decided else 0.0, 0.0, -self.extent[2] / 2])
        out = self.frame.copy()
        out[:3, 3] = self.frame[:3, 3] + self.frame[:3, :3] @ shift
        return out

    def at(self, fraction: ArrayLike) -> NDArray[np.float64]:
        """The point `fraction` (x, y, z) of the extent from the pivot, in the part's axes."""
        pivot = self.pivot
        return pivot[:3, 3] + pivot[:3, :3] @ (np.asarray(fraction, dtype=float) * self.extent)


def candidates(
    points: ArrayLike, frame: ArrayLike | None = None, table_z: float | None = None
) -> list[Part]:
    """An object's part candidates from its points (world, (N, 3)): in `frame`'s coordinates (the
    object's frame, a 4x4 or 7-vector world pose, z up) when given, else in the world's (measured
    in the footprint rectangle's frame). `table_z` (world) is where a seen object stands (default:
    its lowest point)."""
    world = np.asarray(points, dtype=float)
    measured = footprint_frame(world, table_z) if frame is None else _matrix(frame)
    local = (world - measured[:3, 3]) @ measured[:3, :3]
    base = float(local[:, 2].min()) if table_z is None else table_z - float(measured[2, 3])
    found = _merged(_Shape(local, base).candidates())
    given = frame is not None
    return [
        Part(
            kinds=kinds,
            frame=box if given else measured @ box,
            extent=extent,
            decided=decided,
            points=local[sel] if given else world[sel],
        )
        for kinds, sel, box, extent, decided in found
    ]


def footprint_frame(points: ArrayLike, table_z: float | None = None) -> NDArray[np.float64]:
    """The frame of points' footprint rectangle (4x4 world): x along its long side, z up, the
    origin at its center at the points' bottom (`table_z` when given)."""
    summary = summarize(points, table_z)
    out = np.eye(4)
    out[:3, :3] = _yaw(np.radians(summary.yaw_deg))
    out[:3, 3] = (summary.center[0], summary.center[1], summary.bottom_z)
    return out


def sample_hulls(hulls: Sequence[ArrayLike], spacing: float = 0.002) -> NDArray[np.float64]:
    """Points over the surfaces of convex shapes, each given by its vertices (world, (M, 3): one
    collision shape's), on a grid over each face at most `spacing` apart (under `CELL`, so every
    cell a face covers gets a point of it), the vertices included."""
    out = [np.zeros((0, 3))]
    for vertices in hulls:
        v = np.asarray(vertices, dtype=float).reshape(-1, 3)
        out.append(v)
        if len(v) < 4:
            continue
        try:
            faces = v[ConvexHull(v, qhull_options="QJ").simplices]
        except QhullError:  # all on one line
            continue
        for a, b, c in faces:
            n = max(1, int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a)) / spacing)))
            i, j = np.nonzero(np.add.outer(np.arange(n + 1), np.arange(n + 1)) <= n)
            out.append(a + np.outer(i / n, b - a) + np.outer(j / n, c - a))
    return np.concatenate(out)


def candidate_mask(points: ArrayLike, view: View) -> NDArray[np.bool_]:
    """The pixels of a view that show points (world, (N, 3)) not hidden behind something, gaps
    closed."""
    image = view.image
    h, w = image.shape
    uv, z, front = project(np.asarray(points, dtype=float), image.K, image.cam2world)
    ok = front & np.isfinite(uv).all(axis=1)
    pixel = np.floor(uv[ok]).astype(np.int64)
    inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
    u, v, depth = pixel[inside, 0], pixel[inside, 1], z[ok][inside]
    seen = np.abs(view.depth[v, u] - depth) < VISIBLE
    mask = np.zeros((h, w), dtype=bool)
    mask[v[seen], u[seen]] = True
    grown = ndimage.binary_dilation(mask, iterations=1)
    return ndimage.binary_closing(grown, iterations=2) | mask


Found = tuple[
    tuple[PartKind, ...], NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64], bool
]
"""A candidate as found: kinds, its points (a selection), box frame (4x4, object frame), extent,
decided."""


class _Shape:
    """An object's points in its frame (`p`), their heights above `base`, its height map (the
    highest height over each cell, -inf where none) and its footprint (the cells, gaps closed)."""

    def __init__(self, local: NDArray[np.float64], base: float) -> None:
        self.p = local
        self.h = local[:, 2] - base
        self.base = base
        ij = np.floor(local[:, :2] / CELL).astype(np.int64)
        self.origin = ij.min(axis=0) - 2
        self.ij = ij - self.origin
        self.height = np.full(tuple(self.ij.max(axis=0) + 3), -np.inf)
        np.maximum.at(self.height, (self.ij[:, 0], self.ij[:, 1]), self.h)
        self.seen = np.isfinite(self.height)
        closed = ndimage.binary_closing(self.seen, _EIGHT, 2) | self.seen
        self.footprint = ndimage.binary_fill_holes(closed)
        self.summary = summarize(np.column_stack([local[:, :2], self.h]), 0.0)
        self.top = float(self.h.max())
        self.radius = 0

    def at_cells(self, cells: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """The points over cells."""
        return cells[self.ij[:, 0], self.ij[:, 1]]

    def candidates(self) -> list[Found]:
        _, short_side, tall = self.summary.size
        found = self._ends()
        if short_side < THIN and tall > 2 * short_side:
            return found + self._bulges()
        body = self._body(short_side)
        found += self._handles(body)
        inside = self._inside(short_side)
        if inside is not None and self._hollow(inside):
            return found + self._rim_and_opening(inside)
        necks = self._necks()
        return found + necks + ([] if necks else self._lid(body)) + self._tops(necks)

    def _ends(self) -> list[Found]:
        s = self.summary
        if s.aspect < LONG:
            return []
        axis = _yaw(np.radians(s.yaw_deg))[:, 0]
        offset = self.p[:, :2] - np.asarray(s.center[:2])
        along = offset @ axis[:2]
        across = offset @ np.array([-axis[1], axis[0]])
        middle = np.abs(along) < 0.2 * s.size[0]
        out = []
        for sign in (-1.0, 1.0):
            sel = sign * along > (0.5 - END) * s.size[0]
            if sel.sum() < MIN_POINTS:
                continue
            kinds: tuple[PartKind, ...] = ("end",)
            tip = sign * along > (0.5 - END / 2) * s.size[0]  # its outer half: past a shoulder
            if middle.sum() >= MIN_POINTS and tip.sum() >= MIN_POINTS:
                narrow = _spread(across[tip]) < NARROW * _spread(across[middle])
                low = np.percentile(self.h[tip], 95) < NARROW * np.percentile(self.h[middle], 95)
                if narrow and low:
                    kinds = ("end", "neck")
            out.append(self._framed(kinds, sel, sign * axis))
        return out

    def _body(self, short_side: float) -> NDArray[np.bool_]:
        share, low, high = OPENING
        self.radius = r = int(np.rint(np.clip(share * short_side, low, high) / CELL))
        disc = np.hypot(*np.mgrid[-r : r + 1, -r : r + 1]) <= r
        return ndimage.binary_opening(self.footprint, disc)

    def _handles(self, body: NDArray[np.bool_]) -> list[Found]:
        labels, count = ndimage.label(self.footprint & ~body, structure=_EIGHT)
        if not count or not body.any():
            return []
        reach = ndimage.distance_transform_edt(~body)
        far = ndimage.maximum(reach, labels, index=np.arange(1, count + 1))
        of_point = labels[self.ij[:, 0], self.ij[:, 1]]
        center = (np.argwhere(body).mean(axis=0) + self.origin + 0.5) * CELL
        out = []
        for k in range(1, count + 1):
            sel = of_point == k
            if far[k - 1] < REACH * self.radius or sel.sum() < MIN_POINTS:
                continue
            direction = self.p[sel, :2].mean(axis=0) - center
            out.append(self._framed(("handle",), sel, np.array([*direction, 0.0])))
        return out

    def _bulges(self) -> list[Found]:
        """What stands past a plate's faces (the median extent across it, cell by cell along)."""
        s = self.summary
        axis = _yaw(np.radians(s.yaw_deg))[:, 0]
        normal = np.array([-axis[1], axis[0], 0.0])
        offset = self.p[:, :2] - np.asarray(s.center[:2])
        across = offset @ normal[:2]
        bins = np.floor(offset @ axis[:2] / CELL).astype(np.int64)
        bins -= bins.min()
        low = np.full(bins.max() + 1, np.inf)
        high = np.full(bins.max() + 1, -np.inf)
        np.minimum.at(low, bins, across)
        np.maximum.at(high, bins, across)
        used = np.isfinite(low)
        faces = (np.percentile(low[used], 100 - FACE), np.percentile(high[used], FACE))
        out = []
        for sign, face in zip((-1.0, 1.0), faces, strict=True):
            past = sign * (across - face) > BULGE
            runs, count = ndimage.label(np.isin(np.arange(len(low)), bins[past]))
            for k in range(1, count + 1):
                sel = past & (runs[bins] == k)
                if sel.sum() < MIN_POINTS:
                    continue
                tilted = _tilted(self.h[sel], sign * (across[sel] - face))
                kind: PartKind = "rocker" if tilted else "button"
                out.append(self._framed((kind,), sel, sign * normal))
        return out

    def _inside(self, short_side: float) -> NDArray[np.bool_] | None:
        share, least = INSIDE
        distance = ndimage.distance_transform_edt(self.footprint) * CELL
        inside = distance >= max(least, share * short_side)
        return inside if inside.sum() >= 0.1 * self.footprint.sum() else None

    def _edge_top(self, inside: NDArray[np.bool_]) -> float:
        edge = self.footprint & ~inside & self.seen
        return float(np.percentile(self.height[edge], 90)) if edge.any() else self.top

    def _hollow(self, inside: NDArray[np.bool_]) -> bool:
        """Whether most of the inside that shows lies well below the edge's top, or almost none
        of it shows (a container, whose bottom a camera looking in may not see; a closed top it
        would)."""
        known = inside & self.seen
        if known.sum() < SHOWN * inside.sum():
            return True
        share, least = DEPTH
        deep = self.height[known] < self._edge_top(inside) - max(least, share * self.top)
        return bool(deep.mean() >= 0.6)

    def _rim_and_opening(self, inside: NDArray[np.bool_]) -> list[Found]:
        """The rim: the top band of the footprint around the opening; the opening: the
        footprint's cells below the rim's depth (or unseen) that reach the inside."""
        top = self._edge_top(inside)
        share, least = DEPTH
        below = self.footprint & (self.height < top - max(least, share * self.top))
        labels, _ = ndimage.label(below, structure=_EIGHT)
        opening = np.isin(labels, np.unique(labels[inside & below])) & (labels > 0)
        rim = self.at_cells(self.footprint & ~opening) & (self.h >= top - RIM_BAND)
        out = [self._framed(("rim",), rim, None)] if rim.sum() >= MIN_POINTS else []
        axes = _yaw(np.radians(self.summary.yaw_deg))
        corners = (np.argwhere(opening) + self.origin + 0.5) * CELL @ axes[:2, :2]
        low, high = corners.min(axis=0) - CELL / 2, corners.max(axis=0) + CELL / 2
        known = opening & self.seen
        bottom = float(np.percentile(self.height[known], 10)) if known.any() else 0.0
        box = np.eye(4)
        box[:3, :3] = axes
        box[:2, 3] = axes[:2, :2] @ ((low + high) / 2)
        box[2, 3] = self.base + (bottom + top) / 2
        extent = np.array([*(high - low), max(top - bottom, CELL)])
        out.append((("opening",), self.at_cells(opening), box, extent, False))
        return out

    def _necks(self) -> list[Found]:
        heights = self.height[self.footprint & self.seen]
        body_area = (heights >= 0.3 * self.top).sum()
        levels = np.linspace(0.3 * self.top, self.top, 60)
        wide = [t for t in levels if (heights >= t).sum() >= 0.5 * body_area]
        if not wide:
            return []
        shoulder = max(wide)
        length = self.top - shoulder
        if length < 0.2 * self.top:
            return []
        if (heights >= shoulder + 0.25 * length).sum() > NECK_AREA * body_area:
            return []
        sel = self.h >= shoulder
        upper = self.h >= shoulder + length / 2  # the neck itself, past a sloping shoulder
        if sel.sum() < MIN_POINTS or upper.sum() < MIN_POINTS:
            return []
        if length < NECK_RATIO * max(_spread(self.p[upper, 0]), _spread(self.p[upper, 1])):
            return []
        return [self._framed(("neck",), sel, None)]

    def _lid(self, body: NDArray[np.bool_]) -> list[Found]:
        known = body & self.seen
        if known.sum() < 10:
            return []
        level = float(np.percentile(self.height[known], 60))
        flat = known & (np.abs(self.height - level) <= LID_BAND)
        if flat.sum() < LID_AREA * known.sum() or level < 0.6 * self.top:
            return []
        sel = self.at_cells(flat) & (np.abs(self.h - level) <= LID_BAND)
        return [self._framed(("lid",), sel, None)] if sel.sum() >= MIN_POINTS else []

    def _tops(self, necks: list[Found]) -> list[Found]:
        labels, count = ndimage.label(self.height >= self.top - TOP_BAND, structure=_EIGHT)
        of_point = labels[self.ij[:, 0], self.ij[:, 1]]
        area = self.footprint.sum()
        out = []
        for k in range(1, count + 1):
            sel = (of_point == k) & (self.h >= self.top - TOP_BAND)
            if (labels == k).sum() >= BUTTON_AREA * area or sel.sum() < MIN_POINTS:
                continue
            if any((sel & neck[1]).sum() > SHARE * sel.sum() for neck in necks):
                continue
            kind: PartKind = "rocker" if _tilted_plane(self.p[sel]) else "button"
            out.append(self._framed((kind,), sel, None))
        return out

    def _framed(
        self, kinds: tuple[PartKind, ...], sel: NDArray[np.bool_], axis: NDArray[np.float64] | None
    ) -> Found:
        """A candidate's box (object frame): x along `axis` (decided), else its own long side."""
        points = self.p[sel]
        x = oriented_box(points, "upright").axes[0] if axis is None else np.asarray(axis)
        x = np.array([x[0], x[1], 0.0]) / np.hypot(x[0], x[1])
        axes = np.column_stack([x, np.cross([0.0, 0.0, 1.0], x), [0.0, 0.0, 1.0]])
        q = points @ axes
        low, high = q.min(axis=0), q.max(axis=0)
        box = np.eye(4)
        box[:3, :3] = axes
        box[:3, 3] = axes @ ((low + high) / 2)
        return kinds, sel, box, high - low, axis is not None


def _merged(found: list[Found]) -> list[Found]:
    """Candidates in the order found; one lying mostly within another is made one with it: both
    kinds, the larger one's points and box."""
    kept: list[Found] = []
    for candidate in found:
        kinds, sel = candidate[0], candidate[1]
        for i, other in enumerate(kept):
            both = int((sel & other[1]).sum())
            if "opening" in kinds + other[0] or both <= SHARE * min(sel.sum(), other[1].sum()):
                continue
            larger = candidate if sel.sum() > other[1].sum() else other
            kept[i] = (tuple(dict.fromkeys(other[0] + kinds)), *larger[1:])
            break
        else:
            kept.append(candidate)
    return kept


def _matrix(pose: ArrayLike) -> NDArray[np.float64]:
    """A pose as a 7-vector or a 4x4 matrix, as a 4x4 matrix."""
    value = np.asarray(pose, dtype=float)
    return value if value.shape == (4, 4) else pose_to_matrix(value)


def _yaw(yaw: float) -> NDArray[np.float64]:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _spread(values: NDArray[np.float64]) -> float:
    return float(np.percentile(values, 95) - np.percentile(values, 5)) if len(values) else 0.0


def _tilted(height: NDArray[np.float64], depth: NDArray[np.float64]) -> bool:
    """Whether a bulge's surface tilts: how far past the face it stands (its outermost points,
    cell by cell up) changes with height."""
    rows = np.floor(height / CELL).astype(np.int64)
    rows -= rows.min()
    outer = np.full(rows.max() + 1, -np.inf)
    np.maximum.at(outer, rows, depth)
    used = np.flatnonzero(np.isfinite(outer))
    if len(used) < 4:
        return False
    slope = np.polyfit(used * CELL, outer[used], 1)[0]
    return bool(np.degrees(np.arctan(abs(slope))) >= TILT)


def _tilted_plane(points: NDArray[np.float64]) -> bool:
    """Whether the plane through a top part's points tilts."""
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    (gx, gy, _), *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    return bool(np.degrees(np.arctan(np.hypot(gx, gy))) >= TILT)


def far_end(points: ArrayLike, center: ArrayLike) -> NDArray[np.float64] | None:
    """The far end of a footprint (`points` and its box `center` in the object frame): the
    center, at the box center's height, of the footprint cells its points occupy at least
    `FAR_END_BAND` of the farthest one's distance from their center, where that center is
    `FAR_END_MIN_M` off the box center; cells, not points, so that a camera's one-sided view
    weighs no side more. None without points or without such an end."""
    found = np.asarray(points, dtype=float)
    if len(found) == 0:
        return None
    middle_of = np.asarray(center, dtype=float)
    cells = np.unique(np.floor((found[:, :2] - middle_of[:2]) / CELL), axis=0)
    cells = (cells + 0.5) * CELL
    middle = cells.mean(axis=0)
    if np.hypot(*middle) < FAR_END_MIN_M:
        return None
    out = np.linalg.norm(cells - middle, axis=1)
    end = cells[out >= FAR_END_BAND * out.max()].mean(axis=0)
    return np.array([middle_of[0] + end[0], middle_of[1] + end[1], middle_of[2]])
