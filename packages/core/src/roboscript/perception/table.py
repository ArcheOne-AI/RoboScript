"""The table: its plane from a height histogram, the edges the views show, and landmarks.

A level table dominates the heights inside the nominal table area, so the 1 mm histogram peak
finds it and a least-squares plane on the band around the peak refines it; a tilted or mostly
covered table falls back to RANSAC. `extent` spans what the anchor view sees of the plane.

The table's own edges (axis-aligned: `x0`, `x1`, `y0`, `y1`) come only from table pixels next to
a drop: a pixel that sees something `drop` below the plane, so the camera looks past the table
there. Table pixels at the image border or next to an occluder (an object on the table, the
robot, a pixel without depth) show where the view ends, not the table, and never count. An edge
pixel belongs to the side it faces (away from the table seen around it); a side is an edge when
at least `edge_points` pixels agree on it within 1 cm and (almost) no table is seen beyond it;
the others stay None, and the landmarks that need them are left out.

Landmarks sit on the plane: the table center (all four edges), the midpoint of the front edge
(the edge facing the robot, at the smallest y; x0, x1, y0) and the robot-base midline (the
vertical plane through the robot's base, along y) at the table's middle depth (y0, y1). Where a
view shows no edge of a side, the landmarks are estimated (`estimated`): x from the robot's
midline, y from the anchor view's extent, so a view that cuts the table still has them.
Identify-once policy: an episode keeps the table of its initial frame
(`frame_geometry(table=...)`), which the toolset stores as the episode's `TABLE` fact.
"""

from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict
from scipy import ndimage

from roboscript.contracts.geometry import Vec3
from roboscript.perception.config import DEFAULTS, TableConfig
from roboscript.spatial.cloud import fit_plane, ransac_plane

HISTOGRAM_BIN: Final = 0.001
EDGES: Final = ("x0", "x1", "y0", "y1")
BORDER: Final = 2
"""Pixels this close to the image border never make an edge."""
EDGE_CELL: Final = 0.005
EDGE_RADIUS: Final = 0.03
"""An edge pixel faces away from the centroid of the table seen within this distance."""
EDGE_SPREAD: Final = 0.01
"""An edge's pixels agree within this (median absolute deviation), and no more than
`BEYOND_SHARE` of the table points seen lie this far beyond it, or the side stays unknown (the rim
of a hole or a gap between tables bounds no table)."""
BEYOND_SHARE: Final = 0.005
EDGE_COS: Final = float(np.cos(np.radians(35)))
"""An edge pixel faces a side when its direction is within 35 deg of the side's normal (not a
corner)."""


class Table(BaseModel):
    """`z` is the plane's height at the world origin; `extent` (x0, x1, y0, y1) spans the plane's
    inliers in the anchor view (0.5 to 99.5 percentiles); `edges` holds the table's own edges
    where a view shows them (None where none does); `rms` is the inliers' RMS distance to the
    plane. JSON-compatible with the `TABLE` scene fact (`z`, `extent`)."""

    model_config = ConfigDict(frozen=True)

    z: float
    extent: tuple[float, float, float, float]
    edges: dict[str, float | None]
    normal: Vec3
    method: Literal["histogram", "ransac"]
    inliers: int
    rms: float
    landmarks: dict[str, Vec3]

    def height(self, points: ArrayLike) -> NDArray[np.float64]:
        """Signed distance of points (..., 3) to the table plane, positive above it."""
        normal = np.asarray(self.normal)
        return np.asarray(points, dtype=float) @ normal - normal[2] * self.z

    def plane_z(self, xy: ArrayLike) -> NDArray[np.float64]:
        """The plane's height under points (..., 2)."""
        nx, ny, nz = self.normal
        p = np.asarray(xy, dtype=float)
        return self.z - (nx * p[..., 0] + ny * p[..., 1]) / nz

    def side(self, name: str) -> float:
        """The edge `name` (x0, x1, y0, y1) where known, else the anchor view's extent there."""
        edge = self.edges[name]
        return self.extent[EDGES.index(name)] if edge is None else edge

    @property
    def middle(self) -> tuple[float, float]:
        """The middle of the table as far as it is known: its center when all four edges are,
        else between the known edges and the anchor view's extent."""
        x0, x1, y0, y1 = (self.side(name) for name in EDGES)
        return (x0 + x1) / 2, (y0 + y1) / 2

    def over(self, xy: ArrayLike) -> bool:
        """Whether a point's xy lies over the table: within its known edges (an edge no view
        showed does not bound it)."""
        x, y = np.asarray(xy, dtype=float)[:2]
        x0, x1, y0, y1 = (self.edges[name] for name in EDGES)
        return (
            (x0 is None or x >= x0)
            and (x1 is None or x <= x1)
            and (y0 is None or y >= y0)
            and (y1 is None or y <= y1)
        )


def fit_table(points: ArrayLike, config: TableConfig = DEFAULTS.table) -> Table:
    """The table plane in the world points (N, 3) of a view that sees it; its edges are unknown
    until `with_edges`."""
    x0, x1, y0, y1 = config.area
    z0, z1 = config.heights
    p = np.asarray(points, dtype=float)
    p = p[(p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)]
    p = p[(p[:, 2] > z0) & (p[:, 2] < z1)]
    if len(p) < config.min_points:
        raise ValueError(
            f"{len(p)} points in the table area {config.xy_box} at heights {config.z_range}: "
            "no table in view"
        )
    hist, bins = np.histogram(p[:, 2], bins=np.arange(z0, z1 + HISTOGRAM_BIN, HISTOGRAM_BIN))
    peak = bins[np.argmax(hist)] + HISTOGRAM_BIN / 2
    band = p[np.abs(p[:, 2] - peak) < config.band]
    method: Literal["histogram", "ransac"] = "histogram"
    fit = fit_plane(band) if len(band) >= config.min_points else None
    if fit is None or _tilt_deg(fit[0]) > config.max_tilt_deg:
        method = "ransac"
        normal, offset, _ = ransac_plane(p, config.ransac_threshold)
        if _tilt_deg(normal) > config.max_tilt_deg:
            raise ValueError(
                f"the dominant plane in the table area is tilted {_tilt_deg(normal):.1f} deg: "
                "no level table in view"
            )
        band = p
    else:
        normal, offset = fit
    distance = band @ normal + offset
    near = np.abs(distance) < config.inlier
    lo, hi = np.percentile(band[near, :2], [0.5, 99.5], axis=0)
    return Table(
        z=float(-offset / normal[2]),
        extent=(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])),
        edges=dict.fromkeys(EDGES),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        method=method,
        inliers=int(near.sum()),
        rms=float(np.sqrt(np.mean(distance[near] ** 2))),
        landmarks={},
    )


def edge_pixels(
    xyz: NDArray[np.float64],
    usable: NDArray[np.bool_],
    table: Table,
    config: TableConfig = DEFAULTS.table,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One view's (edge points, table points), each (N, 3): the table pixels (usable, in the
    table area, within `band` of the plane) and those of them next to a drop (a usable pixel
    `drop` below the plane among the 8 neighbors), away from the image border. `xyz` (H, W, 3)
    is NaN without depth; `usable` has depth and is not the robot."""
    height = table.height(xyz)
    x0, x1, y0, y1 = config.area
    x, y = xyz[..., 0], xyz[..., 1]
    top = usable & (x > x0) & (x < x1) & (y > y0) & (y < y1) & (np.abs(height) < config.band)
    drop = usable & (height < -config.drop)
    edge = top & ndimage.binary_dilation(drop, np.ones((3, 3), dtype=bool))
    edge[:BORDER], edge[-BORDER:], edge[:, :BORDER], edge[:, -BORDER:] = False, False, False, False
    return xyz[edge], xyz[top]


def with_edges(
    table: Table,
    edge: NDArray[np.float64],
    seen: NDArray[np.float64],
    config: TableConfig = DEFAULTS.table,
    robot_base: ArrayLike | None = None,
) -> Table:
    """The table with the edges that the edge points (N, 3) of a frame's views show, `seen` being
    all the table points the views show (`edge_pixels`); `robot_base` is the world position of
    the robot's base (the midline landmark needs it)."""
    x0, x1, y0, y1 = config.area
    origin = np.array([x0, y0])
    shape = tuple(int(np.ceil(s / EDGE_CELL)) for s in (x1 - x0, y1 - y0))

    def cells(points: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        ij = np.floor((points[:, :2] - origin) / EDGE_CELL).astype(np.int64)
        ij = np.clip(ij, 0, np.array(shape) - 1)
        return ij[:, 0], ij[:, 1]

    def window_mean(a: NDArray[np.float64]) -> NDArray[np.float64]:
        return ndimage.uniform_filter(a, 2 * round(EDGE_RADIUS / EDGE_CELL) + 1, mode="constant")

    occupied = np.zeros(shape)
    occupied[cells(seen)] = 1.0
    i, j = np.indices(shape, dtype=float)
    ei, ej = cells(edge)  # edge points are table points: their cells are occupied
    count = window_mean(occupied)[ei, ej]
    inward = np.stack([window_mean(occupied * i)[ei, ej], window_mean(occupied * j)[ei, ej]], 1)
    outward = np.stack([ei, ej], axis=1) - inward / count[:, None]
    axis = np.argmax(np.abs(outward), axis=1)
    along = outward[np.arange(len(outward)), axis]
    length = np.linalg.norm(outward, axis=1)  # ~3 cells at a straight edge, 0 inside the table
    clear = (np.abs(along) >= EDGE_COS * length) & (length >= 1.0)
    side = 2 * axis + (along > 0)  # 0: x0, 1: x1, 2: y0, 3: y1
    edges: dict[str, float | None] = {}
    for k, name in enumerate(EDGES):
        coords = edge[clear & (side == k), k // 2]
        edges[name] = None
        if len(coords) >= config.edge_points:
            middle = float(np.median(coords))
            outward = 1.0 if k % 2 else -1.0
            beyond = outward * (seen[:, k // 2] - middle) > EDGE_SPREAD
            agree = float(np.median(np.abs(coords - middle))) <= EDGE_SPREAD
            if agree and beyond.mean() <= BEYOND_SHARE:
                edges[name] = middle
    return table.model_copy(
        update={"edges": edges, "landmarks": landmarks(table.z, edges, robot_base, table.extent)}
    )


def landmarks(
    z: float,
    edges: dict[str, float | None],
    robot_base: ArrayLike | None = None,
    extent: tuple[float, float, float, float] | None = None,
) -> dict[str, Vec3]:
    """The landmarks the known edges allow; with `extent`, also those an edge no view shows
    leaves out, estimated (`estimated`): a missing x edge takes the center's x from the robot's
    midline (from the view's extent without a robot base), a missing y edge its side of the
    extent."""
    known = [edges[name] for name in EDGES]
    sides = [
        edge if edge is not None or extent is None else extent[k] for k, edge in enumerate(known)
    ]
    x0, x1, y0, y1 = sides
    found: dict[str, Vec3] = {}
    base_x = None if robot_base is None else float(np.asarray(robot_base, dtype=float)[0])
    mid_x = None if x0 is None or x1 is None else (x0 + x1) / 2
    if (known[0] is None or known[1] is None) and base_x is not None and extent is not None:
        mid_x = base_x
    if mid_x is not None and y0 is not None:
        found["front_edge_mid"] = (mid_x, y0, z)
        if y1 is not None:
            found["table_center"] = (mid_x, (y0 + y1) / 2, z)
    if base_x is not None and y0 is not None and y1 is not None:
        found["robot_midline"] = (base_x, (y0 + y1) / 2, z)
    return found


_NEEDS: Final = {
    "table_center": EDGES,
    "front_edge_mid": ("x0", "x1", "y0"),
    "robot_midline": ("y0", "y1"),
}
"""The edges each landmark is exact with."""


def estimated(table: "Table") -> list[str]:
    """The table's landmarks that an edge no view shows leaves estimated (see `landmarks`)."""
    return sorted(
        name
        for name in table.landmarks
        if any(table.edges.get(edge) is None for edge in _NEEDS.get(name, EDGES))
    )


def _tilt_deg(normal: NDArray[np.float64]) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
