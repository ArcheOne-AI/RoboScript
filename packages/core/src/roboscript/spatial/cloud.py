"""Point clouds: (N, 3) float arrays in meters; "height" is the z of a z-up (world) frame."""

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

from roboscript.spatial.rotation import Array, quat_from_axis_angle, quat_from_matrix

# A dense height map beyond this many cells means the points were not cropped to a workspace.
_MAX_GRID_CELLS = 25_000_000


class Obb(NamedTuple):
    """An oriented box: `quat` (wxyz) maps box axes to the world, `axes[i]` is box axis i."""

    center: Array  # (3,)
    size: Array  # (3,) full extents along axes[0], axes[1], axes[2]
    quat: Array  # (4,)
    axes: Array  # (3, 3), rows


def voxel_downsample(points: ArrayLike, voxel: float) -> Array:
    """Centroid of the points in each occupied cube of edge `voxel`, in voxel-index order."""
    points = np.asarray(points, dtype=float)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)  # numpy 2.0.0 returned it with an extra axis
    sums = [np.bincount(inverse, weights=points[:, i], minlength=len(counts)) for i in range(3)]
    return np.stack(sums, axis=-1) / counts[:, None]


def radius_inliers(points: ArrayLike, radius: float, min_neighbors: int) -> NDArray[np.bool_]:
    """Mask of the points with at least `min_neighbors` other points within `radius`."""
    points = np.asarray(points, dtype=float)
    counts = cKDTree(points).query_ball_point(points, radius, return_length=True)
    return np.asarray(counts) - 1 >= min_neighbors


def statistical_inliers(
    points: ArrayLike, k: int = 20, std_ratio: float = 2.0
) -> NDArray[np.bool_]:
    """Mask of the points whose mean distance to their k nearest neighbors is at most the
    mean of that distance over all points plus `std_ratio` standard deviations."""
    points = np.asarray(points, dtype=float)
    k = min(k, len(points) - 1)
    if k < 1:
        return np.ones(len(points), dtype=bool)
    dist, _ = cKDTree(points).query(points, k=k + 1)  # the nearest one is the point itself
    mean = dist[:, 1:].mean(axis=1)
    return mean <= mean.mean() + std_ratio * mean.std()


def fit_plane(points: ArrayLike) -> tuple[Array, float]:
    """Least-squares plane as (unit normal n, offset d) with n . x + d = 0; n has z >= 0."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError(f"a plane needs at least 3 points, got {len(points)}")
    centroid = points.mean(axis=0)
    centered = points - centroid
    normal = np.linalg.eigh(centered.T @ centered)[1][:, 0]  # smallest-variance direction
    if normal[2] < 0:
        normal = -normal
    return normal, float(-normal @ centroid)


def ransac_plane(
    points: ArrayLike, threshold: float, iterations: int = 200, seed: int = 0
) -> tuple[Array, float, NDArray[np.bool_]]:
    """Dominant plane: RANSAC over point triples, then `fit_plane` on the best consensus set.

    Returns (normal, offset, inliers), inliers being the points within `threshold` meters of the
    refitted plane.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError(f"a plane needs at least 3 points, got {len(points)}")
    rng = np.random.default_rng(seed)
    tri = points[rng.integers(0, len(points), size=(iterations, 3))]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    usable = norms > 1e-12  # skip collinear samples
    if not usable.any():
        raise ValueError("no plane: every sampled triple of points is collinear")
    normals = normals[usable] / norms[usable, None]
    offsets = -np.einsum("ij,ij->i", normals, tri[usable, 0])
    counts = [
        np.count_nonzero(np.abs(points @ n + d) <= threshold)
        for n, d in zip(normals, offsets, strict=True)
    ]
    best = int(np.argmax(counts))
    consensus = np.abs(points @ normals[best] + offsets[best]) <= threshold
    normal, offset = fit_plane(points[consensus])
    return normal, offset, np.abs(points @ normal + offset) <= threshold


def height_components(
    points: ArrayLike, cell: float, min_height: float, min_cells: int = 1, close: int = 0
) -> NDArray[np.int64]:
    """Label points by the 8-connected component of their xy cell on a height map.

    Points at z >= `min_height` occupy the `cell`-sized grid square under them; the occupied
    squares, closed with a (2 * `close` + 1)-square window when `close` > 0 (bridging gaps of up
    to about 2 * `close` cells), are labeled with `scipy.ndimage.label`. Returns one label per
    point: 0, 1, ... with the component holding most points first, and -1 for points below
    `min_height` or in components of fewer than `min_cells` occupied squares. Crop the points to
    the workspace first: the grid spans their xy extent.
    """
    points = np.asarray(points, dtype=float)
    labels = np.full(len(points), -1, dtype=np.int64)
    high = points[:, 2] >= min_height
    if not high.any():
        return labels
    ij = np.floor(points[high, :2] / cell).astype(np.int64)
    ij += close - ij.min(axis=0)  # a margin of `close` cells keeps the closing off the border
    shape = ij.max(axis=0) + 1 + close
    if shape.prod() > _MAX_GRID_CELLS:
        raise ValueError(
            f"height map of {shape[0]}x{shape[1]} cells: crop the points or use a larger cell"
        )
    grid = np.zeros(shape, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    square = np.ones((3, 3), dtype=bool)
    closed = grid | ndimage.binary_closing(grid, square, iterations=close) if close else grid
    cells, n = ndimage.label(closed, structure=square)
    component = cells[ij[:, 0], ij[:, 1]] - 1
    n_cells = np.bincount(cells[grid] - 1, minlength=n)
    order = np.argsort(-np.bincount(component, minlength=n), kind="stable")
    order = order[n_cells[order] >= min_cells]
    relabel = np.full(n, -1, dtype=np.int64)
    relabel[order] = np.arange(len(order))
    labels[high] = relabel[component]
    return labels


def oriented_box(points: ArrayLike, mode: Literal["upright", "pca"] = "upright") -> Obb:
    """Bounding box of the points in one of two modes.

    `upright`: z stays world z; the xy footprint is the minimum-area rectangle, with axes[0] along
    its longer side and yaw in [-90, 90) degrees. `pca`: axes are the principal directions (largest
    variance first, right-handed, signs fixed so that each of axes[0] and axes[2] has its largest
    component positive).
    """
    points = np.asarray(points, dtype=float)
    if mode == "upright":
        return _upright_box(points)
    if mode == "pca":
        return _pca_box(points)
    raise ValueError(f"mode must be 'upright' or 'pca', got {mode!r}")


def _upright_box(points: Array) -> Obb:
    xy = points[:, :2]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except QhullError:  # fewer than 3 distinct points, or all on one line
        hull = xy
    edges = np.roll(hull, -1, axis=0) - hull
    # The minimum-area rectangle has a side along a hull edge (rotating calipers).
    yaws = np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2))
    c, s = np.cos(yaws), np.sin(yaws)
    uv = np.stack([hull @ np.stack([c, s]), hull @ np.stack([-s, c])], axis=-1)  # (H, yaws, 2)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    best = int(np.argmin(np.prod(hi - lo, axis=-1)))
    yaw, (u, v), (su, sv) = yaws[best], (lo[best] + hi[best]) / 2, hi[best] - lo[best]
    center = [np.cos(yaw) * u - np.sin(yaw) * v, np.sin(yaw) * u + np.cos(yaw) * v]
    if su < sv:  # put axes[0] along the longer side
        yaw, su, sv = yaw + np.pi / 2, sv, su
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    c, s = np.cos(yaw), np.sin(yaw)
    z_lo, z_hi = points[:, 2].min(), points[:, 2].max()
    return Obb(
        center=np.array([*center, (z_lo + z_hi) / 2]),
        size=np.array([su, sv, z_hi - z_lo]),
        quat=quat_from_axis_angle([0.0, 0.0, 1.0], yaw),
        axes=np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]),
    )


def _pca_box(points: Array) -> Obb:
    mean = points.mean(axis=0)
    centered = points - mean
    axes = np.linalg.eigh(centered.T @ centered)[1][:, ::-1].T  # rows, largest variance first
    for i in (0, 2):
        if axes[i, np.argmax(np.abs(axes[i]))] < 0:
            axes[i] = -axes[i]
    axes[1] = np.cross(axes[2], axes[0])
    proj = centered @ axes.T
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    return Obb(
        center=mean + ((lo + hi) / 2) @ axes,
        size=hi - lo,
        quat=quat_from_matrix(axes.T),
        axes=axes,
    )
