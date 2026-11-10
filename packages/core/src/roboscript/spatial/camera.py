"""Pinhole cameras in the OpenCV convention: x right, y down, z forward (the optical axis).

`K` is the 3x3 intrinsic matrix (pixels), `cam2world` the 4x4 camera-to-world transform
(meters). Depth maps hold the camera-frame z in meters, 0 where there is no return. Pixels are
(u, v) = (column, row), with integer coordinates at pixel centers.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.spatial.pose import transform_points
from roboscript.spatial.rotation import Array


def backproject(
    depth: ArrayLike, K: ArrayLike, mask: ArrayLike | None = None
) -> tuple[Array, NDArray[np.int64]]:
    """Camera-frame points (N, 3) of pixels with depth > 0 (and in `mask`), and those pixels (N, 2).

    Points are in row-major pixel order; `mask` is an (H, W) boolean array.
    """
    depth = np.asarray(depth)
    valid = depth > 0
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    v, u = np.nonzero(valid)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(np.asarray(K, float)).T
    return rays * depth[v, u].astype(float)[:, None], np.stack([u, v], axis=-1)


def backproject_world(
    depth: ArrayLike, K: ArrayLike, cam2world: ArrayLike, mask: ArrayLike | None = None
) -> tuple[Array, NDArray[np.int64]]:
    """Like `backproject`, with the points in the world frame."""
    points, pixels = backproject(depth, K, mask)
    return transform_points(cam2world, points), pixels


def project(
    points: ArrayLike, K: ArrayLike, cam2world: ArrayLike
) -> tuple[Array, Array, NDArray[np.bool_]]:
    """Project world points (..., 3) to pixels (..., 2), camera depth z (...) and z > 0 (...).

    Pixels of points that are not in front of the camera are NaN.
    """
    cam = transform_points(np.linalg.inv(np.asarray(cam2world, dtype=float)), points)
    z = cam[..., 2]
    in_front = z > 0
    uvw = cam @ np.asarray(K, dtype=float).T
    uv = np.full((*uvw.shape[:-1], 2), np.nan)
    np.divide(uvw[..., :2], z[..., None], out=uv, where=in_front[..., None])
    return uv, z, in_front
