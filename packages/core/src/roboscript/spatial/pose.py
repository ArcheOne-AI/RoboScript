"""Rigid poses as 7-vectors [x, y, z, qw, qx, qy, qz] or 4x4 matrices.

A pose maps its child frame into the parent frame: x_parent = R(q) x_child + p.
Functions broadcast over leading axes.
"""

import numpy as np
from numpy.typing import ArrayLike

from roboscript.spatial.rotation import (
    Array,
    quat_conjugate,
    quat_from_matrix,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_slerp,
    quat_to_matrix,
)


def pose_to_matrix(pose: ArrayLike) -> Array:
    """(..., 7) -> (..., 4, 4)."""
    pose = np.asarray(pose, dtype=float)
    m = np.zeros((*pose.shape[:-1], 4, 4))
    m[..., :3, :3] = quat_to_matrix(pose[..., 3:])
    m[..., :3, 3] = pose[..., :3]
    m[..., 3, 3] = 1.0
    return m


def pose_from_matrix(m: ArrayLike) -> Array:
    """(..., 4, 4) -> (..., 7) with qw >= 0."""
    m = np.asarray(m, dtype=float)
    return np.concatenate([m[..., :3, 3], quat_from_matrix(m[..., :3, :3])], axis=-1)


def pose_compose(a: ArrayLike, b: ArrayLike) -> Array:
    """a then b in a's frame, the matrix product A @ B (e.g. world_T_ee @ ee_T_tcp)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    p = a[..., :3] + quat_rotate(a[..., 3:], b[..., :3])
    q = quat_normalize(quat_multiply(a[..., 3:], b[..., 3:]))
    return np.concatenate([p, q], axis=-1)


def pose_inverse(pose: ArrayLike) -> Array:
    pose = np.asarray(pose, dtype=float)
    q = quat_conjugate(quat_normalize(pose[..., 3:]))
    return np.concatenate([-quat_rotate(q, pose[..., :3]), q], axis=-1)


def transform_points(transform: ArrayLike, points: ArrayLike) -> Array:
    """Map points (..., 3) by one transform, a 7-vector pose or a 4x4 matrix."""
    t = np.asarray(transform, dtype=float)
    points = np.asarray(points, dtype=float)
    if t.shape == (7,):
        return quat_rotate(t[3:], points) + t[:3]
    if t.shape == (4, 4):
        return points @ t[:3, :3].T + t[:3, 3]
    raise ValueError(f"transform must be a 7-vector pose or a 4x4 matrix, got shape {t.shape}")


def pose_interpolate(a: ArrayLike, b: ArrayLike, t: ArrayLike) -> Array:
    """Linear position and slerp orientation from a (t=0) to b (t=1); t is a scalar or (N,)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t = np.asarray(t, dtype=float)
    p = a[..., :3] + t[..., None] * (b[..., :3] - a[..., :3])
    return np.concatenate([p, quat_slerp(a[..., 3:], b[..., 3:], t)], axis=-1)


def as_matrix(pose: ArrayLike) -> Array:
    """A pose given as a 7-vector or already a 4x4 matrix, as a 4x4 matrix."""
    value = np.asarray(pose, dtype=float)
    return value if value.shape == (4, 4) else pose_to_matrix(value)


def transform(rotation: ArrayLike | None = None, position: ArrayLike | None = None) -> Array:
    """The 4x4 matrix of a rotation (3x3; None: none) and a position (None: the origin)."""
    m = np.eye(4)
    if rotation is not None:
        m[:3, :3] = rotation
    if position is not None:
        m[:3, 3] = position
    return m
