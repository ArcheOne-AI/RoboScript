"""Quaternions in (w, x, y, z) order, vectorized over leading axes.

Hamilton convention: q rotates v as q v q*, the same map as the matrix `quat_to_matrix(q)`
(child frame to parent frame). scipy's `Rotation` is (x, y, z, w) unless `scalar_first=True`.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def quat_normalize(q: ArrayLike) -> Array:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_multiply(a: ArrayLike, b: ArrayLike) -> Array:
    """Hamilton product a b: rotate by b, then by a."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=float), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conjugate(q: ArrayLike) -> Array:
    """The inverse rotation of a unit quaternion."""
    return np.asarray(q, dtype=float) * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_matrix(q: ArrayLike) -> Array:
    """(..., 4) -> (..., 3, 3); exact for quaternions of any nonzero norm."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = np.moveaxis(q, -1, 0)
    s = 2.0 / np.sum(q * q, axis=-1)
    m = np.stack(
        [
            1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
            s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w),
            s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y),
        ],
        axis=-1,
    )  # fmt: skip
    return m.reshape((*q.shape[:-1], 3, 3))


def quat_from_matrix(m: ArrayLike) -> Array:
    """(..., 3, 3) rotation matrices -> (..., 4) unit quaternions with w >= 0."""
    m = np.asarray(m, dtype=float)
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # Row c equals 4 * q_c * q; dividing by the largest |q_c| is the stable choice (Shepperd).
    rows = np.stack(
        [
            np.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], axis=-1),
            np.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], axis=-1),
            np.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], axis=-1),
        ],
        axis=-2,
    )
    pivot = np.argmax(np.diagonal(rows, axis1=-2, axis2=-1), axis=-1)
    q = quat_normalize(np.take_along_axis(rows, pivot[..., None, None], axis=-2)[..., 0, :])
    return np.where(q[..., :1] < 0, -q, q)


def quat_from_axis_angle(axis: ArrayLike, angle: ArrayLike) -> Array:
    """Rotation by `angle` radians about `axis` (any nonzero length)."""
    half = np.asarray(angle, dtype=float)[..., None] / 2
    xyz = np.sin(half) * quat_normalize(axis)
    w = np.broadcast_to(np.cos(half), (*xyz.shape[:-1], 1))
    return np.concatenate([w, xyz], axis=-1)


def quat_rotate(q: ArrayLike, v: ArrayLike) -> Array:
    """Rotate vectors (..., 3) by quaternions (..., 4), broadcasting leading axes."""
    q = quat_normalize(q)
    v = np.asarray(v, dtype=float)
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_angle_deg(a: ArrayLike, b: ArrayLike) -> Array:
    """Angle of the rotation between orientations a and b, degrees in [0, 180]."""
    r = quat_multiply(quat_conjugate(a), b)
    return np.degrees(2 * np.arctan2(np.linalg.norm(r[..., 1:], axis=-1), np.abs(r[..., 0])))


def quat_slerp(a: ArrayLike, b: ArrayLike, t: ArrayLike) -> Array:
    """Interpolate from a (t=0) to b (t=1) along the shorter arc; t is a scalar or (N,) array."""
    a, b = quat_normalize(a), quat_normalize(b)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    theta = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    t = np.asarray(t, dtype=float)[..., None]
    near = theta < 1e-9  # identical orientations: the linear weights are exact
    sin = np.where(near, 1.0, np.sin(theta))
    wa = np.where(near, 1 - t, np.sin((1 - t) * theta) / sin)
    wb = np.where(near, t, np.sin(t * theta) / sin)
    return quat_normalize(wa * a + wb * b)


def rpy_to_matrix(rpy_deg: ArrayLike) -> Array:
    """The rotation of roll, pitch and yaw (degrees) about the fixed x, y and z axes, applied in
    that order: Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.radians(np.asarray(rpy_deg, dtype=float))
    cr, sr, cp, sp, cy, sy = (np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch),
                              np.cos(yaw), np.sin(yaw))  # fmt: skip
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(m: ArrayLike) -> Array:
    """`rpy_to_matrix`'s inverse: roll, pitch, yaw (degrees; pitch in [-90, 90])."""
    r = np.asarray(m, dtype=float)
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    if abs(r[2, 0]) > 1 - 1e-12:  # gimbal lock: the yaw alone takes the turn about z
        return np.degrees([0.0, pitch, np.arctan2(-r[0, 1], r[1, 1])])
    return np.degrees([np.arctan2(r[2, 1], r[2, 2]), pitch, np.arctan2(r[1, 0], r[0, 0])])


def yaw_matrix(angle: float) -> Array:
    """The rotation by `angle` radians about z (3x3)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
