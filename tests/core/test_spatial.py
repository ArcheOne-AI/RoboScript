"""roboscript.spatial: quaternions and poses (checked against scipy) and pinhole cameras."""

import numpy as np
from scipy.spatial.transform import Rotation

from roboscript.spatial.camera import backproject, backproject_world, project
from roboscript.spatial.pose import (
    pose_compose,
    pose_inverse,
    pose_to_matrix,
    transform_points,
)
from roboscript.spatial.rotation import (
    quat_from_axis_angle,
    quat_to_matrix,
)


def random_quats(n: int, seed: int) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=(n, 4))  # uniform on the 3-sphere once normalized
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def random_poses(n: int, seed: int) -> np.ndarray:
    p = np.random.default_rng(seed + 100).uniform(-1, 1, (n, 3))
    return np.concatenate([p, random_quats(n, seed)], axis=1)


def scipy_rotation(q: np.ndarray) -> Rotation:
    return Rotation.from_quat(q, scalar_first=True)


def test_to_matrix_matches_scipy_for_any_norm() -> None:
    q = random_quats(64, 0)
    np.testing.assert_allclose(quat_to_matrix(q), scipy_rotation(q).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix(3 * q), quat_to_matrix(q), atol=1e-12)


def test_compose_and_inverse_follow_matrix_algebra() -> None:
    a, b = random_poses(16, 12), random_poses(16, 13)
    ma, mb = pose_to_matrix(a), pose_to_matrix(b)
    np.testing.assert_allclose(pose_to_matrix(pose_compose(a, b)), ma @ mb, atol=1e-12)
    np.testing.assert_allclose(pose_to_matrix(pose_compose(a[0], b)), ma[0] @ mb, atol=1e-12)
    np.testing.assert_allclose(pose_to_matrix(pose_inverse(a)), np.linalg.inv(ma), atol=1e-12)


def test_tcp_sits_ahead_of_the_ee_along_its_x_axis() -> None:
    # TCP = ee pose moved 0.12 m along the ee's own x axis
    ee = np.array([0.1, -0.2, 0.9, *quat_from_axis_angle([0, 1, 0], np.pi / 2)])
    tcp = pose_compose(ee, [0.12, 0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(tcp[:3], [0.1, -0.2, 0.9 - 0.12], atol=1e-12)  # x turned to -z
    np.testing.assert_allclose(tcp[3:], ee[3:], atol=1e-12)


# A camera 1 m above the world origin looking straight down: camera x = world x, camera y = world
# -y, camera z (forward) = world -z.
K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
CAM2WORLD = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]])


def test_backproject_inverts_project() -> None:
    depth = np.ones((480, 640), dtype=np.float32)  # the plane z = 0 seen from 1 m
    depth[100, 202] = 0.0  # no return
    mask = np.zeros(depth.shape, dtype=bool)
    mask[100:110, 200:205] = True
    points, pixels = backproject_world(depth, K, CAM2WORLD, mask)
    assert points.shape == (49, 3) and pixels.dtype == np.int64
    assert pixels[:3].tolist() == [[200, 100], [201, 100], [203, 100]]  # (u, v), row-major
    np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-12)
    uv, _, _ = project(points, K, CAM2WORLD)
    np.testing.assert_allclose(uv, pixels, atol=1e-9)
    camera_points, camera_pixels = backproject(depth, K, mask)
    np.testing.assert_allclose(transform_points(CAM2WORLD, camera_points), points, atol=1e-12)
    np.testing.assert_array_equal(camera_pixels, pixels)
    assert backproject(depth, K)[0].shape == (480 * 640 - 1, 3)
