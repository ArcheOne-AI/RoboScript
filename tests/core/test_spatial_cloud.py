"""roboscript.spatial.cloud: a RANSAC plane for the table under objects, and an upright box's
footprint and yaw."""

import numpy as np
import pytest

from roboscript.spatial.cloud import (
    oriented_box,
    ransac_plane,
)
from roboscript.spatial.rotation import quat_from_axis_angle, quat_to_matrix


def box_grid(
    center: list[float], size: list[float], rotation: np.ndarray, n: int = 9
) -> np.ndarray:
    """A regular n^3 grid filling a box, corners included (so extents and PCA axes are exact)."""
    axis = np.linspace(-0.5, 0.5, n)
    local = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return (local * size) @ rotation.T + center


def yaw_matrix(degrees: float) -> np.ndarray:
    return quat_to_matrix(quat_from_axis_angle([0, 0, 1], np.radians(degrees)))


def test_ransac_plane_finds_the_table_under_objects() -> None:
    rng = np.random.default_rng(1)
    table = np.column_stack(
        [rng.uniform(-0.5, 0.5, (2000, 2)), 0.7 + rng.normal(scale=1e-3, size=2000)]
    )
    box = box_grid([0.1, 0.1, 0.8], [0.1, 0.1, 0.1], yaw_matrix(20))  # z from 0.75 to 0.85
    stray = rng.uniform(-1, 1, (50, 3))
    points = np.vstack([table, box, stray])
    normal, offset, inliers = ransac_plane(points, threshold=0.005)
    np.testing.assert_allclose(normal, [0, 0, 1], atol=1e-2)
    assert -offset == pytest.approx(0.7, abs=1e-3)
    assert inliers[:2000].mean() > 0.99
    assert not inliers[2000 : 2000 + len(box)].any()
    again = ransac_plane(points, threshold=0.005)
    np.testing.assert_array_equal(again[2], inliers)  # seeded: reproducible


@pytest.mark.parametrize("yaw", [0.0, 30.0, -60.0, 89.0, 120.0])
def test_upright_box_recovers_footprint_and_yaw(yaw: float) -> None:
    points = box_grid([0.3, -0.1, 0.8], [0.2, 0.1, 0.05], yaw_matrix(yaw))
    box = oriented_box(points, "upright")
    np.testing.assert_allclose(box.size, [0.2, 0.1, 0.05], atol=1e-9)
    np.testing.assert_allclose(box.center, [0.3, -0.1, 0.8], atol=1e-9)
    found = np.degrees(np.arctan2(box.axes[0, 1], box.axes[0, 0]))
    assert -90 <= found < 90
    assert (found - yaw + 90) % 180 - 90 == pytest.approx(0, abs=1e-6)  # equal up to a half turn
    np.testing.assert_allclose(box.axes[2], [0, 0, 1])
    np.testing.assert_allclose(quat_to_matrix(box.quat), box.axes.T, atol=1e-12)
