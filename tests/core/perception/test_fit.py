"""perception.fit: the summary and shape fits, on whole surfaces
and on what a camera sees."""

import numpy as np
import pytest

from roboscript.perception.fit import fit
from roboscript.spatial.camera import backproject_world
from roboscript.spatial.rotation import quat_to_matrix
from roboscript.testing.scenes import (
    HEAD,
    TABLE_Z,
    Box,
    Cylinder,
    Sphere,
    pinhole,
    render,
    surface,
)


def visible(*shapes: Box | Cylinder | Sphere) -> np.ndarray:
    """The points of the shapes (one object) the head camera sees."""
    camera = pinhole("overhead", HEAD)
    _, depth, index = render(shapes, camera)
    points, _ = backproject_world(depth, camera.K, camera.cam2world, index >= 0)
    return points


def test_box_fit() -> None:
    box = Box((-0.1, 0.1, TABLE_Z + 0.02), (0.08, 0.04, 0.04), yaw_deg=-60)
    f = fit(visible(box), "auto", TABLE_Z)
    assert f.shape == "box" and f.upright and f.symmetry == "pi"
    assert f.residual < 1e-3
    np.testing.assert_allclose(f.center, box.center, atol=1e-3)
    np.testing.assert_allclose(f.bottom_center, [-0.1, 0.1, TABLE_Z], atol=1e-3)
    np.testing.assert_allclose(f.obb.size, box.size, atol=1e-3)
    assert f.yaw_deg == pytest.approx(-60, abs=0.5)
    np.testing.assert_allclose(
        f.axis, [np.cos(np.radians(-60)), np.sin(np.radians(-60)), 0], atol=1e-2
    )
    frame = quat_to_matrix(f.frame.q)
    np.testing.assert_allclose(f.frame.p, f.bottom_center)  # standing: origin at the bottom center
    np.testing.assert_allclose(frame[:, 2], [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(frame[:, 0], f.axis, atol=1e-9)
    cube = fit(
        visible(Box((0.0, 0.0, TABLE_Z + 0.025), (0.05, 0.05, 0.05), yaw_deg=10)), "box", TABLE_Z
    )
    assert cube.symmetry == "half_pi"
    tall = fit(surface(Box((0, 0, TABLE_Z + 0.1), (0.04, 0.03, 0.2))), "box", TABLE_Z)
    assert tall.axis == (0.0, 0.0, 1.0)


def test_standing_cylinder_fit() -> None:
    can = Cylinder((0.05, 0.12, TABLE_Z + 0.06), radius=0.03, length=0.12)
    points = visible(can)
    f = fit(points, "auto", TABLE_Z)
    assert f.shape == "cylinder" and f.upright and f.symmetry == "continuous"
    assert f.radius == pytest.approx(0.03, abs=1e-3)
    np.testing.assert_allclose(f.center, can.center, atol=1e-3)
    np.testing.assert_allclose(f.top_center, [0.05, 0.12, TABLE_Z + 0.12], atol=1e-3)
    assert f.axis == (0.0, 0.0, 1.0)
    assert f.residual < fit(points, "box", TABLE_Z).residual / 3


def test_plane_and_line_fits() -> None:
    mat = Box((0.2, 0.1, TABLE_Z + 0.0005), (0.12, 0.08, 0.001), yaw_deg=15)
    f = fit(visible(mat), "auto", TABLE_Z)
    assert f.shape == "plane" and f.upright and f.symmetry == "pi"
    np.testing.assert_allclose(f.normal, [0, 0, 1], atol=1e-3)
    np.testing.assert_allclose(f.center[:2], mat.center[:2], atol=1e-3)
    np.testing.assert_allclose(f.obb.size[:2], [0.12, 0.08], atol=2e-3)
    assert f.yaw_deg == pytest.approx(15, abs=0.5)
    wall = np.stack(
        np.meshgrid(np.linspace(-0.1, 0.1, 30), [0.3], np.linspace(0.75, 0.85, 20)), -1
    ).reshape(-1, 3)
    upright = fit(wall, "plane")
    assert not upright.upright and abs(upright.normal[1]) == pytest.approx(1.0, abs=1e-6)
    t = np.linspace(0, 0.2, 50)[:, None]
    rod = np.array([0.0, 0.0, TABLE_Z + 0.1]) + t * np.array([0.6, 0.8, 0.0])
    line = fit(rod, "auto")
    assert line.shape == "line" and line.residual < 1e-9
    assert abs(float(np.dot(line.axis, [0.6, 0.8, 0.0]))) == pytest.approx(1.0)
    np.testing.assert_allclose(line.center, rod.mean(axis=0), atol=1e-9)
    # Frames and boxes are upright whatever the shape: the wall's box stands, the rod's lies.
    for f, size in ((upright, (0.2, 0.0, 0.1)), (line, (0.2, 0.0, 0.0))):
        np.testing.assert_allclose(quat_to_matrix(f.frame.q)[:, 2], [0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(quat_to_matrix(f.obb.q)[:, 2], [0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(f.obb.size, size, atol=1e-6)
    with pytest.raises(ValueError, match="at least 3 points"):
        fit(rod[:2])
