"""perception.lift and perception.views: masks, boxes and pixels to world points, across views."""

import numpy as np
import pytest

from roboscript.contracts.ids import FrameId
from roboscript.perception.fit import summarize
from roboscript.perception.lift import box_mask, lift
from roboscript.perception.robot import RobotShape
from roboscript.perception.views import frame_geometry
from roboscript.testing.scenes import (
    HEAD,
    PERCEPTION,
    TABLE_Z,
    Box,
    Cylinder,
    capture,
    look_at,
    pinhole,
    surface,
)

HEAD_CAMERA = pinhole("overhead", HEAD)
SIDE = pinhole("wrist_left", look_at([-0.45, 0.0, 0.95], [0.0, 0.05, TABLE_Z]), 320, 240)


def scene(shapes, cameras=(HEAD_CAMERA,), spheres=None):  # type: ignore[no-untyped-def]
    """Geometry of rendered shapes; `spheres` (M, 4) stand for the robot."""
    images, index = capture(shapes, cameras)
    robot = None if spheres is None else RobotShape(np.asarray(spheres, dtype=float))
    return frame_geometry(FrameId("frame-test"), 3, images, PERCEPTION, robot=robot), index


def test_a_mask_lifts_to_the_object_it_shows() -> None:
    shapes = [
        Box((-0.15, 0.05, TABLE_Z + 0.025), (0.05, 0.05, 0.05), yaw_deg=20),
        Cylinder((0.05, 0.12, TABLE_Z + 0.06), radius=0.03, length=0.12),
    ]
    geometry, index = scene(shapes)
    assert geometry.frame == "frame-test" and geometry.step == 3 and geometry.anchor == "overhead"
    for i, shape in enumerate(shapes):
        mask = index["overhead"] == i
        got = lift(geometry, {"overhead": mask}, PERCEPTION)
        assert got is not None and got.kind == "solid"
        assert set(got.masks) == {"overhead"} and (got.masks["overhead"] <= mask).all()
        est, truth = summarize(got.points, geometry.table.z), summarize(surface(shape), TABLE_Z)
        assert np.hypot(*np.subtract(est.center, truth.center)[:2]) < 1e-3
        assert est.top_z == pytest.approx(truth.top_z, abs=5e-4)


def test_leaks_onto_the_table_and_another_object_fall_away() -> None:
    shapes = [
        Box((0.0, 0.0, TABLE_Z + 0.03), (0.06, 0.06, 0.06)),
        Box((0.0, 0.2, TABLE_Z + 0.02), (0.05, 0.05, 0.04), rgb=(40, 160, 70)),  # behind it
    ]
    geometry, index = scene(shapes)
    target = index["overhead"] == 0
    v, u = np.nonzero(target)
    loose = box_mask((u.min() - 40, v.min() - 120, u.max() + 40, v.max() + 20), target.shape)
    assert (loose & (index["overhead"] == 1)).any() and (loose & (index["overhead"] == -1)).any()
    got = lift(geometry, {"overhead": loose}, PERCEPTION)
    assert got is not None
    np.testing.assert_array_equal(got.masks["overhead"], target)


def test_robot_pixels_are_never_lifted() -> None:
    box = Box((0.0, 0.0, TABLE_Z + 0.03), (0.06, 0.06, 0.06))
    sphere = np.array([[0.02, 0.0, TABLE_Z + 0.06, 0.03]])  # a "finger" over the box's top
    clean, index = scene([box])
    filtered, _ = scene([box], spheres=sphere)
    assert not clean.view("overhead").robot.any()
    robot = filtered.view("overhead").robot
    assert robot.any() and not (robot & (index["overhead"] != 0)).any()
    mask = index["overhead"] == 0
    got = lift(filtered, {"overhead": mask}, PERCEPTION)
    assert got is not None and not (got.masks["overhead"] & robot).any()
    assert filtered.robot is not None
    np.testing.assert_array_equal(filtered.robot.spheres, sphere)
    np.testing.assert_array_equal(
        filtered.view("overhead").usable, clean.view("overhead").usable & ~robot
    )
