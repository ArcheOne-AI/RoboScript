"""perception.table: the plane and far edge the overhead view shows, and no table in view."""

import numpy as np
import pytest

from roboscript.contracts.ids import FrameId
from roboscript.perception.config import TableConfig
from roboscript.perception.table import estimated, fit_table
from roboscript.perception.views import frame_geometry
from roboscript.testing.scenes import (
    HEAD,
    PERCEPTION,
    TABLE_HALF,
    TABLE_Z,
    Box,
    Cylinder,
    capture,
    pinhole,
)

HEAD_CAMERA = pinhole("overhead", HEAD)
"""High above the table: sees all of it and the floor around it."""
NARROW = pinhole("overhead", HEAD, width=400)
"""The same camera cropped to its middle: the table's sides run out of the image."""

TABLE = PERCEPTION.table


def table_points(z: float = TABLE_Z, tilt_deg: float = 0.0, step: float = 0.004) -> np.ndarray:
    x, y = np.meshgrid(
        np.arange(-TABLE_HALF[0], TABLE_HALF[0], step),
        np.arange(-TABLE_HALF[1], TABLE_HALF[1], step),
    )
    heights = z + np.tan(np.radians(tilt_deg)) * x
    return np.stack([x, y, heights], axis=-1).reshape(-1, 3)


def test_the_overhead_sees_the_plane_and_only_the_far_edge() -> None:
    shapes = [
        Box((0.1, 0.0, TABLE_Z + 0.03), (0.2, 0.1, 0.06)),
        Cylinder((-0.2, 0.1, TABLE_Z + 0.1), 0.04, 0.2),
    ]
    images, _ = capture(shapes, [NARROW])
    table = frame_geometry(FrameId("f"), 0, images, PERCEPTION, robot_base=[0.0, -0.6, 0.0]).table
    assert table.method == "histogram"
    assert table.z == pytest.approx(TABLE_Z, abs=1e-4)
    assert table.rms < 5e-4 and table.inliers > 100_000
    assert np.degrees(np.arccos(table.normal[2])) < 0.1
    x0, x1, y0, _ = table.extent
    assert x0 < -0.3 and x1 > 0.3 and y0 > -0.3  # the near side ends at the image's bottom
    # The sides run out of the image and the near edge is below it: only the far edge is known,
    # so the landmarks are estimated: x on the robot's midline, the near side at the extent.
    assert table.edges["y1"] == pytest.approx(TABLE_HALF[1], abs=3e-3)
    assert [table.edges[name] for name in ("x0", "x1", "y0")] == [None, None, None]
    middle = (y0 + table.edges["y1"]) / 2
    assert table.landmarks["table_center"] == pytest.approx((0.0, middle, table.z))
    assert table.landmarks["front_edge_mid"] == pytest.approx((0.0, y0, table.z))
    assert table.landmarks["robot_midline"] == pytest.approx((0.0, middle, table.z))
    assert estimated(table) == ["front_edge_mid", "robot_midline", "table_center"]
    assert table.side("y1") == table.edges["y1"] and table.side("y0") == y0
    assert table.over([0.0, 0.3]) and not table.over([0.0, 0.45])
    assert table.over([2.0, -2.0])  # an unknown edge bounds nothing
    np.testing.assert_allclose(table.height([[0.0, 0.0, TABLE_Z + 0.05]]), 0.05, atol=1e-4)
    assert table.plane_z([0.3, 0.1]) == pytest.approx(table.z, abs=1e-5)


def test_no_table_in_view() -> None:
    with pytest.raises(ValueError, match="no table in view"):
        fit_table(table_points(z=1.2), TABLE)
    with pytest.raises(ValueError, match="lower < upper"):
        TableConfig(z_range=(0.8, 0.6))
    with pytest.raises(ValueError, match="xy_box is unset"):
        fit_table(table_points(), TableConfig())  # a scene's binding sets where the table is
