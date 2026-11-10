"""perception.robot: a rendered arm is filtered and nothing else."""

import math

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import FrameId
from roboscript.perception.robot import (
    RobotShape,
)
from roboscript.perception.views import frame_geometry
from roboscript.spatial.rotation import quat_from_axis_angle
from roboscript.testing.scenes import (
    HEAD,
    PERCEPTION,
    TABLE_Z,
    Box,
    Sphere,
    capture,
    pinhole,
)

URDF = """<robot name="toy">
  <link name="link1"/><link name="link6"/><link name="finger_a"/><link name="finger_b"/>
  <joint name="j6" type="revolute"><parent link="link1"/><child link="link6"/>
    <axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="ja" type="prismatic"><parent link="link6"/><child link="finger_a"/>
    <axis xyz="0 2 0"/><limit lower="0" upper="0.05"/></joint>
  <joint name="jb" type="prismatic"><parent link="link6"/><child link="finger_b"/>
    <axis xyz="0 -1 0"/><limit lower="0" upper="0.05"/></joint>
</robot>"""


BASE = Pose(p=(0.0, -0.6, 0.0), q=(1.0, 0.0, 0.0, 0.0))


def turned(p: tuple[float, float, float], degrees: float) -> Pose:
    return Pose(p=p, q=tuple(quat_from_axis_angle([0, 0, 1], math.radians(degrees)).tolist()))


def test_a_rendered_arm_is_filtered_and_nothing_else() -> None:
    # An "arm" of spheres reaching over the table next to a box; the filter uses the same spheres
    # with a 1 cm margin: every arm pixel goes, no box or table pixel does.
    arm = [
        Sphere((x, -0.1, TABLE_Z + 0.25 - 0.8 * (x + 0.3)), 0.03)
        for x in np.linspace(-0.3, -0.05, 8)
    ]
    box = Box((0.1, 0.0, TABLE_Z + 0.03), (0.06, 0.06, 0.06))
    camera = pinhole("overhead", HEAD)
    images, index = capture([*arm, box], [camera])
    spheres = np.array([[*s.center, s.radius] for s in arm])
    geometry = frame_geometry(
        FrameId("f"), 0, images, PERCEPTION, robot=RobotShape(spheres, margin=0.01)
    )
    robot = geometry.view("overhead").robot
    on_arm = (index["overhead"] >= 0) & (index["overhead"] < len(arm))
    assert on_arm.sum() > 1000
    assert (robot & on_arm).sum() == on_arm.sum()  # recall 1
    assert not (robot & (index["overhead"] == len(arm))).any()  # the box stays
    assert geometry.table.z == pytest.approx(TABLE_Z, abs=1e-4)
