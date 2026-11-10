"""Waypoint selection and straight-line sampling, on a robot whose joints are its TCP position
(forward kinematics is the identity, so the TCP along a joint line is the chord itself)."""

import itertools
import math

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.manipulation.waypoints import line, polyline_distance, select
from roboscript.spatial.rotation import quat_angle_deg, quat_from_axis_angle

calls: list[int] = []


async def identity(joints: np.ndarray) -> np.ndarray:
    calls.append(len(joints))
    return joints


def arc(radius: float, degrees: float, n: int) -> np.ndarray:
    angles = np.radians(np.linspace(0.0, degrees, n))
    return np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)], axis=1)


async def test_a_curved_path_gets_the_waypoints_its_chords_need() -> None:
    calls.clear()
    path = arc(0.1, 90.0, 500)
    chosen = await select(path, path, identity, 0.003)
    # A chord deviates from a 0.1 m arc by 0.1 (1 - cos(a / 2)): 3 mm allows 28 degrees.
    assert chosen[-1] == 499 and len(chosen) == 4, chosen
    points = path[[0, *chosen]]
    steps = np.linspace(0, 1, 200)[:, None]
    chords = np.concatenate([a + steps * (b - a) for a, b in itertools.pairwise(points)])
    assert polyline_distance(chords, path).max() <= 0.003
    assert len(calls) == 1 + 2 * 3 and max(calls) <= 4096  # the end, then a grid per waypoint


def test_a_line_is_sampled_in_steps_of_distance_and_angle() -> None:
    down = quat_from_axis_angle([0.0, 1.0, 0.0], math.pi / 2)
    start = Pose(p=(0.0, 0.0, 1.0), q=tuple(down))
    ahead = Pose(p=(0.0, 0.0, 0.9), q=tuple(down))
    heights = [pose.p[2] for pose in line(start, ahead, 0.02, 5.0, 64)]
    assert heights == pytest.approx([0.98, 0.96, 0.94, 0.92, 0.90])
    yawed = quat_from_axis_angle([0.0, 0.0, 1.0], math.radians(30))
    turned = Pose(p=(0.0, 0.0, 0.99), q=tuple(yawed))
    poses = line(start, turned, 0.02, 5.0, 64)
    steps = [quat_angle_deg(a.q, b.q) for a, b in itertools.pairwise([start, *poses])]
    assert len(poses) == math.ceil(float(quat_angle_deg(start.q, turned.q)) / 5.0)
    assert max(steps) <= 5.0 + 1e-9
    assert poses[-1].as_list() == pytest.approx(turned.as_list())
    assert len(line(start, Pose(p=(0.0, 0.0, -1.0), q=start.q), 0.02, 5.0, 64)) == 64
