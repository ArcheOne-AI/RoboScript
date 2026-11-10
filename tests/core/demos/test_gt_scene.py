"""The ground-truth scene: boxes measured from hull points and carried by true poses, moving
links as parts."""

import math

import numpy as np
import pytest

from roboscript.demos.gt_scene import (
    measure,
    measure_part,
    truth_proxy,
)
from roboscript.manipulation.posemath import axangle
from roboscript.testing.scenes import TABLE_Z, Box, surface


def rigid(rotation: np.ndarray, position: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, position
    return m


def test_a_measured_box_follows_the_true_pose() -> None:
    # An asset whose own x runs along the box's long side, turned 30 deg on the table.
    asset = rigid(axangle([0, 0, 1], math.radians(30)), (0.1, 0.05, TABLE_Z))
    local = surface(Box((0.0, 0.0, 0.02), (0.12, 0.05, 0.04)))
    world = local @ asset[:3, :3].T + asset[:3, 3]
    box = measure(asset, world, TABLE_Z)
    assert box.symmetry in ("pi", "none")
    proxy = truth_proxy("box", asset, box)
    assert proxy.pose[:3, 0] == pytest.approx(asset[:3, 0], abs=1e-3)  # x the asset's own way
    assert proxy.whole.size == pytest.approx([0.12, 0.05, 0.04], abs=2e-3)
    assert not proxy.whole.x_known  # a symmetric box: either end, though x follows the asset
    assert (proxy.pose @ proxy.whole.frame)[:3, 3] == pytest.approx(
        [0.1, 0.05, TABLE_Z + 0.02], abs=2e-3
    )
    # Lifted and turned by the hand: the same box, carried rigidly.
    held = rigid(axangle([0, 0, 1], math.radians(100)), (0.0, -0.1, 0.9)) @ asset
    carried = truth_proxy("box", held, box)
    assert carried.pose == pytest.approx(held @ np.linalg.inv(asset) @ proxy.pose)
    # The same box as an asset turned the other way: x follows the asset.
    turned = asset @ rigid(axangle([0, 0, 1], math.pi), (0, 0, 0))
    flipped = truth_proxy("box", turned, measure(turned, world, TABLE_Z))
    assert flipped.pose[:3, 0] == pytest.approx(-asset[:3, 0], abs=1e-3)


def test_a_link_box_follows_the_link() -> None:
    # A lid tilted 30 deg about the world's x, measured along its own axes.
    link = rigid(axangle([1, 0, 0], math.radians(30)), (0.0, 0.1, 0.8))
    local = surface(Box((0.0, -0.1, 0.005), (0.3, 0.2, 0.01)))
    world = local @ link[:3, :3].T + link[:3, 3]
    frame, size = measure_part(link, world)
    assert size == pytest.approx([0.3, 0.2, 0.01], abs=1e-6)
    assert frame[:3, 3] == pytest.approx([0.0, -0.1, 0.005], abs=1e-6)
    assert frame[:3, 2] == pytest.approx([0, 0, 1])  # the link axis nearest up
