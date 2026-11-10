"""Grasp synthesis's mirror on RoboTwin's own grasp directions."""

import math

import numpy as np

from roboscript.manipulation import mirror
from roboscript.manipulation.posemath import axangle
from roboscript.spatial.rotation import quat_angle_deg, quat_from_matrix, quat_to_matrix


def rigid(rotation: np.ndarray, position: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def test_the_mirror_maps_robotwins_left_grasp_directions_onto_the_right_arms() -> None:
    # Gripper orientations, x the approach.
    pairs = [
        ((-0.61239, 0.353523, -0.61239, -0.353524), (-0.353523, 0.61239, -0.353524, -0.61239)),
        ((-0.924, 0, 0, -0.383), (-0.383, 0, 0, -0.924)),
        ((-0.853532, 0.146484, -0.353542, -0.3536), (-0.353518, 0.353564, -0.14642, -0.853568)),
    ]
    to_contact = np.array([[0.0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    world_x = rigid(axangle([0, 0, 1], math.pi / 2), (0, 0, 0))  # a frame whose x-z plane is x=0
    for left, right in pairs:
        grasp = rigid(quat_to_matrix(left), (0.0, 0.0, 0.0))
        contact = grasp @ np.linalg.inv(to_contact)
        back = mirror(contact, world_x) @ to_contact
        assert quat_angle_deg(quat_from_matrix(back[:3, :3]), right) < 0.05
