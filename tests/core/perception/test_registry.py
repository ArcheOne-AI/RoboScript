"""perception.registry: registration, a held object moving with its TCP, and the JSON round trip."""

import math

import numpy as np
import pytest

from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import FrameId
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.manipulation.posemath import axangle
from roboscript.perception.registry import (
    SceneRegistry,
)
from roboscript.spatial.pose import pose_compose
from roboscript.spatial.rotation import (
    quat_to_matrix,
)
from roboscript.testing.scenes import TABLE_Z, Box, robot_state, surface, table_model

F1, F2 = FrameId("frame-1"), FrameId("frame-2")
HOLD = load_embodiment(MOCK_BIMANUAL).arms["left"].gripper.holding_threshold
"""The robot's holding threshold, from its embodiment profile."""
TABLE = table_model()


def upper(points: np.ndarray) -> np.ndarray:
    """What cameras see of a resting shape's surface: everything but its underside."""
    return points[points[:, 2] > points[:, 2].min() + 1e-6]


def block(xy: tuple[float, float], size: float = 0.04, bottom: float = TABLE_Z) -> np.ndarray:
    return upper(surface(Box((*xy, bottom + size / 2), (size, size, size))))


def yawed(
    points: np.ndarray, degrees: float, about: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    out = points.copy()
    xy = points[:, :2] - about
    out[:, 0], out[:, 1] = (
        c * xy[:, 0] - s * xy[:, 1] + about[0],
        s * xy[:, 0] + c * xy[:, 1] + about[1],
    )
    return out


def hammer(yaw: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Handle and head points of a hammer lying on the table, the head toward +x (turned by
    `yaw` degrees about the table origin)."""
    handle = surface(Box((0.0, 0.0, TABLE_Z + 0.01), (0.16, 0.02, 0.02)))
    head = surface(Box((0.1, 0.0, TABLE_Z + 0.02), (0.04, 0.06, 0.04)))
    return yawed(handle, yaw), yawed(head, yaw)


def test_register_fits_and_numbers_objects() -> None:
    registry = SceneRegistry()
    first = registry.register(
        "red block",
        surface(Box((0.1, 0.0, TABLE_Z + 0.025), (0.05,) * 3)),
        frame=F1,
        step=2,
        table_z=TABLE_Z,
    )
    assert first.id == "o1" and first.name == "red block" and first.frame == F1 and first.step == 2
    assert first.fit.shape == "box" and first.confidence == 1.0 and first.held_by is None
    assert first.pose == first.fit.frame and len(first.history) == 1
    np.testing.assert_allclose(first.pose.p, [0.1, 0.0, TABLE_Z], atol=1e-3)  # bottom center
    second = registry.register(
        "mat",
        surface(Box((0, 0.2, TABLE_Z), (0.1, 0.1, 0.0))),
        frame=F1,
        step=2,
        shape="plane",
        confidence=0.5,
    )
    assert second.id == "o2" and second.fit.shape == "plane" and second.confidence == 0.5
    assert registry.get("o1") is first
    with pytest.raises(
        LookupError, match="no registered object 'o9'; registered: \\['o1', 'o2'\\]"
    ):
        registry.get("o9")


def test_a_held_object_moves_with_its_tcp() -> None:
    block = surface(Box((0.1, 0.0, TABLE_Z + 0.02), (0.04, 0.04, 0.04)))
    registry = SceneRegistry()
    obj = registry.register("block", block, frame=F1, step=1, table_z=TABLE_Z, center_on_close=True)
    # The TCP 4 mm off center along the closing axis (TCP y is world y: the gripper points down).
    closed = robot_state(step=1, gripper=0.5, tcp={"left": (0.1, 0.004, TABLE_Z + 0.02)})
    held = registry.attach(obj.id, "left", closed, closing_axis=(0.0, 1.0, 0.0))
    assert held.held_by == "left" and list(held.grasps) == ["left"]
    assert held.taken and not obj.taken  # for good: a surface change does not tell it apart
    assert held.grasps["left"].p[1] == pytest.approx(0.0, abs=1e-12)  # centered by the fingers
    with pytest.raises(ValueError, match="held by the left arm: give the robot state"):
        registry.geometry(obj.id)
    robot = robot_state(step=7, gripper=0.5, tcp={"left": (0.1, 0.004, TABLE_Z + 0.12)})
    now = registry.geometry(obj.id, robot)
    assert now.held_by == ("left",) and now.stale_age == 6
    np.testing.assert_allclose(now.center, [0.1, 0.004, TABLE_Z + 0.12], atol=1e-3)
    moved = now.points.mean(axis=0) - block.mean(axis=0)
    np.testing.assert_allclose(moved, [0, 0.004, 0.1], atol=1e-3)
    with pytest.raises(ValueError, match="held by the left arm: give the robot state"):
        registry.remeasure(obj.id, block, frame=F2, step=7)
    refined = registry.refresh(
        obj.id, np.add(block, [0.0, 0.004, 0.1]), frame=F2, step=7, robot=robot
    )
    assert refined.held_by == "left"  # still held; the grasp transform follows the measurement
    placed = robot_state(step=9, gripper=1.0, tcp={"left": (0.3, -0.1, TABLE_Z + 0.02)})
    released = registry.detach(obj.id, "left", placed)
    assert released.held_by is None and released.grasps == {}
    tcp = placed.arms["left"].tcp.as_list()
    grasp = refined.grasps["left"].as_list()
    np.testing.assert_allclose(released.pose.as_list(), pose_compose(tcp, grasp))
    center = registry.geometry(obj.id).center
    np.testing.assert_allclose(center, [0.3, -0.1, TABLE_Z + 0.02], atol=1e-3)
    with pytest.raises(ValueError, match="o1 is not held by the left arm"):
        registry.detach(obj.id, "left", placed)


def test_the_registry_round_trips_through_json() -> None:
    handle, head = hammer(30)
    registry = SceneRegistry()
    obj = registry.register("hammer", np.vstack([handle, head]), frame=F1, step=0, table_z=TABLE_Z)
    registry.register_part(obj.id, "head", head)
    registry.register_part(obj.id, "handle", handle, x_toward="head")
    registry.attach(obj.id, "right", robot_state(tcp={"right": (0.0, 0.0, TABLE_Z + 0.02)}))
    text = registry.model_dump_json()
    back = SceneRegistry.model_validate_json(text)
    assert back.model_dump_json() == text
    assert back.counter == 1 and list(back.objects) == ["o1"]
    got, want = back.get("o1"), registry.get("o1")
    np.testing.assert_allclose(got.points, want.points, atol=1e-5)
    np.testing.assert_allclose(got.parts["handle"].points, want.parts["handle"].points, atol=1e-5)
    assert got.parts["handle"].pose == want.parts["handle"].pose
    assert got.fit == want.fit and got.grasps == want.grasps and got.held_by == "right"


def boxed(center: tuple[float, float, float], size: tuple[float, float, float]) -> np.ndarray:
    return upper(surface(Box(center, size)))


def laptop(registry: SceneRegistry) -> str:
    """A flat base with a closed lid on it, the lid registered as an articulated part."""
    base = boxed((0.0, 0.1, TABLE_Z + 0.01), (0.3, 0.2, 0.02))
    lid_points = boxed((0.0, 0.1, TABLE_Z + 0.025), (0.3, 0.2, 0.01))
    record = registry.register("laptop", np.vstack([base, lid_points]), frame=F1, step=0)
    lid = lid_points[lid_points[:, 2] > TABLE_Z + 0.022]
    registry.register_part(record.id, "lid", lid, articulated=True)
    return record.id


def turned_about(point: np.ndarray, axis: np.ndarray, degrees: float, tcp: np.ndarray) -> Pose:
    """The TCP (pointing down) at `tcp` turned rigidly about a line."""
    rotation = axangle(axis, math.radians(degrees))
    down = quat_to_matrix(robot_state().arms["left"].tcp.q)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation @ down, rotation @ (tcp - point) + point
    return Pose.from_matrix(m)
