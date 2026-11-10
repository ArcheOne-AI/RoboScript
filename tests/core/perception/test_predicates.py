"""perception.predicates and perception.relations over a scene of registered objects."""

import math

import numpy as np
import pytest

from roboscript.contracts.ids import FrameId
from roboscript.contracts.robot import RobotState
from roboscript.perception.predicates import PREDICATES, Scene, check
from roboscript.perception.registry import SceneRegistry
from roboscript.perception.relations import relation
from roboscript.testing.scenes import TABLE_Z, Box, Cylinder, robot_state, surface, table_model

F = FrameId("frame-1")
TABLE = table_model()


def scene_of(
    shapes: dict[str, Box | Cylinder], robot: RobotState | None = None
) -> tuple[Scene, SceneRegistry]:
    registry = SceneRegistry()
    for name, shape in shapes.items():
        registry.register(name, surface(shape), frame=F, step=0, table_z=TABLE_Z)
    robot = robot or robot_state(step=0)
    objects = {i: registry.geometry(i, robot) for i in registry.objects}
    return Scene(robot=robot, table=TABLE, objects=objects, records=registry.objects), registry


def gripper(robot: RobotState, opening: float | None, command: float) -> RobotState:
    arm = robot.arms["left"].model_copy(update={"gripper": opening, "gripper_cmd": command})
    return robot.model_copy(update={"arms": {**robot.arms, "left": arm}})


def test_holding_and_gripper_open() -> None:
    base = robot_state()
    cases = [(0.3, 0.0, True), (0.1, 0.0, False), (0.9, 1.0, False), (None, 0.0, None)]
    for opening, command, expected in cases:
        scene = Scene(robot=gripper(base, opening, command), table=TABLE, objects={}, records={})
        result = check(scene, "holding", arm="left", threshold=0.2)
        assert result.value is expected, (opening, command, result.reason)
        assert result.predicate == "holding" and result.measured["command"] == command
    scene = Scene(robot=gripper(base, 0.3, 0.0), table=TABLE, objects={}, records={})
    assert (
        check(scene, "holding", arm="left", threshold=0.2).reason
        == "measured opening 0.3000 >= 0.2000"
    )
    scene = Scene(robot=gripper(base, 0.2, 0.0), table=TABLE, objects={}, records={})
    assert check(scene, "holding", arm="left", threshold=0.2).value is True  # at: held
    opened = {"arm": "left", "fraction": 0.8, "full": 1.0}
    assert check(scene, "gripper_open", **opened).value is False
    scene = Scene(robot=gripper(base, 0.95, 1.0), table=TABLE, objects={}, records={})
    result = check(scene, "gripper_open", **opened)
    assert result.value is True and result.reason == "measured opening 95% of full (>= 80%)"
    scene = Scene(robot=gripper(base, None, 1.0), table=TABLE, objects={}, records={})
    assert check(scene, "gripper_open", **opened).value is None


def test_on_inside_beside_and_relations() -> None:
    scene, _ = scene_of(
        {
            "base": Box((0.0, 0.0, TABLE_Z + 0.025), (0.08, 0.08, 0.05)),
            "top": Box((0.01, 0.0, TABLE_Z + 0.07), (0.04, 0.04, 0.04)),
            "left": Box((-0.15, 0.02, TABLE_Z + 0.02), (0.04, 0.04, 0.04)),
            "basket": Box((0.3, 0.0, TABLE_Z + 0.05), (0.14, 0.14, 0.10)),
            "item": Box((0.31, 0.01, TABLE_Z + 0.02), (0.04, 0.04, 0.04)),
        }
    )
    base, top, left, basket, item = "o1", "o2", "o3", "o4", "o5"
    on = check(scene, "on", a=top, b=base, tol=0.015, min_overlap=0.5)
    assert on.value is True and on.measured["gap_z"] == pytest.approx(0.0, abs=1e-3)
    assert on.measured["overlap"] == 1.0
    assert check(scene, "on", a=left, b=base, tol=0.015, min_overlap=0.5).value is False
    assert check(scene, "inside", a=item, b=basket).value is True
    assert check(scene, "inside", a=top, b=base).value is False  # on it, not in it
    beside = check(scene, "beside", a=left, b=base, direction="left", dmin=0.1, dmax=0.2)
    assert beside.value is True
    assert beside.measured["along"] == pytest.approx(0.15, abs=2e-3)
    assert beside.measured["across"] == pytest.approx(0.02, abs=2e-3)
    assert (
        check(scene, "beside", a=left, b=base, direction="right", dmin=0.1, dmax=0.2).value is False
    )
    r = relation(scene.object(top), scene.object(base))
    assert r.on and r.above and not r.inside and r.center_over
    np.testing.assert_allclose(r.displacement, [0.01, 0.0, 0.045], atol=1e-3)
    s = relation(scene.object(left), scene.object(base))
    assert s.left_of and not s.in_front_of and not s.on and s.overlap == 0.0
    assert s.distance_xy == pytest.approx(math.hypot(0.15, 0.02), abs=2e-3)
    point = relation((0.0, 0.0, TABLE_Z + 0.2), scene.object(base))
    assert point.above and point.center_over and point.overlap == 1.0
    assert not relation(scene.object(base), (0.0, 0.0, TABLE_Z)).on  # a point has no footprint


def test_lifted_near_and_moved() -> None:
    shapes = {"block": Box((0.1, 0.0, TABLE_Z + 0.02), (0.04, 0.04, 0.04))}
    scene, registry = scene_of(shapes)
    assert check(scene, "lifted", obj="o1", height=0.05).value is False
    registry.attach(
        "o1", "left", robot_state(gripper=0.5, tcp={"left": (0.1, 0.0, TABLE_Z + 0.02)})
    )
    raised = robot_state(step=3, gripper=0.5, tcp={"left": (0.1, 0.0, TABLE_Z + 0.1)})
    now = Scene(
        robot=raised,
        table=TABLE,
        objects={"o1": registry.geometry("o1", raised)},
        records=registry.objects,
    )
    lifted = check(now, "lifted", obj="o1", height=0.05)
    assert lifted.value is True and lifted.measured["bottom_above_table"] == pytest.approx(
        0.08, abs=1e-3
    )
    near = check(now, "near", arm="left", obj="o1", distance=0.03)
    assert near.value is True and near.measured["distance"] == 0.0  # the TCP is inside the box
    far = check(now, "near", arm="right", obj="o1", distance=0.03)
    assert far.value is False and far.measured["distance"] > 0.2
    assert check(now, "moved", obj="o1", distance=0.01).value is True  # held and raised 8 cm
    registry.detach("o1", "left", raised)
    registry.refresh(
        "o1",
        surface(Box((0.14, 0.0, TABLE_Z + 0.02), (0.04, 0.04, 0.04))),
        frame=F,
        step=5,
        table_z=TABLE_Z,
    )
    later = Scene(
        robot=robot_state(step=5),
        table=TABLE,
        objects={"o1": registry.geometry("o1")},
        records=registry.objects,
    )
    moved = check(later, "moved", obj="o1", distance=0.01)
    assert moved.value is True and moved.measured["displacement"] == pytest.approx(0.04, abs=1e-3)
    assert (
        check(later, "moved", obj="o1", distance=0.01, since=5).value is False
    )  # since its last measurement
    assert check(later, "moved", obj="o1", distance=0.01, since=-1).value is None


def test_unknown_names() -> None:
    scene, _ = scene_of({"a": Box((0.0, 0.0, TABLE_Z + 0.02), (0.04, 0.04, 0.04))})
    with pytest.raises(LookupError, match="no predicate 'hold'; predicates: \\['beside'"):
        check(scene, "hold", arm="left")
    with pytest.raises(LookupError, match="no registered object 'o7'"):
        check(scene, "lifted", obj="o7", height=0.05)
    assert set(PREDICATES) == {
        "holding",
        "gripper_open",
        "lifted",
        "on",
        "inside",
        "beside",
        "near",
        "pressed",
        "moved",
        "yaw_aligned",
    }
