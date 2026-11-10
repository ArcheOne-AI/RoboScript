"""Receipts from synthetic plans and robot states: a hard failure outranks an earlier miss, what
did not execute is not checked, the holding rule at its threshold, and a look before a rejection
does not count as moving."""

import functools

import numpy as np
import pytest

from roboscript.contracts.action import ActionStep, StepState, Stopped
from roboscript.contracts.geometry import Pose
from roboscript.contracts.receipt import ActionReceipt
from roboscript.contracts.robot import ArmState, RobotState
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.manipulation import GripCheck, MoveCheck, Plan, receipt
from roboscript.manipulation.receipts import chained
from roboscript.perception.predicates import Scene, holding
from roboscript.perception.registry import SceneRegistry
from roboscript.testing.skills import MOCK_EXECUTOR

PROFILE = load_embodiment(MOCK_BIMANUAL)
DOWN = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)
START, TARGET = (0.0, 0.0, 1.0), (0.0, 0.0, 0.9)


def tcp(xyz: tuple[float, float, float]) -> Pose:
    return Pose(p=xyz, q=DOWN)


def state(
    step: int, left: tuple[float, float, float] = START, opening: float = 1.0, command: float = 1.0
) -> RobotState:
    def arm(xyz: tuple[float, float, float], grip: float, cmd: float) -> ArmState:
        ee = Pose(p=(xyz[0], xyz[1], xyz[2] + 0.1), q=DOWN)
        return ArmState(
            ee=ee,
            tcp=tcp(xyz),
            joints=[0.0] * 6,
            joints_cmd=[0.0] * 6,
            gripper=grip,
            gripper_cmd=cmd,
        )

    arms = {"left": arm(left, opening, command), "right": arm((0.3, 0.0, 1.0), 1.0, 1.0)}
    return RobotState(step=step, step_limit=40, arms=arms, qpos_cmd=[0.0] * 14)


def step(command: float = 1.0) -> ActionStep:
    qpos = [0.0] * 14
    qpos[6], qpos[13] = command, 1.0
    return ActionStep(qpos=qpos)


def move(end: int = 0, contact: bool = False) -> MoveCheck:
    path = np.array([START, TARGET])
    target = Pose(p=(0.0, 0.0, 1.0), q=DOWN)
    return MoveCheck("left", end, target, tcp(TARGET), path, contact)


def report(
    plan: Plan,
    states: list[RobotState],
    stopped: Stopped = "completed",
    indices: list[int] | None = None,
) -> ActionReceipt:
    numbered = zip(indices or range(len(states)), states, strict=True)
    steps = [StepState(index=i, robot=robot) for i, robot in numbered]
    return receipt(plan, steps, stopped, PROFILE, MOCK_EXECUTOR)


def test_a_hard_failure_outranks_an_earlier_miss() -> None:
    grip = GripCheck("left", 1, 0.0, grasp=True)
    plan = Plan(state(0), [step(), step(0.0)], [move()], [grip], ["left"])
    states = [
        state(1, left=(0.0, 0.0, 0.95)),
        state(2, left=(0.0, 0.0, 0.95), opening=0.0, command=0.0),
    ]
    failed = report(plan, states)
    assert failed.reason == "empty_grasp"
    assert failed.note == (  # the other issues' notes follow
        "left closed on nothing: the fingers settled at 0.000; "
        "left stopped 5.0 cm short of its target"
    )
    assert failed.commanded is not None and failed.commanded.tcp.p == pytest.approx(TARGET)


def test_what_did_not_execute_is_not_checked() -> None:
    plan = Plan(state(0), [step(), step()], [move(end=1)], arms=["left"])
    done = report(plan, [state(1, left=(0.0, 0.0, 0.97))], stopped="success")
    assert done.reason == "ok" and "success" not in done.note  # the episode's end says it
    limited = report(plan, [state(1, left=(0.0, 0.0, 0.97))], stopped="step_limit")
    assert limited.reason == "budget_exhausted" and limited.steps_left == 39
    ended = report(plan, [state(1, left=(0.0, 0.0, 0.97))], stopped="ended")
    assert ended.reason == "budget_exhausted"  # the benchmark ended it: no action counts
    assert (
        ended.note.startswith("the benchmark ended the episode") and "step limit" not in ended.note
    )


def test_one_holding_rule_at_the_threshold() -> None:
    """Held iff the settled opening is at or above the profile's `holding_threshold`: receipts,
    the `holding` predicate and the scene registry agree exactly at the threshold."""
    threshold = PROFILE.arms["left"].gripper.holding_threshold
    grip = GripCheck("left", 0, 0.0, grasp=True)
    plan = Plan(state(0), [step(0.0)], grips=[grip], arms=["left"])
    below = state(1, opening=threshold - 1e-6, command=0.0)
    at = state(1, opening=threshold, command=0.0)
    assert report(plan, [below]).reason == "empty_grasp"
    assert report(plan, [at]).reason == "ok"
    for robot, held, why in (
        (below, False, "the fingers closed on nothing"),
        (at, True, "nothing registered within 0.0 m"),  # past the rule, nothing to attach
    ):
        scene = Scene(robot=robot, table=None, objects={}, records={})  # type: ignore[arg-type]
        assert holding(scene, "left", threshold).value is held
        steps: list[dict[str, object]] = []
        SceneRegistry().on_close(
            "left", robot, threshold=threshold, reach=0.0, log=functools.partial(_record, steps)
        )
        assert steps[0]["reason"] == why


def _record(steps: list[dict[str, object]], name: str, **data: object) -> None:
    steps.append(data)


def test_a_look_before_a_rejection_does_not_count_as_moving() -> None:
    """`unmoved` allows the steps of a look (a camera move), summed through `chained`; any other
    step makes the rejection one that moved."""

    look = ActionReceipt(reason="ok", steps_used=4, look_steps=4)
    refused = ActionReceipt(reason="unreachable")
    assert chained(look, refused).unmoved
    assert chained(look, refused).look_steps == 4
    assert not chained(look, ActionReceipt(reason="unreachable", steps_used=2)).unmoved
    assert not ActionReceipt(reason="empty_grasp", steps_used=4, look_steps=4).unmoved
