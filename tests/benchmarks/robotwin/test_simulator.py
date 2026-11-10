"""`Simulator` logic on a fake task object, without RoboTwin: a failing `take_action` and the
fault it leaves on the episode, the recorder of the expert's API calls, the physics step counter,
and a seed whose expert pre-run fails to plan counted unsolvable."""

import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.errors import InfraError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import BatchId, TaskName
from roboscript.contracts.robot import ArmState, RobotState
from roboscript_robotwin.benchmark import ROBOTWIN, TASK_CONFIGS
from roboscript_robotwin.sim import official, referee, simulator
from roboscript_robotwin.sim.expert import Recorder
from roboscript_robotwin.sim.simulator import Simulator

TASK = TaskName("beat_block_hammer")
REST = [0.0] * 6 + [1.0]


class _Joint:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class _Entity:
    def __init__(self, names: list[str]) -> None:
        self.joints = [_Joint(name) for name in names]

    def get_active_joints(self) -> list[_Joint]:
        return self.joints

    def get_qpos(self) -> np.ndarray:
        return np.full(len(self.joints), 0.045)  # open fingers


class _Robot:
    """The accessors the shim reads, at rest with open grippers."""

    def __init__(self) -> None:
        arms = ROBOTWIN.profile().arms
        names = [*arms["left"].joints, arms["left"].gripper.joint, *arms["right"].joints]
        self.left_entity = self.right_entity = _Entity([*names, arms["right"].gripper.joint])

    def __getattr__(self, name: str) -> Any:
        if name.endswith("jointState"):
            return lambda: list(REST)
        if name.endswith("ee_pose"):
            return lambda: [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        if name.endswith("tcp_pose"):
            return lambda: [0.12, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        raise AttributeError(name)


class _Task:
    """A task object whose `take_action` fails on call `fail_at` or takes `seconds`."""

    def __init__(self, fail_at: int | None = None, seconds: float = 0.0) -> None:
        self.robot, self.take_action_cnt, self.step_lim, self.eval_success = _Robot(), 0, 400, False
        self.fail_at, self.seconds = fail_at, seconds
        self.closed: list[bool] = []

    def close_env(self, clear_cache: bool = False) -> None:
        self.closed.append(clear_cache)

    def take_action(self, action: np.ndarray, action_type: str) -> None:
        if self.take_action_cnt == self.fail_at:
            raise RuntimeError("CUDA error at /home/someone/private/file.cu")
        time.sleep(self.seconds)
        self.take_action_cnt += 1


def _simulator(task: _Task, monkeypatch: pytest.MonkeyPatch) -> Simulator:
    sim = Simulator(Path("/nonexistent"), TASK_CONFIGS["clean"], ROBOTWIN.profile(), TASK)
    sim._task, sim._episode = task, simulator._Episode({}, {}, {}, 0.74, None)
    for read in ("task_objects", "centers", "joint_positions", "joint_limits", "link_poses"):
        monkeypatch.setattr(referee, read, lambda _: {})  # RoboTwin's
    return sim


def _holds(n: int) -> ActionBatch:
    return ActionBatch(batch_id=BatchId(f"b{n}"), steps=[ActionStep(qpos=REST + REST)] * n)


def test_a_simulator_error_faults_the_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _simulator(_Task(fail_at=1), monkeypatch)
    result = sim.execute(_holds(3))
    assert (result.stopped, result.executed) == ("error", 1)  # the states so far
    assert result.error == "the simulator raised RuntimeError; see the simulator's log"
    for call in (sim.robot, lambda: sim.observe(["head_camera"]), lambda: sim.execute(_holds(1))):
        with pytest.raises(InfraError, match="sim_fault") as caught:
            call()
        assert "/home" not in str(caught.value)
    assert sim.judge()[-1] is True  # faulted
    assert sim.diagnose()[2] == ["the simulator faulted in this episode: an infrastructure error"]


class _Action:
    """RoboTwin's `Action`: a move with a target (and constraint) or a gripper command."""

    def __init__(self, target: list[float] | None = None, grip: float | None = None, **args: Any):
        self.action = "move" if target is not None else "gripper"
        self.target_pose, self.target_gripper_pos, self.args = target, grip, args


class _Scene:
    def step(self) -> None:
        pass


POSE = [0.1, -0.1, 0.9, 0.5, 0.5, 0.5, 0.5]
LOCK = [1, 1, 1, 0, 0, 0]


class _Expert:
    """A task whose `move` runs pairs as RoboTwin's does: two moves together, anything else
    densely, returning at the first refused plan (`refuse`: the (pair, arm) plans to refuse); a
    refused two-move pair still runs the other arm. Every executed arm move steps 7 times."""

    def __init__(self) -> None:
        self.scene, self.plan_success = _Scene(), True
        self.drive = {"left": [0.0] * 6, "right": [0.0] * 6}  # commanded joints
        self.refuse: set[tuple[int, str]] = set()

    def grasp_actor(self, actor: str, arm_tag: str) -> tuple[str, list[_Action]]:
        return arm_tag, [_Action(POSE), _Action(POSE, constraint_pose=LOCK), _Action(grip=0.0)]

    def move_by_displacement(self, arm_tag: str, z: float = 0.0) -> tuple[str, list[_Action]]:
        return arm_tag, [_Action([*POSE[:3], 0.65, -0.25, 0.25, 0.65])]  # norm 0.985

    place_actor = move_to_pose = back_to_origin = move_by_displacement

    def open_gripper(self, arm_tag: str, pos: float = 1.0) -> tuple[str, list[_Action]]:
        return arm_tag, [_Action(grip=pos)]

    close_gripper = open_gripper

    def move(self, first: Any, second: Any = None, save_freq: Any = -1) -> bool:
        if not self.plan_success:
            return False
        lists = {str(group[0]): group[1] for group in (first, second) if group is not None}
        n = max(len(listed) for listed in lists.values())
        for listed in lists.values():
            listed += [None] * (n - len(listed))  # RoboTwin pads the lists in place
        for k in range(n):
            pair = {arm: listed[k] for arm, listed in lists.items() if listed[k] is not None}
            moves = [arm for arm, action in pair.items() if action.action == "move"]
            refused = {arm for arm in moves if (k, arm) in self.refuse}
            if len(moves) == 2:
                self.together_move_to_pose(moves, refused)
            elif refused:
                self.plan_success = False
            else:
                self.take_dense_action(moves)
            if not self.plan_success:
                return False
        return True

    def together_move_to_pose(self, moves: list[str], refused: set[str]) -> None:
        self.plan_success = not refused
        for arm in moves:
            if arm not in refused:
                self._run(arm)

    def take_dense_action(self, moves: list[str]) -> None:
        for arm in moves:
            self._run(arm)
        self.scene.step()

    def _run(self, arm: str) -> None:
        self.drive[arm] = [q + 0.1 for q in self.drive[arm]]
        for _ in range(7):
            self.scene.step()

    def together_close_gripper(
        self, save_freq: Any = -1, left_pos: float = 0, right_pos: float = 0
    ) -> None:
        for _ in range(299):
            self.scene.step()
        self.take_dense_action([])  # outside `move`: no pair of a move call

    def together_open_gripper(
        self, save_freq: Any = -1, left_pos: float = 1, right_pos: float = 1
    ) -> None:
        self.take_dense_action([])

    def delay(self, delay_time: int, save_freq: Any = None) -> None:
        for _ in range(delay_time):
            self.together_close_gripper(left_pos=0.3, right_pos=1.0)

    def play_once(self) -> None:
        self.move(self.grasp_actor("hammer", "left"), ("right", [_Action(POSE)]))
        self.move(self.move_by_displacement("left", z=0.1))
        self.delay(2)
        self.together_open_gripper(right_pos=0.5)
        # Two arms move together; the right arm's locked approach is refused: the left arm still
        # runs its second move, and the call keeps only that.
        self.refuse = {(1, "right")}
        self.move(("left", [_Action(POSE), _Action(POSE)]), self.grasp_actor("block", "right"))
        self.plan_success, self.refuse = True, {(1, "left")}  # the expert takes another route
        self.move(self.grasp_actor("block", "left"))  # refused after the pre-grasp ran
        self.move(self.move_by_displacement("left", z=0.1))  # refused before anything ran


def robot(task: _Expert) -> RobotState:
    """The recorder's view of the fake: commanded joints, fixed finger openings."""
    here = Pose(p=(0.0, 0.0, 1.0), q=(1.0, 0.0, 0.0, 0.0))

    def arm(name: str, opening: float) -> ArmState:
        return ArmState(
            ee=here,
            tcp=here,
            joints=None,
            joints_cmd=task.drive[name],
            gripper=opening,
            gripper_cmd=0.0,
        )

    arms = {"left": arm("left", 0.31), "right": arm("right", 1.0)}
    return RobotState(step=0, step_limit=None, arms=arms, qpos_cmd=[0.0] * 14)


@pytest.fixture
def named(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake's task objects: its actors are their names."""
    monkeypatch.setattr(
        referee, "task_objects", lambda task: {"hammer": "hammer", "block": "block"}
    )
    monkeypatch.setattr(referee, "poses", lambda objects: {})
    monkeypatch.setattr(referee, "link_poses", lambda objects: {})


def test_the_recorder_traces_the_experts_api_calls(named: None) -> None:
    task, recorder = _Expert(), Recorder(robot)
    recorder.install(task)
    task.play_once()
    recorder.uninstall()
    calls = recorder.calls
    assert [(c.kind, c.api, c.sim_steps, c.refused, c.partial) for c in calls] == [
        ("move", {"left": "grasp_object", "right": "raw"}, 23, False, False),
        ("move", {"left": "move_by"}, 8, False, False),
        ("hold", {}, 600, False, False),
        ("grippers", {}, 1, False, False),
        ("move", {"left": "raw", "right": "grasp_object"}, 21, False, True),
        ("move", {"left": "grasp_object"}, 8, False, True),
        ("move", {}, 0, True, False),
    ]
    assert calls[0].openings == {"left": 0.31, "right": 1.0} and calls[2].openings == {}
    assert calls[0].built["left"].obj == "hammer" and set(calls[0].built) == {"left"}
    assert calls[1].built["left"].args == {"z": 0.1} and calls[1].built["left"].obj is None
    assert calls[4].built["right"].obj == "block"  # a partial call keeps how it was built
    [lifted] = calls[1].actions["left"]  # a literal quaternion, normalized
    assert lifted.target is not None and lifted.target.q == pytest.approx(
        (0.66, -0.254, 0.254, 0.66), abs=1e-3
    )
    pre, approach, close = calls[0].actions["left"]
    assert pre.target is not None and pre.target.as_list() == pytest.approx(POSE)
    assert (pre.constraint, approach.constraint, close.gripper) == ("free", "orientation", 0.0)
    opened = [*calls[3].actions["left"], *calls[3].actions["right"]]
    assert [action.gripper for action in opened] == [1.0, 0.5]
    # What ran of the partial calls: both left moves and the right pre-grasp; the pre-grasp.
    together, alone = calls[4].actions, calls[5].actions
    assert (len(together["left"]), len(together["right"])) == (2, 1)
    assert together["right"][0].constraint == "free" and len(alone["left"]) == 1
    assert "move" not in vars(task) and "delay" not in vars(task) and "step" not in vars(task.scene)
    assert "take_dense_action" not in vars(task)
    task.plan_success = True
    recorder.install(task)
    with pytest.raises(ValueError, match="unknown constraint_pose"):
        task.move(("left", [_Action(POSE, constraint_pose=[0, 0, 0, 1, 1, 1])]))


class _Episodes(_Expert, _Task):
    """A task object to open episodes on, with the fake expert's API."""

    def __init__(self) -> None:
        _Task.__init__(self)
        _Expert.__init__(self)

    def setup_demo(self, **kwargs: Any) -> None:
        self.plan_success, self.refuse = True, set()

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def get_obs(self) -> dict[str, Any]:
        return {}


def test_stepping_counts_the_scenes_physics_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _Episodes()
    sim = _simulator(task, monkeypatch)
    with sim.stepping() as stepped:
        task.move(task.grasp_actor("hammer", "left"))  # 2 dense moves of 8 steps, a grip of 1
        assert stepped() == 17
        task.scene.step()
        assert stepped() == 18
    assert "step" not in vars(task.scene)  # the class's own again
    task.scene.step()
    assert stepped() == 18


def test_an_error_after_a_failed_plan_makes_the_seed_unsolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """put_bottles_dustbin's `play_once` indexes the actions a grasp returns, and RoboTwin's
    `grasp_actor` returns none once a plan failed: that IndexError is the failed plan's (the
    seed is unsolvable), while the same error with every plan good is a fault and propagates."""
    create_actor = types.ModuleType("envs.utils.create_actor")

    class UnStableError(Exception):
        pass

    create_actor.UnStableError = UnStableError  # type: ignore[attr-defined]
    for name in ("envs", "envs.utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name, types.ModuleType(name)))
    monkeypatch.setitem(sys.modules, "envs.utils.create_actor", create_actor)

    class Script:
        def __init__(self, plan_success: bool) -> None:
            self.plan_success, self.closed = plan_success, False

        def setup_demo(self, **args: Any) -> None:
            pass

        def play_once(self) -> Any:
            actions: list[Any] = []  # what `grasp_actor` gives after a failed plan
            return actions[1]

        def close_env(self) -> None:
            self.closed = True

        def check_success(self) -> bool:
            return True

    failed = Script(plan_success=False)
    assert official.expert_prerun(failed, 7, {}, 0) is None and failed.closed
    with pytest.raises(IndexError):
        official.expert_prerun(Script(plan_success=True), 7, {}, 0)
