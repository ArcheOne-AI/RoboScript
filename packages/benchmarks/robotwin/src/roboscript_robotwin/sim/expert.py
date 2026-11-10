"""Tracing the expert's official API calls during a pre-run (ground truth: control plane only).

`Recorder.install(task)` wraps, on the task instance only, the data-generation API builders (to
name the API that built each action list and keep its arguments, the task objects' true poses then
and, for `place_actor`, the object's pose once placed), `move` (each call with its per-arm action
lists), the calls that bypass it (`delay`, `together_open_gripper`, `together_close_gripper`), the
two that run `move`'s pairs (`take_dense_action`, `together_move_to_pose`: which pairs ran) and the
scene's `step` (to count simulation steps); `uninstall` removes every wrapper before the task is
set up again. After each call it reads the robot with `state` (`shim.robot_state`): the measured
finger openings, and the commanded joints that tell which arm of a refused pair still moved.
RoboTwin's files are never touched.

`Base_Task.move` pads the shorter action list with None in place (a list passed again comes back
padded: None is an idle arm in that pair), runs the k-th actions of both lists as pair k, and
returns at the first pair whose plan fails. A call the planner refused before anything ran is
`refused`; one refused after some pairs ran is `partial` and keeps only what ran: the pairs
before, and of the failing pair the arm that moved (a pair of two moves runs the arm whose plan
succeeded).
"""

import inspect
from collections.abc import Callable
from typing import Any, Final, Literal

import numpy as np
from pydantic import JsonValue

from roboscript.contracts.env import ExpertAction, ExpertBuilt, ExpertCall
from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm, RobotState
from roboscript.spatial.pose import pose_from_matrix
from roboscript.spatial.rotation import quat_normalize
from roboscript_robotwin.sim import referee

BUILDERS: Final = (
    "grasp_actor",
    "place_actor",
    "move_by_displacement",
    "move_to_pose",
    "open_gripper",
    "close_gripper",
    "back_to_origin",
)
LOCK: Final = [1, 1, 1, 0, 0, 0]
"""`constraint_pose` of an orientation lock, the only constraint RoboTwin's API sets."""
_BYPASSES: Final[dict[str, Literal["hold", "grippers"]]] = {
    "delay": "hold",
    "together_open_gripper": "grippers",
    "together_close_gripper": "grippers",
}
"""Calls that move the robot without `move`, and the kind of call each records."""
_PAIRS: Final = ("take_dense_action", "together_move_to_pose")
_ARMS: Final[tuple[Arm, Arm]] = ("left", "right")
CORE_NAMES: Final = {
    "grasp_actor": "grasp_object",
    "place_actor": "place_object",
    "move_by_displacement": "move_by",
    "back_to_origin": "return_home",
}
"""The official builders' names as a trace reports them: the core's. Everything above the
simulator side sees only these (`ExpertCall.api`); the others are the same in both."""
CORE_ARGS: Final = {
    "grasp_actor": {"pre_grasp_dis": "approach_m", "grasp_dis": "grasp_offset_m"},
    "place_actor": {
        "pre_dis": "approach_m",
        "dis": "stop_back_m",
        "pre_dis_axis": "approach_axis",
        "actor_axis": "object_axis",
        "actor_axis_type": "object_axis_type",
        "functional_point_id": "placement_frame",
    },
}
"""The official builders' arguments as a trace reports them (`ExpertBuilt.args`)."""
CORE_VALUES: Final = {"object_axis_type": {"actor": "object"}, "approach_axis": {"fp": "target_z"}}


class Recorder:
    """The calls of one pre-run: `calls` fills while installed on a task set up for it."""

    def __init__(self, state: Callable[[Any], RobotState]) -> None:
        self.state = state
        self.calls: list[ExpertCall] = []
        self._task: Any = None
        self._built: dict[int, tuple[Any, str, ExpertBuilt]] = {}  # id -> (list, API, how)
        self._steps = 0
        self._inside = False  # `delay` calls `together_close_gripper` itself
        self._pairs: list[set[Arm]] | None = None  # while in `move`: the arms each pair ran

    def install(self, task: Any) -> None:
        self.calls, self._built, self._steps, self._inside, self._task = [], {}, 0, False, task
        step = task.scene.step

        def counted(*args: Any, **kwargs: Any) -> Any:
            self._steps += 1
            return step(*args, **kwargs)

        task.scene.step = counted
        for name in BUILDERS:
            setattr(task, name, self._builder(getattr(task, name), name))
        task.move = self._move(task.move)
        for name, kind in _BYPASSES.items():
            setattr(task, name, self._bypass(getattr(task, name), kind))
        task.take_dense_action = self._pair(task.take_dense_action, together=False)
        task.together_move_to_pose = self._pair(task.together_move_to_pose, together=True)

    def uninstall(self) -> None:
        if self._task is None:
            return
        for name in (*BUILDERS, "move", *_BYPASSES, *_PAIRS):
            vars(self._task).pop(name, None)
        del self._task.scene.step
        self._task = None

    def _builder(self, build: Callable[..., Any], api: str) -> Callable[..., Any]:
        signature = inspect.signature(build)

        def built(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            how = self._how(api, dict(bound.arguments))
            actions = build(*args, **kwargs)
            self._built[id(actions)] = (actions, api, how)  # kept alive: its id is not reused
            return actions

        return built

    def _how(self, api: str, arguments: dict[str, Any]) -> ExpertBuilt:
        """A builder call's arguments (the actor by its task-object name) and ground truth."""
        objects = referee.task_objects(self._task)
        names = {id(actor): name for name, actor in reversed(list(objects.items()))}
        actor, arm = arguments.pop("actor", None), arguments.pop("arm_tag", None)
        arguments.update(arguments.pop("args", {}))  # place_actor's **args
        placed = None
        if api == "place_actor" and actor is not None and self._task.plan_success:
            placed = self._placed(actor, arm, arguments)
        return ExpertBuilt(
            obj=None if actor is None else names.get(id(actor)),
            args=core_args(api, arguments),
            objects=referee.poses(objects),
            links=referee.link_poses(objects),
            placed=placed,
        )

    def _placed(self, actor: Any, arm: Any, arguments: dict[str, Any]) -> Pose:
        """Where `place_actor` would leave `actor`: carried by the ee to the official placement
        at `pre_dis` 0 (`get_place_pose` reads the state and plans nothing)."""
        passed = {k: v for k, v in arguments.items() if k not in ("dis", "is_open", "pre_dis")}
        goal = self._task.get_place_pose(actor, arm, **passed, pre_dis=0.0)
        ee = getattr(self._task.robot, f"get_{arm}_ee_pose")()
        now = actor.get_pose().to_transformation_matrix().astype(float)
        m = Pose.from_list(goal).matrix() @ np.linalg.inv(Pose.from_list(ee).matrix()) @ now
        return Pose.from_list(pose_from_matrix(m))

    def _move(self, move: Callable[..., Any]) -> Callable[..., Any]:
        def recorded(first: Any, second: Any = None, save_freq: Any = -1) -> Any:
            groups = [g for g in (first, second) if g is not None and g[0] is not None]
            entries = {_arm(g[0]): self._entry(g) for g in groups}
            apis: dict[Arm, str] = {arm: api for arm, (api, _) in entries.items()}
            how = {arm: built for arm, (_, built) in entries.items() if built is not None}
            if not self._task.plan_success:  # `move` returns at once; targets may be placeholders
                self.calls.append(ExpertCall(kind="move", sim_steps=0, refused=True))
                return move(first, second, save_freq=save_freq)
            # Read before `move` pads the lists; each action with the pair it belongs to.
            pairs = {
                _arm(g[0]): [(k, _action(a)) for k, a in enumerate(g[1]) if a is not None]
                for g in groups
            }
            steps, self._pairs = self._steps, []
            done = move(first, second, save_freq=save_freq)
            ran, self._pairs = self._pairs, None
            taken = self._steps - steps
            if self._task.plan_success or taken == 0:
                refused = not self._task.plan_success
                actions = {arm: [a for _, a in listed] for arm, listed in pairs.items()}
                self._record("move", actions, taken, apis, how, refused=refused)
                return done
            kept = {
                arm: [a for k, a in listed if k < len(ran) and arm in ran[k]]
                for arm, listed in pairs.items()
            }
            kept = {arm: listed for arm, listed in kept.items() if listed}
            kept_how = {arm: built for arm, built in how.items() if arm in kept}
            apis = {arm: apis[arm] for arm in kept}
            self._record("move", kept, taken, apis, kept_how, partial=True)
            return done

        return recorded

    def _pair(self, run: Callable[..., Any], *, together: bool) -> Callable[..., Any]:
        """Notes which arms a pair of `move` ran: both, or after a refused plan of a two-move
        pair (which still runs the other arm), the arm whose commanded joints changed."""

        def recorded(*args: Any, **kwargs: Any) -> Any:
            if self._pairs is None:  # a gripper call outside `move`
                return run(*args, **kwargs)
            before = self.state(self._task) if together else None
            done = run(*args, **kwargs)
            if before is None or self._task.plan_success:
                self._pairs.append(set(_ARMS))
            else:
                after = self.state(self._task)
                self._pairs.append(
                    {
                        arm
                        for arm in _ARMS
                        if not np.allclose(before.arms[arm].joints_cmd, after.arms[arm].joints_cmd)
                    }
                )
            return done

        return recorded

    def _bypass(
        self, call: Callable[..., Any], kind: Literal["hold", "grippers"]
    ) -> Callable[..., Any]:
        def recorded(*args: Any, **kwargs: Any) -> Any:
            if self._inside:
                return call(*args, **kwargs)
            self._inside, steps = True, self._steps
            done = call(*args, **kwargs)
            self._inside = False
            actions: dict[Arm, list[ExpertAction]] = {}
            if kind == "grippers":
                bound = inspect.signature(call).bind(*args, **kwargs)
                bound.apply_defaults()
                actions = {
                    arm: [ExpertAction(gripper=float(bound.arguments[f"{arm}_pos"]))]
                    for arm in _ARMS
                }
            self._record(kind, actions, self._steps - steps)
            return done

        return recorded

    def _record(
        self,
        kind: Literal["move", "grippers", "hold"],
        actions: dict[Arm, list[ExpertAction]],
        steps: int,
        api: dict[Arm, str] | None = None,
        how: dict[Arm, ExpertBuilt] | None = None,
        *,
        refused: bool = False,
        partial: bool = False,
    ) -> None:
        arms = self.state(self._task).arms if actions else {}
        openings = {arm: opening for arm in actions if (opening := arms[arm].gripper) is not None}
        self.calls.append(
            ExpertCall(
                kind=kind,
                actions=actions,
                api={arm: CORE_NAMES.get(name, name) for arm, name in (api or {}).items()},
                sim_steps=steps,
                refused=refused,
                partial=partial,
                openings=openings,
                built=how or {},
            )
        )

    def _entry(self, group: Any) -> tuple[str, ExpertBuilt | None]:
        """The API that built an action list and how, or `raw` for one the task built itself."""
        entry = self._built.pop(id(group), None)
        return (entry[1], entry[2]) if entry is not None and entry[0] is group else ("raw", None)


def core_args(api: str, arguments: dict[str, Any]) -> dict[str, JsonValue]:
    """An official builder's arguments under the core's names and values (`CORE_ARGS`)."""
    names = CORE_ARGS.get(api, {})
    args: dict[str, JsonValue] = {}
    for key, value in arguments.items():
        name = names.get(key, key)
        shown = jsonable(value)
        if isinstance(shown, str):
            shown = CORE_VALUES.get(name, {}).get(shown, shown)
        args[name] = shown
    return args


def _arm(tag: Any) -> Arm:
    """RoboTwin's `ArmTag` (or its string) as an `Arm`."""
    arm = str(tag)
    if arm not in _ARMS:
        raise ValueError(f"an expert action for an unknown arm {arm!r}")
    return arm  # `in _ARMS` narrowed it


def _action(action: Any) -> ExpertAction:
    """RoboTwin's `Action` (`envs/utils/action.py`) as an `ExpertAction`. Its quaternion is
    normalized, as RoboTwin's planning does: tasks write some by hand, and those can have a
    norm away from 1."""
    if action.action != "move":
        return ExpertAction(gripper=float(action.target_gripper_pos))
    constraint = action.args.get("constraint_pose")
    if constraint is not None and list(constraint) != LOCK:
        raise ValueError(f"an expert move with an unknown constraint_pose {constraint}")
    target = np.asarray(action.target_pose, dtype=float)
    pose = Pose.from_list([*target[:3], *quat_normalize(target[3:])])
    return ExpertAction(target=pose, constraint="free" if constraint is None else "orientation")


def jsonable(value: Any) -> JsonValue:
    """An API argument as JSON: poses (SAPIEN's, arrays, lists) as lists, arm tags as names."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()  # type: ignore[no-any-return]
    if hasattr(value, "p") and hasattr(value, "q"):  # sapien.Pose
        return [float(v) for v in (*value.p, *value.q)]
    if isinstance(value, np.ndarray | list | tuple):
        return [jsonable(v) for v in (value.tolist() if isinstance(value, np.ndarray) else value)]
    return str(value)  # ArmTag
