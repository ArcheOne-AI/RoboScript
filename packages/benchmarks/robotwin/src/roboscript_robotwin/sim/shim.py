"""The only code that reads or drives a RoboTwin task object for the agent plane
(docs/concepts/embodied-core.md and docs/architecture.md): camera
frames from `get_obs()`, proprioception from the robot's own accessors, and `take_action`. Nothing
here sees an object pose, a segmentation or a success check. The env server (`simulator`) and P3's
`eval()` (`roboscript_robotwin.eval.policy`) run batches through the same `run_batch`.

`ALLOWED` lists every attribute chain this module uses on the task object (`task.<chain>`);
`tests/benchmarks/robotwin` parses this file and fails on any other use of `task`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from roboscript.contracts.action import ActionBatch, ExecuteResult, StepState
from roboscript.contracts.geometry import Pose
from roboscript.contracts.robot import Arm, ArmState, RobotState
from roboscript.embodiments import EmbodimentProfile
from roboscript_robotwin.frames import Frame

log = logging.getLogger(__name__)

ALLOWED: Final = frozenset(
    {
        ("get_obs",),
        ("get_instruction",),
        ("take_action",),
        ("take_action_cnt",),
        ("step_lim",),
        ("eval_success",),
        # Our own flag on the episode (`mark_infra`), not RoboTwin state.
        ("rs_infra",),
        # What get_obs reports as joint_action (drive targets and gripper commands) and endpose.
        ("robot", "get_left_arm_jointState"),
        ("robot", "get_right_arm_jointState"),
        ("robot", "get_left_ee_pose"),
        ("robot", "get_right_ee_pose"),
        # Measured joints and fingers, and the TCP (from link poses).
        ("robot", "get_left_arm_real_jointState"),
        ("robot", "get_right_arm_real_jointState"),
        ("robot", "get_left_tcp_pose"),
        ("robot", "get_right_tcp_pose"),
        ("robot", "left_entity", "get_qpos"),
        ("robot", "left_entity", "get_active_joints"),
        ("robot", "right_entity", "get_qpos"),
        ("robot", "right_entity", "get_active_joints"),
    }
)


@dataclass(frozen=True)
class _Reading:
    command: list[float]  # joint drive targets, then the normalized gripper command
    measured: list[float]  # measured joints, then (again) the gripper command
    ee: list[float]
    tcp: list[float]
    qpos: NDArray[np.float64]  # of the whole articulation
    joint_names: list[str]  # its active joints, in qpos order


def frames(task: Any, cameras: list[str]) -> tuple[dict[str, Frame], str | None]:
    """The named cameras' frames (RoboTwin renders all of them) and the episode's instruction."""
    return frames_of(task.get_obs(), cameras), instruction(task)


def frames_of(obs: dict[str, Any], cameras: list[str]) -> dict[str, Frame]:
    """The named cameras' frames in a `get_obs()` result (P3's official loop takes one itself)."""
    captured = {}
    for name in cameras:
        camera = obs["observation"][name]
        world2cam = np.vstack([camera["extrinsic_cv"], [0.0, 0.0, 0.0, 1.0]])
        captured[name] = Frame(
            K=np.asarray(camera["intrinsic_cv"], dtype=float),
            cam2world=np.linalg.inv(world2cam),
            rgb=np.ascontiguousarray(camera["rgb"]),
            depth=_depth_m(camera["depth"]),
        )
    return captured


def _depth_m(millimetres: NDArray[np.float64]) -> NDArray[np.float32]:
    """A RoboTwin depth image (millimetres, 0 where nothing was hit) in metres, 0 also where the
    renderer gives no valid depth (non-finite, or behind the camera when it sits inside or
    against a surface), as a depth camera reports no return there."""
    depth = millimetres / 1000.0
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)


def instruction(task: Any) -> str | None:
    return task.get_instruction()  # type: ignore[no-any-return]


def robot_state(task: Any, profile: EmbodimentProfile) -> RobotState:
    """Proprioception without rendering: commands as in `get_obs`, measured joints, fingers
    and TCP from the articulation."""
    readings = {"left": _left(task), "right": _right(task)}
    arms: dict[Arm, ArmState] = {}
    for arm, spec in profile.arms.items():
        reading = readings[arm]
        finger = reading.qpos[reading.joint_names.index(spec.gripper.joint)]
        arms[arm] = ArmState(
            ee=Pose.from_list(reading.ee),
            tcp=Pose.from_list(reading.tcp),
            joints=[float(v) for v in reading.measured[:-1]],
            joints_cmd=[float(v) for v in reading.command[:-1]],
            gripper=float(spec.gripper.opening(float(finger))),
            gripper_cmd=float(reading.command[-1]),
        )
    command = readings["left"].command + readings["right"].command  # get_obs joint_action/vector
    return RobotState(
        step=task.take_action_cnt,
        step_limit=task.step_lim,
        arms=arms,
        qpos_cmd=[float(v) for v in command],
    )


def take_action(task: Any, qpos: list[float]) -> None:
    """One counted action: RoboTwin's joint-space `take_action`."""
    task.take_action(np.asarray(qpos, dtype=float), action_type="qpos")


def mark_infra(task: Any) -> None:
    """Note that infrastructure lost the task's current episode (`infra`)."""
    task.rs_infra = True


def clear_infra(task: Any) -> None:
    """Start an episode with no infrastructure loss noted (`mark_infra`)."""
    task.rs_infra = False


def infra(task: Any) -> bool:
    """Whether infrastructure lost the current episode (`mark_infra`; set by `clear_infra`)."""
    return bool(task.rs_infra)


def progress(task: Any) -> tuple[int, int | None, bool]:
    """Actions taken, the step limit (None outside eval mode) and the success latch."""
    return task.take_action_cnt, task.step_lim, bool(task.eval_success)


def run_batch(
    task: Any,
    profile: EmbodimentProfile,
    batch: ActionBatch,
    *,
    refresh: bool = False,
    stop: Callable[[], bool] | None = None,
    fault: Callable[[str], None] | None = None,
) -> ExecuteResult:
    """Run `batch` one counted action at a time. A step with `until_gripper_stable` repeats (the
    same command) while the measured opening of a gripper commanded below open still moved by
    1e-4 or more (open fingers creep towards their stop for many calls, holding nothing). Stops
    at the success latch (after it RoboTwin's `take_action` does nothing), at the step limit,
    when `stop()` says so, or with the states so far at a simulator error, whose type `fault`
    gets. `refresh` takes a `get_obs()` after every action, so that the frame `take_action`
    writes to the official video first is the current one (P3)."""
    states: list[StepState] = []

    def result(
        stopped: Literal["completed", "success", "step_limit", "error"], error: str | None = None
    ) -> ExecuteResult:
        return ExecuteResult(
            batch_id=batch.batch_id,
            states=states,
            executed=len(states),
            stopped=stopped,
            error=error,
        )

    for index, step in enumerate(batch.steps):
        watched = _closing(step.qpos, profile)
        for _ in range(1 + (step.max_repeats if step.until_gripper_stable else 0)):
            taken, limit, success = progress(task)
            if success:
                return result("success")
            if limit is not None and taken >= limit:
                return result("step_limit")
            if stop is not None and stop():
                return result("error", "the env server is shutting down")
            previous = states[-1].robot if states else robot_state(task, profile)
            try:
                take_action(task, step.qpos)
                if refresh:
                    task.get_obs()
                state = robot_state(task, profile)
            except Exception as error:
                log.exception("take_action failed (batch %s, step %d)", batch.batch_id, index)
                if fault is not None:
                    fault(type(error).__name__)
                detail = f"the simulator raised {type(error).__name__}; see the simulator's log"
                return result("error", detail)
            states.append(StepState(index=index, robot=state))
            moved = np.abs(_openings(state, watched) - _openings(previous, watched))
            if np.max(moved, initial=0.0) < 1e-4:
                break
    return result("success" if progress(task)[2] else "completed")


def hold(task: Any, profile: EmbodimentProfile, actions: int) -> int:
    """Keep the robot still for up to `actions` counted `take_action`s of the current drive
    targets, fewer at the success latch or the step limit; how many it took. How P3 spends the
    steps left once the agent is done (the official loop ends only at one of the two): a chunk
    per `eval()`, so that the loop's own `get_obs()` between chunks keeps its video going."""
    qpos = robot_state(task, profile).qpos_cmd
    taken, limit, success = progress(task)
    if limit is None:
        raise ValueError("no step limit to hold until: the official loop runs in eval mode")
    held = 0
    while held < actions and not success and taken < limit:
        take_action(task, qpos)
        held += 1
        taken, limit, success = progress(task)
    return held


def _closing(qpos: list[float], profile: EmbodimentProfile) -> list[Arm]:
    """The arms whose gripper `qpos` commands below open."""
    return [a for a, s in profile.arms.items() if qpos[s.gripper_index] < s.gripper.open]


def _openings(state: RobotState, arms: list[Arm]) -> NDArray[np.float64]:
    return np.array([state.arms[arm].gripper for arm in arms], dtype=float)


def _left(task: Any) -> _Reading:
    return _Reading(
        command=task.robot.get_left_arm_jointState(),
        measured=task.robot.get_left_arm_real_jointState(),
        ee=task.robot.get_left_ee_pose(),
        tcp=task.robot.get_left_tcp_pose(),
        qpos=task.robot.left_entity.get_qpos(),
        joint_names=[joint.get_name() for joint in task.robot.left_entity.get_active_joints()],
    )


def _right(task: Any) -> _Reading:
    return _Reading(
        command=task.robot.get_right_arm_jointState(),
        measured=task.robot.get_right_arm_real_jointState(),
        ee=task.robot.get_right_ee_pose(),
        tcp=task.robot.get_right_tcp_pose(),
        qpos=task.robot.right_entity.get_qpos(),
        joint_names=[joint.get_name() for joint in task.robot.right_entity.get_active_joints()],
    )
