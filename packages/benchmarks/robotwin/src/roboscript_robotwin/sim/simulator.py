"""Everything done with the RoboTwin task object; `host` calls it on one worker thread only.

Episodes open the official way (expert pre-run, `close_env`, `setup_demo` at the same seed, the
instruction), with RoboTwin's reset options (`RoboTwinOptions`); agent-plane calls reach the task
object only through `shim`; the referee reads ground truth for the control plane (and for
re-recording a demonstration, `truth` and `expert_trajectory`), and `expert` traces the pre-run's
API calls when asked. A simulator error during an episode faults it: later agent-plane calls
fail with `sim_fault`, and `judge` and `diagnose` say so.
"""

import contextlib
import importlib
import math
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.env import EpisodeSpec, ExpertCall, GtLink, JointTruth, ObjectTruth
from roboscript.contracts.errors import InfraError, ProtocolError
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.ids import TaskName
from roboscript.contracts.robot import RobotState
from roboscript.embodiments import EmbodimentProfile
from roboscript.envs import reset_options
from roboscript_robotwin.sim import expert, official, referee, shim


class RoboTwinOptions(BaseModel):
    """RoboTwin's reset options (`EpisodeSpec.options`), named as its own task config names them:
    `instruction_type`, which of its instruction sets the text is drawn from; `eval_mode`, its
    evaluation-time setup (the step limit, and unseen textures on randomized scenes)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction_type: Literal["seen", "unseen"] = "seen"
    eval_mode: bool = True


@dataclass(frozen=True)
class Opened:
    solvable: bool
    instruction: str | None
    step_limit: int | None
    prerun_attempts: int


@dataclass
class _Episode:
    initial: dict[str, Pose]
    initial_centers: dict[str, Vec3]
    initial_joints: dict[str, list[float]]
    table_top_z: float
    trace: list[ExpertCall] | None  # the pre-run's API calls, when traced
    success_step: int | None = None
    fault: str | None = None  # the simulator error that ended it, if one did


class Simulator:
    """RoboTwin's task object for one `task`, built once by `start`."""

    def __init__(
        self, root: Path, task_config: Path, profile: EmbodimentProfile, task: TaskName
    ) -> None:
        self.root, self.task_config, self.profile, self.task = root, task_config, profile, task
        self._task: Any = None
        self._episode: _Episode | None = None
        self._opened = 0
        self._prerun_passed: tuple[str, int] | None = None  # (task, seed) of the last passing one
        self._stopping = threading.Event()

    def start(self) -> str | None:
        """Import RoboTwin, build the task and render once: cuRobo's warm-up and the process's
        first render take most of a minute. Returns RoboTwin's commit."""
        official.use_robotwin(self.root)
        importlib.import_module("envs")
        from envs.utils.create_actor import UnStableError

        self._task = official.make_task(self.task)
        args = official.setup_args(self.root, self.task, self.task_config, eval_mode=True)
        for seed in range(10):
            try:
                self._task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
                break
            except UnStableError:
                continue
        else:
            raise RuntimeError(f"{self.task}: objects were unstable on seeds 0-9")
        self._task.get_obs()
        self._task.close_env()
        return _commit(self.root)

    def open(self, spec: EpisodeSpec) -> Opened:
        """Reset for `spec`: up to `prerun_attempts` expert pre-runs, then the episode. Skipping
        the pre-run reuses what the last one set on the task object, so a repeat skips it and is
        allowed only to reopen the (task, seed) whose pre-run passed last."""
        options = reset_options(RoboTwinOptions, spec)
        if spec.repeat and (spec.task, spec.seed) != self._prerun_passed:
            detail = "a repeat reopens only the seed whose pre-run passed last"
            raise ProtocolError("invalid_request", detail)
        self.close()
        task = self._task
        args = official.setup_args(
            self.root, spec.task, self.task_config, eval_mode=options.eval_mode
        )
        info, attempts = None, 0
        recorder = expert.Recorder(self._robot) if spec.trace_expert else None
        if not spec.repeat:
            self._prerun_passed = None
            while info is None and attempts < spec.prerun_attempts:
                attempts += 1
                info = official.expert_prerun(task, spec.seed, args, self._opened, recorder)
            if info is None:
                return Opened(False, None, None, attempts)
            self._prerun_passed = (spec.task, spec.seed)
        task.setup_demo(now_ep_num=self._opened, seed=spec.seed, is_test=True, **args)
        # Drawn even on a repeat, so numpy's generator stays where it would be.
        drawn = official.instruction(spec.task, info, options.instruction_type) if info else None
        instruction = spec.instruction or drawn
        assert instruction is not None  # EpisodeSpec requires one for a repeat
        task.set_instruction(instruction=instruction)
        # A scene's first render is slow (seconds after `close_env` cleared SAPIEN's cache), so
        # `open` pays for it rather than the agent's first `observe`.
        task.get_obs()
        self._opened += 1
        objects = referee.task_objects(task)
        self._episode = _Episode(
            referee.poses(objects),
            referee.centers(objects),
            referee.joint_positions(objects),
            referee.table_top(task),
            recorder.calls if recorder is not None else None,
        )
        return Opened(True, instruction, task.step_lim, attempts)

    def close(self) -> None:
        """End the episode and clear SAPIEN's render cache, every episode (the official loop does
        it every fifth): each rendered scene's new assets stay on the GPU until cleared, so a
        process that renders a new scene per episode grows without bound."""
        if self._episode is not None:
            self._episode = None
            self._task.close_env(clear_cache=True)

    def expert_trace(self) -> list[ExpertCall]:
        """The traced API calls of this episode's pre-run."""
        episode = self._episode
        assert episode is not None
        if episode.trace is None:
            detail = "this episode was opened without trace_expert"
            raise ProtocolError("invalid_request", detail)
        return episode.trace

    def fault(self, error: str) -> None:
        """Mark the episode as ended by a simulator error (`error` names its type)."""
        if self._episode is not None and self._episode.fault is None:
            self._episode.fault = error

    def stop(self) -> None:
        """Make a running `execute` return before its next action (the server is stopping)."""
        self._stopping.set()

    def robot(self) -> RobotState:
        self._healthy()
        return shim.robot_state(self._task, self.profile)

    def observe(self, cameras: list[str]) -> tuple[dict[str, shim.Frame], RobotState, str | None]:
        self._healthy()
        frames, instruction = shim.frames(self._task, cameras)
        return frames, shim.robot_state(self._task, self.profile), instruction

    def execute(self, batch: ActionBatch) -> ExecuteResult:
        """Run `batch` (`shim.run_batch`): a simulator error faults the episode."""
        self._healthy()
        episode = self._episode
        assert episode is not None
        stop = self._stopping.is_set
        result = shim.run_batch(self._task, self.profile, batch, stop=stop, fault=self.fault)
        if episode.success_step is None and result.stopped == "success" and result.states:
            episode.success_step = result.states[-1].robot.step
        return result

    def judge(self) -> tuple[bool, int | None, int, int | None, bool]:
        """Success latch, the step it fired at, actions taken, the step limit, faulted."""
        assert self._episode is not None
        taken, limit, success = shim.progress(self._task)
        return success, self._episode.success_step, taken, limit, self._episode.fault is not None

    def diagnose(self) -> tuple[bool, list[ObjectTruth], list[str]]:
        """The task objects at the start and now (a method's explainer reads them in the try's
        own terms), and notes that name no object."""
        episode = self._episode
        assert episode is not None
        objects = referee.task_objects(self._task)
        final, centers = referee.poses(objects), referee.centers(objects)
        joints, limits = referee.joint_positions(objects), referee.joint_limits(objects)
        truths = [
            ObjectTruth(
                name=name,
                initial=initial,
                final=final[name],
                displacement_m=float(np.linalg.norm(np.subtract(final[name].p, initial.p))),
                initial_center=episode.initial_centers[name],
                final_center=centers[name],
                joints=[
                    JointTruth(initial=start, final=now, low=low, high=high)
                    for start, now, (low, high) in zip(
                        episode.initial_joints.get(name, []),
                        joints.get(name, []),
                        limits.get(name, []),
                        strict=True,
                    )
                    if math.isfinite(low) and math.isfinite(high)  # a free-turning joint: none
                ],
            )
            for name, initial in episode.initial.items()
        ]
        notes = []
        if episode.fault is not None:
            notes.append("the simulator faulted in this episode: an infrastructure error")
        return shim.progress(self._task)[2], truths, notes

    def snapshot(
        self, geometry: bool
    ) -> tuple[
        int, dict[str, Pose], float, dict[str, list[list[Vec3]]], dict[str, dict[str, GtLink]]
    ]:
        """Poses now; with `geometry` the objects' and moving links' convex hulls too."""
        assert self._episode is not None
        objects = referee.task_objects(self._task)
        step = shim.progress(self._task)[0]
        hulls = referee.hulls(objects) if geometry else {}
        links = referee.links(objects, hulls=geometry)
        return step, referee.poses(objects), self._episode.table_top_z, hulls, links

    def _robot(self, task: Any) -> RobotState:
        return shim.robot_state(task, self.profile)

    def truth(self) -> referee.Truth:
        """The task objects now: poses, joint positions, arm contacts and categories."""
        objects = referee.task_objects(self._task)
        return referee.Truth(
            poses=referee.poses(objects),
            joints=referee.joint_positions(objects),
            touched=referee.contacts(self._task, objects, self.profile),
            categories=referee.categories(objects),
        )

    @contextlib.contextmanager
    def stepping(self) -> Iterator[Callable[[], int]]:
        """Count the physics steps the open episode's scene simulates meanwhile (`scene.step`,
        once a step, on this object only): yields the count so far."""
        scene, count = self._task.scene, [0]
        step = scene.step

        def counted(*args: Any, **kwargs: Any) -> Any:
            count[0] += 1
            return step(*args, **kwargs)

        scene.step = counted
        try:
            yield lambda: count[0]
        finally:
            del scene.step

    def expert_trajectory(self, seed: int, attempts: int) -> tuple[list[RobotState], Any] | None:
        """Run the expert at `seed` like a pre-run, up to `attempts` times (its planning is not
        deterministic), taking the robot state wherever RoboTwin's data collector saves a frame
        (`_take_picture`: each motion's start and end, and every `save_freq` physics steps), so
        that the states line up with the expert data's frames. Returns them with the expert's
        episode info, or None if every run failed. Ends the open episode."""
        self.close()
        task, states = self._task, []
        args = official.setup_args(self.root, self.task, self.task_config, eval_mode=True)
        # Only the saving calls are replaced, on this object; RoboTwin's files stay untouched.
        task._take_picture = lambda: states.append(shim.robot_state(task, self.profile))
        try:
            for _ in range(attempts):
                states.clear()
                if (info := official.expert_prerun(task, seed, args, self._opened)) is not None:
                    return list(states), info
        finally:
            del task._take_picture
        return None

    def prerun_passes(self, seed: int, runs: int) -> int:
        """Run the expert pre-run at `seed` up to `runs` times, as `open` does, stopping at the
        first failure: how many passed. Ends the open episode."""
        self.close()
        self._prerun_passed = None  # the next open pre-runs again
        args = official.setup_args(self.root, self.task, self.task_config, eval_mode=True)
        for n in range(runs):
            if official.expert_prerun(self._task, seed, args, self._opened) is None:
                return n
        return runs

    def _healthy(self) -> None:
        if self._episode is not None and self._episode.fault is not None:
            detail = f"the simulator failed in this episode ({self._episode.fault})"
            raise InfraError("sim_fault", detail)


def _commit(root: Path) -> str | None:
    done = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return done.stdout.strip() if done.returncode == 0 else None
