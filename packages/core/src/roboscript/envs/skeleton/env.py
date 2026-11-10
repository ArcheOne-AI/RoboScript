"""The env planes over a `SimBackend`: what every entry skeleton serves. One task per process, one
worker thread for everything simulated, episodes bookkept by `roboscript.envs.Episodes`.

It offers no expert and records nothing (`open` refuses `trace_expert` and `record`, runs no
expert pre-run and says so: `prerun_attempts` 0; a reset is solvable unless the backend says the
benchmark's own rule skips it, `Opened.unsolvable`, opened with no instruction), judges as the
backend's `Verdict` says, and serves ground truth from `SimBackend.object_poses` where the
benchmark declares it (`EnvCapabilities.truth`; otherwise `diagnose` and `gt_snapshot` are
refused). A benchmark whose resets may be opened once (`EnvCapabilities.reruns` False) has each
reset opened once, unless `once_per_reset` turns the rule off for a local copy, whose `discover()`
then declares reruns (what it serves).

The simulation thread is a worker by default. A simulator that must be driven from the main
thread (`sim_thread="main"`: Isaac Sim, whose `SimulationApp` installs signal handlers and whose
Kit loop stalls while the main thread waits on another) gets the main thread, and the planes are
served from a second thread (`run_server`).
"""

import argparse
import asyncio
import functools
import hashlib
import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, TypeVar

import anyio
import numpy as np
from pydantic import BaseModel, ConfigDict, SecretStr

from roboscript.blobs import LocalBlobStore
from roboscript.contracts import versions
from roboscript.contracts.action import ActionBatch, ExecuteResult, StepState
from roboscript.contracts.camera import CameraModel
from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.capabilities import SensingCapabilities, unoffered_reset
from roboscript.contracts.env import (
    Diagnosis,
    EnvDiscovery,
    EpisodeOpened,
    EpisodeSpec,
    ExpertTrace,
    GtSnapshot,
    JudgeResult,
    ObjectTruth,
    ObserveRequest,
)
from roboscript.contracts.errors import InfraError, ProtocolError, RSError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import EpisodeId, FrameId, TaskName, new_id
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import RobotState
from roboscript.envs import (
    Batches,
    Benchmark,
    Episodes,
    ServerConfig,
    control_token,
    reset_options,
    serve_env,
)
from roboscript.envs.skeleton.backend import (
    EndRules,
    Opened,
    SimBackend,
    StageProgress,
    SuccessConditions,
    TruthGeometry,
)
from roboscript.settings import EnvSettings, load_config_file

log = logging.getLogger(__name__)
T = TypeVar("T")
SimThread = Literal["worker", "main"]


class MainThreadExecutor(Executor):
    """An executor whose jobs run on the thread that calls `run`, in order, one at a time: the
    main thread, for a simulator that has to be driven from there."""

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue[
            tuple[Future[Any], Callable[..., Any], tuple[Any, ...]] | None
        ]
        self._jobs = queue.SimpleQueue()
        self._shut = False

    def submit(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        if self._shut:
            raise RuntimeError("the simulation thread has shut down")
        future: Future[T] = Future()
        self._jobs.put((future, functools.partial(fn, **kwargs), args))
        return future

    def run(self) -> None:
        """Run jobs until `shutdown`."""
        while (job := self._jobs.get()) is not None:
            future, fn, args = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as error:
                future.set_exception(error)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._shut = True
        if cancel_futures:
            while True:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    job[0].cancel()
        self._jobs.put(None)


class SkeletonConfig(BaseModel):
    """An adapter's env server config extends this: the one `task` the process serves, where
    frames go (default: the stack's `RS_BLOB_DIR`), whether each reset opens once
    (`once_per_reset`; None: as the benchmark declares, `EnvCapabilities.reruns`; False on a
    benchmark that declares one open per reset: a local copy that reopens), and where it
    listens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    blob_dir: Path | None = None
    once_per_reset: bool | None = None
    server: ServerConfig = ServerConfig()


class SkeletonEnv:
    """`EnvPublic`, `EnvControl` and `EnvAgent` over one backend."""

    def __init__(
        self,
        benchmark: Benchmark,
        config: SkeletonConfig,
        backend: SimBackend,
        blobs: LocalBlobStore,
        sensing: SensingCapabilities,
        sim_thread: SimThread = "worker",
    ) -> None:
        self.benchmark, self.config, self.backend = benchmark, config, backend
        self.profile = benchmark.profile()
        self.sensing = sensing
        self.blobs = blobs
        self.episodes = Episodes()
        self.agent_url = ""  # set by `serve_env`
        self._batches = Batches()
        self._thread: Executor = (
            MainThreadExecutor()
            if sim_thread == "main"
            else ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
        )
        self._discovery: EnvDiscovery | None = None
        self._initial: dict[str, Pose] = {}
        self._instruction: str | None = None
        reruns = benchmark.capabilities.reruns
        self.once_per_reset = not reruns if config.once_per_reset is None else config.once_per_reset
        self._opened_resets: set[tuple[str, str, int]] = set()
        self._fault: str | None = None

    async def start(self) -> None:
        """Import the official package and warm up, before any request is served."""
        upstream = await self._call(self.backend.start)
        simulated = {
            "embodiment": self.profile.model_dump(mode="json"),
            "upstream": upstream,
            "task": self.config.task,
        }
        self._discovery = EnvDiscovery(
            contracts={p: versions.VERSIONS[p] for p in ("env.public", "env.control", "env.agent")},
            benchmark=self.benchmark.name,
            embodiment=self.profile.name,
            upstream_commit=upstream,
            config_hash=hashlib.sha256(canonical_json(simulated).encode()).hexdigest(),
            capabilities=self.benchmark.discovered(self.sensing).model_copy(
                update={"reruns": not self.once_per_reset}
            ),
        )

    async def discover(self) -> EnvDiscovery:
        assert self._discovery is not None  # `start` ran before serving
        return self._discovery

    async def open(self, spec: EpisodeSpec) -> EpisodeOpened:
        if spec.task != self.config.task:
            detail = f"this env serves task {self.config.task!r}, not {spec.task!r}"
            raise ProtocolError("invalid_request", detail)
        if (refused := unoffered_reset(spec.reset_by, self.benchmark.capabilities)) is not None:
            raise ProtocolError("invalid_request", refused)
        if spec.record or spec.trace_expert:
            what = "recording episodes" if spec.record else "the expert trace"
            raise ProtocolError("invalid_request", f"{what} is not offered by this env")
        options = reset_options(self.backend.options, spec)
        reset = (spec.task, spec.reset_by, spec.seed)
        if self.once_per_reset and reset in self._opened_resets:
            at = f"{spec.reset_by} {spec.seed}"
            detail = f"{spec.task} at {at} was opened already; each reset opens once"
            raise ProtocolError("invalid_request", detail)
        self.episodes.end()  # a refused or failed open ends the open episode too
        opened: Opened = await self._call(self.backend.open, spec.task, spec.seed, options)
        self._opened_resets.add(reset)
        self._fault = None
        self._initial = await self._call(self.backend.object_poses)
        self._instruction = spec.instruction or opened.instruction
        self._batches = Batches()
        episode_id, token = self.episodes.open()
        if opened.unsolvable is not None:
            why = opened.unsolvable
            log.info("%s %s %d does not count: %s", spec.task, spec.reset_by, spec.seed, why)
        return EpisodeOpened(
            episode_id=episode_id,
            agent_url=self.agent_url,
            agent_token=token,
            instruction=None if opened.unsolvable is not None else self._instruction,
            step_limit=opened.step_limit,
            embodiment=self.profile.name,
            cameras=list(self.profile.cameras),
            solvable=opened.unsolvable is None,  # no expert: the benchmark's own skip rule
        )

    async def judge(self, episode_id: EpisodeId) -> JudgeResult:
        self.episodes.check(episode_id)
        verdict = await self._call(self.backend.verdict)
        return JudgeResult(
            episode_id=episode_id,
            success=verdict.success,
            success_step=verdict.success_step,
            steps_used=verdict.steps_used,
            step_limit=verdict.step_limit,
            faulted=self._fault is not None,
            score=verdict.score,
            ended=verdict.ended,
        )

    async def diagnose(self, episode_id: EpisodeId) -> Diagnosis:
        self._truth(episode_id)
        verdict = await self.judge(episode_id)
        final = await self._in_episode(episode_id, self.backend.object_poses)
        objects = [
            ObjectTruth(
                name=name,
                initial=initial,
                final=final[name],
                displacement_m=float(np.linalg.norm(np.subtract(final[name].p, initial.p))),
            )
            for name, initial in self._initial.items()
        ]
        notes = [f"the simulator faulted: {self._fault}"] if self._fault else []
        if verdict.ended is not None:
            notes.append(f"the benchmark ended the episode: {verdict.ended}")
        conditions = []
        if isinstance(self.backend, SuccessConditions):
            try:
                conditions = await self._in_episode(episode_id, self.backend.conditions)
            except InfraError as error:  # optional: the objects' truth still comes back
                notes.append(f"the success checks could not be read: {error}")
        return Diagnosis(
            episode_id=episode_id, success=verdict.success, objects=objects, notes=notes,
            conditions=conditions,
        )  # fmt: skip

    async def gt_snapshot(self, episode_id: EpisodeId, geometry: bool = False) -> GtSnapshot:
        self._truth(episode_id)
        poses = await self._in_episode(episode_id, self.backend.object_poses)
        robot = await self._in_episode(episode_id, self.backend.robot)
        rules = []
        if isinstance(self.backend, EndRules):
            rules = await self._in_episode(episode_id, self.backend.end_rules)
        progress = None
        if isinstance(self.backend, StageProgress):
            progress = await self._in_episode(episode_id, self.backend.progress)
        if not isinstance(self.backend, TruthGeometry):
            return GtSnapshot(
                episode_id=episode_id,
                step=robot.step,
                objects=poses,
                end_rules=rules,
                progress=progress,
            )
        hulls = await self._in_episode(episode_id, self.backend.object_hulls) if geometry else {}
        table = await self._in_episode(episode_id, self.backend.table_top_z)
        return GtSnapshot(
            episode_id=episode_id,
            step=robot.step,
            objects=poses,
            table_top_z=table,
            hulls=hulls,
            end_rules=rules,
            progress=progress,
        )

    async def expert_trace(self, episode_id: EpisodeId) -> ExpertTrace:
        self.episodes.check(episode_id)
        raise ProtocolError("invalid_request", "this env offers no expert pre-run or trace")

    async def close(self, episode_id: EpisodeId) -> None:
        self.episodes.close(episode_id)
        await self._call(self.backend.close)

    async def robot(self, episode_id: EpisodeId) -> RobotState:
        self._healthy(episode_id)
        return await self._in_episode(episode_id, self.backend.robot)

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation:
        self._healthy(episode_id)
        names = request.cameras if request.cameras is not None else list(self.profile.cameras)
        if unknown := sorted(set(names) - set(self.profile.cameras)):
            cameras = sorted(self.profile.cameras)
            raise ProtocolError("invalid_request", f"no cameras {unknown}; cameras: {cameras}")

        def capture() -> tuple[dict[str, Any], RobotState]:
            return self.backend.frames(names), self.backend.robot()

        frames, robot = await self._in_episode(episode_id, capture)
        encode = functools.partial(self._observation, frames, robot, request.depth)
        return await anyio.to_thread.run_sync(encode)

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        self._healthy(episode_id)
        if wrong := {len(step.qpos) for step in batch.steps} - {self.profile.action_dim}:
            detail = f"a qpos has {self.profile.action_dim} values, got {sorted(wrong)}"
            raise ProtocolError("invalid_request", detail)
        run = functools.partial(self._in_episode, episode_id, self._run_batch, batch)
        return await self._batches.run(batch, run)

    def _truth(self, episode_id: EpisodeId) -> None:
        """Ground truth is served only where the benchmark declares it; a closed episode still
        says so first."""
        self.episodes.check(episode_id)
        if not self.benchmark.capabilities.truth:
            raise ProtocolError("invalid_request", "this env offers no ground truth")

    def _healthy(self, episode_id: EpisodeId) -> None:
        """Agent-plane calls on an episode a simulator error ended fail with `sim_fault`; `judge`
        and `diagnose` still answer, and say so."""
        self.episodes.check(episode_id)
        if self._fault is not None:
            raise InfraError("sim_fault", f"the episode faulted: {self._fault}")

    def _observation(self, frames: dict[str, Any], robot: RobotState, depth: bool) -> Observation:
        cameras = {}
        for name, frame in frames.items():
            height, width = frame.rgb.shape[:2]
            camera = CameraModel(
                name=name,
                width=width,
                height=height,
                K=np.asarray(frame.K, dtype=float).tolist(),
                cam2world=np.asarray(frame.cam2world, dtype=float).tolist(),
            )
            cameras[name] = self.blobs.put_frame(camera, frame.rgb, frame.depth if depth else None)
        return Observation(
            frame_id=FrameId(new_id("frame")),
            step=robot.step,
            cameras=cameras,
            robot=robot,
            instruction=self._instruction,
        )

    def _run_batch(self, batch: ActionBatch) -> ExecuteResult:
        """Run `batch` one counted action at a time on the simulation thread. A step with
        `until_gripper_stable` repeats while a closing gripper's measured opening still moved by
        1e-4 or more. Stops at a success latch (where the benchmark has one), at the step limit,
        where the benchmark ended the episode before it (`ended`: no action counts after), or with
        the states so far at a simulator error or an action the wire cannot carry."""
        backend, profile = self.backend, self.profile
        latched = "latch" in self.benchmark.capabilities.success
        states: list[StepState] = []

        def result(stopped: Any, error: str | None = None) -> ExecuteResult:
            return ExecuteResult(
                batch_id=batch.batch_id,
                states=states,
                executed=len(states),
                stopped=stopped,
                error=error,
            )

        for index, step in enumerate(batch.steps):
            watched = [
                arm
                for arm, spec in profile.arms.items()
                if step.qpos[spec.gripper_index] < spec.gripper.open
            ]
            for _ in range(1 + (step.max_repeats if step.until_gripper_stable else 0)):
                verdict = backend.verdict()
                if latched and batch.stop_on_success and verdict.success:
                    return result("success")
                limit = verdict.step_limit
                if limit is not None and verdict.steps_used >= limit:
                    return result("step_limit")
                if verdict.ended is not None:
                    return result("ended")
                previous = states[-1].robot if states else backend.robot()
                try:
                    backend.step(step.qpos)
                    state = backend.robot()
                except NotImplementedError as error:  # not the simulator's fault
                    return result("error", str(error))
                except Exception as error:
                    log.exception("step failed (batch %s, step %d)", batch.batch_id, index)
                    self._fault = self._fault or type(error).__name__
                    detail = f"the simulator raised {type(error).__name__}; see the env server log"
                    return result("error", detail)
                states.append(StepState(index=index, robot=state))
                moved = [_opening(state, arm) - _opening(previous, arm) for arm in watched]
                if max((abs(m) for m in moved), default=0.0) < 1e-4:
                    break
        final = backend.verdict()
        if final.ended is not None:
            return result("ended")
        return result("success" if latched and final.success else "completed")

    async def _in_episode(self, episode_id: EpisodeId, fn: Callable[..., T], *args: Any) -> T:
        """`fn(*args)` on the simulation thread, if `episode_id` is still the open episode there."""
        self.episodes.check(episode_id)

        def job() -> T:
            self.episodes.check(episode_id)  # an `open` may have come first in the queue
            return fn(*args)

        return await self._call(job)

    async def _call(self, fn: Callable[..., T], *args: Any) -> T:
        """`fn(*args)` on the simulation thread; its failures come back as `sim_fault`."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._thread, functools.partial(fn, *args))
        except RSError:
            raise
        except Exception as error:
            log.exception("the simulation thread failed")
            detail = f"the simulator raised {type(error).__name__}; see the env server log"
            raise InfraError("sim_fault", detail) from None

    def shutdown(self) -> None:
        self._thread.shutdown(wait=True, cancel_futures=True)


def _opening(state: RobotState, arm: str) -> float:
    return state.arms[arm].gripper or 0.0


async def _serve(
    env: SkeletonEnv, config: SkeletonConfig, token: SecretStr, ready: Path | None
) -> None:
    try:
        await env.start()
        await serve_env(env, config.server, ready, token)
    finally:
        env.shutdown()


def run_server(
    description: str,
    config_model: type[SkeletonConfig],
    make_env: Callable[[Any, LocalBlobStore], SkeletonEnv],
) -> None:
    """An adapter's `python -m <env_server> --config <file> --ready-file <path>`: the config
    validated as `config_model`, the blob store and the control token resolved before the
    simulator can change the working directory, then `make_env(config, blobs)` served."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="the adapter's config YAML")
    parser.add_argument("--ready-file", type=Path, help="where to write the planes' URLs")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config_file(args.config, config_model)
    blob_dir = config.blob_dir or EnvSettings().blob_dir
    if blob_dir is None:
        raise SystemExit("set blob_dir in the config, or RS_BLOB_DIR (a stack sets it)")
    token = control_token()
    ready = args.ready_file.resolve() if args.ready_file else None
    env = make_env(config, LocalBlobStore(blob_dir.resolve()))
    if not isinstance(env._thread, MainThreadExecutor):
        anyio.run(_serve, env, config, token, ready)
        return
    # The simulator keeps the main thread; the planes run beside it until they stop (an error, or
    # the process ends: signals reach the main thread, whose simulator handles them).
    failed: list[BaseException] = []

    def serve() -> None:
        try:
            anyio.run(_serve, env, config, token, ready)
        except BaseException as error:
            failed.append(error)

    server = threading.Thread(target=serve, name="planes")
    server.start()
    env._thread.run()
    server.join()
    if failed:
        raise failed[0]
