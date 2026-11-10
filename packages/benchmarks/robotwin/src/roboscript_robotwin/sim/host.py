"""The RoboTwin env server: `python -m roboscript_robotwin.sim.host --config <file> --ready-file
<path>`.

Serves the env planes on a control and an agent port (`roboscript.envs.serve_env`) for the one
task of its config. Everything simulated runs on one worker thread through `Simulator`; the event
loop never touches SAPIEN. Paths, settings and the control token are resolved before the
simulator changes the working directory to `ROBOTWIN_ROOT`.
"""

import argparse
import asyncio
import functools
import hashlib
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import anyio
from pydantic import SecretStr

from roboscript.blobs import LocalBlobStore
from roboscript.contracts import versions
from roboscript.contracts.action import ActionBatch, ExecuteResult
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
    ObserveRequest,
)
from roboscript.contracts.errors import InfraError, ProtocolError, RSError
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import RobotState
from roboscript.envs import Batches, Episodes, control_token, serve_env, unsolvable
from roboscript.settings import EnvSettings, load_config_file
from roboscript_robotwin.benchmark import ROBOTWIN, TASK_CONFIGS
from roboscript_robotwin.config import RoboTwinConfig, robotwin_root
from roboscript_robotwin.frames import observation
from roboscript_robotwin.sim.simulator import Simulator

log = logging.getLogger(__name__)
T = TypeVar("T")


class RoboTwinEnv:
    """The planes `EnvPublic`, `EnvControl` and `EnvAgent` over one RoboTwin task object."""

    def __init__(self, config: RoboTwinConfig, root: Path, blob_dir: Path) -> None:
        self.config = config
        self.profile = ROBOTWIN.profile()
        self.episodes = Episodes()
        self.agent_url = ""  # set by `serve_env`
        # Resolved now: the simulator changes the working directory to ROBOTWIN_ROOT.
        self.blobs = LocalBlobStore(blob_dir)
        self._batches = Batches()
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robotwin")
        self._sim = Simulator(root, TASK_CONFIGS[config.mode], self.profile, config.task)
        self._discovery: EnvDiscovery | None = None

    async def start(self) -> None:
        """Import RoboTwin and warm the task up before any request is served."""
        commit = await self._call(self._sim.start)
        simulated = {
            "task_config": TASK_CONFIGS[self.config.mode].read_text(encoding="utf-8"),
            "embodiment": self.profile.model_dump(mode="json"),
        }
        self._discovery = EnvDiscovery(
            contracts={p: versions.VERSIONS[p] for p in ("env.public", "env.control", "env.agent")},
            benchmark=ROBOTWIN.name,
            embodiment=self.profile.name,
            upstream_commit=commit,
            config_hash=hashlib.sha256(canonical_json(simulated).encode()).hexdigest(),
            capabilities=ROBOTWIN.discovered(
                SensingCapabilities(
                    scene_identity=True,
                    measured_proprio=True,
                    depth=True,
                    cameras=list(self.profile.cameras),
                    step_unit="action",
                )
            ),
        )

    async def discover(self) -> EnvDiscovery:
        assert self._discovery is not None  # `start` ran before serving
        return self._discovery

    async def open(self, spec: EpisodeSpec) -> EpisodeOpened:
        if spec.task != self.config.task:
            detail = f"this env serves task {self.config.task!r}, not {spec.task!r}"
            raise ProtocolError("invalid_request", detail)
        if (refused := unoffered_reset(spec.reset_by, ROBOTWIN.capabilities)) is not None:
            raise ProtocolError("invalid_request", refused)
        if spec.record:
            raise ProtocolError(
                "invalid_request", "this env server does not record episodes (EpisodeSpec.record)"
            )
        self.episodes.end()  # a refused or failed open ends the open episode too
        opened = await self._call(self._sim.open, spec)
        if not opened.solvable:
            return unsolvable(self.profile.name, opened.prerun_attempts)
        self._batches = Batches()
        episode_id, token = self.episodes.open()
        return EpisodeOpened(
            episode_id=episode_id,
            agent_url=self.agent_url,
            agent_token=token,
            instruction=opened.instruction,
            step_limit=opened.step_limit,
            embodiment=self.profile.name,
            cameras=list(self.profile.cameras),
            solvable=True,
            prerun_attempts=opened.prerun_attempts,
        )

    async def judge(self, episode_id: EpisodeId) -> JudgeResult:
        success, step, used, limit, faulted = await self._in_episode(episode_id, self._sim.judge)
        return JudgeResult(
            episode_id=episode_id,
            success=success,
            success_step=step,
            steps_used=used,
            step_limit=limit,
            faulted=faulted,
        )

    async def diagnose(self, episode_id: EpisodeId) -> Diagnosis:
        success, objects, notes = await self._in_episode(episode_id, self._sim.diagnose)
        return Diagnosis(episode_id=episode_id, success=success, objects=objects, notes=notes)

    async def gt_snapshot(self, episode_id: EpisodeId, geometry: bool = False) -> GtSnapshot:
        step, objects, table, hulls, links = await self._in_episode(
            episode_id, self._sim.snapshot, geometry
        )
        return GtSnapshot(
            episode_id=episode_id,
            step=step,
            objects=objects,
            table_top_z=table,
            hulls=hulls,
            links=links,
        )

    async def expert_trace(self, episode_id: EpisodeId) -> ExpertTrace:
        calls = await self._in_episode(episode_id, self._sim.expert_trace)
        return ExpertTrace(episode_id=episode_id, calls=calls)

    async def close(self, episode_id: EpisodeId) -> None:
        self.episodes.close(episode_id)
        await self._call(self._sim.close)

    async def robot(self, episode_id: EpisodeId) -> RobotState:
        return await self._in_episode(episode_id, self._sim.robot)

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation:
        names = request.cameras if request.cameras is not None else list(self.profile.cameras)
        if unknown := sorted(set(names) - set(self.profile.cameras)):
            cameras = sorted(self.profile.cameras)
            raise ProtocolError("invalid_request", f"no cameras {unknown}; cameras: {cameras}")
        frames, robot, instruction = await self._in_episode(episode_id, self._sim.observe, names)
        encode = functools.partial(
            observation, frames, robot, instruction, self.blobs, depth=request.depth
        )
        return await anyio.to_thread.run_sync(encode)

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        self.episodes.check(episode_id)
        if wrong := {len(step.qpos) for step in batch.steps} - {self.profile.action_dim}:
            detail = f"a qpos has {self.profile.action_dim} values, got {sorted(wrong)}"
            raise ProtocolError("invalid_request", detail)
        if not batch.stop_on_success:
            detail = (
                "RoboTwin always stops at the success latch (take_action does nothing after it)"
            )
            raise ProtocolError("invalid_request", detail)
        run = functools.partial(self._in_episode, episode_id, self._sim.execute, batch)
        return await self._batches.run(batch, run)

    async def _in_episode(self, episode_id: EpisodeId, fn: Callable[..., T], *args: Any) -> T:
        """`fn(*args)` on the simulation thread, if `episode_id` is still the open episode there."""
        self.episodes.check(episode_id)

        def job() -> T:
            self.episodes.check(episode_id)  # an `open` may have come first in the queue
            try:
                return fn(*args)
            except RSError:
                raise
            except Exception as error:
                self._sim.fault(type(error).__name__)
                raise

        return await self._call(job)

    def shutdown(self) -> None:
        """Stop a running batch before its next action and drop queued simulator work."""
        self._sim.stop()
        self._thread.shutdown(wait=True, cancel_futures=True)

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


async def _serve(
    config: RoboTwinConfig, root: Path, blob_dir: Path, token: SecretStr, ready_file: Path | None
) -> None:
    env = RoboTwinEnv(config, root, blob_dir)
    try:
        await env.start()
        await serve_env(env, config.server, ready_file, token)
    finally:
        env.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve RoboTwin on a control and an agent port.")
    parser.add_argument("--config", type=Path, required=True, help="a RoboTwinConfig YAML file")
    parser.add_argument("--ready-file", type=Path, help="where to write the planes' URLs")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config_file(args.config, RoboTwinConfig)
    # Everything read from the working directory or .env is resolved before the simulator
    # changes the working directory to ROBOTWIN_ROOT.
    settings = EnvSettings()
    checkout = robotwin_root()
    root = config.robotwin_root or checkout
    if root is None:
        raise SystemExit("set ROBOTWIN_ROOT in the environment or .env, or robotwin_root")
    blob_dir = config.blob_dir or settings.blob_dir
    if blob_dir is None:
        raise SystemExit("set blob_dir in the config, or RS_BLOB_DIR (a stack sets it)")
    ready_file = args.ready_file.resolve() if args.ready_file else None
    token = control_token()
    anyio.run(_serve, config, root.resolve(), blob_dir.resolve(), token, ready_file)


if __name__ == "__main__":
    main()
