"""The robot server skeleton (docs/how-to/add-a-real-robot.md): one real robot, its own process.

It serves what an env server does, less what a real robot never has, plus the real-robot
extensions:

- control port (bearer `RS_ENV_TOKEN`): `env.public` (`discover`) and `robot.control`
  (`describe`; `open`, where a person resets the scene; `judge`, where a person judges; `close`;
  `estop`; `home`; `heartbeat`);
- agent port (a token per episode): `env.agent` (`robot`, `observe`, `execute`).

There is no `env.control`: a real robot has no ground truth to diagnose, no expert to pre-run
and no snapshot to take (`/v1/env.control/*` does not exist here, and the capabilities a robot
declares cannot say otherwise). `execute` carries joint positions only (`backend.NOT_CARRIED`),
so a robot whose profile takes no joint positions is refused at construction.

The server is a control loop around a `RobotBackend`. Each action of a batch is stretched over
as many control ticks (`RobotLimits.control_hz`) as the speed caps need, the whole batch is
checked against the limits before the first tick, the measured TCP is checked every tick, and
`safety.Safety` carries the e-stop and the heartbeat rules (read its docstring for the
conventions).
"""

import argparse
import functools
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anyio
import numpy as np
from pydantic import BaseModel, ConfigDict, SecretStr
from starlette.applications import Starlette

from roboscript.blobs import BlobStore, LocalBlobStore, put_frame
from roboscript.contracts.action import ActionBatch, ExecuteResult, StepState
from roboscript.contracts.capabilities import ManualReset, OpenSpec, unoffered_reset
from roboscript.contracts.env import (
    EnvAgent,
    EnvDiscovery,
    EnvPublic,
    EpisodeOpened,
    JudgeResult,
    ObserveRequest,
)
from roboscript.contracts.errors import EpisodeEnded, InfraError, ProtocolError
from roboscript.contracts.ids import EpisodeId, FrameId, TaskName, new_id
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import ArmState, RobotState
from roboscript.contracts.robot_server import RobotControl, RobotDescription
from roboscript.contracts.versions import VERSIONS
from roboscript.envs import Batches, Episodes, ServerConfig, control_token
from roboscript.rpc import make_app, routes, serve, write_ready
from roboscript.settings import config_hash, load_config_file
from roboscript_robots.base.backend import NOT_CARRIED, Measured, RobotBackend
from roboscript_robots.base.config import RobotConfig, load_robot_config
from roboscript_robots.base.operator import ConsoleOperator, Operator, ScriptedOperator
from roboscript_robots.base.safety import MAX_STEP_S, MEASURED_MARGIN_M, Safety
from roboscript_robots.base.spec import BackendKind, RobotSpec, load_robot

log = logging.getLogger(__name__)

_GRIPPER_SETTLED = 1e-4
_SPEED_SCALING_ROUNDS = 8


@dataclass(frozen=True)
class Clock:
    """Time as the server uses it: monotonic seconds and a sleep. Tests inject their own."""

    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = anyio.sleep


@dataclass(frozen=True)
class _Move:
    """One action planned: from `start` to `target` over `ticks` control ticks. Joints move in a
    straight line in joint space; a gripper command takes effect on the first tick."""

    start: list[float]
    target: list[float]
    ticks: int
    grippers: frozenset[int]

    def waypoint(self, tick: int) -> list[float]:
        if tick == self.ticks:
            return list(self.target)  # arrive exactly, not within rounding
        f = tick / self.ticks
        return [
            t if i in self.grippers else s + (t - s) * f
            for i, (s, t) in enumerate(zip(self.start, self.target, strict=True))
        ]


@dataclass
class _Live:
    task: TaskName
    instruction: str
    step_limit: int
    verdict: bool | None = None
    faulted: bool = False


class RobotServer:
    """The planes of a real robot over one `RobotBackend`, one episode at a time."""

    def __init__(
        self,
        config: RobotConfig,
        backend: RobotBackend,
        operator: Operator,
        blobs: BlobStore,
        *,
        clock: Clock | None = None,
    ) -> None:
        if not config.joint_position:
            raise NotImplementedError(
                f"{config.name}: this robot takes no joint-position actions, and the wire "
                "carries nothing else. " + NOT_CARRIED
            )
        self.config = config
        self.profile = config.profile()
        self.backend = backend
        self.operator = operator
        self.blobs = blobs
        self.clock = clock or Clock()
        self.safety = Safety(config.limits, self.clock.now)
        self.episodes = Episodes()
        self.agent_url = ""  # set by `serve_robot`
        self._capabilities = config.capabilities()
        self._batches = Batches()
        self._lock = anyio.Lock()  # one motion at a time; the e-stop never waits for it
        self._live: _Live | None = None
        self._commanded: list[float] | None = None
        self._step = 0
        self._halted_for: str | None = None
        self._joints = [
            i
            for i in range(config.action_dim)
            if i not in {arm.gripper_index for arm in self.profile.arms.values()}
        ]
        self._grippers = frozenset(arm.gripper_index for arm in self.profile.arms.values())

    # Lifecycle ------------------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to the hardware and read where it stands."""
        await self.backend.connect()
        self._commanded = (await self.backend.measure()).qpos

    async def stop(self) -> None:
        """Halt and release the hardware."""
        await self.backend.halt()
        await self.backend.disconnect()

    async def watch(self) -> None:
        """The heartbeat watchdog: halt the robot when the client falls silent, even while no
        action is running. Runs until cancelled."""
        while True:
            await self.clock.sleep(self.config.limits.heartbeat_timeout_s / 4)
            await self.enforce()

    async def enforce(self) -> None:
        """Halt the robot if the e-stop is latched or the heartbeat lapsed."""
        if (why := self.safety.stopped()) is not None:
            await self._halt(why)

    # env.public -----------------------------------------------------------------------------

    async def discover(self) -> EnvDiscovery:
        planes = {
            "env.public": VERSIONS["env.public"],
            "env.agent": VERSIONS["env.agent"],
            RobotControl.PLANE: RobotControl.VERSION,
        }
        return EnvDiscovery(
            contracts=planes,
            benchmark=self.config.name,
            embodiment=self.profile.name,
            upstream_commit=None,
            config_hash=config_hash(self.config),
            capabilities=self._capabilities.model_copy(update={"sensing": self.config.sensing()}),
        )

    # robot.control --------------------------------------------------------------------------

    async def describe(self) -> RobotDescription:
        return self.config.description()

    async def open(self, spec: OpenSpec) -> EpisodeOpened:
        if (refused := unoffered_reset(spec.reset.kind, self._capabilities)) is not None:
            raise ProtocolError("invalid_request", refused)
        assert isinstance(spec.reset, ManualReset)  # the only kind a real robot offers
        if spec.task not in self.config.tasks:
            detail = f"no task {spec.task!r}; tasks: {sorted(self.config.tasks)}"
            raise ProtocolError("invalid_request", detail)
        if self._commanded is None:
            raise InfraError("not_ready", "the robot server has not connected to the hardware")
        if (why := self.safety.estopped) is not None:
            raise InfraError("not_ready", f"emergency stop latched ({why}): restart the server")
        task = self.config.tasks[spec.task]
        self.episodes.end()  # a batch still running from the last episode notices and stops
        self.safety.disarm()
        self._live = None
        async with self._lock:
            if self.config.home_on_open and (why := await self._go_home(None)) is not None:
                raise InfraError("not_ready", f"could not send the robot home: {why}")
            await self.operator.confirm_reset(spec.task, spec.reset.prompt)
            if (why := self.safety.stopped()) is not None:
                raise InfraError("not_ready", why)
            self._commanded = (await self.backend.measure()).qpos
            self._step = 0
            self._batches = Batches()
            self._halted_for = None
            episode_id, token = self.episodes.open()
            instruction = spec.instruction or task.instruction
            self._live = _Live(spec.task, instruction, task.step_limit)
            self.safety.arm()
        return EpisodeOpened(
            episode_id=episode_id,
            agent_url=self.agent_url,
            agent_token=token,
            instruction=instruction,
            step_limit=task.step_limit,
            embodiment=self.profile.name,
            cameras=list(self.config.cameras),
            solvable=True,
        )

    async def judge(self, episode_id: EpisodeId) -> JudgeResult:
        live = self._episode(episode_id)
        faulted = live.faulted or self.safety.stopped() is not None
        if live.verdict is None:  # a person is asked once; a stopped episode is not a success
            live.verdict = (
                False if faulted else await self.operator.judge(live.task, live.instruction)
            )
        return JudgeResult(
            episode_id=episode_id,
            success=live.verdict,
            success_step=None,
            steps_used=self._step,
            step_limit=live.step_limit,
            faulted=faulted,
        )

    async def close(self, episode_id: EpisodeId) -> None:
        self.episodes.close(episode_id)
        self.safety.disarm()

    async def estop(self, reason: str) -> None:
        """Latch the e-stop and halt the hardware now. Takes no lock and waits for nothing: the
        motion in flight sees it on its next tick."""
        self.safety.estop(reason)
        await self._halt(f"emergency stop: {reason}")

    async def home(self, episode_id: EpisodeId) -> None:
        self._episode(episode_id)
        async with self._lock:
            why = await self._interrupted(episode_id) or await self._go_home(episode_id)
        if why is not None:
            raise EpisodeEnded(why)

    async def heartbeat(self, episode_id: EpisodeId) -> None:
        self._episode(episode_id)
        if (why := self.safety.stopped()) is not None:
            await self._halt(why)
            raise EpisodeEnded(why)
        self.safety.beat()

    # env.agent ------------------------------------------------------------------------------

    async def robot(self, episode_id: EpisodeId) -> RobotState:
        live = self._episode(episode_id)
        return self._robot_state(await self.backend.measure(), live)

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation:
        live = self._episode(episode_id)
        names = request.cameras if request.cameras is not None else list(self.config.cameras)
        if unknown := sorted(set(names) - set(self.config.cameras)):
            detail = f"no cameras {unknown}; cameras: {sorted(self.config.cameras)}"
            raise ProtocolError("invalid_request", detail)
        captures = await self.backend.capture(names, request.depth and self.config.depth)
        frames = {
            name: await put_frame(self.blobs, c.camera, c.rgb, c.depth)
            for name, c in captures.items()
        }
        robot = self._robot_state(await self.backend.measure(), live)
        return Observation(
            frame_id=FrameId(new_id("frame")),
            step=robot.step,
            cameras=frames,
            robot=robot,
            instruction=live.instruction,
        )

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        self._episode(episode_id)
        return await self._batches.run(batch, lambda: self._execute(episode_id, batch))

    # Motion ---------------------------------------------------------------------------------

    async def _execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        async with self._lock:
            live = self._episode(episode_id)
            if (why := await self._interrupted(episode_id)) is not None:
                raise EpisodeEnded(why)
            moves = []
            here = list(self._commanded or [])
            for index, step in enumerate(batch.steps):
                try:
                    moves.append(self._plan(here, step.qpos))
                except ProtocolError as error:
                    raise ProtocolError(
                        "invalid_request", f"step {index}: {error.detail}"
                    ) from None
                here = list(step.qpos)  # the next step starts where this one ends
            states: list[StepState] = []
            for index, (step, move) in enumerate(zip(batch.steps, moves, strict=True)):
                closing = [
                    arm.gripper_index
                    for arm in self.profile.arms.values()
                    if step.qpos[arm.gripper_index] < arm.gripper.open
                ]
                for repeat in range(1 + (step.max_repeats if step.until_gripper_stable else 0)):
                    if self._step >= live.step_limit:
                        return _result(batch, states, "step_limit")
                    before = (await self.backend.measure()).qpos
                    hold = _Move(move.target, move.target, 1, self._grippers)
                    if (
                        why := await self._run(move if repeat == 0 else hold, episode_id)
                    ) is not None:
                        return _result(batch, states, "error", why)
                    self._step += 1
                    measured = await self.backend.measure()
                    states.append(StepState(index=index, robot=self._robot_state(measured, live)))
                    moved = [abs(measured.qpos[i] - before[i]) for i in closing]
                    if max(moved, default=0.0) < _GRIPPER_SETTLED:
                        break  # the closing grippers settled
        return _result(batch, states, "completed")

    async def _go_home(self, episode_id: EpisodeId | None) -> str | None:
        """Send the robot to its configured home under the limits; why it stopped, if it did."""
        assert self._commanded is not None
        move = self._plan(self._commanded, self.config.home_vector())
        return await self._run(move, episode_id)

    async def _run(self, move: _Move, episode_id: EpisodeId | None) -> str | None:
        """Drive a planned move tick by tick; why it stopped early (None: it arrived)."""
        limits = self.config.limits
        for tick in range(1, move.ticks + 1):
            if (why := await self._interrupted(episode_id)) is not None:
                return why
            waypoint = move.waypoint(tick)
            await self.backend.command_joints(waypoint)
            self._commanded = waypoint
            await self.clock.sleep(1.0 / limits.control_hz)
            if (why := self._breach(await self.backend.measure())) is not None:
                self.safety.fault(why)
        return await self._interrupted(episode_id)

    async def _interrupted(self, episode_id: EpisodeId | None) -> str | None:
        """Why motion may not go on (the e-stop, a lapsed heartbeat, a measured breach, the
        episode closed); the hardware is halted when this first says so."""
        if (why := self.safety.stopped()) is not None:
            if self._live is not None:
                self._live.faulted = True
            await self._halt(why)
            return why
        if episode_id is not None and self.episodes.current != episode_id:
            return "the episode was closed"
        return None

    async def _halt(self, why: str) -> None:
        if self._halted_for != why:  # once per stop; the hardware is already holding after that
            self._halted_for = why
            await self.backend.halt()

    # Limits ---------------------------------------------------------------------------------

    def _plan(self, start: list[float], target: list[float] | tuple[float, ...]) -> _Move:
        """The move from `start` to `target`, or `ProtocolError` if `target` or the way there
        breaks a limit. Stretched over enough ticks that neither the joint nor the TCP speed cap
        is exceeded; every waypoint's TCP must lie in the workspace box."""
        target = list(target)
        if len(target) != self.config.action_dim:
            detail = f"a qpos has {self.config.action_dim} values, got {len(target)}"
            raise ProtocolError("invalid_request", detail)
        self._check_target(target)
        limits = self.config.limits
        per_tick = limits.max_joint_speed_rad_s / limits.control_hz
        ticks = max(1, math.ceil(max(abs(target[i] - start[i]) for i in self._joints) / per_tick))
        for _ in range(_SPEED_SCALING_ROUNDS):
            if ticks / limits.control_hz > MAX_STEP_S:
                detail = f"this action would take over {MAX_STEP_S:g} s at the speed caps"
                raise ProtocolError("invalid_request", detail)
            move = _Move(start, target, ticks, self._grippers)
            ratio = self._tcp_ratio(move)
            if ratio <= 1.0:
                return move
            ticks = math.ceil(ticks * ratio * 1.05)
        raise ProtocolError("invalid_request", "this action cannot be kept under the TCP speed cap")

    def _check_target(self, target: list[float]) -> None:
        for name, arm in self.profile.arms.items():
            low, high = sorted((arm.gripper.closed, arm.gripper.open))
            if not low - 1e-9 <= target[arm.gripper_index] <= high + 1e-9:
                command = target[arm.gripper_index]
                detail = f"the {name} gripper command {command:g} is outside [{low:g}, {high:g}]"
                raise ProtocolError("invalid_request", detail)
            start, stop = arm.action
            if (past := arm.past_stops([target[start:stop]])) is not None:
                raise ProtocolError("invalid_request", f"the {name} arm: {past}")
        span, torso = self.config.torso_range(), self.config.torso
        if span is not None and torso is not None:
            for joint, value, (low, high) in zip(
                torso.joints, target[span[0] : span[1]], torso.stops, strict=True
            ):
                if not low <= value <= high:
                    detail = (
                        f"torso joint {joint!r} would be at {value:g}, "
                        f"outside its stops [{low:g}, {high:g}]"
                    )
                    raise ProtocolError("invalid_request", detail)

    def _tcp_ratio(self, move: _Move) -> float:
        """The highest TCP speed along the move as a multiple of the cap, refusing a move whose
        TCP leaves the workspace box on the way."""
        limits = self.config.limits
        low, high = np.array(limits.workspace_min), np.array(limits.workspace_max)
        before = self._tcps(move.start)
        fastest = 0.0
        for tick in range(1, move.ticks + 1):
            now = self._tcps(move.waypoint(tick))
            for arm, point in now.items():
                if not (np.all(point >= low) and np.all(point <= high)):
                    x, y, z = point
                    detail = (
                        f"the {arm} TCP would leave the workspace at ({x:.3f}, {y:.3f}, {z:.3f}) m "
                        f"(box {limits.workspace_min} to {limits.workspace_max})"
                    )
                    raise ProtocolError("invalid_request", detail)
                fastest = max(fastest, float(np.linalg.norm(point - before[arm])))
            before = now
        return fastest * limits.control_hz / limits.max_tcp_speed_m_s

    def _tcps(self, qpos: list[float]) -> dict[str, np.ndarray]:
        ee = self.backend.forward_kinematics(qpos)
        return {arm: np.array(self.profile.tcp_pose(arm, pose).p) for arm, pose in ee.items()}

    def _breach(self, measured: Measured) -> str | None:
        """A measured TCP past the box (by `MEASURED_MARGIN_M`), in words."""
        limits = self.config.limits
        low = np.array(limits.workspace_min) - MEASURED_MARGIN_M
        high = np.array(limits.workspace_max) + MEASURED_MARGIN_M
        for arm, ee in measured.ee.items():
            point = np.array(self.profile.tcp_pose(arm, ee).p)
            if not (np.all(point >= low) and np.all(point <= high)):
                where = tuple(round(float(v), 3) for v in point)
                return f"the measured {arm} TCP left the workspace at {where}"
        return None

    # State ----------------------------------------------------------------------------------

    def _episode(self, episode_id: EpisodeId) -> _Live:
        self.episodes.check(episode_id)
        assert self._live is not None  # an open episode has its state
        return self._live

    def _robot_state(self, measured: Measured, live: _Live) -> RobotState:
        assert self._commanded is not None
        arms = {}
        for name, arm in self.profile.arms.items():
            start, stop = arm.action
            ee = measured.ee[name]
            arms[name] = ArmState(
                ee=ee,
                tcp=self.profile.tcp_pose(name, ee),
                joints=measured.qpos[start:stop],
                joints_cmd=self._commanded[start:stop],
                gripper=measured.qpos[arm.gripper_index],
                gripper_cmd=self._commanded[arm.gripper_index],
            )
        return RobotState(
            step=self._step,
            step_limit=live.step_limit,
            arms=arms,
            qpos_cmd=list(self._commanded),
        )


def _result(
    batch: ActionBatch,
    states: list[StepState],
    stopped: Literal["completed", "step_limit", "error"],
    error: str | None = None,
) -> ExecuteResult:
    return ExecuteResult(
        batch_id=batch.batch_id,
        states=states,
        executed=len(states),
        stopped=stopped,
        error=error,
    )


def robot_apps(server: RobotServer, token: SecretStr) -> tuple[Starlette, Starlette]:
    """The control-port app and the agent-port app of `server`."""
    public = routes(EnvPublic, server, token=None)
    control = make_app(public, routes(RobotControl, server, token=token))
    agent = make_app(public, routes(EnvAgent, server, token=None, allow=server.episodes.allow))
    return control, agent


class RobotServerConfig(BaseModel):
    """What a robot server process is told: `robot`, a name in `roboscript.robots`; `backend`,
    `fake` or `hardware`; `robot_file`, a YAML of that robot's configuration class to use instead
    of the registered one (calibrated limits, this machine's tasks); `operator`, who resets and
    judges (`console`: on this terminal; `scripted`: resets at once and judges every episode a
    failure, for tutorials)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    robot: str
    backend: BackendKind = "fake"
    robot_file: Path | None = None
    operator: Literal["console", "scripted"] = "console"
    blob_dir: Path
    server: ServerConfig = ServerConfig()


def build_server(config: RobotServerConfig, spec: RobotSpec | None = None) -> RobotServer:
    """The server `config` describes. `spec`: the robot when it is not registered in
    `roboscript.robots` (a deployment directory's own, as `examples/realRobots/_template`'s);
    it must be named `config.robot`."""
    if spec is None:
        spec = load_robot(config.robot)
    elif spec.name != config.robot:
        raise ValueError(
            f"the server is configured for {config.robot!r}, the robot is {spec.name!r}"
        )
    robot = spec.config
    if config.robot_file is not None:
        robot = load_robot_config(config.robot_file, type(spec.config))
        if robot.name != spec.name:
            raise ValueError(f"{config.robot_file} configures {robot.name!r}, not {spec.name!r}")
    operator: Operator = ConsoleOperator() if config.operator == "console" else ScriptedOperator()
    return RobotServer(
        robot, spec.backend(config.backend, robot), operator, LocalBlobStore(config.blob_dir)
    )


async def serve_robot(
    server: RobotServer, config: ServerConfig, ready_file: Path | None, token: SecretStr
) -> None:
    """Serve `server` until SIGINT or SIGTERM: connect to the hardware, serve both ports, run the
    heartbeat watchdog; on the way out halt and release the hardware. `ready_file` gets both
    ports' URLs once they listen."""
    await server.start()
    try:
        control_app, agent_app = robot_apps(server, token)
        async with anyio.create_task_group() as outer:
            watching = anyio.CancelScope()

            async def watch() -> None:
                with watching:
                    await server.watch()

            outer.start_soon(watch)
            async with anyio.create_task_group() as tg:
                server.agent_url = await tg.start(
                    functools.partial(serve, agent_app, host=config.host, port=config.agent_port)
                )
                url = await tg.start(
                    functools.partial(
                        serve, control_app, host=config.host, port=config.control_port
                    )
                )
                log.info(
                    "robot.control and env.public at %s, env.agent at %s", url, server.agent_url
                )
                if ready_file is not None:
                    planes = {
                        "robot.control": url,
                        "env.public": url,
                        "env.agent": server.agent_url,
                    }
                    write_ready(ready_file, url, planes)
            watching.cancel()
    finally:
        with anyio.CancelScope(shield=True):
            await server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve one real robot on a control and an agent port."
    )
    parser.add_argument("--config", type=Path, required=True, help="a RobotServerConfig YAML file")
    parser.add_argument("--ready-file", type=Path, help="where to write the planes' URLs")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config_file(args.config, RobotServerConfig)
    anyio.run(serve_robot, build_server(config), config.server, args.ready_file, control_token())


if __name__ == "__main__":
    main()
