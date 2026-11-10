"""The client-driven bridge (docs/concepts/embodied-core.md,
docs/concepts/learning.md): the `env.agent` plane of an episode whose loop a benchmark's own
evaluation client drives. Toolsets reach a
`BridgeEnv` as they reach an env server (through the Hub's env client), so they cannot tell the
two apart; the client feeds it through the Agent host:

- `submit(observation)`: the client's observation at the start of its turn. `observe` answers
  with the latest one (a new frame id, the cameras asked for); after an executed batch it waits
  for the next, since the robot has moved.
- `poll()`: the batch `execute` queued, if any. `execute` completes when `report(result)` brings
  the client's states after every step (its step count and success latch); `robot` is the latest
  state, submitted or reported.
- `end(reason)`: the benchmark ended the episode (success latch or step limit), so later agent
  calls get `EpisodeEnded`. A report of a failed execution faults the episode (`sim_fault`).
- `close()`: the agent's episode is over; a batch the client has not polled yet is withdrawn.

Every wait is bounded (`BridgeConfig`): a client that stops answering faults the episode.
"""

import secrets
from collections.abc import Callable
from functools import partial

import anyio
from pydantic import BaseModel, ConfigDict, PositiveFloat, SecretStr

from roboscript.contracts.action import ActionBatch, ExecuteResult
from roboscript.contracts.env import EnvAgent, ObserveRequest
from roboscript.contracts.errors import EpisodeEnded, InfraError, ProtocolError, RSError
from roboscript.contracts.ids import BatchId, EpisodeId, FrameId, new_id
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import RobotState
from roboscript.contracts.tools import EnvBinding
from roboscript.envs.episodes import Batches
from roboscript.records import replay
from roboscript.stack import stall


class BridgeConfig(BaseModel):
    """How long `observe` and `robot` wait for the client's next observation (keep it below the
    sensing tools' timeout, so that a silent client ends the episode as infrastructure, and above
    the time a client's relay may take to hand one over: its `submit_timeout_s`)
    and how long `execute` waits for the report of its batch. Both waits are put off by the time
    the host stood still meanwhile (`roboscript.stack.stall`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observe_timeout_s: PositiveFloat = 50.0
    report_timeout_s: PositiveFloat = 300.0


class BridgeEnv:
    """One episode's agent plane, fed by a benchmark client."""

    def __init__(self, episode_id: EpisodeId, config: BridgeConfig) -> None:
        self.episode_id = episode_id
        self.config = config
        self._batches = Batches()
        self._observation: Observation | None = None
        self._robot: RobotState | None = None
        self._fresh = False  # no step ran since the latest observation
        self._queued: ActionBatch | None = None  # until reported (or withdrawn before a poll)
        self._handed = False  # the client polled the queued batch
        self._reports: dict[BatchId, ExecuteResult] = {}
        self._ended: str | None = None
        self._latched = False  # a reported batch stopped at the success latch
        self._fault: str | None = None
        self._closed = False
        self._changed = anyio.Event()

    # The client's side, through the Agent host

    def submit(self, observation: Observation) -> None:
        self._observation, self._robot, self._fresh = observation, observation.robot, True
        self._notify()

    def poll(self) -> ActionBatch | None:
        """The queued batch, if any; it stays queued (polls hand it out again) until reported."""
        if self._queued is not None:
            self._handed = True
        return self._queued

    def report(self, result: ExecuteResult) -> None:
        """The client ran the batch it polled (also one polled before the episode closed)."""
        queued = self._queued
        if queued is None or queued.batch_id != result.batch_id or not self._handed:
            raise ProtocolError("invalid_request", f"no batch {result.batch_id} awaits a report")
        self._queued = None
        self._reports[result.batch_id] = result
        if result.states:
            self._robot, self._fresh = result.states[-1].robot, False
        self._latched = self._latched or result.stopped == "success"
        if result.stopped == "error":
            self._fault = self._fault or f"the benchmark's client failed: {result.error}"
        self._notify()

    @property
    def ended(self) -> str | None:
        """Why the benchmark ended the episode (None: it runs, so its success latch has not
        fired; the client ends it there), `success` already once a reported batch stopped at the
        latch (the client's `end` follows its report)."""
        return self._ended or ("success" if self._latched else None)

    def end(self, reason: str) -> None:
        """The benchmark ended the episode (e.g. `success`, `step_limit`)."""
        self._ended = self._ended or reason
        self._notify()

    def close(self) -> None:
        """The agent's episode is over: withdraw a batch the client has not polled."""
        self._closed = True
        if not self._handed:
            self._queued = None
        self._notify()

    # The agent plane (EnvAgent)

    async def robot(self, episode_id: EpisodeId) -> RobotState:
        self._check(episode_id)
        await self._await(lambda: self._robot is not None, "observation")
        assert self._robot is not None
        return self._robot

    async def observe(self, episode_id: EpisodeId, request: ObserveRequest) -> Observation:
        self._check(episode_id)
        await self._await(lambda: self._fresh, "observation")
        observation = self._observation
        assert observation is not None
        names = request.cameras if request.cameras is not None else list(observation.cameras)
        if unknown := sorted(set(names) - set(observation.cameras)):
            detail = f"no cameras {unknown}; cameras: {sorted(observation.cameras)}"
            raise ProtocolError("invalid_request", detail)
        cameras = {
            name: observation.cameras[name]
            if request.depth
            else observation.cameras[name].model_copy(update={"depth": None})
            for name in names
        }
        update = {"frame_id": FrameId(new_id("frame")), "cameras": cameras}
        return observation.model_copy(update=update)

    async def execute(self, episode_id: EpisodeId, batch: ActionBatch) -> ExecuteResult:
        self._check(episode_id)
        if (error := self._over()) is not None:
            raise error
        if not batch.stop_on_success:
            detail = "a benchmark's client stops at its success latch: set stop_on_success"
            raise ProtocolError("invalid_request", detail)
        if self._robot is not None:
            size = len(self._robot.qpos_cmd)
            if wrong := sorted({len(step.qpos) for step in batch.steps} - {size}):
                raise ProtocolError("invalid_request", f"a qpos has {size} values, got {wrong}")
        result = await self._batches.run(batch, partial(self._relay, batch))
        tape = replay.active() if result.stopped == "success" else None
        if tape is not None and tape.mode == "replay":
            await self._ending()
        return result

    async def _ending(self) -> None:
        """A replay waits here for the client's `end` that follows a report at the success latch.
        A recorded agent's next call came a model call later, after the `end`; a replayed one
        comes at once and would find the episode still running (a race the recording does not
        have)."""
        with anyio.move_on_after(self.config.report_timeout_s):
            while self._ended is None and not self._closed and self._fault is None:
                await self._changed.wait()

    # Internals

    async def _relay(self, batch: ActionBatch) -> ExecuteResult:
        """Queue `batch` for the client and wait for its report."""
        self._queued, self._handed = batch, False
        self._notify()
        seconds = self.config.report_timeout_s
        try:
            with stall.stretched(seconds) as scope:
                while batch.batch_id not in self._reports and self._over() is None:
                    await self._changed.wait()
            if scope.cancelled_caught:
                raise TimeoutError
        except TimeoutError:
            what = "ran" if self._handed else "polled"
            self._fault = (
                self._fault or f"the benchmark's client never {what} batch {batch.batch_id}"
            )
            raise InfraError("timeout", f"no report of the batch within {seconds:g} s") from None
        finally:
            if self._queued is batch and not self._handed:
                self._queued = None  # withdrawn: the client never saw it
        if (result := self._reports.pop(batch.batch_id, None)) is not None:
            return result
        error = self._over()
        assert error is not None
        raise error

    async def _await(self, ready: Callable[[], bool], what: str) -> None:
        """Wait until `ready()` or the episode is over (then raise why)."""
        seconds = self.config.observe_timeout_s
        try:
            with stall.stretched(seconds) as scope:
                while not ready() and self._over() is None:
                    await self._changed.wait()
            if scope.cancelled_caught:
                raise TimeoutError
        except TimeoutError:
            self._fault = self._fault or f"the benchmark's client sent no {what} in {seconds:g} s"
            raise InfraError(
                "timeout", f"no {what} from the benchmark within {seconds:g} s"
            ) from None
        if (error := self._over()) is not None:
            raise error

    def _over(self) -> RSError | None:
        if self._closed:
            return EpisodeEnded("the episode is closed")
        if self._fault is not None:
            return InfraError("sim_fault", self._fault)
        if self._ended is not None:
            return EpisodeEnded(f"the benchmark ended the episode ({self._ended})")
        return None

    def _check(self, episode_id: EpisodeId) -> None:
        if episode_id != self.episode_id:
            raise ProtocolError("not_found", f"no episode {episode_id}")

    def _notify(self) -> None:
        self._changed.set()
        self._changed = anyio.Event()


class Bridges:
    """The open bridges of a process; `client` is the env client a Hub is built with."""

    def __init__(self) -> None:
        self._open: dict[EpisodeId, BridgeEnv] = {}

    def open(self, config: BridgeConfig) -> tuple[BridgeEnv, EnvBinding]:
        """A new bridge and the binding a Hub episode reaches it by."""
        bridge = BridgeEnv(EpisodeId(new_id("bridge")), config)
        self._open[bridge.episode_id] = bridge
        token = SecretStr(secrets.token_urlsafe(16))  # in-process: nothing checks it
        binding = EnvBinding(url="inproc://bridge", episode_id=bridge.episode_id, token=token)
        return bridge, binding

    def close(self, bridge: BridgeEnv) -> None:
        bridge.close()
        self._open.pop(bridge.episode_id, None)

    def client(self, binding: EnvBinding) -> EnvAgent:
        bridge = self._open.get(binding.episode_id)
        if bridge is None:
            raise ProtocolError("not_found", f"no open bridge for env episode {binding.episode_id}")
        return bridge
