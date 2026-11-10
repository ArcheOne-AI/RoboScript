"""Conformance checks for robots (`roboscript.robots`, docs/how-to/add-a-real-robot.md).

A robot conforms when, served by `RobotServer` over its software backend (`RobotSpec.fake`),
it behaves as a real robot's planes must: it declares what a real robot declares (success judged
from outside, a manual reset, no ground truth and no expert, and no `env.control` plane at all),
is registered as an embodiment profile and a benchmark descriptor under its own name, serves its
planes with their tokens, has a person reset and judge its episodes, counts and limits its
actions and refuses a batch that breaks a limit before anything moves, stops at the step limit,
stops at once when the e-stop is pressed, and stops when its client's heartbeat lapses.

The checks drive the server over RPC in process, with a fake clock (no waiting), the way
`roboscript.testing.env` drives an env. Use them from pytest by subclassing `RobotConformance`
as a `Test*` class with a `robot_target` fixture, or without pytest via
`roboscript.testing.precheck`:

```python
@pytest.fixture(params=registered("roboscript.robots", "roboscript-myrobot"))
def robot_target(request, tmp_path):
    return RobotTarget(request.param, tmp_path)

class TestMyRobot(RobotConformance): ...
```
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import anyio.lowlevel
import httpx
from pydantic import SecretStr

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.capabilities import ManualReset, OpenSpec, SeedReset
from roboscript.contracts.env import EnvAgent, EnvPublic, EpisodeOpened, ObserveRequest
from roboscript.contracts.errors import EpisodeEnded, InfraError, ProtocolError, RSError
from roboscript.contracts.ids import BatchId, TaskName, new_id
from roboscript.contracts.robot_server import RobotControl
from roboscript.embodiments import load_embodiment
from roboscript.envs import load_benchmark
from roboscript.rpc import client
from roboscript.testing import unit
from roboscript_robots.base import Clock, RobotServer, ScriptedOperator, load_robot, robot_apps

TOKEN = SecretStr("robot-control-token-for-tests")


@dataclass(frozen=True)
class RobotTarget:
    """`ref`: the robot's name in `roboscript.robots`; `blob_dir`: where its frames go."""

    ref: str
    blob_dir: Path


class FakeTime:
    """The server's clock in tests: sleeping advances it, so no check waits."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds
        await anyio.lowlevel.checkpoint()

    def advance(self, seconds: float) -> None:
        self.t += seconds


@dataclass(frozen=True)
class Running:
    """A robot served in process: the server, its fake time and operator, and RPC clients."""

    server: RobotServer
    time: FakeTime
    operator: ScriptedOperator
    control: RobotControl
    public: EnvPublic
    control_http: httpx.AsyncClient
    agent_http: httpx.AsyncClient

    def agent(self, opened: EpisodeOpened) -> EnvAgent:
        return client(
            EnvAgent,  # type: ignore[type-abstract]
            opened.agent_url,
            token=opened.agent_token,
            http=self.agent_http,
            timeout=10,
        )

    async def open(self, task: TaskName | None = None) -> EpisodeOpened:
        name = task if task is not None else next(iter(self.server.config.tasks))
        return await self.control.open(
            OpenSpec(task=name, reset=ManualReset(prompt="set the scene"))
        )


@asynccontextmanager
async def running(target: RobotTarget) -> AsyncIterator[Running]:
    """Serve `target`'s robot over its software backend, both ports in process."""
    spec = load_robot(target.ref)
    fake_time = FakeTime()
    operator = ScriptedOperator([True])
    server = RobotServer(
        spec.config,
        spec.backend("fake"),
        operator,
        LocalBlobStore(target.blob_dir),
        clock=Clock(fake_time.now, fake_time.sleep),
    )
    server.agent_url = "http://agent"
    await server.start()
    control_app, agent_app = robot_apps(server, TOKEN)
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(control_app)) as control_http,
        httpx.AsyncClient(transport=httpx.ASGITransport(agent_app)) as agent_http,
    ):
        yield Running(
            server=server,
            time=fake_time,
            operator=operator,
            control=client(
                RobotControl,  # type: ignore[type-abstract]
                "http://control",
                token=TOKEN,
                http=control_http,
                timeout=10,
            ),
            public=client(
                EnvPublic,  # type: ignore[type-abstract]
                "http://control",
                token=None,
                http=control_http,
                timeout=10,
            ),
            control_http=control_http,
            agent_http=agent_http,
        )
    await server.stop()


class RobotConformance:
    """The checks, as `test_*` methods taking a `RobotTarget`."""

    @unit
    async def test_declares_what_a_real_robot_declares(self, robot_target: RobotTarget) -> None:
        spec = load_robot(robot_target.ref)
        assert spec.name == robot_target.ref == spec.config.name
        declared = spec.config.capabilities()
        assert (declared.success, declared.reset) == (
            frozenset({"external"}),
            frozenset({"manual"}),
        )
        assert declared.real_robot and not declared.truth and not declared.expert
        description = spec.config.description()
        assert description.capabilities == declared and description.embodiment == spec.name

    @unit
    async def test_is_registered_as_an_embodiment_and_a_benchmark(
        self, robot_target: RobotTarget
    ) -> None:
        spec = load_robot(robot_target.ref)
        profile = spec.config.profile()
        assert load_embodiment(robot_target.ref) == profile, "the shipped profile drifted"
        benchmark = load_benchmark(robot_target.ref)
        assert benchmark.profile() == profile
        assert benchmark.tasks.keys() == spec.config.tasks.keys()
        assert benchmark.capabilities == spec.config.capabilities()

    @unit
    async def test_discover(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            found = await run.public.discover()
            assert found.embodiment == robot_target.ref and found.capabilities.real_robot
            assert RobotControl.PLANE in found.contracts
            assert not found.capabilities.truth and not found.capabilities.expert
            assert found.capabilities.sensing is not None and found.config_hash
            assert not found.capabilities.sensing.scene_identity, "a person's scene is not a seed's"

    @unit
    async def test_control_plane_needs_its_token(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            for token in (None, SecretStr("not-the-control-token")):
                control = client(
                    RobotControl,  # type: ignore[type-abstract]
                    "http://control",
                    token=token,
                    http=run.control_http,
                    timeout=10,
                )
                await _raises(ProtocolError, "unauthorized", control.estop("probe"))

    @unit
    async def test_has_no_ground_truth_plane(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            headers = {"Authorization": f"Bearer {TOKEN.get_secret_value()}"}
            for method in ("diagnose", "gt_snapshot", "expert_trace"):
                answer = await run.control_http.post(
                    f"http://control/v1/env.control/{method}", headers=headers, json={}
                )
                assert answer.status_code == 404, f"env.control/{method} is served"

    @unit
    async def test_refuses_a_reset_it_does_not_offer(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            task = next(iter(run.server.config.tasks))
            spec = OpenSpec(task=task, reset=SeedReset(seed=1))
            await _raises(ProtocolError, "invalid_request", run.control.open(spec))
            unknown = OpenSpec(task=TaskName("no_such_task"), reset=ManualReset(prompt="x"))
            await _raises(ProtocolError, "invalid_request", run.control.open(unknown))

    async def test_lifecycle(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            profile = run.server.profile
            opened = await run.open()
            assert opened.solvable and opened.instruction and opened.agent_url
            assert opened.embodiment == robot_target.ref
            assert opened.step_limit is not None and opened.cameras == list(profile.cameras)
            assert run.operator.resets, "a person was asked to set the scene up"
            agent = run.agent(opened)
            state = await agent.robot(opened.episode_id)
            assert set(state.arms) == set(profile.arms) and state.step == 0
            assert len(state.qpos_cmd) == profile.action_dim
            seen = await agent.observe(opened.episode_id, ObserveRequest())
            assert set(seen.cameras) == set(profile.cameras) and seen.instruction
            await _raises(
                ProtocolError,
                "invalid_request",
                agent.observe(opened.episode_id, ObserveRequest(cameras=["no_such_camera"])),
            )
            verdict = await run.control.judge(opened.episode_id)
            assert verdict.success and verdict.success_step is None and not verdict.faulted
            assert (await run.control.judge(opened.episode_id)) == verdict, "a person is asked once"
            await run.control.close(opened.episode_id)
            await _raises(EpisodeEnded, "episode_ended", agent.robot(opened.episode_id))
            token = SecretStr("not-this-episodes-token")
            other = client(
                EnvAgent,  # type: ignore[type-abstract]
                opened.agent_url,
                token=token,
                http=run.agent_http,
                timeout=10,
            )
            await _raises(ProtocolError, "unauthorized", other.robot(opened.episode_id))

    async def test_execute_counts_steps_and_is_idempotent(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            opened = await run.open()
            agent = run.agent(opened)
            hold = run.server.config.home_vector()
            batch = _batch(hold, hold)
            result = await agent.execute(opened.episode_id, batch)
            assert (result.stopped, result.executed) == ("completed", 2)
            assert [s.robot.step for s in result.states] == [1, 2]
            assert await agent.execute(opened.episode_id, batch) == result, (
                "same batch, same result"
            )
            assert (await agent.robot(opened.episode_id)).step == 2
            other = _batch(hold)
            changed = ActionBatch(batch_id=batch.batch_id, steps=other.steps)
            await _raises(
                ProtocolError, "invalid_request", agent.execute(opened.episode_id, changed)
            )

    async def test_a_batch_that_breaks_a_limit_is_refused_before_anything_moves(
        self, robot_target: RobotTarget
    ) -> None:
        async with running(robot_target) as run:
            opened = await run.open()
            agent = run.agent(opened)
            config, profile = run.server.config, run.server.profile
            hold = config.home_vector()
            first = next(iter(profile.arms.values()))
            past_stop = list(hold)
            past_stop[first.action[0]] = first.stops[0][1] + 1.0  # type: ignore[index]
            wide_grip = list(hold)
            wide_grip[first.gripper_index] = first.gripper.open + 1.0
            before = await agent.robot(opened.episode_id)
            for bad in (hold[:-1], past_stop, wide_grip):
                batch = _batch(hold, bad)  # the second step is bad: the first must not run either
                await _raises(
                    ProtocolError, "invalid_request", agent.execute(opened.episode_id, batch)
                )
                assert await agent.robot(opened.episode_id) == before, "something moved"

    async def test_stops_at_the_step_limit(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            opened = await run.open()
            agent = run.agent(opened)
            hold = run.server.config.home_vector()
            assert opened.step_limit is not None
            result = await agent.execute(
                opened.episode_id, _batch(*[hold] * (opened.step_limit + 3))
            )
            assert (result.stopped, result.executed) == ("step_limit", opened.step_limit)

    async def test_home_returns_to_the_home_pose(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            opened = await run.open()
            await run.control.home(opened.episode_id)
            state = await run.agent(opened).robot(opened.episode_id)
            assert state.qpos_cmd == run.server.config.home_vector() and state.step == 0

    async def test_the_estop_has_priority(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            opened = await run.open()
            agent = run.agent(opened)
            hold = run.server.config.home_vector()
            await run.control.estop("test")
            await _raises(
                EpisodeEnded, "episode_ended", agent.execute(opened.episode_id, _batch(hold))
            )
            await _raises(EpisodeEnded, "episode_ended", run.control.home(opened.episode_id))
            await _raises(EpisodeEnded, "episode_ended", run.control.heartbeat(opened.episode_id))
            verdict = await run.control.judge(opened.episode_id)
            assert verdict.faulted and not verdict.success, "a stopped episode is not a success"
            assert (await agent.robot(opened.episode_id)).step == 0, "reading still works"
            await run.control.close(opened.episode_id)
            await _raises(InfraError, "not_ready", run.open())  # the latch holds until a restart

    async def test_a_missed_heartbeat_stops_the_robot(self, robot_target: RobotTarget) -> None:
        async with running(robot_target) as run:
            timeout = run.server.config.limits.heartbeat_timeout_s
            opened = await run.open()
            agent = run.agent(opened)
            hold = run.server.config.home_vector()
            for _ in range(3):  # heartbeats keep it moving
                run.time.advance(timeout * 0.6)
                await run.control.heartbeat(opened.episode_id)
                await agent.execute(opened.episode_id, _batch(hold))
            run.time.advance(timeout * 1.1)  # silence
            await _raises(
                EpisodeEnded, "episode_ended", agent.execute(opened.episode_id, _batch(hold))
            )
            await _raises(EpisodeEnded, "episode_ended", run.control.heartbeat(opened.episode_id))
            assert (await run.control.judge(opened.episode_id)).faulted
            await run.control.close(opened.episode_id)
            again = await run.open()  # a fault ends the episode, not the server
            result = await run.agent(again).execute(again.episode_id, _batch(hold))
            assert result.stopped == "completed"


def _batch(*steps: list[float]) -> ActionBatch:
    return ActionBatch(
        batch_id=BatchId(new_id("batch")), steps=[ActionStep(qpos=list(s)) for s in steps]
    )


async def _raises(kind: type[RSError], code: str | None, call: Any) -> None:
    try:
        await call
    except kind as error:
        assert code is None or error.code == code, f"expected {code}, got {error.code}: {error}"
        return
    raise AssertionError(f"expected {kind.__name__}({code})")
