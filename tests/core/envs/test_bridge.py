"""The client-driven bridge: the agent plane of an episode that a benchmark's own client drives. The
client here plays with `MockEnv` as its simulator: it submits observations, polls the batches that
`execute` queues, runs them and reports the states."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch, ActionStep, ExecuteResult
from roboscript.contracts.env import EpisodeSpec, ObserveRequest
from roboscript.contracts.errors import EpisodeEnded, InfraError, ProtocolError
from roboscript.contracts.ids import BatchId, EpisodeId, TaskName, new_id
from roboscript.envs import BridgeConfig, BridgeEnv
from roboscript.envs.mock import MockConfig, MockEnv


@dataclass
class Client:
    """A benchmark's client over `MockEnv` (the "simulator" of its episode)."""

    env: MockEnv
    episode: EpisodeId
    bridge: BridgeEnv

    async def submit(self) -> None:
        self.bridge.submit(await self.env.observe(self.episode, ObserveRequest()))

    async def serve_one(self) -> ExecuteResult:
        """Poll until a batch is queued, run it and report it."""
        result = await self.env.execute(self.episode, await polled(self.bridge))
        self.bridge.report(result)
        return result


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[Client]:
    config = MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs")
    env = MockEnv(config, LocalBlobStore(tmp_path / "blobs"))
    opened = await env.open(EpisodeSpec(task=TaskName("lift_red"), seed=3))
    quick = BridgeConfig(observe_timeout_s=2, report_timeout_s=2)
    yield Client(env, opened.episode_id, BridgeEnv(EpisodeId("bridge-1"), quick))


async def polled(bridge: BridgeEnv) -> ActionBatch:
    """The batch the bridge hands out once one is queued."""
    while True:
        if (batch := bridge.poll()) is not None:
            return batch
        await anyio.sleep(0.01)


def holds(qpos: list[float], n: int, batch_id: str | None = None) -> ActionBatch:
    steps = [ActionStep(qpos=qpos)] * n
    return ActionBatch(batch_id=BatchId(batch_id or new_id("batch")), steps=steps)


async def test_execute_is_relayed_through_poll_and_report(client: Client) -> None:
    bridge, me = client.bridge, EpisodeId("bridge-1")
    await client.submit()
    rest = (await bridge.robot(me)).qpos_cmd
    assert bridge.poll() is None  # nothing queued yet
    batch = holds(rest, 3)
    async with anyio.create_task_group() as tg:
        tg.start_soon(client.serve_one)
        done = await bridge.execute(me, batch)
    assert (done.executed, [s.robot.step for s in done.states]) == (3, [1, 2, 3])
    assert (await bridge.robot(me)).step == 3  # the reported state
    assert await bridge.execute(me, batch) == done  # the same batch id: its first outcome
    with anyio.fail_after(1):  # the robot moved: observe waits for the client's next frame
        async with anyio.create_task_group() as tg:
            tg.start_soon(client.submit)
            assert (await bridge.observe(me, ObserveRequest())).step == 3
    with pytest.raises(ProtocolError, match="no batch b-x awaits a report"):
        bridge.report(
            ExecuteResult(batch_id=BatchId("b-x"), states=[], executed=0, stopped="completed")
        )


async def test_the_benchmark_ends_the_episode(client: Client) -> None:
    bridge, me = client.bridge, EpisodeId("bridge-1")
    await client.submit()
    rest = (await bridge.robot(me)).qpos_cmd
    async with anyio.create_task_group() as tg:
        tg.start_soon(client.serve_one)
        await bridge.execute(me, holds(rest, 2))

    async def end_soon() -> None:
        await anyio.sleep(0.05)
        bridge.end("success")

    async with anyio.create_task_group() as tg:  # an observe waiting for the next frame
        tg.start_soon(end_soon)
        with anyio.fail_after(1), pytest.raises(EpisodeEnded, match=r"episode \(success\)"):
            await bridge.observe(me, ObserveRequest())
    for call in (bridge.robot(me), bridge.execute(me, holds(rest, 1))):
        with pytest.raises(EpisodeEnded, match="success"):
            await call


async def test_a_silent_client_faults_the_episode(tmp_path: Path) -> None:
    quick = BridgeConfig(observe_timeout_s=0.2, report_timeout_s=0.2)
    bridge, me = BridgeEnv(EpisodeId("bridge-1"), quick), EpisodeId("bridge-1")
    with pytest.raises(InfraError, match="timeout"):
        await bridge.observe(me, ObserveRequest())
    with pytest.raises(InfraError, match="sim_fault"):  # and it stays faulted
        await bridge.robot(me)

    client_env = MockEnv(
        MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path), LocalBlobStore(tmp_path)
    )
    opened = await client_env.open(EpisodeSpec(task=TaskName("lift_red"), seed=3))
    unpolled = BridgeEnv(EpisodeId("bridge-2"), quick)
    unpolled.submit(await client_env.observe(opened.episode_id, ObserveRequest()))
    rest = (await unpolled.robot(EpisodeId("bridge-2"))).qpos_cmd
    with pytest.raises(InfraError, match=r"no report of the batch within 0\.2 s"):
        await unpolled.execute(EpisodeId("bridge-2"), holds(rest, 1))
    assert unpolled.poll() is None  # withdrawn: the client never saw it
