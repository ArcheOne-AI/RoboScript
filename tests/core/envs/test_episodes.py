"""Episode bookkeeping: per-episode tokens, and batches that never execute twice."""

import anyio
import pytest
from pydantic import SecretStr

from roboscript.contracts.action import ActionBatch, ActionStep, ExecuteResult
from roboscript.contracts.errors import EpisodeEnded, InfraError, ProtocolError
from roboscript.contracts.ids import BatchId, EpisodeId
from roboscript.envs import Batches, Episodes, unsolvable


def batch(name: str, steps: int = 1) -> ActionBatch:
    return ActionBatch(batch_id=BatchId(name), steps=[ActionStep(qpos=[0.0] * 14)] * steps)


def test_tokens_open_only_their_episode() -> None:
    episodes = Episodes()
    first, first_token = episodes.open()
    second, second_token = episodes.open()  # closes the first
    assert episodes.allow(second_token, "robot", {"episode_id": second})
    assert not episodes.allow(first_token, "robot", {"episode_id": second})
    assert not episodes.allow(None, "robot", {"episode_id": second})
    assert not episodes.allow(SecretStr("tökén"), "robot", {"episode_id": second})  # no TypeError
    assert episodes.allow(first_token, "robot", {"episode_id": first})  # then EpisodeEnded:
    with pytest.raises(EpisodeEnded):
        episodes.check(first)
    with pytest.raises(ProtocolError, match="no episode"):
        episodes.check(EpisodeId("ep-unknown"))
    gone = unsolvable("mock_bimanual", 3)
    assert (gone.solvable, gone.prerun_attempts, gone.agent_url) == (False, 3, "")


async def test_a_failed_batch_is_never_executed_again() -> None:
    batches, calls = Batches(), []

    async def fail() -> ExecuteResult:
        calls.append("fail")
        await anyio.sleep(0.05)
        raise InfraError("sim_fault", "the simulator raised RuntimeError")

    async with anyio.create_task_group() as tg:

        async def first() -> None:
            with pytest.raises(InfraError, match="sim_fault"):
                await batches.run(batch("b1"), fail)

        tg.start_soon(first)
        await anyio.sleep(0.01)
        with pytest.raises(InfraError, match="busy"):
            await batches.run(batch("b2"), fail)
    for _ in range(2):  # a replay gets the same error, without running the steps
        with pytest.raises(InfraError, match="sim_fault"):
            await batches.run(batch("b1"), fail)
    with pytest.raises(ProtocolError, match="ran other steps"):
        await batches.run(batch("b1", steps=2), fail)
    assert calls == ["fail"]
