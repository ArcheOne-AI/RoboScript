"""Serve a `SkeletonEnv` over RPC in process (both ports), as `tests/core/envs/test_mock_env.py`
does for MockEnv, and hand the conformance suite its target."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast

import httpx
from pydantic import SecretStr

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.env import EnvAgent, EnvControl, EpisodeOpened, EpisodeSpec
from roboscript.contracts.ids import BatchId, EpisodeId, TaskName, new_id
from roboscript.envs import env_apps
from roboscript.envs.skeleton import SkeletonEnv
from roboscript.rpc import client
from roboscript.testing.env import EnvTarget

TOKEN = SecretStr("control-token-for-tests")


@asynccontextmanager
async def served(env: SkeletonEnv, *, seed: int, reset_by: str) -> AsyncIterator[EnvTarget]:
    await env.start()
    control_app, agent_app = env_apps(env, TOKEN)
    env.agent_url = "http://agent"
    try:
        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(control_app)) as control_http,
            httpx.AsyncClient(transport=httpx.ASGITransport(agent_app)) as agent_http,
        ):

            def connect(interface: type[Any], url: str, token: SecretStr | None) -> Any:
                http = agent_http if url == env.agent_url else control_http
                return client(interface, url, token=token, http=http, timeout=10)

            yield EnvTarget(
                url="http://control",
                token=TOKEN,
                connect=connect,
                profile=env.profile,
                task=TaskName(env.config.task),
                seed=seed,
                reset_by=reset_by,  # type: ignore[arg-type]
            )
    finally:
        env.shutdown()


@dataclass(frozen=True)
class Episode:
    """An open episode and the clients to drive it."""

    opened: EpisodeOpened
    control: EnvControl
    agent: EnvAgent

    @property
    def id(self) -> EpisodeId:
        return self.opened.episode_id

    async def rest(self) -> list[float]:
        return (await self.agent.robot(self.id)).qpos_cmd


async def open_episode(target: EnvTarget, seed: int | None = None) -> Episode:
    control = cast(EnvControl, target.connect(EnvControl, target.url, target.token))
    spec = EpisodeSpec(
        task=target.task,
        seed=target.seed if seed is None else seed,
        reset_by=target.reset_by,
    )
    opened = await control.open(spec)
    agent = cast(EnvAgent, target.connect(EnvAgent, opened.agent_url, opened.agent_token))
    return Episode(opened, control, agent)


def batch(*qpos: list[float]) -> ActionBatch:
    return ActionBatch(
        batch_id=BatchId(new_id("batch")), steps=[ActionStep(qpos=list(q)) for q in qpos]
    )
