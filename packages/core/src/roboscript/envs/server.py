"""Serving an env implementation (docs/concepts/embodied-core.md): two ports in one
process.

The control port serves `env.control` (bearer token `RS_ENV_TOKEN` from `.env`) and
`env.public`; the agent port serves `env.agent`, where each call must carry the token of the
episode it names (issued by `open`), and `env.public` again for toolsets' version checks. The
ready file lists each plane's URL.
"""

import functools
import logging
from pathlib import Path
from typing import Protocol

import anyio
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from starlette.applications import Starlette

from roboscript.contracts.env import EnvAgent, EnvControl, EnvPublic
from roboscript.envs.episodes import Episodes
from roboscript.rpc import make_app, routes, serve, write_ready
from roboscript.settings import EnvSettings

log = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """Where an env server listens; port 0 picks a free one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    control_port: int = Field(default=0, ge=0, le=65535)
    agent_port: int = Field(default=0, ge=0, le=65535)


class Env(Protocol):
    """An env implementation: the methods of `EnvPublic`, `EnvControl` and `EnvAgent` (checked
    when mounted), its episodes, and the agent-plane URL it hands out in `EpisodeOpened`."""

    episodes: Episodes
    agent_url: str


def env_apps(env: Env, control_token: SecretStr) -> tuple[Starlette, Starlette]:
    """The control-port app and the agent-port app of `env`."""
    public = routes(EnvPublic, env, token=None)
    control = make_app(public, routes(EnvControl, env, token=control_token))
    agent = make_app(public, routes(EnvAgent, env, token=None, allow=env.episodes.allow))
    return control, agent


def control_token() -> SecretStr:
    """`RS_ENV_TOKEN` from the environment or `.env`, as the orchestrator reads it. A service
    resolves it at startup, before anything changes its working directory (`.env` is found from
    there, or from `RS_ROOT`)."""
    token = EnvSettings().env_token
    if token is None or len(token.get_secret_value()) < 16:
        raise ValueError("set RS_ENV_TOKEN in .env to a random string of 16 or more characters")
    return token


async def serve_env(
    env: Env, config: ServerConfig, ready_file: Path | None, token: SecretStr
) -> None:
    """Serve `env` until SIGINT or SIGTERM, its control plane with bearer `token`; `ready_file`
    gets both ports' URLs once they listen."""
    control_app, agent_app = env_apps(env, token)
    async with anyio.create_task_group() as tg:
        env.agent_url = await tg.start(
            functools.partial(serve, agent_app, host=config.host, port=config.agent_port)
        )
        url = await tg.start(
            functools.partial(serve, control_app, host=config.host, port=config.control_port)
        )
        log.info("env.control and env.public at %s, env.agent at %s", url, env.agent_url)
        if ready_file is not None:
            planes = {"env.control": url, "env.public": url, "env.agent": env.agent_url}
            write_ready(ready_file, url, planes)
