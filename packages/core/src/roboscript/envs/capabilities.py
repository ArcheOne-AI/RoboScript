"""The startup capability check (docs/concepts/embodied-core.md): before a run starts, the env
must offer every optional face its method `requires`, or the run stops with a message naming each
one missing.

What an env offers comes from its `Benchmark` descriptor (in process: no server needs to run yet)
or, for an env served elsewhere, from `discover()` on its public plane at its address.
"""

import httpx

from roboscript import rpc
from roboscript.contracts.capabilities import EnvCapabilities, missing
from roboscript.contracts.env import EnvPublic
from roboscript.envs.benchmark import Benchmark


class MissingCapabilities(Exception):
    """The env lacks faces the method needs; `missing` lists them, one line each."""

    def __init__(self, who: str, env: str, lines: list[str]) -> None:
        super().__init__(f"{who} needs what {env} does not offer: {'; '.join(lines)}")
        self.missing = lines


def check_capabilities(
    required: EnvCapabilities, offered: EnvCapabilities, *, who: str, env: str
) -> None:
    """Raise `MissingCapabilities` unless `offered` covers `required`."""
    lines = missing(required, offered)
    if lines:
        raise MissingCapabilities(who, env, lines)


async def discovered(
    url: str, *, timeout: float = 30.0, http: httpx.AsyncClient | None = None
) -> EnvCapabilities:
    """What the env server whose public plane is at `url` declares (`discover()`); `http`: a
    client to reach it with (default: its own connections, closed after)."""
    public = rpc.client(EnvPublic, url, token=None, http=http, timeout=timeout)  # type: ignore[type-abstract]
    try:
        return (await public.discover()).capabilities
    finally:
        if http is None:
            await rpc.aclose(public)


async def offered_by(benchmark: Benchmark, url: str | None = None) -> EnvCapabilities:
    """What a run's env offers: the server's own declaration when it runs at `url`, else the
    benchmark's descriptor."""
    return benchmark.capabilities if url is None else await discovered(url)
