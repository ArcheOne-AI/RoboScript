"""The toolsets a run mounts, in this process or served by another
(docs/concepts/agent-runtime.md).

`provider` makes one toolset's provider from its registered name (`roboscript.toolsets`), its
config, the skillsets mounted on it and the URL of a server that already serves it (a
`RemoteProvider`, whose config is its server's); `run_providers` makes those of a `RunSpec`'s
`toolsets`, each with the run's `skillsets`. The Agent host and the learning runner drive their
Hub themselves, so they mount remote toolsets with `remote_toolsets` around their started Hub:
each server is checked as the run starts (it answers, and lists exactly the tools the run
mounts from it), and the Hub serves their calls back on loopback. Without a URL nothing of this
runs, and the `mcp` extra is not needed.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.hub import Hub
from roboscript.kernel import entrypoints
from roboscript.learning.runspec import RunSpec
from roboscript.tools import LocalProvider, ToolProvider

TOOLSETS = "roboscript.toolsets"


def provider(
    name: str,
    config: Mapping[str, Any] | None = None,
    skillsets: Sequence[str] = (),
    url: str | None = None,
    profile: EmbodimentProfile | None = None,
) -> ToolProvider:
    """The toolset registered as `name`: in this process with `config`, or, given `url`, the
    one its server serves there (the server holds its config, so `config` must be empty); given
    `profile`, the robot it mounts for, its tools that do not fit it left out."""
    if url is None:
        return LocalProvider(entrypoints.load(TOOLSETS, name)(), config, skillsets, profile)
    if config:
        raise ValueError(f"toolset {name}: a remote toolset's config is its server's: drop it")
    from roboscript.integrations.mcp.remote import RemoteProvider  # the `mcp` extra

    return RemoteProvider(entrypoints.load(TOOLSETS, name), url, skillsets, profile)


def run_providers(spec: RunSpec) -> list[ToolProvider]:
    """The providers of `spec.toolsets`, in order, each with the run's skillsets mounted (their
    names: a skillset's bindings travel as data, so a remote toolset's server names them too),
    for the run's embodiment."""
    skillsets = [item.name for item in spec.skillsets]
    profile = load_embodiment(spec.embodiment.name)
    return [
        provider(item.name, item.overrides, skillsets, item.url, profile) for item in spec.toolsets
    ]


@asynccontextmanager
async def remote_toolsets(hub: Hub, providers: Sequence[ToolProvider]) -> AsyncIterator[None]:
    """Mount the remote toolsets among `providers` (those of `hub`, started) for the block
    (`roboscript.integrations.mcp.remote.serve_remote`; tokens from `RS_HUB_SECRET`); nothing
    happens when every toolset runs in this process."""
    if all(isinstance(one, LocalProvider) for one in providers):
        yield
        return
    from roboscript.integrations.mcp.remote import serve_remote  # the `mcp` extra

    async with serve_remote(hub, providers):
        yield
