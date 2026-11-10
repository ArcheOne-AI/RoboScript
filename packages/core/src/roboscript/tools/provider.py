"""Tool providers: where the Hub sends a toolset's calls. `LocalProvider` runs one in-process."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from roboscript.contracts.ids import EpisodeId
from roboscript.embodiments import EmbodimentProfile
from roboscript.kernel import KernelError
from roboscript.tools.context import ToolContext, ToolsetRuntime
from roboscript.tools.skillsets import fit_problems, guides, mounted
from roboscript.tools.spec import Toolset, ToolSpec


class ToolProvider(Protocol):
    """A toolset as the Hub sees it: `name` is the toolset's, `config` its validated config,
    `skillsets` the names of the skillsets mounted on it, `unmounted` the tools it leaves out
    (for lack of a binding, or not fitting the robot), with why (`tools.skillsets`)."""

    name: str
    config: BaseModel
    skillsets: tuple[str, ...]
    unmounted: Mapping[str, str]

    def specs(self) -> list[ToolSpec]: ...

    async def start(self, runtime: ToolsetRuntime) -> None: ...

    async def stop(self) -> None: ...

    async def call(
        self, name: str, args: Mapping[str, Any] | BaseModel, ctx: ToolContext
    ) -> BaseModel:
        """Run tool `name` with its params model, or a mapping validated into one."""
        ...

    async def close_episode(self, episode_id: EpisodeId) -> None:
        """The episode is closed: free what the toolset keeps for it (the Hub drops `ctx.state`
        of toolsets in its process itself)."""
        ...


class LocalProvider:
    """A toolset instance in this process, with `config` validated against its `Config`, and the
    skillsets named in `skillsets` mounted on it (`tools.skillsets`): their tools for it, their
    bindings set on it; given `profile`, the robot it mounts for, its tools that do not fit it
    left out."""

    def __init__(
        self,
        toolset: Toolset,
        config: BaseModel | Mapping[str, Any] | None = None,
        skillsets: Sequence[str] = (),
        profile: EmbodimentProfile | None = None,
    ) -> None:
        self.toolset = toolset
        self.name = toolset.name
        schema = type(toolset).Config
        self.config = config if isinstance(config, schema) else schema.model_validate(config or {})
        found = mounted(type(toolset), skillsets, profile)
        if toolset.specs() and not found.specs:
            reasons = sorted(set(found.unmounted.values()))
            raise KernelError(f"toolset {toolset.name} mounts none of its tools: {reasons}")
        toolset.bindings = found.bindings
        self.skillsets = tuple(skillsets)
        self.unmounted = found.unmounted
        self._specs = {spec.name: spec for spec in found.specs}
        extensions = {ext: ext(toolset) for ext in set(found.owners.values()) if ext is not None}
        self._owners: dict[str, object] = {
            name: toolset if ext is None else extensions[ext] for name, ext in found.owners.items()
        }

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    async def start(self, runtime: ToolsetRuntime) -> None:
        self.toolset.config = runtime.config
        self.toolset.runtime = runtime
        await self.toolset.start(runtime)

    async def stop(self) -> None:
        await self.toolset.stop()

    async def call(
        self, name: str, args: Mapping[str, Any] | BaseModel, ctx: ToolContext
    ) -> BaseModel:
        spec = self._specs[name]
        params = args if isinstance(args, spec.params) else spec.params.model_validate(args)
        # Iterating a model yields its fields, nested models kept as models.
        return await spec.handler(self._owners[name], ctx, **dict(params))

    async def close_episode(self, episode_id: EpisodeId) -> None:
        await self.toolset.close_episode(episode_id)


def unfit_tools(providers: Sequence[ToolProvider], profile: EmbodimentProfile) -> list[str]:
    """The tools `providers` mounted that do not fit `profile`'s robot, one line each (empty: all
    fit; `tools.skillsets.fit_problems`): a stack built without its robot checks them as it
    starts."""
    mounts = [(p.name, [spec.name for spec in p.specs()], p.skillsets) for p in providers]
    return fit_problems(mounts, profile)


def provider_guides(providers: Sequence[ToolProvider]) -> list[str]:
    """The guides of the skillsets mounted on `providers`, each skillset once, in mount order
    (`tools.skillsets.guides`)."""
    return guides([ref for provider in providers for ref in provider.skillsets])
