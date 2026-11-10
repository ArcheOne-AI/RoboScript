"""Plugins and the kernel that loads them in dependency order."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from roboscript.kernel.context import Context, KernelError, PluginContext, ServiceKey


class NoConfig(BaseModel):
    """The config of a plugin without settings: an empty mapping."""

    model_config = ConfigDict(extra="forbid")


class Plugin(ABC):
    """A unit of composition: the services it requires and provides, its config model, and a
    `setup` that registers through the given context (undone when the plugin unloads)."""

    name: ClassVar[str]
    requires: ClassVar[tuple[ServiceKey[Any], ...]] = ()
    provides: ClassVar[tuple[ServiceKey[Any], ...]] = ()
    Config: ClassVar[type[BaseModel]] = NoConfig

    @abstractmethod
    async def setup(self, ctx: PluginContext, config: Any) -> None:
        """Provide the declared services; `config` is an instance of `Config`."""


class Kernel:
    """Loads plugins on `async with` (providers before their users) and unloads them in reverse.

    `config` maps plugin names to raw config mappings; services already in `ctx` count as
    provided. Wiring and config errors are raised here, before any plugin runs.
    """

    def __init__(
        self,
        plugins: Sequence[Plugin],
        config: Mapping[str, Mapping[str, Any]] | None = None,
        ctx: Context | None = None,
    ) -> None:
        self.ctx = Context() if ctx is None else ctx
        config = dict(config or {})
        names = [plugin.name for plugin in plugins]
        for name in names:
            if names.count(name) > 1:
                raise KernelError(f"plugin {name!r} is listed twice")
        unknown = sorted(set(config) - set(names))
        if unknown:
            raise KernelError(f"config for plugins that are not loaded: {unknown}")
        self.order = _dependency_order(plugins, self.ctx)
        self._configs = {plugin.name: _config(plugin, config) for plugin in self.order}
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> Context:
        async with AsyncExitStack() as stack:
            for plugin in self.order:
                plugin_stack = await stack.enter_async_context(AsyncExitStack())
                view = PluginContext(self.ctx, plugin.name, plugin.provides, plugin_stack)
                await plugin.setup(view, self._configs[plugin.name])
                missing = [key.name for key in plugin.provides if not self.ctx.has(key)]
                if missing:
                    raise KernelError(f"plugin {plugin.name!r} did not provide {missing}")
            self._stack = stack.pop_all()
        return self.ctx

    async def __aexit__(self, *exc_info: object) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()


def _config(plugin: Plugin, config: Mapping[str, Mapping[str, Any]]) -> BaseModel:
    try:
        return plugin.Config.model_validate(config.get(plugin.name, {}))
    except ValidationError as error:
        raise KernelError(f"invalid config for plugin {plugin.name!r}: {error}") from error


def _dependency_order(plugins: Sequence[Plugin], ctx: Context) -> list[Plugin]:
    """Topological order, stable with respect to the given order."""
    providers: dict[ServiceKey[Any], Plugin] = {}
    for plugin in plugins:
        for key in plugin.provides:
            if key in providers:
                raise KernelError(
                    f"service {key.name!r} is provided by both {providers[key].name!r} "
                    f"and {plugin.name!r}"
                )
            if ctx.has(key):
                raise KernelError(
                    f"plugin {plugin.name!r} provides {key.name!r}, which the context already has"
                )
            providers[key] = plugin
    needs: dict[str, list[Plugin]] = {}
    for plugin in plugins:
        needs[plugin.name] = []
        for key in plugin.requires:
            if key in providers:
                needs[plugin.name].append(providers[key])
            elif not ctx.has(key):
                raise KernelError(
                    f"plugin {plugin.name!r} requires {key.name!r}, which no plugin provides"
                )
    order: list[Plugin] = []
    pending = list(plugins)
    while pending:
        ready = next((p for p in pending if all(d in order for d in needs[p.name])), None)
        if ready is None:
            raise KernelError(f"plugins depend on each other in a cycle: {_cycle(pending, needs)}")
        order.append(ready)
        pending.remove(ready)
    return order


def _cycle(pending: list[Plugin], needs: dict[str, list[Plugin]]) -> str:
    # Every pending plugin waits on another pending one, so following those edges must loop.
    path = [pending[0]]
    while True:
        nxt = next(d for d in needs[path[-1].name] if d in pending)
        if nxt in path:
            loop = [*path[path.index(nxt) :], nxt]
            return " -> ".join(plugin.name for plugin in loop)
        path.append(nxt)
