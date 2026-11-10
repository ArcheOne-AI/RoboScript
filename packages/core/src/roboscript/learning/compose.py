"""A run's composition from its plugins (docs/concepts/agent-runtime.md): the run's
`RunSpec.plugins` load through the kernel (`roboscript.kernel.Kernel`), in order, each adding its
part to the run's `RunParts`; the result is the Hub's guard pipeline, the event sinks and the
prechecks.

Every run starts from the built-ins: the guards of `roboscript.hub.DEFAULT_GUARDS` and the
prechecks of `DEFAULT_PRECHECKS`, with no sinks beside the run's own logs. A plugin item names its
entry-point group and its name; its `overrides` are its config. Nothing listed: the built-in
pipeline exactly (`roboscript.hub.DEFAULT_PIPELINE`).

```python
async with compose_run(spec.plugins) as run:
    problems = await run_prechecks(run.prechecks, facts)
    with open_sinks(run.sinks, run_dir) as sinks:
        runner = ProtocolRunner(..., pipeline=run.pipeline, sinks=sinks)
```
"""

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from roboscript.hub import GUARDS, Guard, Pipeline, assemble, load_guard
from roboscript.kernel import Kernel, Plugin, PluginContext, ServiceKey
from roboscript.learning.prechecks import DEFAULT_PRECHECKS, Precheck, load_precheck
from roboscript.learning.runspec import PluginItem
from roboscript.records.sinks import EventSink, Sink, load_sink


@dataclass
class RunParts:
    """What the run's plugins added, in order, each with its config values."""

    guards: list[tuple[Guard, dict[str, Any]]] = field(default_factory=list)
    sinks: list[tuple[Sink, dict[str, Any]]] = field(default_factory=list)
    prechecks: list[tuple[Precheck, dict[str, Any]]] = field(default_factory=list)


PARTS: Final = ServiceKey[RunParts]("run.parts")


class _Values(BaseModel):
    """A plugin's config as given; its own model validates it when the part is built."""

    model_config = ConfigDict(extra="allow", frozen=True)


class _Builtins(Plugin):
    name: ClassVar[str] = "run.builtins"
    provides = (PARTS,)

    async def setup(self, ctx: PluginContext, config: Any) -> None:
        guards: list[tuple[Guard, dict[str, Any]]] = [(g, {}) for g in vars(GUARDS).values()]
        ctx.provide(PARTS, RunParts(guards=guards, prechecks=[(p, {}) for p in DEFAULT_PRECHECKS]))


class _Part(Plugin):
    """One plugin item: loads its entry point and adds it to the run's parts (taken out again
    when the kernel unloads it)."""

    requires = (PARTS,)
    Config = _Values

    def __init__(self, item: PluginItem) -> None:
        self.item = item
        self.name = f"{item.group}:{item.name}"  # type: ignore[misc]  # per instance

    async def setup(self, ctx: PluginContext, config: Any) -> None:
        parts = ctx.get(PARTS)
        values = dict(self.item.overrides)
        if self.item.group == "guards":
            entry: Any = (load_guard(self.item.name), values)
            target: list[Any] = parts.guards
        elif self.item.group == "event_sinks":
            entry, target = (load_sink(self.item.name), values), parts.sinks
        else:
            entry, target = (load_precheck(self.item.name), values), parts.prechecks
        if any(part.name == entry[0].name for part, _ in target):
            raise ValueError(f"{self.name} is part of the run already (built in)")
        target.append(entry)
        ctx.defer(lambda: target.remove(entry))


@dataclass(frozen=True)
class Composition:
    """The run's Hub pipeline, event sinks (to open in a run directory, `open_sinks`) and
    prechecks, each with its config values."""

    pipeline: Pipeline
    sinks: tuple[tuple[Sink, dict[str, Any]], ...]
    prechecks: tuple[tuple[Precheck, dict[str, Any]], ...]


@asynccontextmanager
async def compose_run(plugins: Sequence[PluginItem]) -> AsyncIterator[Composition]:
    """Load `plugins` over the built-ins (unloaded after the block). Every config is validated
    here, before anything runs."""
    kernel = Kernel([_Builtins(), *(_Part(item) for item in plugins)])
    async with kernel as ctx:
        parts = ctx.get(PARTS)
        guards = [guard for guard, _ in parts.guards]
        configs = {guard.name: values for guard, values in parts.guards if values}
        pipeline = assemble(guards, configs)
        for sink, values in parts.sinks:
            _config(sink.name, sink.Config, values)
        for precheck, values in parts.prechecks:
            _config(precheck.name, precheck.Config, values)
        yield Composition(pipeline, tuple(parts.sinks), tuple(parts.prechecks))


@contextmanager
def open_sinks(
    sinks: Sequence[tuple[Sink, dict[str, Any]]], run_dir: Path
) -> Iterator[tuple[EventSink, ...]]:
    """The sinks opened in `run_dir`, flushed and closed after the block."""
    opened: list[EventSink] = []
    try:
        for sink, values in sinks:
            opened.append(sink.make(_config(sink.name, sink.Config, values), run_dir))
        yield tuple(opened)
    finally:
        for one in reversed(opened):
            one.flush()
            one.close()


def _config(name: str, model: type[BaseModel], values: dict[str, Any]) -> BaseModel:
    try:
        return model.model_validate(values)
    except ValidationError as error:
        raise ValueError(f"invalid config for plugin {name!r}: {error}") from error
