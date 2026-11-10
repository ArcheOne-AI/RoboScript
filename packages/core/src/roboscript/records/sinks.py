"""The event-sink extension point (`roboscript.event_sinks`): exporters that receive a copy of
every event a run writes, beside its own logs (which stay the source of truth).

A registered `Sink` names the sink, its config model, and `make(config, run_dir)`, which opens an
`EventSink` (`write` each event, `flush`, `close`). Built-ins register through the same group:
`jsonl` (one more JSONL file of the whole run, `JsonlSink`) and the example `console_summary`
(a log line per `every` events and one at close counting events by kind; off by default).
"""

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import BaseModel, ConfigDict, PositiveInt

from roboscript.contracts.events import Event
from roboscript.kernel import entrypoints
from roboscript.records.jsonl import JsonlSink

GROUP: Final = "roboscript.event_sinks"
log = logging.getLogger(__name__)


class EventSink(Protocol):
    def write(self, event: Event) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Sink:
    """An event sink as a plugin: `make(config, run_dir)` opens it, `config` an instance of
    `Config`."""

    name: str
    Config: type[BaseModel]
    make: Callable[[Any, Path], EventSink]


def load_sink(ref: str) -> Sink:
    """The sink registered as `ref` in `roboscript.event_sinks`, or by import path
    `package.module:attribute`."""
    found = entrypoints.load_ref(ref) if ":" in ref else entrypoints.load(GROUP, ref)
    if not isinstance(found, Sink):
        where = ref if ":" in ref else f"{ref} in {GROUP}"
        raise TypeError(f"{where} is a {type(found).__name__}, not a Sink")
    return found


class JsonlConfig(BaseModel):
    """`file`: relative to the run directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = "exported.jsonl"


def _jsonl(config: JsonlConfig, run_dir: Path) -> EventSink:
    return JsonlSink(run_dir / config.file)


JSONL: Final = Sink(name="jsonl", Config=JsonlConfig, make=_jsonl)


class SummaryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    every: PositiveInt = 1000


class ConsoleSummary:
    """Logs the events seen so far by kind, every `every` events and at close."""

    def __init__(self, config: SummaryConfig) -> None:
        self.every = config.every
        self.kinds: Counter[str] = Counter()

    def write(self, event: Event) -> None:
        self.kinds[event.kind] += 1
        if sum(self.kinds.values()) % self.every == 0:
            self._log()

    def flush(self) -> None:
        """Nothing is buffered."""

    def close(self) -> None:
        self._log()

    def _log(self) -> None:
        counts = ", ".join(f"{kind} {n}" for kind, n in self.kinds.most_common())
        log.info("events: %d (%s)", sum(self.kinds.values()), counts)


def _summary(config: SummaryConfig, run_dir: Path) -> EventSink:
    return ConsoleSummary(config)


CONSOLE_SUMMARY: Final = Sink(name="console_summary", Config=SummaryConfig, make=_summary)
