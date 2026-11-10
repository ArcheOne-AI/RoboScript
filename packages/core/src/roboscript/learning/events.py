"""The run's event log, split by trace (docs/concepts/operations.md): each episode runs
under a trace of its own, and everything logged on it (Hub, toolsets, model calls) goes to that
episode's `events.jsonl` while it runs; a task's protocol events go to the task's log the same way,
and anything else to the run's."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from roboscript.contracts.events import Event
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.trace import TraceContext, get_trace
from roboscript.records import EventWriter
from roboscript.records.sinks import EventSink


class EventRouter(EventWriter):
    """An `EventWriter` for the run's log that hands events of a routed trace to that route's
    own writer."""

    def __init__(
        self, path: Path, service: str = "orchestrator", sinks: Sequence[EventSink] = ()
    ) -> None:
        super().__init__(path, service)
        self._routes: dict[str, EventWriter] = {}
        self._sinks = tuple(sinks)

    @contextmanager
    def route(self, trace: TraceContext, path: Path) -> Iterator[EventWriter]:
        """Send the events of `trace` (by trace id) to `path` inside the block."""
        if trace.trace_id in self._routes:
            raise ValueError(f"trace {trace.trace_id} is routed already")
        with EventWriter(path, self.service) as writer:
            self._routes[trace.trace_id] = writer
            try:
                yield writer
            finally:
                del self._routes[trace.trace_id]

    def emit(
        self,
        kind: str,
        data: dict[str, Any],
        *,
        episode_id: EpisodeId | None = None,
        trace: TraceContext | None = None,
    ) -> Event:
        trace = trace or get_trace()
        writer = None if trace is None else self._routes.get(trace.trace_id)
        if writer is None:
            event = super().emit(kind, data, episode_id=episode_id, trace=trace)
        else:
            event = writer.emit(kind, data, episode_id=episode_id, trace=trace)
        for sink in self._sinks:
            sink.write(event)
        return event
