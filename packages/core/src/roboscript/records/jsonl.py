"""JSONL event logs: one `rs.event/1` object per line, appended by one service each."""

import heapq
import logging
import queue
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from typing_extensions import Self

from roboscript.contracts.events import Event
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.trace import TraceContext, get_trace

log = logging.getLogger(__name__)


class JsonlSink:
    """Appends events to a JSONL file, one line per event with one unbuffered write each.

    Events are dumped in the default context, so WireSecret fields come out masked.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("ab", buffering=0)
        self._lock = threading.Lock()

    def write(self, event: Event) -> None:
        self.append(event.model_dump_json().encode() + b"\n")

    def append(self, line: bytes) -> None:
        """Append one whole line."""
        rest = memoryview(line)
        with self._lock:
            while rest:  # a raw write may be short
                rest = rest[self._file.write(rest) :]

    def flush(self) -> None:
        """Nothing to wait for: every write is through when it returns."""

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class BackgroundSink:
    """A `JsonlSink` whose appends run on a thread of its own, in order, so that `write` never
    waits for the disk: an event loop must not, and a saturated file system journal can hold an
    append for tens of seconds. `flush` waits (at most `wait` seconds) until the lines written
    before it are appended; `close` flushes the same way and stops the thread.

    The log never blocks or kills the service: a line whose append fails for any reason, one
    written while `max_queued` lines already wait, and one written after `close` are dropped
    and counted in `dropped`."""

    def __init__(self, path: Path, *, max_queued: int = 10_000, wait: float = 10.0) -> None:
        self._sink = JsonlSink(path)
        self.path = path
        self.max_queued = max_queued
        self.wait = wait
        self.dropped = 0
        self._lines: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._done = threading.Condition()
        self._queued = self._appended = 0
        self._closed = False
        self._thread = threading.Thread(target=self._append, name=f"jsonl:{path.name}", daemon=True)
        self._thread.start()

    def write(self, event: Event) -> None:
        line = event.model_dump_json().encode() + b"\n"
        with self._done:
            if self._closed or self._queued - self._appended >= self.max_queued:
                self.dropped += 1
                return
            self._queued += 1
            self._lines.put(line)  # under the lock: lines go out in the order they were counted

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for the lines written so far; False if `timeout` (default `wait`) ran out first."""
        with self._done:
            target = self._queued
            return self._done.wait_for(
                lambda: self._appended >= target, self.wait if timeout is None else timeout
            )

    def close(self) -> None:
        """Flush (bounded by `wait`) and stop; lines still queued then are lost. Idempotent."""
        with self._done:
            if self._closed:
                return
            self._closed = True
            self._lines.put(None)
        self._thread.join(self.wait)
        if self._thread.is_alive():
            log.error("%s: the disk is stuck (%ss); queued lines are lost", self.path, self.wait)
        else:
            self._sink.close()

    def _append(self) -> None:
        while (line := self._lines.get()) is not None:
            try:
                self._sink.append(line)
            except Exception as error:  # the service goes on; its log misses this line
                log.error("cannot append to %s: %s", self.path, error)
                with self._done:
                    self.dropped += 1
            with self._done:
                self._appended += 1
                self._done.notify_all()


class EventWriter:
    """A service's event log: `emit` stamps time, `seq` and trace, then appends the event (with
    `background`, on a thread of its own: an async service's `emit` never waits for the disk,
    and `flush` waits for the events emitted so far).

    Thread-safe. `seq` counts this writer's events from 0. Callers pass plain JSON data
    (and BlobRef dumps), never secrets.
    """

    def __init__(
        self, path: Path, service: str, *, background: bool = False, max_queued: int = 10_000
    ) -> None:
        self.service = service
        self._sink: JsonlSink | BackgroundSink = (
            BackgroundSink(path, max_queued=max_queued) if background else JsonlSink(path)
        )
        self._lock = threading.Lock()
        self._seq = 0

    @property
    def path(self) -> Path:
        return self._sink.path

    def emit(
        self,
        kind: str,
        data: dict[str, Any],
        *,
        episode_id: EpisodeId | None = None,
        trace: TraceContext | None = None,
    ) -> Event:
        """Append one event; `trace` defaults to the current trace."""
        with self._lock:
            event = Event(
                ts=time.time(),
                seq=self._seq,
                service=self.service,
                kind=kind,
                episode_id=episode_id,
                trace=trace or get_trace(),
                data=data,
            )
            self._sink.write(event)
            self._seq += 1
        return event

    def flush(self) -> None:
        """Wait until the events emitted so far are in the file."""
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_events(path: Path, start: int = 0) -> Iterator[Event]:
    """The events of one log, in file order, from byte `start` (where a line begins, such as the
    log's size when a reader began to care); a line that is not an event is an error. A last line
    without its newline is one still being written, and is not read yet."""
    with path.open("rb") as lines:
        lines.seek(start)
        for number, line in enumerate(lines, 1):
            if not line.endswith(b"\n"):
                return
            try:
                yield Event.model_validate_json(line)
            except ValidationError as error:
                where = f"{path}:{number}" if not start else f"{path}, line {number} from {start}"
                raise ValueError(f"{where}: not an rs.event/1 line: {error}") from None


def merge_events(paths: Iterable[Path], episode_id: EpisodeId | None = None) -> list[Event]:
    """The events of several logs merged by time, each log keeping its own order.

    A log's order is its writer's, so a clock that steps back cannot reorder a service's events.
    With `episode_id`: that episode's events plus every event on the traces they belong to,
    whatever episode id it carries (an env server tags events with its own env episode id).
    """
    logs = [list(read_events(path)) for path in paths]
    if episode_id is not None:
        traces = {_trace_id(e) for log in logs for e in log if e.episode_id == episode_id}
        traces.discard(None)
        logs = [
            [e for e in log if e.episode_id == episode_id or _trace_id(e) in traces] for log in logs
        ]
    return list(heapq.merge(*logs, key=lambda e: (e.ts, e.service)))


def _trace_id(event: Event) -> str | None:
    return None if event.trace is None else event.trace.trace_id
