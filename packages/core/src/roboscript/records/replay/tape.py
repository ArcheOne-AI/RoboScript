"""The tape: what a run's external, nondeterministic edges answered, recorded once and served
again on other code (docs/concepts/operations.md).

Three edges are taped: model calls (through the gateway client's HTTP transport,
`records.replay.http`), the motion service's plans, IK and FK, and the vision service's
detections and masks (through `rpc.taped`). With `record` the run calls them as always and the
tape writes each canonical request with what came back; with `replay` they are not called: the
tape serves the recorded answer of the same canonical request, repeated requests in the order
they were recorded, and a request it has no answer for stops the run (`ReplayMiss`). Both modes
also write the comparison stream (`observe`: the env's actions and results, the tool results the
model got, the Memory's edits and versions), which `records.replay.compare` diffs between two
runs.

Off (the default: `RS_REPLAY` unset) nothing here runs and nothing changes. It is switched per
process by the environment, so that a pipeline job's processes (its worker, the P3 Agent host it
starts) share it: `RS_REPLAY` (`record` or `replay`), `RS_REPLAY_DIR` (where this run writes),
`RS_REPLAY_FROM` (replay: the recorded run's `RS_REPLAY_DIR`), `RS_REPLAY_SCOPE` (the job's
directory under both), `RS_REPLAY_BLOB_DIR` (the blob store whose masks the vision answers name)
and `RS_REPLAY_VOLATILE` (a JSON object of literal strings the canonical form replaces, e.g. the
run root). A test switches a tape on for its context with `use`.

Files, under `<dir>/<scope>/`: `calls.<pid>.jsonl` (record: a line per answered request),
`compare.<pid>.jsonl` (both modes), `blobs/` (the masks the vision answers name),
`clock.<pid>.jsonl` (record: the values of the pinned clocks), `misses/` (replay: each miss with
the request and the nearest recorded one) and `summary.<pid>.json` (replay: what was served and
what was left over, written at exit).
"""

import atexit
import contextlib
import contextvars
import difflib
import functools
import heapq
import json
import logging
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.blob import BlobRef
from roboscript.records.replay.canonical import canonicalize, request_key

if TYPE_CHECKING:
    from roboscript.blobs.local import LocalBlobStore

log = logging.getLogger(__name__)

Mode = Literal["record", "replay"]

ENV_MODE: Final = "RS_REPLAY"
ENV_DIR: Final = "RS_REPLAY_DIR"
ENV_FROM: Final = "RS_REPLAY_FROM"
ENV_SCOPE: Final = "RS_REPLAY_SCOPE"
ENV_BLOBS: Final = "RS_REPLAY_BLOB_DIR"
ENV_VOLATILE: Final = "RS_REPLAY_VOLATILE"

REPLAYED: Final = frozenset(
    {
        "motion.plan",
        "motion.plan_batch",
        "motion.ik",
        "motion.fk",
        "vision.detect",
        "vision.segment",
    }
)
"""The RPC methods a replay serves from the tape (`<plane>.<method>`). `info` and the other
planes are called as always: metadata, or the simulator itself, which runs for real."""

OBSERVED_PLANES: Final = frozenset({"env.agent", "env.control"})
"""The planes whose calls go to the comparison stream only (never served): the env's."""

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")


class ReplayMiss(BaseException):
    """A replay was asked for a request the tape has no (more) answers for. Not an `Exception`,
    so that no retry, rerun or error outcome in between turns it into something else: it ends
    the run, after the tape logged it with the nearest recorded request and wrote it to
    `misses/`."""


class TapeConfig(BaseModel):
    """Where and how a process tapes (see the module docstring for the environment)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Mode
    directory: Path
    source: Path | None = None
    scope: str = ""
    blob_dir: Path | None = None
    volatile: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "TapeConfig | None":
        """The tape `environ` switches on, or None when `RS_REPLAY` is unset or empty."""
        mode = environ.get(ENV_MODE, "")
        if not mode:
            return None
        if mode not in ("record", "replay"):
            raise ValueError(f"{ENV_MODE} must be 'record' or 'replay', got {mode!r}")
        directory = environ.get(ENV_DIR)
        if not directory:
            raise ValueError(f"{ENV_MODE}={mode} needs {ENV_DIR}, the tape directory")
        source = environ.get(ENV_FROM) or None
        if mode == "replay" and source is None:
            raise ValueError(f"{ENV_MODE}=replay needs {ENV_FROM}, the recorded run's tape")
        volatile = json.loads(environ.get(ENV_VOLATILE) or "{}")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            directory=Path(directory),
            source=None if source is None else Path(source),
            scope=environ.get(ENV_SCOPE, ""),
            blob_dir=Path(environ[ENV_BLOBS]) if environ.get(ENV_BLOBS) else None,
            volatile=volatile,
        )

    def environ(self) -> dict[str, str]:
        """The environment variables that switch this tape on in a child process."""
        found = {
            ENV_MODE: self.mode,
            ENV_DIR: str(self.directory),
            ENV_SCOPE: self.scope,
            ENV_VOLATILE: json.dumps(self.volatile, sort_keys=True),
        }
        if self.source is not None:
            found[ENV_FROM] = str(self.source)
        if self.blob_dir is not None:
            found[ENV_BLOBS] = str(self.blob_dir)
        return found


@dataclass
class _Recorded:
    stream: str
    key: str
    request: Any
    response: dict[str, Any]
    served: bool = False


@dataclass
class _Source:
    """A recorded scope: its answers by key in recorded order, and by stream."""

    by_key: dict[str, deque[_Recorded]] = field(default_factory=dict)
    by_stream: dict[str, list[_Recorded]] = field(default_factory=dict)
    clocks: dict[str, deque[Any]] = field(default_factory=dict)
    observed: dict[str, deque[Any]] = field(default_factory=dict)


T = TypeVar("T")


class Tape:
    """One process's tape (see the module docstring). Thread-safe; files are written as the
    first line arrives."""

    def __init__(self, config: TapeConfig) -> None:
        self.config = config
        self.mode: Mode = config.mode
        self.out = config.directory / config.scope if config.scope else config.directory
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._seq = 0
        self._source: _Source | None = None
        self._misses = 0
        self._served = 0

    # Canonical form

    def canonical(self, value: Any) -> Any:
        return canonicalize(value, self.config.volatile)

    def replays(self, stream: str) -> bool:
        """Whether this tape serves `stream` (`model:<path>`, `<plane>.<method>`)."""
        return self.mode == "replay" and (stream.startswith("model:") or stream in REPLAYED)

    # Record

    def record(self, stream: str, request: Any, response: dict[str, Any]) -> None:
        """An answered request (record mode): its canonical form, its key and the answer, with
        the bytes of every blob the answer names (a mask) copied into the tape."""
        canonical = self.canonical(request)
        self._save_blobs(response)
        line = {
            "stream": stream,
            "key": request_key(stream, canonical),
            "request": canonical,
            "response": response,
        }
        self._append("calls", line)

    def observe(self, kind: str, data: Any) -> None:
        """A comparison record (both modes), canonicalized."""
        self._append("compare", {"kind": kind, "data": self.canonical(data)})

    # Replay

    def serve(self, stream: str, request: Any) -> dict[str, Any]:
        """The recorded answer to `request` in `stream`, the next unserved one of its key, with
        the blobs it names put into the blob store; a miss raises `ReplayMiss`."""
        canonical = self.canonical(request)
        key = request_key(stream, canonical)
        source = self._loaded()
        with self._lock:
            queue = source.by_key.get(key)
            found = queue.popleft() if queue else None
            if found is not None:
                found.served = True
                self._served += 1
        if found is None:
            self._miss(stream, key, canonical, source)
        assert found is not None
        self._restore_blobs(found.response)
        return found.response

    def clock(self, name: str, real: Callable[[], T]) -> Callable[[], T]:
        """`real`, pinned: record mode records each value it returns (a float or a datetime),
        replay returns the recorded ones in order (a run that reads it more often than the
        recording did is a miss)."""
        if self.mode == "record":

            def recorded() -> T:
                value = real()
                text = value.isoformat() if isinstance(value, datetime) else value
                self._append("clock", {"clock": name, "value": text})
                return value

            return recorded

        def replayed() -> T:
            values = self._loaded().clocks.get(name)
            if not values:
                message = f"clock {name!r}: read more often than the recording did"
                self._write_miss({"clock": name, "detail": message})
                raise ReplayMiss(message)
            value = values.popleft()
            parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
            return cast(T, parsed)

        return replayed

    def pin(self, name: str, value: Any) -> Any:
        """A value the simulator draws without a seed we control (an instruction): record mode
        records `value` and returns it; replay returns the recorded ones in order, whatever
        `value` is (a run that asks more often than the recording did is a miss)."""
        if self.mode == "record":
            self._append("clock", {"clock": name, "value": value})
            return value
        values = self._loaded().clocks.get(name)
        if not values:
            message = f"pinned {name!r}: asked more often than the recording did"
            self._write_miss({"clock": name, "detail": message})
            raise ReplayMiss(message)
        return values.popleft()

    def next_observed(self, kind: str) -> Any:
        """Replay: the recorded run's next comparison record of `kind` (its canonical data), for
        an env answer the simulator draws nondeterministically and a replay follows (an expert
        pre-run's verdict); None when the recording has no more."""
        values = self._loaded().observed.get(kind)
        return values.popleft() if values else None

    def keep(self, name: str, data: bytes) -> None:
        """Record mode: a file the run made that a replay follows (`kept`), under `files/`."""
        folder = self.out / "files"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)

    def kept(self, name: str) -> bytes | None:
        """Replay: the file the recording kept as `name`, None when it kept none."""
        assert self.config.source is not None
        path = self.config.source / self.config.scope / "files" / name
        return path.read_bytes() if path.is_file() else None

    def summary(self) -> dict[str, Any]:
        """Replay: answers served, misses, and the recorded answers left over per stream."""
        left: dict[str, int] = {}
        if self._source is not None:
            for stream, records in self._source.by_stream.items():
                if unserved := sum(not r.served for r in records):
                    left[stream] = unserved
        return {"served": self._served, "misses": self._misses, "unserved": left}

    def close(self) -> None:
        """Write the replay's summary (record mode has nothing to finish)."""
        if self.mode == "replay" and self._source is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            path = self.out / f"summary.{self._pid}.json"
            path.write_text(
                json.dumps(self.summary(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )

    # Internals

    def _append(self, kind: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            line = {"seq": self._seq, "t": time.time(), **record}
            self.out.mkdir(parents=True, exist_ok=True)
            with (self.out / f"{kind}.{self._pid}.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")

    def _loaded(self) -> _Source:
        with self._lock:
            if self._source is None:
                assert self.config.source is not None
                where = self.config.source / self.config.scope
                self._source = load_source(where)
            return self._source

    def _blobs(self) -> "LocalBlobStore":
        from roboscript.blobs.local import LocalBlobStore  # blobs imports rpc, which imports us

        return LocalBlobStore(self.out / "blobs")

    def _store(self) -> "LocalBlobStore":
        from roboscript.blobs.local import LocalBlobStore

        if self.config.blob_dir is None:
            raise ValueError(
                f"an answer names blobs (vision masks): set {ENV_BLOBS} to the run's blob store"
            )
        return LocalBlobStore(self.config.blob_dir)

    def _save_blobs(self, response: dict[str, Any]) -> None:
        refs = blob_refs(response)
        if refs:
            store, tape = self._store(), self._blobs()
            for ref in refs:
                tape.put_bytes(store.get_bytes(ref), ref.media_type)

    def _restore_blobs(self, response: dict[str, Any]) -> None:
        refs = blob_refs(response)
        if refs:
            assert self.config.source is not None
            store = self._store()
            recorded = type(store)(self.config.source / self.config.scope / "blobs")
            for ref in refs:
                if not store.exists(ref):
                    store.put_bytes(recorded.get_bytes(ref), ref.media_type)

    def _miss(self, stream: str, key: str, canonical: Any, source: _Source) -> None:
        recorded = source.by_stream.get(stream, [])
        nearest = _nearest(canonical, recorded)
        lines = [
            f"replay miss in {stream} (scope {self.config.scope or '-'}): no recorded answer "
            f"left for key {key[:16]}; the stream has {len(recorded)} recorded, "
            f"{sum(r.served for r in recorded)} served",
        ]
        if nearest is None:
            lines.append("nothing recorded in this stream to compare with")
        else:
            served = " (already served)" if nearest.served else ""
            lines.append(f"diff against the nearest recorded request{served} (- recorded, + now):")
            lines.extend(diff_lines(nearest.request, canonical))
        message = "\n".join(lines)
        self._write_miss(
            {
                "stream": stream,
                "key": key,
                "request": canonical,
                "nearest": None if nearest is None else nearest.request,
                "report": message,
            }
        )
        log.error("%s", message)
        raise ReplayMiss(message)

    def _write_miss(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._misses += 1
            number = self._misses
        folder = self.out / "misses"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self._pid}-{number:03d}.json"
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_source(where: Path) -> _Source:
    """A recorded scope's answers and clock values in recorded order: each process's file in its
    own order, the files merged by time."""
    if not where.is_dir():
        raise FileNotFoundError(f"no recorded tape at {where}")
    source = _Source()
    for line in merged_lines(where, "calls"):
        record = _Recorded(line["stream"], line["key"], line["request"], line["response"])
        source.by_key.setdefault(record.key, deque()).append(record)
        source.by_stream.setdefault(record.stream, []).append(record)
    for line in merged_lines(where, "clock"):
        source.clocks.setdefault(line["clock"], deque()).append(line["value"])
    for line in merged_lines(where, "compare"):
        source.observed.setdefault(line["kind"], deque()).append(line["data"])
    return source


def merged_lines(where: Path, kind: str) -> list[dict[str, Any]]:
    """The lines of every process's `<kind>.<pid>.jsonl` in `where`: each file in its order, the
    files merged by the time of their lines."""
    files = [list(_lines(path)) for path in sorted(where.glob(f"{kind}.*.jsonl"))]
    return list(heapq.merge(*files, key=lambda line: line["t"]))


def blob_refs(value: Any) -> list[BlobRef]:
    """The blob references in a JSON value: objects with a sha256 `id`, a `size` and a
    `media_type`."""
    found: list[BlobRef] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if (
                isinstance(item.get("id"), str)
                and _SHA256.match(item["id"])
                and "size" in item
                and "media_type" in item
            ):
                found.append(BlobRef.model_validate(item))
                return
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return found


def diff_lines(recorded: Any, now: Any, limit: int = 120) -> list[str]:
    """A unified diff of two JSON values, pretty-printed with sorted keys, at most `limit`
    lines."""
    a = _pretty(recorded).splitlines()
    b = _pretty(now).splitlines()
    lines = list(difflib.unified_diff(a, b, "recorded", "now", n=3, lineterm=""))
    if len(lines) > limit:
        lines = [*lines[:limit], f"... {len(lines) - limit} more diff lines"]
    return lines


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False)


def _nearest(canonical: Any, recorded: list[_Recorded]) -> _Recorded | None:
    """The recorded request most like `canonical`: the first unserved one in the stream (what a
    replay in step would ask next) unless another is closer, among the next few unserved."""
    unserved = [r for r in recorded if not r.served]
    candidates = (unserved or recorded)[:20]
    if not candidates:
        return None
    text = _pretty(canonical)
    scores = [
        difflib.SequenceMatcher(None, _pretty(r.request), text, autojunk=False).quick_ratio()
        for r in candidates
    ]
    return candidates[max(range(len(candidates)), key=lambda i: (scores[i], -i))]


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


# The process's tape

_override: contextvars.ContextVar[Tape | None] = contextvars.ContextVar("rs_tape", default=None)


@functools.lru_cache(maxsize=1)
def _from_environ() -> Tape | None:
    config = TapeConfig.from_environ(os.environ)
    if config is None:
        return None
    tape = Tape(config)
    atexit.register(tape.close)
    log.warning(
        "replay tape on: %s into %s%s",
        config.mode,
        tape.out,
        "" if config.source is None else f" from {config.source}",
    )
    return tape


def active() -> Tape | None:
    """The tape this context uses (`use`), else the process's (the environment's), else None."""
    tape = _override.get()
    return tape if tape is not None else _from_environ()


@contextlib.contextmanager
def use(tape: Tape) -> Iterator[Tape]:
    """Switch `tape` on for this context (tests; a driver taping in-process)."""
    token = _override.set(tape)
    try:
        yield tape
    finally:
        _override.reset(token)
        tape.close()
