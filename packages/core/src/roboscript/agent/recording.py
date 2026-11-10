"""The record of what a model saw and did (docs/concepts/operations.md).

`llm_calls.jsonl` holds, in order:

- `context` lines: the `instructions` and `tools` of the requests, once per episode for each
  distinct pair, by the sha256 of their canonical JSON;
- a `request` line before each model call is sent: the request without `instructions` and
  `tools` (they are the `context` it names), images as `rs-blob:<sha256>` URLs with their blob
  references beside it;
- a `response` line once the call is over: the response (output items, usage, deployment,
  latency) or the error.

A call the context budget shrank has a `model.context` event before its request (estimates
before and after, the budget, the shrink state). A call infrastructure failed and sent again has
a request and a response line per attempt, with `attempt` (from 2), and a `model.retry` event
before each.

So a call that never returns (a crash, a cancellation) still leaves what the model was sent.
`turns.jsonl` holds each turn's observation, calls and what they returned; `model.request` and
`model.response` events go to the event log. `rebuild_requests` turns the log back into the
requests as they were sent, byte for byte.

A `Recorder` made with `background` appends on a thread of its own (an event loop that serves
others, the P3 Agent host's, must not wait for a stalled disk); `close` then waits a bounded
time for the lines queued.
"""

import base64
import hashlib
import json
import logging
import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from roboscript.blobs import BlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.events import EventKind
from roboscript.contracts.ids import EpisodeId
from roboscript.records import EventWriter

log = logging.getLogger(__name__)

BLOB_URL: Final = "rs-blob:"
"""Prefix of an image URL that stands for the blob with that sha256."""

CONTEXT: Final = ("instructions", "tools")
"""Request fields logged once per episode in `context` lines."""

_MODEL_IMAGES: Final = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})


class Recorder:
    """Writes an episode's `llm_calls.jsonl` and `turns.jsonl` under `directory` and its model
    events to `events`."""

    def __init__(
        self, directory: Path, events: EventWriter | None = None, *, background: bool = False
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.events = events
        self.calls = directory / "llm_calls.jsonl"
        self.turns = directory / "turns.jsonl"
        self._contexts: set[str] = set()
        self._appender = _Appender() if background else None

    def close(self, wait: float = 10.0) -> bool:
        """With `background`: wait up to `wait` s for the queued lines and stop the thread;
        whether they are all in. Without, nothing to wait for."""
        return True if self._appender is None else self._appender.close(wait)

    def _append(self, path: Path, record: Mapping[str, Any]) -> None:
        if self._appender is None:
            _append(path, record)
        else:
            self._appender.put(path, json.dumps(record, ensure_ascii=False) + "\n")

    def request(
        self,
        episode_id: EpisodeId,
        call: int,
        turn: int,
        request: Mapping[str, Any],
        refs: Mapping[str, BlobRef],
        *,
        attempt: int = 1,
    ) -> None:
        """Before model call `call` is sent: its `request` line and `model.request` event."""
        context = {key: request[key] for key in CONTEXT}
        context_digest = digest(context)
        if context_digest not in self._contexts:
            self._contexts.add(context_digest)
            self._append(self.calls, {"kind": "context", "sha256": context_digest, **context})
        rest = {key: value for key, value in request.items() if key not in CONTEXT}
        line = {
            "kind": "request",
            "call": call,
            "turn": turn,
            "ts": time.time(),
            "context": context_digest,
            "request": rest,
            "blobs": {blob: ref.model_dump(mode="json") for blob, ref in refs.items()},
            **_attempt(attempt),
        }
        self._append(self.calls, line)
        if self.events is not None:
            data = {
                "call": call,
                "turn": turn,
                "model": request.get("model"),
                "sha256": digest(request),
                "items": len(request.get("input", ())),
                "images": sorted(refs),
                **_attempt(attempt),
            }
            self.events.emit(EventKind.MODEL_REQUEST, data, episode_id=episode_id)

    def response(
        self,
        episode_id: EpisodeId,
        call: int,
        turn: int,
        response: Mapping[str, Any] | None,
        error: str | None,
        *,
        attempt: int = 1,
    ) -> None:
        """After model call `call`: its `response` line and `model.response` event."""
        line = {
            "kind": "response",
            "call": call,
            "turn": turn,
            "ts": time.time(),
            "response": response,
            "error": error,
            **_attempt(attempt),
        }
        self._append(self.calls, line)
        if self.events is not None:
            got = response or {}
            data = {
                "call": call,
                "turn": turn,
                "status": got.get("status"),
                "deployment": got.get("deployment"),
                "latency_s": got.get("latency_s"),
                "usage": got.get("usage"),
                "calls": [
                    item.get("name")
                    for item in got.get("output", [])
                    if item.get("type") == "function_call"
                ],
                "error": error,
                **_attempt(attempt),
            }
            self.events.emit(EventKind.MODEL_RESPONSE, data, episode_id=episode_id)

    def retry(
        self, episode_id: EpisodeId, call: int, turn: int, attempt: int, wait_s: float, error: str
    ) -> None:
        """Model call `call` is sent again as `attempt` after `wait_s`: a `model.retry` event."""
        if self.events is not None:
            data = {"call": call, "turn": turn, "attempt": attempt, "wait_s": wait_s}
            self.events.emit(EventKind.MODEL_RETRY, {**data, "error": error}, episode_id=episode_id)

    def context(self, episode_id: EpisodeId, data: Mapping[str, Any]) -> None:
        """The context budget shrank a request's input: a `model.context` event."""
        if self.events is not None:
            self.events.emit(EventKind.MODEL_CONTEXT, dict(data), episode_id=episode_id)

    def turn(self, record: Mapping[str, Any]) -> None:
        self._append(self.turns, record)


def _attempt(attempt: int) -> dict[str, int]:
    """A retry's `attempt` field; a call's first attempt has none."""
    return {} if attempt == 1 else {"attempt": attempt}


def digest(value: Any) -> str:
    """sha256 hex of a value's canonical JSON (a request's names its images by their URLs)."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


async def materialize(
    request: Mapping[str, Any], refs: Mapping[str, BlobRef], blobs: BlobStore
) -> dict[str, Any]:
    """`request` with every `rs-blob:<sha256>` image URL replaced by a data URL of the blob."""
    urls = {blob: await data_url(ref, blobs) for blob, ref in refs.items()}
    replaced: dict[str, Any] = _replace(request, urls)
    return replaced


async def rebuild_requests(path: Path, blobs: BlobStore) -> list[dict[str, Any]]:
    """The requests an `llm_calls.jsonl` records, in order, as they were sent."""
    lines = read_lines(path)
    contexts = {line["sha256"]: line for line in lines if line["kind"] == "context"}
    requests = []
    for line in lines:
        if line["kind"] == "request":
            context = {key: contexts[line["context"]][key] for key in CONTEXT}
            refs = {blob: BlobRef.model_validate(ref) for blob, ref in line["blobs"].items()}
            requests.append(await materialize({**line["request"], **context}, refs, blobs))
    return requests


async def data_url(ref: BlobRef, blobs: BlobStore) -> str:
    if ref.media_type not in _MODEL_IMAGES:
        raise ValueError(f"blob {ref.id} is {ref.media_type}; models take {sorted(_MODEL_IMAGES)}")
    data = base64.b64encode(await blobs.aget_bytes(ref)).decode("ascii")
    return f"data:{ref.media_type};base64,{data}"


def read_lines(path: Path) -> list[dict[str, Any]]:
    """The records of a JSONL file (`llm_calls.jsonl`, `turns.jsonl`)."""
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


def _replace(value: Any, urls: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        url = value.get("image_url")
        if value.get("type") == "input_image" and isinstance(url, str) and url.startswith(BLOB_URL):
            return {**value, "image_url": urls[url.removeprefix(BLOB_URL)]}
        return {key: _replace(item, urls) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item, urls) for item in value]
    return value


def _append(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


class _Appender:
    """Appends lines to files on a daemon thread, in the order they were put; a line that cannot
    be appended is logged and dropped."""

    def __init__(self) -> None:
        self._lines: queue.SimpleQueue[tuple[Path, str] | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def put(self, path: Path, line: str) -> None:
        self._lines.put((path, line))

    def close(self, wait: float) -> bool:
        self._lines.put(None)
        self._thread.join(wait)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while (item := self._lines.get()) is not None:
            path, line = item
            try:
                with path.open("a", encoding="utf-8") as file:
                    file.write(line)
            except OSError as error:
                log.error("cannot append to %s: %s", path, error)
