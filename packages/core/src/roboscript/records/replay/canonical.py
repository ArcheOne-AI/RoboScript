"""The canonical form of a recorded request or result: what two runs of the same behavior agree
on (docs/concepts/operations.md).

`canonicalize` returns a copy of a JSON value with what differs between such runs replaced by a
placeholder, so a request's key (`request_key`) is the same in the baseline and in the replay of
it. It changes nothing a model or a tool computes; it only decides what a key hashes and what the
comparator diffs. The rules, all listed here for review:

- `STRIPPED_KEYS`: a mapping key in this set has its value replaced by `"<stripped>"`, at any
  depth: ids of runs, episodes, sessions, traces, requests and action batches, tokens and keys,
  URLs, ports and process ids, timestamps and measured durations.
- `volatile`: literal strings the caller names (run roots, the code SHA, ...) are replaced by
  `<name>` inside every string, longest first.
- inside every string (`_PATTERNS`, in order): data URLs become the sha256 of their decoded bytes
  (content stays, bytes go); UUIDs become `<uuid>`; ids made by `contracts.ids.new_id`
  (`ep-3f9a0c1b2d4e`) `<prefix>-<id>`; ISO-8601 times `<time>`; `http(s)://host:port` `<url>`.

Kept on purpose: blob sha256 ids (content addresses), Hub aliases (`obj_3`, `frame_1`, counters),
ids an upstream model returned (they come back from the record), every number.
"""

import base64
import hashlib
import re
from collections.abc import Callable, Mapping
from typing import Any, Final

from roboscript.contracts.canonical import canonical_json

STRIPPED: Final = "<stripped>"

STRIPPED_KEYS: Final = frozenset(
    {
        # ids of runs, episodes, sessions, traces, requests and action batches
        "episode_id",
        "env_episode",
        "session_id",
        "run_id",
        "run_dir",
        "trace",
        "trace_id",
        "span_id",
        "parent_id",
        "traceparent",
        "request_id",
        "batch_id",
        "prompt_cache_key",
        # secrets and where services listen
        "token",
        "api_key",
        "authorization",
        "url",
        "base_url",
        "agent_url",
        "host_url",
        "port",
        "pid",
        # clocks and measured durations
        "ts",
        "timestamp",
        "created",
        "created_at",
        "frozen_at",
        "at",
        "elapsed_s",
        "latency_s",
        "duration_s",
        "queued_s",
        "time_s",
    }
)

_DATA_URL: Final = re.compile(r"data:([\w.+/-]+);base64,([A-Za-z0-9+/=]+)")
_NEW_ID: Final = re.compile(r"\b([a-z][a-z0-9]*)-[0-9a-f]{12}\b")
_UUID: Final = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_TIME: Final = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_URL: Final = re.compile(r"https?://(?:\[[0-9a-fA-F:]+\]|[\w.-]+):\d+")


def _data_url(found: re.Match[str]) -> str:
    digest = hashlib.sha256(base64.b64decode(found[2])).hexdigest()
    return f"data:{found[1]};sha256,{digest}"


_PATTERNS: Final[tuple[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...]] = (
    (_DATA_URL, _data_url),
    (_UUID, "<uuid>"),  # before new ids: a UUID's last group looks like one
    (_NEW_ID, r"\1-<id>"),
    (_TIME, "<time>"),
    (_URL, "<url>"),
)


def canonicalize(value: Any, volatile: Mapping[str, str] | None = None) -> Any:
    """`value` (JSON: dicts, lists, strings, numbers, booleans, None) with the module's rules
    applied; `volatile` maps a placeholder name to a literal string to replace."""
    literals = sorted(((v, k) for k, v in (volatile or {}).items() if v), key=lambda p: -len(p[0]))
    return _walk(value, literals)


def request_key(stream: str, canonical: Any) -> str:
    """The key of a canonical request in `stream` (e.g. `model:/v1/responses`, `motion.plan`)."""
    text = canonical_json({"stream": stream, "request": canonical})
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _walk(value: Any, literals: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {
            key: STRIPPED if key.lower() in STRIPPED_KEYS else _walk(item, literals)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_walk(item, literals) for item in value]
    if isinstance(value, str):
        return _string(value, literals)
    return value


def _string(text: str, literals: list[tuple[str, str]]) -> str:
    for literal, name in literals:
        text = text.replace(literal, f"<{name}>")
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
