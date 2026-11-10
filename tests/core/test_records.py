import json
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr

from roboscript.contracts.events import Event, EventKind
from roboscript.contracts.ids import EpisodeId, WireSecret
from roboscript.contracts.trace import new_trace, use_trace
from roboscript.records import (
    EventWriter,
    JsonlSink,
    merge_events,
    read_events,
)


class Opened(BaseModel):
    episode_id: str
    agent_token: WireSecret


def test_writer_appends_one_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "events" / "hub.jsonl"
    root = new_trace()
    with EventWriter(path, "hub") as writer:
        first = writer.emit(EventKind.EPISODE_OPEN, {"task": "lift_red"})
        with use_trace(root):
            second = writer.emit(
                EventKind.TOOL_CALL, {"tool": "observe"}, episode_id=EpisodeId("e1")
            )
        explicit = root.child()
        third = writer.emit(EventKind.TOOL_RESULT, {}, trace=explicit)
        assert len(path.read_bytes().splitlines()) == 3  # on disk before close
    assert [e.seq for e in (first, second, third)] == [0, 1, 2]
    assert first.trace is None and second.trace == root and third.trace == explicit
    assert list(read_events(path)) == [first, second, third]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[1])["episode_id"] == "e1"
    with pytest.raises(ValueError, match="closed file"):
        writer.emit("late", {})
    with EventWriter(path, "hub") as again:  # reopening appends
        again.emit("restart", {})
    assert [e.kind for e in read_events(path)][-2:] == [EventKind.TOOL_RESULT, "restart"]


def test_events_never_carry_secrets(tmp_path: Path) -> None:
    path = tmp_path / "hub.jsonl"
    with EventWriter(path, "hub") as writer:
        writer.emit("env.open", {"opened": Opened(episode_id="e1", agent_token=SecretStr("tok-9"))})
    text = path.read_text(encoding="utf-8")
    assert "tok-9" not in text and "**********" in text


def test_a_line_still_being_written_is_not_read_yet(tmp_path: Path) -> None:
    path = tmp_path / "hub.jsonl"
    with EventWriter(path, "hub") as writer:
        written = [writer.emit("tick", {"i": i}) for i in range(2)]
    with path.open("ab") as log:
        log.write(b'{"ts": 1.0, "seq": 2, "serv')  # half a line, its writer still at work
    assert list(read_events(path)) == written


def write_log(path: Path, events: list[Event]) -> Path:
    with JsonlSink(path) as sink:
        for event in events:
            sink.write(event)
    return path


def event(service: str, seq: int, ts: float, **kw: object) -> Event:
    return Event(ts=ts, seq=seq, service=service, kind="k", **kw)


def test_merge_by_time_and_by_episode_trace(tmp_path: Path) -> None:
    ep, other = EpisodeId("ep-1"), EpisodeId("ep-2")
    trace, other_trace = new_trace(), new_trace()
    hub = [
        event("hub", 0, 1.0, episode_id=ep, trace=trace),
        event("hub", 1, 3.0, episode_id=other, trace=other_trace),
        event("hub", 2, 3.0, episode_id=ep, trace=trace.child()),
    ]
    perception = [
        event("perception", 0, 2.0, trace=trace.child()),  # served ep-1 without knowing it
        event("perception", 1, 3.0, trace=other_trace.child()),
        event("perception", 2, 3.0),  # no trace, no episode: startup
    ]
    # The env server tags its events with its own episode id; the trace joins them to ep-1.
    env = [event("env", 0, 2.5, episode_id=EpisodeId("env-7"), trace=trace.child())]
    paths = [
        write_log(tmp_path / f"{name}.jsonl", events)
        for name, events in (("hub", hub), ("perception", perception), ("env", env))
    ]
    merged = merge_events(paths)
    assert [(e.ts, e.service, e.seq) for e in merged] == sorted(
        (e.ts, e.service, e.seq) for e in hub + perception + env
    )
    assert merge_events(paths, ep) == [hub[0], perception[0], env[0], hub[2]]
    assert merge_events(paths, other) == [hub[1], perception[1]]
