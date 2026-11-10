import json

import pytest
from pydantic import ValidationError

from roboscript.contracts.events import Event, EventKind
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.trace import new_trace


def test_event_json_shape() -> None:
    trace = new_trace()
    event = Event(
        ts=1.5,
        seq=3,
        service="hub",
        kind=EventKind.TOOL_CALL,
        episode_id=EpisodeId("ep-1"),
        trace=trace,
        data={"tool": "observe"},
    )
    line = event.model_dump_json()
    assert list(json.loads(line)) == [
        "schema", "ts", "seq", "service", "kind", "episode_id", "trace", "data",
    ]  # fmt: skip
    assert json.loads(line)["schema"] == "rs.event/1"
    assert Event.model_validate_json(line) == event
    assert Event(ts=0, seq=0, service="s", kind="k").data == {}


def test_reader_keeps_fields_of_newer_minors_and_refuses_other_majors() -> None:
    line = '{"schema": "rs.event/1", "ts": 1, "seq": 0, "service": "s", "kind": "k", "new": 7}'
    event = Event.model_validate_json(line)
    assert json.loads(event.model_dump_json())["new"] == 7
    with pytest.raises(ValidationError):
        Event.model_validate_json(line.replace("rs.event/1", "rs.event/2"))
