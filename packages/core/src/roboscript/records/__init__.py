"""Event logs (docs/architecture.md, docs/concepts/operations.md) and
files replaced whole.

Each service appends `rs.event/1` lines to its own JSONL file (`EventWriter`; with `background`,
on a thread of its own, so that an event loop never waits for the disk); `merge_events`
joins the logs of an episode by episode id and trace for replay and post-mortems. `JsonlSink`
writes ready-made events and is the JSONL exporter for the `roboscript.event_sinks` group.
`write_atomic` replaces a file at once (ready files, gateway state, Memory versions).
`log_in_background` moves a process's logging onto a thread of its own.
"""

from roboscript.records.files import write_atomic
from roboscript.records.jsonl import EventWriter, JsonlSink, merge_events, read_events
from roboscript.records.logs import log_in_background

__all__ = [
    "EventWriter",
    "JsonlSink",
    "log_in_background",
    "merge_events",
    "read_events",
    "write_atomic",
]
