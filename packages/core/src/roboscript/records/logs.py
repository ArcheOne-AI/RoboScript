"""A process's logging on a thread of its own: a service whose log file sits on a stalled file
system must not wait for it where it serves (an event loop, a request's thread). A saturated
journal can hold a single append for minutes; a relay that logs its own timeout there answers
nobody until it clears.

```python
log_in_background(logging.INFO, "%(asctime)s %(levelname)s %(name)s: %(message)s")
```
"""

import atexit
import logging
import queue
import sys
import threading
from typing import IO, Final

STOP_WAIT_S: Final = 5.0
"""How long the process's exit waits for the lines still queued (a stuck disk loses them)."""


class _Deferred(logging.Handler):
    """Hands every record to the writer thread as it is: formatting it (a traceback reads source
    files) happens there too."""

    def __init__(self, records: "queue.SimpleQueue[logging.LogRecord | None]") -> None:
        super().__init__()
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.put(record)


def log_in_background(
    level: int, fmt: str, stream: IO[str] | None = None
) -> logging.Handler | None:
    """`logging.basicConfig(level=level, format=fmt)`, but the root logger queues each record for
    a daemon thread that writes it to `stream` (stderr). Like `basicConfig`, nothing happens if
    the root logger has handlers already (a test runner's). At exit the queued lines get
    `STOP_WAIT_S` to go out. The root's new handler, or None."""
    root = logging.getLogger()
    if root.handlers:
        return None
    records: queue.SimpleQueue[logging.LogRecord | None] = queue.SimpleQueue()
    writer = logging.StreamHandler(stream or sys.stderr)
    writer.setFormatter(logging.Formatter(fmt))

    def write() -> None:
        while (record := records.get()) is not None:
            writer.handle(record)  # its own errors go to `handleError`

    thread = threading.Thread(target=write, name="log-writer", daemon=True)
    thread.start()

    def stop() -> None:
        records.put(None)
        thread.join(STOP_WAIT_S)

    atexit.register(stop)
    handler = _Deferred(records)
    root.addHandler(handler)
    root.setLevel(level)
    return handler
