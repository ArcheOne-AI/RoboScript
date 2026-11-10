"""Server-sent events as OpenAI APIs stream them: whole events out of a byte stream, their
fields, and the error event that ends a stream early."""

import json
import re
from typing import Final

_BOUNDARY: Final = re.compile(rb"\r\n\r\n|\n\n|\r\r")
# Only these end a line: `str.splitlines` also cuts at U+2028 and others JSON strings may hold.
_LINE: Final = re.compile(r"\r\n|\r|\n")


class Splitter:
    """Cuts a byte stream into whole events, each with the blank line that ends it, scanning
    each byte about once however the stream is chunked."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        start = max(0, len(self._buffer) - 3)  # a boundary may begin in the previous chunk
        self._buffer += chunk
        events, end = [], 0
        for boundary in _BOUNDARY.finditer(self._buffer, start):
            events.append(bytes(self._buffer[end : boundary.end()]))
            end = boundary.end()
        del self._buffer[:end]
        return events

    def rest(self) -> bytes:
        """Bytes after the last whole event (a stream that ended without a blank line)."""
        rest = bytes(self._buffer)
        self._buffer.clear()
        return rest


def data(event: bytes) -> str:
    """The joined `data` lines of one event."""
    lines = []
    for line in _LINE.split(event.decode("utf-8", errors="replace")):
        field, _, value = line.partition(":")
        if field == "data":
            lines.append(value.removeprefix(" "))
    return "\n".join(lines)


def encode(name: str, data: object) -> bytes:
    """A named event with JSON data."""
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


def error_event(wire: str, code: str, message: str) -> bytes:
    """An event that makes OpenAI clients raise: openai-python raises on any event whose data
    has an `error` object; a Responses stream also gets the fields of its own `error` event."""
    error = {"type": "rs_gateway_error", "code": code, "message": message}
    if wire == "responses":
        return encode("error", {"type": "error", "code": code, "message": message, "error": error})
    return f"data: {json.dumps({'error': error})}\n\n".encode()
