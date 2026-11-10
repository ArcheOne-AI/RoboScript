"""W3C trace context: ids, `traceparent` headers and the trace of the current task.

One episode is one trace. A service handling a call runs under a child span of the caller's
(`from_traceparent(header).child()`), so its events join the episode by `trace_id`.
"""

import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, ConfigDict, Field

_TRACEPARENT = re.compile(r"([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}(-.*)?")


class TraceContext(BaseModel):
    """One span: `span_id` is this span, `parent_id` the span it was started under."""

    model_config = ConfigDict(frozen=True)

    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    parent_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")

    def child(self) -> "TraceContext":
        return TraceContext(
            trace_id=self.trace_id, span_id=secrets.token_hex(8), parent_id=self.span_id
        )

    def to_traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-01"

    @classmethod
    def from_traceparent(cls, header: str) -> "TraceContext":
        """The caller's span from a `traceparent` header; `ValueError` if it is invalid."""
        match = _TRACEPARENT.fullmatch(header)
        # Version ff is invalid, version 00 has exactly four fields, all-zero ids are invalid.
        if (
            match is None
            or match[1] == "ff"
            or (match[1] == "00" and match[4] is not None)
            or not int(match[2], 16)
            or not int(match[3], 16)
        ):
            raise ValueError(f"invalid traceparent header: {header!r}")
        return cls(trace_id=match[2], span_id=match[3])


def new_trace() -> TraceContext:
    """A root span of a new trace."""
    return TraceContext(trace_id=secrets.token_hex(16), span_id=secrets.token_hex(8))


current_trace: ContextVar[TraceContext | None] = ContextVar("rs_current_trace", default=None)


def get_trace() -> TraceContext | None:
    return current_trace.get()


@contextmanager
def use_trace(ctx: TraceContext) -> Iterator[TraceContext]:
    """Make `ctx` the current trace inside the block."""
    token = current_trace.set(ctx)
    try:
        yield ctx
    finally:
        current_trace.reset(token)
