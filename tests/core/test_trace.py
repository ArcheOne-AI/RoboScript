import pytest

from roboscript.contracts.trace import TraceContext, new_trace

W3C_EXAMPLE = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_traceparent_round_trip() -> None:
    ctx = new_trace().child()
    parsed = TraceContext.from_traceparent(ctx.to_traceparent())
    assert (parsed.trace_id, parsed.span_id, parsed.parent_id) == (ctx.trace_id, ctx.span_id, None)
    example = TraceContext.from_traceparent(W3C_EXAMPLE)
    assert example.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert example.span_id == "00f067aa0ba902b7"
    assert example.to_traceparent() == W3C_EXAMPLE
    # A later version may append fields; its first four are still read.
    future = TraceContext.from_traceparent(W3C_EXAMPLE.replace("00-", "cc-", 1) + "-more")
    assert future.span_id == example.span_id


@pytest.mark.parametrize(
    "header",
    [
        W3C_EXAMPLE.upper(),
        W3C_EXAMPLE.replace("00-", "ff-", 1),  # version ff is invalid
        W3C_EXAMPLE + "-extra",  # version 00 has exactly four fields
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
        "",
    ],
)
def test_invalid_traceparent(header: str) -> None:
    with pytest.raises(ValueError, match="invalid traceparent"):
        TraceContext.from_traceparent(header)
