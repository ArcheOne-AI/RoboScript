"""Deadlines that do not count the time the whole host stood still.

Linux reports, per resource, how long at least one task has been stalled on it since boot
(pressure stall information, `/proc/pressure/<io|memory>`, the `some` line's `total`, in µs). When
a shared disk stalls (a nearly full volume busy with other processes), simulators block in their
writes, and actions waiting on them reach their timeout although nothing of ours failed. A
`stretched` deadline is put off by the time the host was
stalled while it ran, up to a bound, so that a real hang (our process or service gone, the host
otherwise quiet) still ends on time and a frozen host does not end what was only waiting for it.
An `Excuse` puts off a deadline that a poll watches the same way, counting only the polls that
waited on the host (a service that did not answer), up to the same bound: the `some` totals count
every process's stalls, other tenants' too, so an unbounded excuse would never give up on a frozen
service on a busy disk. Where the kernel reports no pressure (macOS, older kernels) both are plain
deadlines.
"""

import asyncio
import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import anyio

PRESSURE: Final = Path("/proc/pressure")
_RESOURCES: Final = ("io", "memory")
_LEAD_S: Final = 0.5  # how long before a deadline it looks at the stalls
_LEAST_S: Final = 1.0  # stalls shorter than this do not move a deadline


def stall_totals() -> tuple[float, ...] | None:
    """Seconds since boot during which some task of the host was stalled, per resource (I/O,
    memory); None where the kernel does not report it."""
    totals = []
    for name in _RESOURCES:
        try:
            text = (PRESSURE / name).read_text(encoding="ascii")
        except OSError:
            return None
        some = next((line for line in text.splitlines() if line.startswith("some ")), None)
        if some is None:
            return None
        totals.append(int(some.rpartition("total=")[2]) / 1e6)
    return tuple(totals)


def stalled_since(start: tuple[float, ...] | None) -> float:
    """Seconds the host was stalled on I/O or memory (the longer) since `stall_totals()` gave
    `start`; 0 where the kernel reports none."""
    now = stall_totals()
    if start is None or now is None:
        return 0.0
    return max(b - a for a, b in zip(start, now, strict=True))


class Excuse:
    """The time a deadline of `seconds` is put off for the host's stalls, as `stretched` puts one
    off, but counted per interval, for the intervals its caller says waited on the host alone
    (`mark`): `most` seconds in all at most (default `seconds`), stalls shorter than a second in
    an interval left out. A watch that asks a service every few seconds marks each poll, excusing
    the intervals that ended without an answer (a frozen host freezes event loops), not one that
    ended in a refusal, so that a host busy with other processes' I/O cannot keep a service that is
    gone from being given up for more than `most`."""

    def __init__(self, seconds: float, most: float | None = None) -> None:
        self.bound = seconds if most is None else most
        self.granted = 0.0
        self._at = stall_totals()

    def mark(self, excused: bool) -> float:
        """End the interval since the last mark (or since it began); the host's stall in it
        counts if `excused`. Returns the seconds granted in all."""
        now, start = stall_totals(), self._at
        self._at = now
        if excused and start is not None and now is not None:
            stalled = max(b - a for a, b in zip(start, now, strict=True))
            if stalled >= _LEAST_S:
                self.granted = min(self.bound, self.granted + stalled)
        return self.granted


@contextlib.contextmanager
def stretched(seconds: float, most: float | None = None) -> Iterator[anyio.CancelScope]:
    """A cancel scope whose deadline is `seconds` from now, put off by the time the host was
    stalled on I/O or memory since it began (the longer of the two), `most` seconds in all at
    most (default: `seconds`, i.e. twice the time at worst). It looks again at each deadline it
    reaches. The caller tells a deadline that ran out by `scope.cancelled_caught`."""
    scope = anyio.CancelScope(deadline=anyio.current_time() + seconds)
    start = stall_totals()
    bound = seconds if most is None else most
    if start is None or bound <= 0:
        with scope:
            yield scope
        return
    loop = asyncio.get_running_loop()
    granted = 0.0

    def look() -> None:
        nonlocal granted, handle
        extra = min(stalled_since(start), bound) - granted
        if extra >= _LEAST_S:
            granted += extra
            scope.deadline += extra
            handle = loop.call_at(max(loop.time(), scope.deadline - _LEAD_S), look)

    handle = loop.call_at(max(loop.time(), scope.deadline - _LEAD_S), look)
    try:
        with scope:
            yield scope
    finally:
        handle.cancel()
