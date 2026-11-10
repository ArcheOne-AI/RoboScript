"""Signals as requests to stop, for a process that must clean up before it exits (`rpc.serve`,
the supervisor, launchers; `roboscript.stack` re-exports it).

While an `on_signals` context is open, the signals it names call its callback instead of their
previous handlers, which come back when the last context for a signal closes. A signal the
process ignores stays ignored: SIGHUP under `nohup`, so that a run started that way outlives its
ssh session. Contexts nest and overlap (several servers in one process, a server under an
evaluation), and a signal calls every open one. Handlers exist only in the main thread; elsewhere
a context does nothing. `cancel_on_signals` turns a signal into the cancellation of an async
`with` body, so that its `finally` blocks (shielded cleanup included) run before `Stopped` names
the signal.
"""

import asyncio
import contextlib
import signal
import threading
from collections.abc import Callable, Iterable, Iterator
from types import FrameType
from typing import Any

import anyio

Callback = Callable[[int, FrameType | None], None]

_callbacks: dict[int, list[Callback]] = {}  # per signal, the open contexts' callbacks in order
_previous: dict[int, Any] = {}  # the handlers they replaced


class Stopped(Exception):
    """A signal stopped the body of `cancel_on_signals`."""

    def __init__(self, sig: signal.Signals) -> None:
        super().__init__(f"stopped by {sig.name}")
        self.signal = sig


@contextlib.contextmanager
def on_signals(callback: Callback, signals: Iterable[int]) -> Iterator[None]:
    """While open, each of `signals` calls `callback(signal, frame)` (in the main thread), unless
    the process ignores it (`nohup`)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    wanted = [
        sig
        for sig in dict.fromkeys(signals)
        if sig in _callbacks or signal.getsignal(sig) != signal.SIG_IGN
    ]
    for sig in wanted:
        if sig not in _callbacks:
            _previous[sig] = signal.signal(sig, _dispatch)
            _callbacks[sig] = []
        _callbacks[sig].append(callback)
    try:
        yield
    finally:
        for sig in wanted:
            _callbacks[sig].remove(callback)
            if not _callbacks[sig]:
                del _callbacks[sig]
                signal.signal(sig, _previous.pop(sig))


@contextlib.contextmanager
def cancel_on_signals(*signals: signal.Signals) -> Iterator[None]:
    """Run the body (of an async function) until one of `signals` cancels it; `Stopped` follows
    once its `finally` blocks have run. Outside the main thread signals do not reach it, and an
    ignored signal stays ignored."""
    loop = asyncio.get_running_loop()
    received: list[signal.Signals] = []
    with anyio.CancelScope() as scope:

        def cancel(sig: int, frame: FrameType | None) -> None:
            received.append(signal.Signals(sig))
            loop.call_soon_threadsafe(scope.cancel)  # handlers run between bytecodes: defer

        with on_signals(cancel, signals):
            yield
    if received:
        raise Stopped(received[0])


def _dispatch(sig: int, frame: FrameType | None) -> None:
    for callback in list(_callbacks.get(sig, ())):
        callback(sig, frame)
