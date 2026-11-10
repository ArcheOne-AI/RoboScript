"""Hooks whose dispatch mode is fixed where they are declared, and the bus that runs them."""

import asyncio
import bisect
import inspect
import itertools
import logging
from collections.abc import Callable, Sequence
from typing import Any, Generic, Literal, NamedTuple, TypeVar, get_args

T = TypeVar("T")
Mode = Literal["emit", "serial", "parallel", "middleware", "bail"]
_MODES: tuple[str, ...] = get_args(Mode)

logger = logging.getLogger(__name__)


class Revoke:
    """Undoes one registration; calling it again does nothing."""

    __slots__ = ("_undo",)

    def __init__(self, undo: Callable[[], None]) -> None:
        self._undo: Callable[[], None] | None = undo

    def __call__(self) -> None:
        undo, self._undo = self._undo, None
        if undo is not None:
            undo()


class Hook(Generic[T]):
    """A named extension point whose handlers take a T. The mode fixes how `EventBus.call` runs
    them (handlers may be sync or async):

    - `emit`: observers, in order; a failing handler is logged and the others still run.
      Returns None.
    - `serial`: awaited in order; returns their results.
    - `parallel`: run concurrently; returns their results in handler order, or raises the
      first failure (in handler order) once all have finished.
    - `middleware`: `handler(value, next)`; `await next(value)` runs the rest of the chain,
      which ends in the `terminal` passed to `call`; not calling `next` short-circuits.
      Returns the first handler's result.
    - `bail`: in order until one returns something other than None, which is returned.

    Handlers with a higher priority run first (outermost for middleware); ties run in
    registration order.
    """

    __slots__ = ("mode", "name")

    def __init__(self, name: str, mode: Mode) -> None:
        if mode not in _MODES:
            raise ValueError(f"hook {name!r}: mode must be one of {_MODES}, got {mode!r}")
        self.name = name
        self.mode = mode

    def __repr__(self) -> str:
        return f"Hook({self.name!r}, {self.mode!r})"


class _Entry(NamedTuple):
    rank: int  # -priority: ascending order puts high priorities first
    seq: int
    handler: Callable[..., Any]


class EventBus:
    """Handlers per hook, kept in dispatch order."""

    def __init__(self) -> None:
        self._entries: dict[Hook[Any], list[_Entry]] = {}
        self._seq = itertools.count()

    def on(self, hook: Hook[T], handler: Callable[..., Any], *, priority: int = 0) -> Revoke:
        entry = _Entry(-priority, next(self._seq), handler)
        entries = self._entries.setdefault(hook, [])
        bisect.insort(entries, entry)

        def remove() -> None:
            entries.remove(entry)
            if not entries and self._entries.get(hook) is entries:
                del self._entries[hook]

        return Revoke(remove)

    def hooks(self) -> list[Hook[Any]]:
        """The hooks that have handlers."""
        return list(self._entries)

    async def call(
        self, hook: Hook[T], value: T, *, terminal: Callable[[T], Any] | None = None
    ) -> Any:
        """Run the handlers of `hook` on `value` as its mode says (see `Hook`)."""
        if terminal is not None and hook.mode != "middleware":
            raise TypeError(f"{hook!r} is not a middleware hook: it takes no terminal")
        handlers = [entry.handler for entry in self._entries.get(hook, ())]
        if hook.mode == "emit":
            for handler in handlers:
                try:
                    await _invoke(handler, value)
                except Exception:
                    logger.exception("handler %r of %r failed", handler, hook)
            return None
        if hook.mode == "serial":
            return [await _invoke(handler, value) for handler in handlers]
        if hook.mode == "parallel":
            results = await asyncio.gather(
                *(_invoke(handler, value) for handler in handlers), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return results
        if hook.mode == "middleware":
            if terminal is None:
                raise TypeError(f"{hook!r} is a middleware hook: pass a terminal")
            return await _chain(handlers, 0, value, terminal)
        if hook.mode == "bail":
            for handler in handlers:
                result = await _invoke(handler, value)
                if result is not None:
                    return result
            return None
        raise ValueError(f"{hook!r}: unknown mode {hook.mode!r}")


async def _chain(
    handlers: Sequence[Callable[..., Any]], index: int, value: Any, terminal: Callable[[Any], Any]
) -> Any:
    if index == len(handlers):
        return await _invoke(terminal, value)
    return await _invoke(
        handlers[index], value, lambda new: _chain(handlers, index + 1, new, terminal)
    )


async def _invoke(function: Callable[..., Any], *args: Any) -> Any:
    result = function(*args)
    if inspect.isawaitable(result):
        return await result
    return result
