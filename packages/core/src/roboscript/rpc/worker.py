"""A thread that runs a service's calls one at a time, e.g. the one that owns its CUDA state."""

import asyncio
import functools
import queue
import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Worker:
    """Runs calls in order, one at a time, on one daemon thread: a daemon, so that a service
    stopped during a warm-up exits at once instead of waiting for it. A caller that stops waiting
    leaves its call to finish; the result goes nowhere."""

    def __init__(self, name: str) -> None:
        self._jobs: queue.SimpleQueue[
            tuple[Callable[[], Any], asyncio.Future[Any], asyncio.AbstractEventLoop]
        ] = queue.SimpleQueue()
        threading.Thread(target=self._loop, name=name, daemon=True).start()

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        self._jobs.put((functools.partial(fn, *args), future, loop))
        return await future

    def _loop(self) -> None:
        while True:
            call, future, loop = self._jobs.get()
            try:
                result = call()
            except Exception as error:  # handed to the caller
                loop.call_soon_threadsafe(_settle, future, None, error)
            else:
                loop.call_soon_threadsafe(_settle, future, result, None)


def _settle(future: asyncio.Future[Any], result: Any, error: Exception | None) -> None:
    if future.cancelled():  # the caller stopped waiting
        return
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)
