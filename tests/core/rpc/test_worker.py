"""`rpc.Worker`: calls in order on one daemon thread."""

import subprocess
import sys
import threading
import time

import anyio
import pytest

from roboscript.rpc import Worker


async def test_the_worker_runs_calls_in_order_on_one_thread() -> None:
    worker = Worker("test-worker")
    seen: list[tuple[int, int]] = []

    def job(i: int) -> int:
        seen.append((i, threading.get_ident()))
        return i * i

    results: dict[int, int] = {}

    async def call(i: int) -> None:
        results[i] = await worker.run(job, i)

    async with anyio.create_task_group() as tg:
        for i in range(5):
            tg.start_soon(call, i)
    assert results == {i: i * i for i in range(5)}
    assert [i for i, _ in seen] == list(range(5))  # queued in call order
    assert len({thread for _, thread in seen} - {threading.get_ident()}) == 1


async def test_errors_reach_the_caller_and_the_worker_goes_on() -> None:
    worker = Worker("test-worker")

    def fail() -> None:
        raise ValueError("bad goal")

    with pytest.raises(ValueError, match="bad goal"):
        await worker.run(fail)
    assert await worker.run(sum, [1, 2]) == 3


async def test_a_cancelled_caller_leaves_the_job_to_finish() -> None:
    worker = Worker("test-worker")
    release, finished = threading.Event(), threading.Event()

    def slow() -> str:
        release.wait(5)
        finished.set()
        return "late"

    with anyio.move_on_after(0.05) as scope:
        await worker.run(slow)
    assert scope.cancelled_caught and not finished.is_set()
    release.set()
    assert await worker.run(str, "next") == "next"  # the late result went nowhere
    assert finished.is_set()


def test_a_busy_worker_does_not_hold_the_process_open() -> None:
    # A service stopped while its worker loads weights must exit at once.
    code = (
        "import asyncio, time\n"
        "from roboscript.rpc import Worker\n"
        "async def main():\n"
        "    worker = Worker('busy')\n"
        "    task = asyncio.ensure_future(worker.run(time.sleep, 60))\n"
        "    await asyncio.sleep(0.2)\n"
        "    task.cancel()\n"
        "asyncio.run(main())\n"
    )
    started = time.monotonic()
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
    assert time.monotonic() - started < 10
