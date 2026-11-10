"""A benchmark's env server for one task, as a subprocess: started in its own directory (its log
there), waited for until its ready file names its planes, stopped with its process group."""

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from roboscript.rpc import Ready, read_ready

UP_S: Final = 900.0
"""An env server not ready by then has failed to start (many starting at once take minutes)."""


@contextlib.contextmanager
def serving(command: list[str], env: dict[str, str], directory: Path, task: str) -> Iterator[Ready]:
    """`command` (given `--ready-file`) run in `directory` with `env`, until it is ready."""
    ready_file, output = directory / "ready.json", directory / "host.log"
    ready_file.unlink(missing_ok=True)
    started = time.monotonic()
    with output.open("wb") as out:
        process = subprocess.Popen(
            [*command, "--ready-file", str(ready_file)],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=directory,
            start_new_session=True,
        )
    try:
        while (ready := read_ready(ready_file)) is None:
            if (code := process.poll()) is not None or time.monotonic() - started > UP_S:
                tail = output.read_text(errors="replace", encoding="utf-8")[-3000:]
                raise SystemExit(f"the {task} host did not come up ({code}):\n{tail}")
            time.sleep(1.0)
        print(f"{task} host ready in {time.monotonic() - started:.0f} s", flush=True)
        yield ready
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
