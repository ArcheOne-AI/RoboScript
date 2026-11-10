"""The process supervisor: start a stack in dependency order, gate on readiness, stop by group.

Services inherit `os.environ` and get their arguments on the command line, both readable by other
local users (`ps`, `/proc`): keep secrets out of both; services read them from `.env` files.
"""

import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

import anyio
import anyio.abc
import httpx
import yaml
from pydantic import TypeAdapter

from roboscript.records import write_atomic
from roboscript.rpc import Ready, read_ready
from roboscript.settings import repo_root
from roboscript.stack.config import ServiceSpec, StackConfig

_POLL_S: Final = 0.05
_PROBE_TIMEOUT_S: Final = 2.0
_KILL_WAIT_S: Final = 5.0  # how long a group may take to empty after SIGKILL
_ENDPOINTS: Final = TypeAdapter(dict[str, Ready])


class StackError(RuntimeError):
    """A service did not come up, or exited while the stack was running."""


@dataclass(frozen=True)
class Running:
    endpoints: dict[str, str]  # service name -> primary URL
    urls: dict[str, dict[str, str]]  # service name -> plane -> URL, for planes served apart
    pids: dict[str, int]  # service name -> pid, which is also its process group id
    run_dir: Path
    blob_dir: Path  # the blob store directory the services share (their RS_BLOB_DIR)


@dataclass(frozen=True)
class _Child:
    spec: ServiceSpec
    process: anyio.abc.Process
    log: Path
    ready_file: Path


@asynccontextmanager
async def up(
    config: StackConfig | str | Path, *, on_start: Callable[[str, int], None] | None = None
) -> AsyncIterator[Running]:
    """Run a stack (a `StackConfig` or the path of its YAML file) for the `async with` body.

    Services start in dependency order, each in its own process group with stdout and stderr in
    `<run_dir>/logs/<name>.log`; `on_start(name, pid)` hears of each as soon as its process
    runs (for a launcher's own record, before it is ready). Each must write its ready file
    (`roboscript.rpc.write_ready`) and answer `GET /readyz` with 200 on every URL in it within
    `ready_timeout_s` before the next starts; `<run_dir>/.roboscript/endpoints.json` holds the
    ready files of those up. A
    service that exits cancels the body and raises `StackError` with its last log lines, chained
    to the body's own error. On exit every group gets SIGTERM, then SIGKILL if it has not emptied
    within its `stop_grace_s`. Every service gets the stack's blob store directory, created
    here, as `RS_BLOB_DIR`.
    """
    config = _load(config)
    config.shared_blob_dir().mkdir(parents=True, exist_ok=True)
    children: list[_Child] = []
    ready: dict[str, Ready] = {}
    body = anyio.CancelScope()
    watcher = None
    error: BaseException | None = None
    try:
        async with httpx.AsyncClient(trust_env=False) as http:
            for spec in config.start_order():
                child = await _start(config, spec)
                children.append(child)
                if on_start is not None:
                    on_start(spec.name, child.process.pid)
                ready[spec.name] = await _wait_ready(child, children, http)
                endpoints = _ENDPOINTS.dump_json(ready, indent=2)
                write_atomic(config.run_dir / ".roboscript" / "endpoints.json", endpoints)
        # A task of its own, not a task group: those wrap the body's exceptions in groups.
        watcher = asyncio.get_running_loop().create_task(_watch(children, body))
        with body:
            yield Running(
                endpoints={name: r.url for name, r in ready.items()},
                urls={name: dict(r.urls) for name, r in ready.items()},
                pids={child.spec.name: child.process.pid for child in children},
                run_dir=config.run_dir,
                blob_dir=config.shared_blob_dir(),
            )
    except BaseException as caught:
        error = caught
        raise
    finally:
        with anyio.CancelScope(shield=True):
            if watcher is not None:
                watcher.cancel()
                await asyncio.wait({watcher})
            exited = await _exit_report(children)  # before the stop below ends them all
            await _stop(children)
        # Cancellation, interrupts and a StackError from startup propagate as they are.
        replace = error is None or isinstance(error, Exception)
        if exited is not None and replace and not isinstance(error, StackError):
            raise StackError(exited) from error


def _load(config: StackConfig | str | Path) -> StackConfig:
    if isinstance(config, StackConfig):
        return config
    path = Path(config)
    try:
        return StackConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def ready_path(run_dir: Path, service: str) -> Path:
    """Where a stack in `run_dir` has `service` write its ready file (`--ready-file`)."""
    return run_dir / ".roboscript" / "ready" / f"{service}.json"


async def _start(config: StackConfig, spec: ServiceSpec) -> _Child:
    log = config.run_dir / "logs" / f"{spec.name}.log"
    ready_file = ready_path(config.run_dir, spec.name)
    command = [*_command(config, spec), "--ready-file", str(ready_file)]
    with _open_log(log, ready_file, command) as out:
        process = await anyio.open_process(
            command,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=_env(config, spec),
            start_new_session=True,  # its own process group, stopped as a whole
        )
    return _Child(spec, process, log, ready_file)


def _command(config: StackConfig, spec: ServiceSpec) -> list[str]:
    if isinstance(spec.interpreter, Path):
        python = spec.interpreter
    else:
        named = getattr(config.interpreters, spec.interpreter)
        assert named is not None  # StackConfig checks it
        python = named
    head = ["-m", spec.module] if spec.module is not None else spec.argv or []
    return [str(python), *head, *spec.args]


def _env(config: StackConfig, spec: ServiceSpec) -> dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # log lines land before a crash
    # Services find `.env` through RS_ROOT whatever their working directory (a simulator may change
    # it).
    env["RS_ROOT"] = str(repo_root())
    env["RS_BLOB_DIR"] = str(config.shared_blob_dir())
    if spec.interpreter in ("sim", "vision"):
        # A user site would shadow the env's packages; our src dirs are not installed there.
        env["PYTHONNOUSERSITE"] = "1"
        paths = config.sim_pythonpath if spec.interpreter == "sim" else config.vision_pythonpath
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
    if spec.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)
    return env | spec.env


def _open_log(log: Path, ready_file: Path, command: list[str]) -> IO[bytes]:
    log.parent.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.unlink(missing_ok=True)  # left by an earlier run
    out = log.open("ab")
    out.write(f"$ {shlex.join(command)}\n".encode())
    out.flush()
    return out


async def _wait_ready(child: _Child, children: list[_Child], http: httpx.AsyncClient) -> Ready:
    """The child's ready file once all its URLs are ready; fails if any child exits meanwhile."""
    spec = child.spec
    deadline = anyio.current_time() + spec.ready_timeout_s
    while True:
        exited = _exited(children)
        if exited is not None:
            raise StackError(exited)
        ready = read_ready(child.ready_file)
        unready = "no ready file" if ready is None else await _unready(http, ready)
        if ready is not None and unready is None:
            return ready
        if anyio.current_time() > deadline:
            message = f"service {spec.name} not ready after {spec.ready_timeout_s} s: {unready}"
            raise StackError(message + _tail(child.log))
        await anyio.sleep(_POLL_S)


async def _unready(http: httpx.AsyncClient, ready: Ready) -> str | None:
    for url in dict.fromkeys([ready.url, *ready.urls.values()]):
        try:
            response = await http.get(f"{url}/readyz", timeout=_PROBE_TIMEOUT_S)
        except httpx.TransportError:  # timeouts included: not ready yet
            return f"{url}/readyz does not answer"
        if response.status_code != 200:
            return f"{url}/readyz answers {response.status_code}"
    return None


async def _watch(children: list[_Child], body: anyio.CancelScope) -> None:
    """Cancel the body as soon as a child exits; `up` then raises the `StackError`."""
    while True:
        await anyio.sleep(_POLL_S)
        if _exited(children) is not None:
            body.cancel()
            return


async def _exit_report(children: list[_Child]) -> str | None:
    """Like `_exited`, but also for a child the OS has reaped and asyncio not yet recorded."""
    for child in children:
        if child.process.returncode is None and not _pid_alive(child):
            with anyio.move_on_after(_KILL_WAIT_S):
                await child.process.wait()
    return _exited(children)


def _exited(children: list[_Child]) -> str | None:
    for child in children:
        code = child.process.returncode
        if code is not None:
            how = f"was killed by {signal.Signals(-code).name}" if code < 0 else f"exited ({code})"
            return f"service {child.spec.name} {how}" + _tail(child.log)
    return None


def _tail(log: Path, lines: int = 20) -> str:
    with log.open("rb") as file:
        file.seek(max(0, file.seek(0, os.SEEK_END) - 4096))
        text = file.read().decode(errors="replace")
    return f"\n--- last lines of {log} ---\n" + "\n".join(text.splitlines()[-lines:])


async def _stop(children: list[_Child]) -> None:
    await stop_groups([(child.process, child.spec.stop_grace_s) for child in reversed(children)])


async def stop_groups(groups: Sequence[tuple[anyio.abc.Process, float]]) -> None:
    """Stop process groups, each led by its process (`start_new_session`), with a bound: SIGTERM
    to every group at once, SIGKILL to each one still not empty after its grace seconds, and
    return at most `_KILL_WAIT_S` after the longest grace even if a group never empties (a process
    stuck in the kernel). The leaders are reaped."""
    start = anyio.current_time()
    last = max((grace for _, grace in groups), default=0.0) + _KILL_WAIT_S
    try:
        for process, _ in groups:
            _signal_group(process.pid, signal.SIGTERM)
        pending = list(groups)
        while pending and anyio.current_time() - start < last:
            await anyio.sleep(_POLL_S)
            elapsed = anyio.current_time() - start
            for process, grace in pending:
                if elapsed > grace:
                    _signal_group(process.pid, signal.SIGKILL)
            pending = [(process, grace) for process, grace in pending if _alive(process.pid)]
        for process, _ in groups:  # the leaders are reaped; let asyncio record their exit
            with anyio.move_on_after(_KILL_WAIT_S):
                await process.wait()
    finally:  # even if the stop above is interrupted, no process survives
        for process, _ in groups:
            _signal_group(process.pid, signal.SIGKILL)


def _pid_alive(child: _Child) -> bool:
    try:
        os.kill(child.process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def _alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_group(group: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):  # the whole group is gone
        os.killpg(group, sig)
