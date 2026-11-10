"""How the coordinator reaches a worker: run a command there, copy listed files from it and to it.
Transports are an extension point (entry-point group `roboscript.dispatch_transports`; the
conformance checks are `roboscript.testing.transport`): `ssh` (an ssh alias, rsync over ssh)
and `local` (this machine, for tests and for a share of the coordinator's own machine)."""

import shlex
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import anyio

from roboscript.kernel import entrypoints

GROUP: Final = "roboscript.dispatch_transports"
SSH_OPTIONS: Final = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ServerAliveInterval=30",
)
_UNREACHABLE: Final = frozenset({255})
"""ssh's exit status when it could not run the command (rsync reports the same through ssh as
12 or 255)."""


class TransportError(RuntimeError):
    """The worker could not be reached (as opposed to a command that ran and failed)."""


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class Transport(Protocol):
    """`run` executes `argv` in `cwd` on the worker (`input` on its stdin) and raises
    `TransportError` only when the worker cannot be reached or `timeout_s` passes; `pull` copies
    `files` (relative paths) from `remote_root` to `local_root`, `push` the other way, creating
    directories as needed and raising `TransportError` when unreachable."""

    host: str

    async def run(
        self, argv: Sequence[str], *, cwd: str, input: bytes | None = None, timeout_s: float
    ) -> RunResult: ...

    async def pull(
        self, remote_root: str, local_root: Path, files: Sequence[str], *, timeout_s: float
    ) -> None: ...

    async def push(
        self, local_root: Path, remote_root: str, files: Sequence[str], *, timeout_s: float
    ) -> None: ...


def open_transport(name: str, host: str) -> Transport:
    """The transport registered as `name`, for `host`."""
    factory = entrypoints.load(GROUP, name)
    transport: Transport = factory(host)
    return transport


async def _exec(
    argv: Sequence[str], input: bytes | None, timeout_s: float, cwd: str | None = None
) -> RunResult:
    try:
        with anyio.fail_after(timeout_s):
            done = await anyio.run_process(list(argv), input=input, check=False, cwd=cwd)
    except TimeoutError as error:
        raise TransportError(f"{argv[0]} timed out after {timeout_s:g} s") from error
    return RunResult(done.returncode, done.stdout, done.stderr)


class SshTransport:
    """A worker behind an ssh alias (keys and the alias in ~/.ssh/config; never a password)."""

    def __init__(self, host: str) -> None:
        self.host = host

    async def run(
        self, argv: Sequence[str], *, cwd: str, input: bytes | None = None, timeout_s: float
    ) -> RunResult:
        remote = f"cd {shlex.quote(cwd)} && exec {shlex.join(argv)}"
        found = await _exec(["ssh", *SSH_OPTIONS, self.host, remote], input, timeout_s)
        if found.returncode in _UNREACHABLE:
            raise TransportError(
                f"ssh {self.host}: {found.stderr.decode(errors='replace').strip()}"
            )
        return found

    async def pull(
        self, remote_root: str, local_root: Path, files: Sequence[str], *, timeout_s: float
    ) -> None:
        await anyio.Path(local_root).mkdir(parents=True, exist_ok=True)
        await self._rsync(f"{self.host}:{remote_root}/", f"{local_root}/", files, timeout_s)

    async def push(
        self, local_root: Path, remote_root: str, files: Sequence[str], *, timeout_s: float
    ) -> None:
        mkdir = f"--rsync-path=mkdir -p {shlex.quote(remote_root)} && rsync"
        await self._rsync(f"{local_root}/", f"{self.host}:{remote_root}/", files, timeout_s, mkdir)

    async def _rsync(
        self, source: str, target: str, files: Sequence[str], timeout_s: float, *more: str
    ) -> None:
        if not files:
            return
        argv = ["rsync", "-rlt", "--files-from=-", "-e", shlex.join(["ssh", *SSH_OPTIONS]), *more]
        listed = ("\n".join(files) + "\n").encode()
        found = await _exec([*argv, source, target], listed, timeout_s)
        if found.returncode in (12, 255):
            raise TransportError(
                f"rsync {self.host}: {found.stderr.decode(errors='replace').strip()}"
            )
        if found.returncode != 0:
            raise RuntimeError(f"rsync {source} {target}: {found.stderr.decode(errors='replace')}")


class LocalTransport:
    """This machine: commands run as subprocesses, copies are file copies."""

    def __init__(self, host: str = "localhost") -> None:
        self.host = host

    async def run(
        self, argv: Sequence[str], *, cwd: str, input: bytes | None = None, timeout_s: float
    ) -> RunResult:
        return await _exec(argv, input, timeout_s, cwd)

    async def pull(
        self, remote_root: str, local_root: Path, files: Sequence[str], *, timeout_s: float
    ) -> None:
        await anyio.to_thread.run_sync(_copy, Path(remote_root), local_root, list(files))

    async def push(
        self, local_root: Path, remote_root: str, files: Sequence[str], *, timeout_s: float
    ) -> None:
        await anyio.to_thread.run_sync(_copy, local_root, Path(remote_root), list(files))


def _copy(source: Path, target: Path, files: list[str]) -> None:
    for name in files:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source / name, destination)
        except FileNotFoundError as error:
            raise RuntimeError(f"copy {source / name}: {error}") from error


__all__ = [
    "GROUP",
    "LocalTransport",
    "RunResult",
    "SshTransport",
    "Transport",
    "TransportError",
    "open_transport",
]
