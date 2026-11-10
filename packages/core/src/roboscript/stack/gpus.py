"""The GPUs as the processes sharing a machine see them: which one a service goes on when its
config names none (`least_loaded`: the least loaded of the GPUs a machine lends, by `nvidia-smi`:
memory used in MiB + 100 x utilization in %, so that single services spread over the cards
instead of piling onto one); their memory (`nvidia_smi`: per GPU and per process) and the process
table that tells whose processes are whose (`ps`, `family`); the claims file every tenant of the
cards shares (`SharedClaims`), so that tenants placing work by memory
under a cap see each other's pending claims; and whether a process's logs show it ran out of GPU
memory (`LogMarks`)."""

import contextlib
import fcntl
import logging
import math
import os
import re
import subprocess
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import anyio

from roboscript.records import write_atomic

log = logging.getLogger(__name__)

GPUS: Final = (
    "nvidia-smi",
    "--query-gpu=index,uuid,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
APPS: Final = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,used_memory",
    "--format=csv,noheader,nounits",
)
"""The processes using each GPU; a simulator (Vulkan and CUDA, `C+G`) is listed too."""
PS: Final = ("ps", "-A", "-o", "pid=,ppid=,pgid=")
LOCK_WARN_S: Final = 30.0
"""A wait for a lock this long is logged: someone holds it long (a slow disk under its file)."""
CLAIM_TTL_S: Final = 180
"""How long a claim counts in the shared claims file: the time a process has to
show its memory in `nvidia-smi`."""
CLAIMS_MINE: Final = "gpu_claims.mine"
"""A run's own claims in the shared claims file, by line, in its run directory (`SharedClaims`)."""
OOM: Final = re.compile(
    rb"out of memory|cannot create buffer|OutOfMemoryError|ErrorOutOfDeviceMemory|ErrorDeviceLost",
    re.IGNORECASE,
)
"""What a process that ran out of GPU memory logs: CUDA's (torch's, cuRobo's), Vulkan's (a render
that could not allocate ends with the device lost) and a simulator's (`cannot create buffer`)."""
NOT_OOM: Final = re.compile(rb"^.*OIDN Error: out of memory.*$", re.MULTILINE | re.IGNORECASE)
"""What does not tell a process ran out of GPU memory: a renderer's denoiser (OIDN) logs it as an
env server starts, whether or not the server comes up."""
OOM_TAIL: Final = 4 << 20
"""How much of what a log gained at most is searched for `OOM`: the end, where a crash is."""
HOLD_TTL_S: Final = 60
"""How long a line `SharedClaims.publish` writes counts: it is stamped `CLAIM_TTL_S - HOLD_TTL_S`
back, so that the holds of a pipeline that stopped publishing go within this, not three minutes."""


@dataclass(frozen=True)
class GpuMemory:
    """A GPU's memory in MiB: in use (by any process), in all, and by process id."""

    used_mib: float
    total_mib: float
    processes: Mapping[int, float] = field(default_factory=dict)

    @property
    def free_mib(self) -> float:
        return self.total_mib - self.used_mib


GpuProbe = Callable[[], Awaitable[Mapping[int, GpuMemory]]]
"""Every GPU's memory now, by index; RuntimeError when it cannot be read."""
ProcessTable = Callable[[], Awaitable[Mapping[int, tuple[int, int]]]]
"""Every process now: its parent's pid and its process group, by pid."""


def parse(gpus: str, apps: str = "") -> dict[int, GpuMemory]:
    """The GPUs an `nvidia-smi` report lists (`GPUS`), with their processes' memory (`APPS`); a
    line it cannot read (a `[N/A]`) leaves its GPU or process out."""
    processes: dict[str, dict[int, float]] = {}
    for line in apps.splitlines():
        try:
            uuid, pid, used = (part.strip() for part in line.split(","))
            by_pid = processes.setdefault(uuid, {})
            by_pid[int(pid)] = by_pid.get(int(pid), 0.0) + float(used)
        except ValueError:
            continue
    memory = {}
    for line in gpus.splitlines():
        try:
            index, uuid, used, total = (part.strip() for part in line.split(","))
            found = GpuMemory(float(used), float(total), processes.get(uuid, {}))
        except ValueError:
            continue
        memory[int(index)] = found
    return memory


async def nvidia_smi() -> dict[int, GpuMemory]:
    """Every GPU's memory now, by `nvidia-smi`."""
    try:
        gpus = (await anyio.run_process(GPUS)).stdout.decode()
        apps = (await anyio.run_process(APPS)).stdout.decode()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"nvidia-smi could not read the GPUs' memory: {error}") from error
    return parse(gpus, apps)


async def ps() -> dict[int, tuple[int, int]]:
    """Every process's parent and process group, by `ps`."""
    try:
        report = (await anyio.run_process(PS)).stdout.decode()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"ps could not list the processes: {error}") from error
    table = {}
    for line in report.splitlines():
        with contextlib.suppress(ValueError):  # a process that ended as ps listed it
            pid, ppid, pgid = (int(part) for part in line.split())
            table[pid] = (ppid, pgid)
    return table


def family(table: Mapping[int, tuple[int, int]], roots: Iterable[int]) -> set[int]:
    """The processes of `roots` (pids or process groups) and all their descendants."""
    roots = set(roots)
    found = {pid for pid, (_, pgid) in table.items() if pid in roots or pgid in roots}
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        children.setdefault(ppid, []).append(pid)
    todo = list(found)
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in found:
                found.add(child)
                todo.append(child)
    return found


@contextlib.asynccontextmanager
async def flocked(path: Path) -> AsyncIterator[None]:
    """`path` locked (`flock`, exclusive) for the body. The wait blocks in the kernel, in a worker
    thread, as the other tenants' `flock` does: a waiter that polls the lock loses it to them
    every time it changes hands.
    A waiter cancelled meanwhile lets go of the lock as soon as its thread gets it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    guard, held, left = threading.Lock(), threading.Event(), threading.Event()

    def take() -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        with guard:
            if left.is_set():  # its waiter was cancelled: let go at once
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            else:
                held.set()

    started = time.monotonic()
    try:
        await anyio.to_thread.run_sync(take, abandon_on_cancel=True)
    except BaseException:
        with guard:
            left.set()
            if held.is_set():  # taken just as the wait was cancelled
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        raise
    if (waited := time.monotonic() - started) > LOCK_WARN_S:
        log.warning("waited %.0f s for the lock %s", waited, path)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class SharedClaims:
    """The claims file the tenants of the cards share: in `directory`
    (best on a tmpfs, e.g. /dev/shm: a slow disk keeps every tenant waiting for the lock),
    `lock` (flock, waited for fairly: `flocked`) serialises every choice of a GPU, and `claims`
    holds a line `<epoch> <gpu> <GiB>` per claim, which counts for `CLAIM_TTL_S` (whole GiB; a
    line may carry more fields, which are not read). Another tenant's writer may keep three fields
    when it rewrites the file, so ours are told apart
    by `mine`, a file of our own that keeps each of our claims' line (across launches: a launch
    again counts those of the one before as ours, not as others'). Within `hold`, `others` are
    the claims that count but ours, `add` writes one of ours, and `publish` replaces all of ours
    with what we hold now, a line per GPU that counts for `HOLD_TTL_S`."""

    def __init__(self, directory: Path, mine: Path) -> None:
        self.directory, self.mine = directory, mine
        self._unreadable = False

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator["SharedClaims"]:
        async with flocked(self.directory / "lock"):
            yield self

    def others(self) -> dict[int, float]:
        """GiB claimed per GPU by others, their claims that still count (under `hold`)."""
        now = time.time()
        mine = Counter(self._live(self.mine, now))
        claimed: dict[int, float] = {}
        for at, gpu, gib in self._live(self.directory / "claims", now):
            if mine[at, gpu, gib] > 0:
                mine[at, gpu, gib] -= 1
                continue
            claimed[gpu] = claimed.get(gpu, 0.0) + gib
        return claimed

    def add(self, gpu: int, mib: float) -> None:
        """Claim `mib` on `gpu` (under `hold`), rounded up to whole GiB."""
        claim = (int(time.time()), gpu, math.ceil(mib / 1024))
        line = " ".join(map(str, claim)) + "\n"
        with (self.directory / "claims").open("a", encoding="utf-8") as file:
            file.write(line)
        kept = [" ".join(map(str, c)) + "\n" for c in self._live(self.mine, claim[0])]
        write_atomic(self.mine, "".join([*kept, line]).encode())  # the expired ones go

    def publish(self, holds: Mapping[int, float]) -> None:
        """Replace our claims (under `hold`) with `holds`, MiB per GPU that nvidia-smi does not
        show as ours yet, each rounded up to whole GiB; the other lines stay as they are."""
        now = int(time.time())
        stamp = now - (CLAIM_TTL_S - HOLD_TTL_S)
        fresh = [
            f"{stamp} {gpu} {math.ceil(mib / 1024)}\n" for gpu, mib in holds.items() if mib > 0
        ]
        path = self.directory / "claims"
        ours = Counter(self._live(self.mine, now))
        kept = []
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            fields = line.split()[:3]
            claim = tuple(int(f) for f in fields) if all(f.isdigit() for f in fields) else ()
            if len(claim) == 3 and ours[claim] > 0:
                ours[claim] -= 1
                continue
            kept.append(line + "\n")
        mode = path.stat().st_mode if path.is_file() else None
        write_atomic(path, "".join([*kept, *fresh]).encode())
        if mode is not None:
            os.chmod(path, mode)  # the other tenants' access, as it was
        write_atomic(self.mine, "".join(fresh).encode())

    def _live(self, path: Path, now: float) -> list[tuple[int, int, int]]:
        """The claims in `path` that still count."""
        found = []
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            try:
                at, gpu, gib = (int(part) for part in line.split()[:3])
            except ValueError:
                if not self._unreadable:
                    log.warning("%s: skipping a line that is not '<epoch> <gpu> <GiB>'", path)
                    self._unreadable = True
                continue
            if now - at < CLAIM_TTL_S:
                found.append((at, gpu, gib))
        return found


class LogMarks:
    """The logs under a directory (`*.log`, at any depth; none under `apart`, another job's) as
    they were, to tell whether what they gained since shows a process that ran out of GPU memory
    (`OOM`)."""

    def __init__(self, directory: Path, apart: Sequence[Path] = ()) -> None:
        self.directory, self.apart = directory, tuple(apart)
        self.sizes = self._sizes()

    def out_of_memory(self) -> bool:
        for path, size in self._sizes().items():
            start = self.sizes.get(path, 0)
            if size <= start:
                continue
            with contextlib.suppress(FileNotFoundError), path.open("rb") as file:
                file.seek(max(start, size - OOM_TAIL))
                if OOM.search(NOT_OOM.sub(b"", file.read(size - start))):
                    return True
        return False

    def _sizes(self) -> dict[Path, int]:
        sizes = {}
        for path in self.directory.rglob("*.log"):
            if any(path.is_relative_to(other) for other in self.apart):
                continue
            with contextlib.suppress(FileNotFoundError):  # set aside as we looked
                sizes[path] = path.stat().st_size
        return sizes


QUERY: Final = (
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
)


def rank(report: str, lent: Sequence[int]) -> list[int]:
    """The `lent` GPUs an `nvidia-smi` report (`QUERY`) lists, least loaded first (ties: the
    lower index); a line it cannot read (a `[N/A]`) leaves its GPU out."""
    loads = []
    for line in report.splitlines():
        try:
            index, memory, utilization = (float(part) for part in line.split(","))
        except ValueError:
            continue
        if int(index) in lent:
            loads.append((memory + 100 * utilization, int(index)))
    return [index for _, index in sorted(loads)]


async def least_loaded(lent: Sequence[int]) -> int:
    """The least loaded of the `lent` GPUs now (`rank`); RuntimeError when `nvidia-smi` does not
    run or lists none of them (give the GPU in the config then)."""
    try:
        report = (await anyio.run_process(QUERY)).stdout.decode()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"nvidia-smi could not rank GPUs {list(lent)}: {error}") from error
    ranked = rank(report, lent)
    if not ranked:
        detail = f"nvidia-smi lists none of GPUs {list(lent)}: give the GPU in the config"
        raise RuntimeError(detail)
    return ranked[0]
