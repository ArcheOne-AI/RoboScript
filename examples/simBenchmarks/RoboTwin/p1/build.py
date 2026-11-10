"""P1's skill builder on RoboTwin (`roboscript_methods.skill_builder.builder`: README, driver): the
driver's `Bench` for RoboTwin. The tasks are those with an exported demonstration
(`demo_export.py`'s `--recordings`; `--tasks` narrows them). The starting skill set is the core's
entries bound for RoboTwin by its binding data alone (`roboscript_robotwin.skills.bindings_only`):
no written skill. A sweep runs, per task, `cover_demos.py`, then `cover_demos.py --reuse`, then
`cover_demos.py --chain` with the written skills (`--skills`); a case run is `cover_demos.py --case`
(a chained gap's with the round's reports and the accepted skills, `--plan`, `--plan-skills`;
`--case ... --transfer`, the gate's transfer cases, on its segment's own gap). Every one of these
imports what an Agent wrote, so each runs in the sandbox (`sandboxed`: writes only a fresh directory
of its own, never a skill's or another job's; reads the repository, the simulator env, RoboTwin's
checkout and its assets, and the recordings; no credentials) on one GPU slot (`--slots-per-gpu` per
GPU, default `SLOTS_PER_GPU`, the free memory less what our jobs there have yet to take,
`Slots.pick`), bounded (`CASE_TIMEOUT_S`, `SWEEP_TASK_TIMEOUT_S`) and stopped at once where its env
server's log shows its simulator broken (`HOST_FATAL`, an infrastructure failure); the driver copies
a job's reports out of its directory. A sweep step (coverage, reuse) longer than `SHARD_S` estimated
is cut into shards by segment or gap, each a job on the slot pool (longest first), then merged into
the report the serial step writes (`cover_demos.py --only`, `--merge`). A job stopped while a
written skill's `act` ran (the last `ACT_MARK` line of its log) is that skill's failure, not
infrastructure. Gate step 1 is `plugins.validate` in the same sandbox, without network. One motion
service per GPU (RoboTwin's planner configuration, `--motion-config`) runs for the whole build
outside the sandbox (our code only); a job is given its GPU's URL (`--planner`).

Accepted skills go to `<run>/written/<name>/` with their provenance. Run directories are under
`--runs`, outside the repository (a builder session may read the repository; it must not read
transcripts, gate files or other sessions).

The per-task closure (builder README, "Per-task closure") runs after each round's acceptance: a
plan trial is `cover_demos.py --chain --trial` on one task's coverage and reuse with a task plan
(`plan_trial`), the task's official success condition the `check_success` of its RoboTwin source
(`success_condition`, read only), and every sweep's chained step reads the run's accepted task
plans (`<run>/plans/<task>.json`).

`--seed-candidates DIR...` makes skills (each a skill's directory another run's gate passed)
candidates of the next round: validated again, then its regression. A run resumes where it
stopped (`--run` again); `<run>/STOP` halts it before its next round. `--regate RUN` gates again,
with this gate (transfer included), every skill a finished run's gate passed, into `--run`'s
directory (`regate.json`, `regate.md`); it only reads RUN.

`--smoke` checks the wiring on a small scale, not a result: the clusters `--only` names, the tasks
`--tasks` names, `SMOKE` as the session's budget, one round. Only with it, `--no-sandbox` runs the
jobs without the sandbox, for debugging: the run records it (`sandbox.record_mode`, `run.json`'s
`"sandbox": "none"`, for good), its summaries are marked `UNSANDBOXED` (`sandbox.mark_summaries`)
and it is never frozen. The Codex session keeps its own permission profile either way. The
session is `--builder-config`'s (`session.SessionConfig`: `codex` with the gateway's model name,
or `fake`, no model; `--backend` overrides its backend); a codex session's gateway address and
token come from the repository's config and `.env`.

Runs in the repository's env:

    ROBOTWIN_ROOT=<RoboTwin> uv run python build.py --run <name> --runs <runs> \\
        --python <sim env>/bin/python --recordings <recordings> --gpus 0,1 \\
        --smoke --tasks <task>,<task> [--dry-run]
"""

import argparse
import ast
import asyncio
import contextlib
import heapq
import itertools
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from roboscript.stack import ServiceSpec, StackConfig, Stopped, cancel_on_signals, up
from roboscript_methods.skill_builder.builder import plugins
from roboscript_methods.skill_builder.builder.driver import (
    CASE_TIMEOUT_S,
    CEILING_RUN_S,
    OUTSIDE,
    ROUNDS,
    STRATEGIES,
    SWEEP_TASK_TIMEOUT_S,
    TASK_STRATEGIES,
    Cost,
    Driver,
    Limits,
    Outside,
    append_line,
    check_outside,
    copy_skill,
    files_hash,
)
from roboscript_methods.skill_builder.builder.plans import CONFIRMED_REPEATS, FIRST_REPEATS
from roboscript_methods.skill_builder.builder.sandbox import (
    SandboxMode,
    env_reads,
    mark_summaries,
    record_mode,
    wrap,
)
from roboscript_methods.skill_builder.demo_coverage.fitting import ACT_MARK
from roboscript_methods.skill_builder.demo_coverage.reuse import gap_records
from roboscript_methods.skill_builder.session import SessionConfig, make_session
from roboscript_robotwin.config import robotwin_root
from roboscript_robotwin.demos.replay import ReplayConfig, sim_interpreters

P1: Final = Path(__file__).resolve().parent
REPO: Final = P1.parents[3]
SWEEP_CODE: Final = (
    "packages/methods/src/roboscript_methods/skill_builder/demo_coverage",
    "packages/methods/src/roboscript_methods/skill_builder/core_entry",
    "packages/methods/src/roboscript_methods/skill_builder/builder/plugins.py",
    "packages/methods/src/roboscript_methods/skill_builder/builder/plans.py",
    "packages/methods/src/roboscript_methods/skill_builder/builder/static_check.py",
    "examples/simBenchmarks/RoboTwin/p1",
    "packages/benchmarks/robotwin/src",
    "packages/core/src",
    "packages/vision/src",
)
"""What makes a sweep's reports (`RoboTwin.sweep_code`, `Driver.stale`): the coverage code
(demo_coverage), what `cover_demos.py` imports of the builder (plugins: the skills' loading and
own gaps; plans: a task plan's data; static_check: the vocabulary), core_entry (the episode, the
criteria), the p1 glue (`cover_demos.py`, `entry_check.py`, `demo_export.py`), the RoboTwin
package (the env server, the bindings) and the core and vision packages (the core's skills,
motion, perception). Left out (`SWEEP_CODE_SKIP`): the scheduling (`build.py`: the shards' merge
writes the serial report) and the freeze."""
SWEEP_CODE_SKIP: Final = (
    "examples/simBenchmarks/RoboTwin/p1/build.py",
    "examples/simBenchmarks/RoboTwin/p1/freeze.py",
)
SLOTS_PER_GPU: Final = 3
POLL_S: Final = 5.0
"""How often a job waiting for a slot looks again."""
JOB_MIB: Final = 7000
"""What a job's simulator takes on its GPU (about a RoboTwin env server's peak): counted as
taken until `nvidia-smi` shows its processes holding it (`Slots.pick`)."""
RESERVE_MIB: Final = 2500
"""Left free beside the jobs, for a simulator that passes `JOB_MIB` while it fills and for other
users' processes on the same GPUs growing. Our jobs' own memory is held back by `Slots.pick`
until they have taken it, so the reserve stays small."""
WAIT_LOG_S: Final = 600.0
"""The first job in the queue that waits this long for a slot says so, then again every
`WAIT_LOG_S`: each GPU's free memory and what our jobs there have yet to take (`Slots.take`)."""
SWEEP_TRIES: Final = 2
"""Tries of each of a sweep's tasks: one that left a report missing is tried again at once (its
shards already out kept), not after every task's first try."""
WATCH_S: Final = 5.0
"""How often a running job's env server logs are read for `HOST_FATAL`."""
HOST_FATAL: Final = (
    "ErrorOutOfDeviceMemory",
    "ErrorDeviceLost",
    "CUDA error: out of memory",
    "CUDA out of memory",
)
"""An env server's log lines after which its simulator is broken: SAPIEN's Vulkan renderer out of
device memory or its device lost, CUDA out of memory. The job is stopped then, an infrastructure
failure (`bounded`)."""
FATAL_MARK: Final = "[build] host fatal:"
SHARD_S: Final = 900.0
"""A sweep step is cut into shards of at most this many estimated seconds (one segment or gap at
least), each a job of its own on the slot pool, so a long task's step does not hold one slot
while the others sit idle."""
TRY_S: Final = 20.0
FRAME_S: Final = 0.14
"""An explain's estimated seconds: `TRY_S` + `FRAME_S` per frame replayed to its segment's
start."""
GAP_TRIES: Final = 3
"""The tries a gap's estimate counts."""
TRIAL_TIMEOUT_S: Final = 3600.0 + (CONFIRMED_REPEATS - FIRST_REPEATS) * CEILING_RUN_S
"""A plan trial (one rollout and the ceiling's runs, more where it confirms, `plans.due`; its
demonstration's stages replayed once) longer than this is stopped: infrastructure."""
MERGE_S: Final = 600.0
"""A step's merge: a scene measured, the ceiling's runs (`cover_demos.REPEATS`, each an env
server's start)."""
CHAIN_S: Final = 800.0
CHAIN_SEGMENT_S: Final = 30.0
"""A chained step's estimated seconds: the ceiling and the demonstration's replay, then per
segment (its rollouts)."""
FIRST: Final = 1e9
"""A merge's priority: it is what a task's next step waits for."""
SMOKE: Final = Limits(case_runs=2, wall_s=1800.0, tokens=1_500_000, relays=0)
VARIANTS: Final = 2
TASK_VARIANTS: Final = 2
SESSIONS_AT_ONCE: Final = 24
"""Competing sessions per gap cluster and per task, and builder sessions at once (at most one per
GPU slot is useful: the slots, not the seats, bound the GPU memory); `--sessions-at-once`
changes it."""
TRANSFER_CASES: Final = 3
"""Other demonstrations a transfer case runs a skill on, at most (`cover_demos.py --transfer`)."""
MOTION_READY_S: Final = 300.0
"""A motion service's start (cuRobo's warm-up) longer than this fails the build."""
BACKENDS: Final = ("codex", "fake")
SIM_HOME_SEED: Final = (".cache/warp", ".cache/torch_extensions", ".cache/nvidia")
"""Caches the simulator's processes would otherwise rebuild in each job's empty HOME (Warp's and
cuRobo's compiled kernels, the driver's shader cache), copied in as a sandbox starts (`home_seed`;
those that do not exist are skipped, the originals never written)."""
MARK_S: Final = 60.0
"""How often a `--no-sandbox` run's new round summaries are marked `UNSANDBOXED` while it runs
(`mark_summaries`; again at its end)."""


def free_mib() -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}


def gpu_usage() -> dict[int, dict[int, int]]:
    """Per GPU, each process's memory by `nvidia-smi` (MiB; Kit's renderer's included), {} where
    it cannot tell."""
    try:
        uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True).stdout  # fmt: skip
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return {}
    try:
        index = {u.strip(): int(i) for i, u in (x.split(",") for x in uuids.strip().splitlines())}
        found: dict[int, dict[int, int]] = {}
        for line in apps.strip().splitlines():
            uuid, pid, mib = (x.strip() for x in line.split(","))
            if uuid in index and pid.isdigit() and mib.isdigit():
                found.setdefault(index[uuid], {})[int(pid)] = int(mib)
    except ValueError:  # an output it cannot read (a driver message): every job counts whole
        return {}
    return found


def process_parents() -> dict[int, int]:
    """Each process's parent (`/proc`), {} without it."""
    found = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text(encoding="utf-8")
        except OSError:  # it ended
            continue
        found[int(d.name)] = int(stat.rsplit(")", 1)[1].split()[1])  # the name may hold ")"
    return found


def family(roots: Iterable[int], parents: dict[int, int]) -> set[int]:
    """`roots` and every process under them."""
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    found, todo = set(), list(roots)
    while todo:
        pid = todo.pop()
        if pid not in found:
            found.add(pid)
            todo += children.get(pid, [])
    return found


def kill_tree(pid: int) -> None:
    """`pid`'s process group and every process under it, found before any is killed (a killed
    parent's children go to init): a job's env server runs in a session of its own, and
    outside the sandbox nothing else takes it down."""
    found = family([pid], process_parents())
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    for p in found - {pid}:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(p, signal.SIGKILL)


class HostLogs:
    """A job's env server logs (`<own>/hosts/*/logs/*.log`, `entry_check.host`'s), read as they
    grow for a `HOST_FATAL` line."""

    def __init__(self, own: Path):
        self.own = own
        self.read: dict[Path, int] = {}

    def fatal(self) -> str | None:
        """The first `HOST_FATAL` line written since the last look, else None."""
        overlap = max(map(len, HOST_FATAL))
        for path in sorted(self.own.glob("hosts/*/logs/*.log")):
            at = self.read.get(path, 0)
            try:
                with path.open("rb") as f:
                    f.seek(max(0, at - overlap))  # a line cut at the last look's end
                    text = f.read().decode("utf-8", errors="replace")
                    self.read[path] = f.tell()
            except OSError:
                continue
            for line in text.splitlines():
                if any(m in line for m in HOST_FATAL):
                    return f"{path.parent.parent.name}: {line.strip()[:300]}"
        return None


@dataclass
class Seat:
    """A slot taken (`Slots.take`): the processes its job started (`bounded` adds each)."""

    pids: list[int] = field(default_factory=list)


SEAT: ContextVar[Seat | None] = ContextVar("SEAT", default=None)
"""The slot the running task holds: `Slots.take` sets it for its block, and `bounded` (in a
thread: `asyncio.to_thread` copies it) adds the job it starts there."""


def cause(log: Path, at: int, code: int | None) -> str:
    """Why a job that did not exit 0 failed, from its log past byte `at` (where it began):
    `host_fatal` (its env server's simulator broke, `HOST_FATAL`), `stuck` (stopped inside a
    written skill's `act`), `timeout`, else `infra` (an exit of its own: a crash, a host that did
    not come up)."""
    text = ""
    if log.is_file():
        with log.open("rb") as f:
            f.seek(at)
            text = f.read().decode("utf-8", errors="replace")
    if FATAL_MARK in text:
        return "host_fatal"
    if code is None:
        return "stuck" if stuck(log) is not None else "timeout"
    return "infra"


def earlier_coverages(task: str, prior: Path | None, earlier: Sequence[Path]) -> dict[str, Path]:
    """A coverage job's inputs: the task's coverage reports of `prior` (`<task>.prior.json`) and
    of `earlier` (`<task>.prior-<n>.json`, newest first), whose place targets the coverage keeps
    where they are measured objects (`cover_demos.earlier_targets`, `fitting.place_target`): a
    target two supports nearly tie for otherwise flips with the measurement, and with its name
    every earlier chain's fix, task plan and written skill's case keyed by the segment's name
    stops applying."""
    found: dict[str, Path] = {}
    if prior is not None and (prior / f"{task}.json").is_file():
        found[f"{task}.prior.json"] = prior / f"{task}.json"
    for n, directory in enumerate(earlier, 1):
        if (directory / f"{task}.json").is_file():
            found[f"{task}.prior-{n}.json"] = directory / f"{task}.json"
    return found


def bounded(
    argv: list[str],
    log: Path,
    timeout_s: float,
    cwd: Path = P1,
    started: "Launch | None" = None,
    watch: Path | None = None,
) -> int | None:
    """`argv` with its output in `log`, killed with every process under it past `timeout_s`: its
    exit code, or None when it ran out of time. `started` is given the child's pid (its group's):
    what it kills when the job is cancelled, at once if it was cancelled already; the slot it
    runs in (`SEAT`) too. `watch` (the job's own directory): where its env servers' logs show a
    `HOST_FATAL` line it is killed at once (the line in `log`, after `FATAL_MARK`), a failure
    (exit -9) that is the infrastructure's, never a skill's (`stuck` reads timeouts only)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    logs = HostLogs(watch) if watch is not None else None
    deadline = time.monotonic() + timeout_s
    with log.open("ab") as out:
        child = subprocess.Popen(
            argv, stdout=out, stderr=subprocess.STDOUT, cwd=cwd, start_new_session=True
        )
        if started is not None:
            started.add(child.pid)
        if (seat := SEAT.get()) is not None:
            seat.pids.append(child.pid)
        while True:
            left = deadline - time.monotonic()
            try:
                return child.wait(timeout=max(0.0, min(left, WATCH_S) if logs else left))
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= deadline:
                kill_tree(child.pid)
                child.wait()
                return None
            if logs is not None and (line := logs.fatal()) is not None:
                out.write(f"\n{FATAL_MARK} {line}; the job is stopped\n".encode())
                out.flush()
                kill_tree(child.pid)
                return child.wait()


class Launch:
    """A job's process groups, killed when it is cancelled: one started before is killed then,
    one started after (the thread was still in `Popen`) as soon as its pid is known."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pids: list[int] = []
        self.cancelled = False

    @staticmethod
    def kill(pid: int) -> None:
        kill_tree(pid)

    def add(self, pid: int) -> None:
        with self.lock:
            self.pids.append(pid)
            if self.cancelled:
                self.kill(pid)

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            for pid in self.pids:
                self.kill(pid)


async def guarded_run(
    argv: list[str], log: Path, timeout_s: float, watch: Path | None = None
) -> int | None:
    """`bounded` in a thread; cancelled, its process group is killed first (a job the driver
    stops must not hold a GPU slot it no longer counts), also one not started yet."""
    launch = Launch()
    try:
        return await asyncio.to_thread(bounded, argv, log, timeout_s, P1, launch, watch)
    except asyncio.CancelledError:
        launch.cancel()
        raise


def merge_missing(
    path: Path, swept: list[str], mine: dict[str, dict[str, Any]], left: list[str]
) -> None:
    """A sweep's `missing.json`: its own entries (`mine`) for the tasks it swept (`swept`), and
    another sweep's into the same directory kept for the tasks still missing there (`left`):
    the driver's sweeps of one round's coverage run at once (`Driver.planned_sweep`), and each
    keeps what the others recorded (`stuck`, which blames a skill)."""
    was = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    kept = {t: x for t, x in was.items() if t not in swept and t in left}
    path.write_text(json.dumps(kept | mine, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def stuck(log: Path) -> str | None:
    """The written skill whose `act` a job's log shows running when it ended (its last
    `ACT_MARK` line opens one), else None."""
    if not log.is_file():
        return None
    marks = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines()
             if x.startswith(ACT_MARK)]  # fmt: skip
    words = marks[-1].split() if marks else []
    return words[-1] if len(words) >= 2 and words[-2] == "begin" else None


def fresh(directory: Path) -> Path:
    """`directory`, emptied: a job's own output directory."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def report(source: Path, target: Path, task: str) -> bool:
    """A job's report copied out of its directory when it is one (JSON, its task's)."""
    try:
        found = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not (isinstance(found, dict) and found.get("task") == task):
        return False
    shutil.copyfile(source, target)
    return True


def explain_s(start: int) -> float:
    return TRY_S + FRAME_S * start


def shards(costs: list[float]) -> list[list[int]]:
    """The indices of `costs`, in order, cut into runs of at most `SHARD_S` estimated seconds
    (one index at least): a serial step's order, so the merge is the serial report."""
    found: list[list[int]] = []
    total = 0.0
    for k, cost in enumerate(costs):
        if not found or total + cost > SHARD_S:
            found.append([])
            total = 0.0
        found[-1].append(k)
        total += cost
    return found


def ran(path: Path, group: list[int]) -> bool:
    """Whether `path` is a shard that ran `group` (one a retry need not run again)."""
    try:
        return bool(json.loads(path.read_text(encoding="utf-8"))["only"] == group)
    except (OSError, ValueError, KeyError, TypeError):
        return False


class Slots:
    """GPU slots: a job takes the GPU with the fewest of ours that has a slot and, by
    `nvidia-smi`, the memory: free memory less what our jobs there have yet to take."""

    def __init__(self, gpus: list[int], per_gpu: int = SLOTS_PER_GPU):
        self.per_gpu = per_gpu
        self.seats: dict[int, list[Seat]] = {g: [] for g in gpus}
        self.lock = asyncio.Lock()
        self.waiting: list[tuple[float, int]] = []
        self.seq = itertools.count()
        self.last: dict[int, tuple[int, int]] = {}
        """Per GPU, its free memory and what our jobs there had yet to take at the last look."""

    def pick(self) -> int | None:
        """A GPU where `JOB_MIB` and `RESERVE_MIB` are free once each of our jobs there holds
        `JOB_MIB` (`yet`): a fixed wait after a job's start would be shorter than its
        simulator's fill (the memory comes once its env server has built the scene, again for
        each env server a job starts), admitting jobs into memory already promised."""
        free, used = free_mib(), gpu_usage()
        parents = process_parents() if used else {}
        seats = {g: list(s) for g, s in self.seats.items()}  # `take` changes them meanwhile
        for gpu in sorted(seats, key=lambda g: len(seats[g])):
            yet = sum(self.yet(s, used.get(gpu, {}), parents) for s in seats[gpu])
            self.last[gpu] = (free[gpu], yet)
            if len(seats[gpu]) < self.per_gpu and free[gpu] - yet >= JOB_MIB + RESERVE_MIB:
                return gpu
        return None

    @staticmethod
    def yet(seat: Seat, used: dict[int, int], parents: dict[int, int]) -> int:
        """What `seat`'s job has yet to take of `JOB_MIB`: less what its processes hold on the
        GPU (`used`, by pid); all of it before it starts one, between its env servers, or
        where `nvidia-smi` or `/proc` cannot tell."""
        held = sum(used.get(p, 0) for p in family(seat.pids, parents))
        return max(0, JOB_MIB - held)

    @contextlib.asynccontextmanager
    async def take(self, priority: float = 0.0) -> AsyncIterator[int]:
        """A slot, the highest `priority` among the waiting served first (longest first)."""
        me = (-priority, next(self.seq))
        heapq.heappush(self.waiting, me)
        seat = Seat()
        since = told = time.monotonic()
        try:
            while True:
                if self.waiting[0] == me and time.monotonic() - told >= WAIT_LOG_S:
                    told = time.monotonic()  # the head of the queue: no GPU has room for it
                    gpus = ", ".join(f"{g}: {f} free, {y} yet to take, {len(self.seats[g])} jobs"
                                     for g, (f, y) in sorted(self.last.items()))  # fmt: skip
                    print(f"[slots] {len(self.waiting)} jobs wait, the first for "
                          f"{told - since:.0f} s (needs {JOB_MIB + RESERVE_MIB} MiB free beside "
                          f"ours): {gpus}", flush=True)  # fmt: skip
                if self.waiting[0] == me:
                    async with self.lock:
                        gpu = await asyncio.to_thread(self.pick)
                        if gpu is not None:
                            self.seats[gpu].append(seat)
                            break
                await asyncio.sleep(POLL_S)
        finally:
            self.waiting.remove(me)
            heapq.heapify(self.waiting)
        token = SEAT.set(seat)
        try:
            yield gpu
        finally:
            SEAT.reset(token)
            self.seats[gpu].remove(seat)


class RoboTwin:
    def __init__(self, args: argparse.Namespace, extra_reads: Sequence[Path] = ()):
        self.args = args
        self.extra_reads = list(extra_reads)
        """Read-only in every job's sandbox besides `reads`' own (`sandbox_read_roots`)."""
        self.gpus = [int(g) for g in args.gpus.split(",")]
        self.case_slots = len(self.gpus) * args.slots_per_gpu
        self.slots = Slots(self.gpus, args.slots_per_gpu)
        self.recording = asyncio.Lock()
        """Held while a sweep merges its `missing.json` (`merge_missing`): the driver runs sweeps
        into one directory at once (`Driver.planned_sweep`)."""
        self.python = Path(sys.executable)
        self.run_dir: Path = args.runs / args.run
        self.written = self.run_dir / "written"
        self.mode: SandboxMode = "none" if args.no_sandbox else "bwrap"
        self.planners: dict[int, str] = {}
        """Each GPU's motion service URL (`motion`), set once they serve."""

    def tasks(self) -> list[str]:
        """The tasks with an exported demonstration (one demonstration each), or those of
        `--tasks` among them."""
        exported = sorted(
            d.name for d in self.args.recordings.iterdir() if (d / "source.json").is_file()
        )
        return [t for t in exported if t in self.args.tasks] if self.args.tasks else exported

    # -- the sandbox ---------------------------------------------------------------------------

    def reads(self) -> list[Path]:
        """Read-only in the sandbox besides the repository: the simulator env, its editable
        installs with their git directories (`env_reads`), RoboTwin's checkout and its assets (as
        given and as resolved: the assets may be a link out of the checkout), the recordings, the
        criteria (`--criteria`) and the builder config's `sandbox_read_roots`."""
        a = self.args
        roots = [a.python.parent.parent, *env_reads(a.python), a.robotwin_root]
        roots += assets(a.robotwin_root)
        roots += [a.recordings, a.recordings.resolve(), *self.extra_reads]
        if a.criteria is not None:  # the jobs' `--criteria`
            roots += [a.criteria, a.criteria.resolve()]
        return list(dict.fromkeys(p for p in roots if p.exists()))

    def sandboxed(
        self,
        argv: list[str],
        write: list[Path],
        read: list[Path],
        gpu: int | None,
        network: bool = True,
    ) -> list[str]:
        """`argv` in the sandbox (`sandbox.wrap`, the build's mode: `none` only with
        `--no-sandbox`); `network`: the host's (a job reaches its GPU's motion service and its own
        env server on loopback)."""
        env = {"ROBOTWIN_ROOT": str(self.args.robotwin_root), "LANG": "C.UTF-8"}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return wrap(argv, write=write, read=[*read, *self.reads()], env=env,
                    home_seed=SIM_HOME_SEED, network=network, mode=self.mode)  # fmt: skip

    def cover(self, extra: list[str], gpu: int | None = None) -> list[str]:
        """`cover_demos.py` with `extra`; on `gpu`, given that GPU's motion service."""
        a = self.args
        argv = [str(self.python), str(P1 / "cover_demos.py"), "--python", str(a.python),
                "--recordings", str(a.recordings)]  # fmt: skip
        if a.criteria is not None:
            argv += ["--criteria", str(a.criteria)]
        if gpu is not None:
            argv += ["--planner", self.planners[gpu]]
        return [*argv, *extra]

    # -- sweeps --------------------------------------------------------------------------------

    @staticmethod
    def swept(out: Path, task: str) -> bool:
        return all((out / f"{task}{x}.json").is_file() for x in ("", ".reuse", ".chain"))

    def missing(self, out: Path) -> list[str]:
        return [t for t in self.tasks() if not self.swept(out, t)]

    async def segments(self, out: Path, task: str) -> list[float] | None:
        """The estimated seconds of each segment of the task's coverage (`cover_demos.py
        --segments`, no scene, no GPU), None when that failed."""
        own = await asyncio.to_thread(fresh, out / "jobs" / task / "segments")
        argv = self.sandboxed(self.cover(["--out", str(own), "--task", task, "--segments"]),
                              [own], [], None)  # fmt: skip
        log = out / "logs" / f"{task}.segments.log"
        code = await asyncio.to_thread(bounded, argv, log, CASE_TIMEOUT_S)
        path = own / f"{task}.segments.json"
        if code != 0 or not path.is_file():
            return None
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [explain_s(int(r["start"])) for r in rows]

    async def sweep(
        self,
        out: Path,
        skills: list[Path],
        tasks: list[str] | None = None,
        prior: Path | None = None,
        earlier: Sequence[Path] = (),
        plans: Path | None = None,
        reuse_before: bool = False,
    ) -> Cost:
        """Coverage + reuse of every task (or of `tasks`) missing from `out`. Each step is cut
        into shards (`shards`: by segment, by gap) run as jobs of their own on the slot pool,
        longest first, each in a directory of its own and bounded (`SWEEP_TASK_TIMEOUT_S`); a
        merge job then writes the step's report as the serial step would (the same rows, the same
        alternative chosen per gap) and runs the ceiling. A step of one shard runs whole. A task
        left missing a report is tried again at once, `SWEEP_TRIES` in all
        (shards already out are kept); those left are `missing.json`'s, with the skill a job was
        stopped in, merged with what other sweeps into `out` recorded there (`merge_missing`). A
        cancelled sweep's jobs are killed (`guarded_run`). `plans`: the run's accepted task plans
        (`<task>.json`), each its task's chained step's (`<task>.plan.json`). `reuse_before`: the
        chained steps take the coverage's measurement of a seed it ran as it is
        (`cover_demos.measured_before`)."""
        await asyncio.to_thread(out.mkdir, parents=True, exist_ok=True)
        skills = [s for s in skills if s.exists()]
        given = ["--skills", *map(str, skills)] if skills else []
        rows: list[dict[str, Any]] = []
        stopped_in: dict[str, str | None] = {}
        failed: dict[str, dict[str, Any]] = {}  # per task, its last job that did not exit 0

        async def unit(task: str, tag: str, extra: list[str], inputs: dict[str, Path],
                       priority: float) -> Path | None:  # fmt: skip
            """One job: `cover_demos.py` with `extra` in `jobs/<task>/<tag>` (`inputs` copied in
            first); that directory when it exited 0."""
            own = await asyncio.to_thread(fresh, out / "jobs" / task / tag)
            for name, source in inputs.items():
                await asyncio.to_thread(shutil.copyfile, source, own / name)
            log = out / "logs" / f"{task}.{tag}.log"
            async with self.slots.take(priority) as gpu:
                started = time.monotonic()
                argv = self.cover(["--out", str(own), "--task", task, *given, *extra], gpu)
                argv = self.sandboxed(argv, [own], skills, gpu)
                at = log.stat().st_size if log.is_file() else 0
                code = await guarded_run(argv, log, SWEEP_TASK_TIMEOUT_S, own)
            if code is None and (skill := stuck(log)) is not None:
                stopped_in[task] = skill
            if code != 0:
                failed[task] = {"step": tag, "exit": code,
                                "cause": await asyncio.to_thread(cause, log, at, code)}  # fmt: skip
            row = {"task": task, "step": tag, "gpu": gpu, "exit": code,
                   "seconds": round(time.monotonic() - started, 1)}  # fmt: skip
            rows.append(row)
            await asyncio.to_thread(append_line, out / "logs" / "_jobs.jsonl", row)
            return own if code == 0 else None

        async def step(task: str, costs: list[float], reuse: bool) -> bool:
            """One step of the task (`reuse`: its reuse, else its coverage), its report copied
            out: whole, or by shards then their merge."""
            tag, flag = ("reuse", ["--reuse"]) if reuse else ("cover", [])
            name = f"{task}.reuse.json" if reuse else f"{task}.json"
            inputs = {f"{task}.json": out / f"{task}.json"} if reuse else {}
            for x in (".reuse", ".chain"):  # what marks a refused fit a gap to write again
                if reuse and prior is not None and (prior / f"{task}{x}.json").is_file():
                    inputs[f"{task}.prior{x}.json"] = prior / f"{task}{x}.json"
            if not reuse:  # earlier coverages: a near-tie target stays the one chosen
                inputs |= earlier_coverages(task, prior, earlier)
            groups = shards(costs)
            if len(groups) <= 1:
                own = await unit(task, tag, flag, inputs, sum(costs) + MERGE_S)
                return own is not None and await asyncio.to_thread(
                    report, own / name, out / name, task
                )
            stem = f"{task}.reuse.part" if reuse else f"{task}.part"
            parts = out / "parts"
            await asyncio.to_thread(parts.mkdir, exist_ok=True)

            async def shard(k: int, group: list[int]) -> bool:
                kept = parts / f"{stem}-{k}.json"
                if await asyncio.to_thread(ran, kept, group):  # a retry keeps what ran
                    return True
                extra = [*flag, "--only", ",".join(map(str, group))]
                cost = sum(costs[i] for i in group)
                own = await unit(task, f"{tag}-{k}", extra, inputs, cost + MERGE_S)
                return own is not None and await asyncio.to_thread(
                    report, own / f"{stem}.json", kept, task
                )

            done = await asyncio.gather(*(shard(k, g) for k, g in enumerate(groups)))
            if not all(done):
                return False
            inputs |= {f"{stem}-{k}.json": parts / f"{stem}-{k}.json" for k in range(len(groups))}
            own = await unit(task, f"{tag}-merge", [*flag, "--merge"], inputs, FIRST)
            return own is not None and await asyncio.to_thread(report, own / name, out / name, task)

        async def chain(task: str, cost: float) -> bool:
            """The task's chained step (`cover_demos.py --chain`) on its coverage and reuse, from
            the chain fixes `prior` holds for it (`<task>.prior.chain.json`) or, deeper, those of
            `earlier` (`<task>.prior-<n>.chain.json`, newest first). Bounded by the job's
            bound (`--budget-s`); a job that did not end keeps its chain and the ceiling runs it
            finished (`<task>.chain.partial.json`, in `parts/`), which the next try is given and
            resumes from where its inputs are the same (`cover_demos.resume`)."""
            inputs = {f"{task}{x}.json": out / f"{task}{x}.json" for x in ("", ".reuse")}
            if prior is not None and (prior / f"{task}.chain.json").is_file():
                inputs[f"{task}.prior.chain.json"] = prior / f"{task}.chain.json"
            for n, directory in enumerate(earlier, 1):
                if (directory / f"{task}.chain.json").is_file():
                    inputs[f"{task}.prior-{n}.chain.json"] = directory / f"{task}.chain.json"
            if plans is not None and (plans / f"{task}.json").is_file():
                inputs[f"{task}.plan.json"] = plans / f"{task}.json"
            partial = f"{task}.chain.partial.json"
            kept = out / "parts" / partial
            if kept.is_file():
                inputs[partial] = kept
            budget = ["--budget-s", f"{SWEEP_TASK_TIMEOUT_S:.0f}"]
            code = ["--sweep-code", await asyncio.to_thread(self.sweep_code)]  # `earlier_tried`
            code += ["--reuse-before"] if reuse_before else []
            own = await unit(task, "chain", ["--chain", *budget, *code], inputs, cost)
            name = f"{task}.chain.json"
            if own is None:  # stopped or failed: what it finished, for the next try
                left = out / "jobs" / task / "chain" / partial
                await asyncio.to_thread(kept.parent.mkdir, exist_ok=True)
                await asyncio.to_thread(report, left, kept, task)
                return False
            return await asyncio.to_thread(report, own / name, out / name, task)

        async def job(task: str, costs: list[float] | None) -> None:
            covered = (out / f"{task}.json").is_file()
            if not covered and (costs is None or not await step(task, costs, False)):
                return
            coverage = json.loads((out / f"{task}.json").read_text(encoding="utf-8"))
            gaps = [GAP_TRIES * explain_s(int(g["frames"][0])) for g in gap_records(coverage)]
            if not (out / f"{task}.reuse.json").is_file() and not await step(task, gaps, True):
                return
            if not (out / f"{task}.chain.json").is_file():
                await chain(task, CHAIN_S + CHAIN_SEGMENT_S * len(coverage["calls"]))

        async def tried(task: str, costs: list[float] | None) -> None:
            """The task's `SWEEP_TRIES` tries, the next as soon as one leaves a report missing
            (a second pass after every task's first would leave GPUs idle meanwhile)."""
            for n in range(SWEEP_TRIES):
                if n and costs is None and not (out / f"{task}.json").is_file():
                    costs = await self.segments(out, task)
                await job(task, costs)
                if await asyncio.to_thread(self.swept, out, task):
                    return

        wanted = [t for t in self.tasks() if tasks is None or t in tasks]
        todo = [t for t in await asyncio.to_thread(self.missing, out) if t in wanted]
        costs = await asyncio.gather(*(  # every estimate first: the queue is longest first
            self.segments(out, t) if not (out / f"{t}.json").is_file() else asyncio.sleep(0)
            for t in todo
        ))  # fmt: skip
        await asyncio.gather(*(tried(t, c) for t, c in zip(todo, costs, strict=True)))
        async with self.recording:
            left = await asyncio.to_thread(self.missing, out)
            mine = {t: {"stuck": stopped_in.get(t)} | failed.get(t, {})
                    for t in left if t in wanted}  # fmt: skip
            await asyncio.to_thread(merge_missing, out / "missing.json", wanted, mine, left)
        return Cost(slot_s=sum(r["seconds"] for r in rows), jobs=len(rows))

    async def plan_trial(
        self, task: str, plan: Path, out: Path, reports: Path, skills: list[Path]
    ) -> dict[str, Any]:
        """P1-4b's plan trial (`Driver`'s per-task stage): `cover_demos.py --chain --trial` on
        the task's coverage and reuse in `reports` with the task plan `plan` and the written
        skills `skills` (the accepted, a session's), in the sandbox on one GPU slot, bounded
        (`TRIAL_TIMEOUT_S`): its chain report, copied to `out`; `infra` and `error` when the
        infrastructure broke it or it timed out outside a skill's `act` (inside: `act_error`, the
        skill's failure)."""
        own = await asyncio.to_thread(fresh, out.with_suffix(""))
        if not (reports / f"{task}.json").is_file():
            return {"task": task, "infra": True, "error": f"no coverage report in {reports}"}
        for x in ("", ".reuse"):  # a task the reuse step never ran has no reuse report
            if (reports / f"{task}{x}.json").is_file():
                await asyncio.to_thread(shutil.copyfile, reports / f"{task}{x}.json",
                                        own / f"{task}{x}.json")  # fmt: skip
        await asyncio.to_thread(shutil.copyfile, plan, own / f"{task}.plan.json")
        skills = [s for s in skills if s.exists()]
        given = ["--skills", *map(str, skills)] if skills else []
        log = out.with_suffix(".log")
        async with self.slots.take() as gpu:
            extra = ["--out", str(own), "--task", task, *given, "--chain", "--trial"]
            argv = self.cover(extra, gpu)
            argv = self.sandboxed(argv, [own], skills, gpu)
            code = await guarded_run(argv, log, TRIAL_TIMEOUT_S, own)
        found = own / f"{task}.chain.json"
        if code is None and (skill := stuck(log)) is not None:
            return {"task": task, "infra": False, "act_error": f"stopped inside {skill}'s act",
                    "rows": [], "ceiling": []}  # fmt: skip
        if code != 0 or not found.is_file():
            return {"task": task, "infra": True, "error": f"exit {code}: {log}"}
        await asyncio.to_thread(shutil.copyfile, found, out)
        return dict(json.loads(found.read_text(encoding="utf-8")))

    def success_condition(self, task: str) -> str:
        """The task's official success condition, as RoboTwin's source states it: the
        `check_success` method of `envs/<task>.py` (read, never changed), else ""."""
        path = self.args.robotwin_root / "envs" / f"{task}.py"
        if not path.is_file():
            return ""
        source = path.read_text(encoding="utf-8")
        found = [ast.get_source_segment(source, node, padded=True) or ""
                 for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef) and node.name == "check_success"]  # fmt: skip
        return "\n\n".join(textwrap.dedent(f) for f in found)

    # -- the gate ------------------------------------------------------------------------------

    async def run_case(
        self, gap: str, skill: Path | None, out: Path, extra: list[str], plan: Path | None = None
    ) -> int | None:
        """`cover_demos.py --case` on `gap` with the skill in `skill` (None: the core's entry
        alone, `--core`), in the sandbox on one GPU slot, writing only `out`'s own fresh directory
        (`out` without its suffix), never the skill's; its rows copied to `out`, its log beside
        it. A gap with a mode (`#chained`, `#stages`) runs from the plan of `plan`'s reports (the
        round's coverage) with the accepted skills (read-only). The exit code, None: timed
        out."""
        own = await asyncio.to_thread(fresh, out.with_suffix(""))
        if skill is not None and await asyncio.to_thread(
            lambda: skill.resolve().is_relative_to(own.resolve())
        ):
            raise ValueError(f"{skill} is inside the case's own directory {own}")
        given = ["--skills", str(skill)] if skill is not None else ["--core"]
        reads = [skill] if skill is not None else []
        if plugins.case_mode(gap.partition("/")[2])[1] is not None:
            if plan is None:
                raise ValueError(f"{gap}: a chained case needs the round's reports (plan)")
            accepted = self.run_dir / "accepted"
            given += ["--plan", str(plan), "--plan-skills", str(accepted)]
            reads += [plan, accepted]
        command = ["--case", gap, *given, "--out", str(own / "rows.json"), *extra]
        async with self.slots.take() as gpu:
            argv = self.sandboxed(self.cover(command, gpu), [own], reads, gpu)
            code = await asyncio.to_thread(bounded, argv, out.with_suffix(".log"),
                                           CASE_TIMEOUT_S, P1, None, own)  # fmt: skip
        if code == 0 and (own / "rows.json").is_file():
            await asyncio.to_thread(shutil.copyfile, own / "rows.json", out)
        return code

    async def case(
        self, gap: str, skill: Path, out: Path, plan: Path | None = None
    ) -> dict[str, Any]:
        if plugins.case_mode(gap.partition("/")[2])[1] is not None and plan is None:
            return {"gap": gap, "explained": False, "infra": True,
                    "error": "a chained case without the round's reports"}  # fmt: skip
        code = await self.run_case(gap, skill, out, [], plan)
        return (await asyncio.to_thread(self.case_rows, gap, skill.name, code, out))[0]

    async def transfer(
        self, gap: str, skill: Path, out: Path, demos: list[int] | None = None
    ) -> list[dict[str, Any]]:
        """A gap with a mode transfers as its segment's own gap (other demonstrations have no
        plan to chain): where its `fit` puts the motion is what tells."""
        gap = plugins.own_gap(gap)
        extra = ["--transfer", str(TRANSFER_CASES)]
        extra += ["--demos", ",".join(map(str, demos))] if demos is not None else []
        code = await self.run_case(gap, skill, out, extra)
        return await asyncio.to_thread(self.case_rows, gap, skill.name, code, out, True)

    async def core_transfer(self, gap: str, out: Path) -> list[dict[str, Any]]:
        gap = plugins.own_gap(gap)
        code = await self.run_case(gap, None, out, ["--transfer", str(TRANSFER_CASES)])
        return await asyncio.to_thread(self.case_rows, gap, "core", code, out, True)

    @staticmethod
    def case_rows(
        gap: str, skill: str, code: int | None, out: Path, transfer: bool = False
    ) -> list[dict[str, Any]]:
        """The skill's rows of `cover_demos.py --case` (one; `--transfer`: one per other
        demonstration that applies, none when none does); `infra` (not a case run: the session
        is told) when the run failed, the env broke it (`not judged`) or it timed out outside the
        skill's `act`. A timeout inside it (`stuck`) is the skill's failure."""
        log = out.with_suffix(".log")
        if code is None:
            if stuck(log) == skill:
                why = f"its act did not return; the case was stopped at {CASE_TIMEOUT_S:.0f} s"
                return [{"gap": gap, "explained": False, "infra": False, "cause": "entry fails",
                         "act_error": why}]  # fmt: skip
            why = f"the case run took more than {CASE_TIMEOUT_S:.0f} s and was stopped"
            return [{"gap": gap, "explained": False, "infra": True, "error": why}]
        if code != 0 or not out.is_file():
            return [{"gap": gap, "explained": False, "infra": True, "error": f"exit {code}: {log}"}]
        found = json.loads(out.read_text(encoding="utf-8"))
        rows = [r for r in found if r.get("skill") == skill and bool(r.get("demo")) == transfer
                and r.get("applicable", True)]  # fmt: skip
        if not rows and not transfer:
            return [{"gap": gap, "explained": False, "infra": True, "error": f"no row: {log}"}]
        return [
            dict(r) | {"gap": gap, "explained": bool(r.get("explained")),
                       "infra": r.get("cause") == "not judged" or "error" in r}
            for r in rows
        ]  # fmt: skip

    def vocab(self) -> Path:
        """`cover_demos.py --vocab`'s file (our code, from the recordings: no skill is
        imported)."""
        path = self.run_dir / "vocabulary.json"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(self.cover(["--vocab", str(path)]), check=True, cwd=P1)
        return path

    async def validate(self, skill: Path) -> dict[str, Any]:
        vocab = await asyncio.to_thread(self.vocab)
        scratch = skill.parent / f".{skill.name}.validate"
        await asyncio.to_thread(scratch.mkdir, parents=True, exist_ok=True)
        python = self.sandboxed(
            [str(self.python)], [scratch], [skill, vocab.parent], None, network=False
        )
        (row,) = await plugins.validate([skill], vocab, python)
        return {
            "ok": bool(row["ok"]),
            "name": row.get("name"),
            "kinds": row.get("kinds", []),
            "cases": row.get("cases", []),
            "findings": row.get("static", []) + row.get("contract", []),
        }

    # -- the rest --------------------------------------------------------------------------------

    def sweep_code(self) -> str:
        """A hash of the files that make a sweep's reports (`SWEEP_CODE`)."""
        return files_hash(REPO, SWEEP_CODE, SWEEP_CODE_SKIP)

    def accept(self, skill: Path, provenance: dict[str, Any]) -> Path:
        """`skill` copied to `<run>/written/<name>/` with its provenance (`provenance.yaml`), its
        version one more than an earlier different one's."""
        target = self.written / skill.name
        record = target / "provenance.yaml"
        version = 1
        if record.is_file():
            old = yaml.safe_load(record.read_text(encoding="utf-8"))
            same = old.get("sha256") == provenance["sha256"]
            version = old["version"] if same else old["version"] + 1
        copy_skill(skill, target)
        record.write_text(
            yaml.safe_dump({"version": version} | provenance, sort_keys=False), encoding="utf-8"
        )
        return target

    def describe(self, out: Path, skills: list[Path]) -> list[str]:
        given = ["--skills", *map(str, skills)]
        where = "without the sandbox" if self.args.no_sandbox else "in the sandbox"
        return [
            f"per task, {where} on GPU slots (longest first), each job bounded "
            f"({SWEEP_TASK_TIMEOUT_S:.0f} s), a step cut into shards of <= {SHARD_S:.0f} s "
            "estimated (--only K,...), then merged (--merge):",
            "  " + shlex.join(self.cover(["--out", str(out), "--task", "<task>", *given])),
            "  ... --planner <the slot's GPU's motion service>",
            "  ... --reuse",
            f"a case, {where} on one GPU slot, bounded "
            f"({CASE_TIMEOUT_S:.0f} s): CUDA_VISIBLE_DEVICES=<slot's GPU> "
            + shlex.join(self.cover(["--case", "<task>/<segment>", *given, "--out", str(out)])),
        ]


@contextlib.asynccontextmanager
async def motion(args: argparse.Namespace, gpus: Sequence[int]) -> AsyncIterator[dict[int, str]]:
    """A motion service per GPU (RoboTwin's planner configuration, `--motion-config`; logs under
    `<runs>/motion`), run in the simulator env for the whole build: their URLs by GPU."""
    services = [
        ServiceSpec(
            name=f"motion{gpu}",
            module="roboscript.motion",
            interpreter="sim",
            args=["--config", str(args.motion_config)],
            gpu=gpu,
            ready_timeout_s=MOTION_READY_S,
        )
        for gpu in gpus
    ]
    interpreters, pythonpath = sim_interpreters(REPO, args.python)
    stack = StackConfig(
        interpreters=interpreters,
        sim_pythonpath=pythonpath,
        run_dir=args.runs / "motion",
        services=services,
    )
    async with up(stack) as running:
        yield {gpu: running.endpoints[f"motion{gpu}"] for gpu in gpus}


STOP_SIGNALS: Final = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


def until_stopped(body: Callable[[], Awaitable[None]]) -> None:
    """Run `body` until it returns or one of `STOP_SIGNALS` (SIGHUP unless ignored) cancels it:
    its context managers then stop what they started (the motion services' process groups, each
    job's), so nothing outlives the build, which exits with 128 plus the signal's number."""

    async def guarded() -> None:
        with cancel_on_signals(*STOP_SIGNALS):
            await body()

    try:
        asyncio.run(guarded())
    except Stopped as stopped:
        print(f"{stopped}: the motion services and jobs are stopped", file=sys.stderr, flush=True)
        raise SystemExit(128 + stopped.signal) from None


def session_config(args: argparse.Namespace) -> SessionConfig:
    """The session's config: `--builder-config`'s (`SessionConfig`), its backend replaced by
    `--backend`; without the file, `--backend fake` alone (a codex session needs its model)."""
    found: dict[str, Any] = {}
    if args.builder_config.is_file():
        found = yaml.safe_load(args.builder_config.read_text(encoding="utf-8")) or {}
    elif args.backend != "fake":
        raise SystemExit(f"{args.builder_config}: no such file (copy configs/builder.example."
                         "yaml), or give --backend fake")  # fmt: skip
    if args.backend is not None:
        found["backend"] = args.backend
    try:
        return SessionConfig.model_validate(found)
    except ValueError as e:
        raise SystemExit(f"{args.builder_config}: {e}") from None


def assets(root: Path) -> list[Path]:
    """RoboTwin's assets, as given and as resolved (they may be a link out of the checkout)."""
    return list(dict.fromkeys([root / "assets", (root / "assets").resolve()]))


def record_run(driver: Driver, run_dir: Path, mode: SandboxMode, builder: dict[str, Any]) -> None:
    """`run.json` (the driver's, `Driver.start`) with the sandbox mode (`sandbox.record_mode`:
    once `none`, always) and how the build runs (`builder`: the session backend, `smoke`; a run
    that was a smoke run stays one)."""
    driver.start()
    record_mode(run_dir, mode)
    meta = run_dir / "run.json"
    record = json.loads(meta.read_text(encoding="utf-8"))
    was = record.get("builder", {})
    builder = builder | {"smoke": bool(was.get("smoke")) or builder["smoke"]}
    meta.write_text(json.dumps(record | {"builder": builder}, indent=1) + "\n", encoding="utf-8")


async def marking(run_dir: Path, mode: SandboxMode) -> None:
    """`mark_summaries` every `MARK_S` while the rounds run (the caller cancels it, then marks
    once more)."""
    while True:
        await asyncio.to_thread(mark_summaries, run_dir, mode)
        await asyncio.sleep(MARK_S)


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    found.add_argument("--run", required=True, help="the run's directory under --runs")
    found.add_argument("--runs", type=Path, required=True, help="outside the repository")
    found.add_argument("--python", type=Path, required=True, help="the simulator env's python")
    found.add_argument("--robotwin-root", type=Path, default=robotwin_root(),
                       help="RoboTwin's checkout (default: ROBOTWIN_ROOT, from the environment "
                       "or .env)")  # fmt: skip
    found.add_argument("--recordings", type=Path, required=True,
                       help="demo_export.py's --out (with --transfer: the transfer "
                       "demonstrations)")  # fmt: skip
    found.add_argument("--criteria", type=Path, metavar="DIR",
                       help="the entry check's measured criteria (DIR/criteria.yaml)")  # fmt: skip
    found.add_argument("--motion-config", type=Path, default=REPO / ReplayConfig().motion_config,
                       help="the motion services' config (default: RoboTwin's planner "
                       "configuration, as the recordings' replay uses)")  # fmt: skip
    found.add_argument("--gpus", required=True, help="comma-separated GPU indices")
    found.add_argument("--slots-per-gpu", type=int, default=SLOTS_PER_GPU,
                       help="jobs (simulators) at once per GPU")  # fmt: skip
    found.add_argument("--builder-config", type=Path, default=REPO / "configs" / "builder.yaml",
                       help="the session's config (configs/builder.example.yaml)")  # fmt: skip
    found.add_argument("--backend", choices=BACKENDS,
                       help="the session backend, replacing the config's")  # fmt: skip
    found.add_argument("--smoke", action="store_true", help="a wiring check (`SMOKE` budget)")
    found.add_argument("--no-sandbox", action="store_true",
                       help="--smoke only: the jobs without the sandbox (debugging)")  # fmt: skip
    found.add_argument("--only", help="--smoke: the clusters, comma-separated")
    found.add_argument("--tasks", help="--smoke: the tasks, comma-separated")
    found.add_argument("--import-baseline", type=Path, help="round 0's coverage, already swept")
    found.add_argument("--variants", type=int, default=VARIANTS,
                       choices=range(1, len(STRATEGIES) + 1),
                       help="competing sessions per gap cluster")  # fmt: skip
    found.add_argument("--task-variants", type=int, default=TASK_VARIANTS,
                       choices=range(1, len(TASK_STRATEGIES) + 1),
                       help="competing per-task sessions per task")  # fmt: skip
    found.add_argument("--sessions-at-once", type=int, default=SESSIONS_AT_ONCE,
                       help="builder sessions at once, every kind together")  # fmt: skip
    found.add_argument("--rounds", type=int, default=ROUNDS,
                       help=f"rounds at most (default {ROUNDS}; 0: no cap)")  # fmt: skip
    found.add_argument("--until-target", action="store_true",
                       help="stop only when every task is at its target (or at STOP), not "
                       "after a round in which nothing grew")  # fmt: skip
    found.add_argument("--outside", nargs="*", type=Outside.parse, default=list(OUTSIDE),
                       metavar="REASON[:NOTE]",
                       help="first-failure causes a task's sessions are paused on "
                       f"(default: {'; '.join(o.label() for o in OUTSIDE)}; "
                       "none: pause nothing)")  # fmt: skip
    found.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    found.add_argument("--regate", type=Path, metavar="RUN",
                       help="gate again what a finished run's gate passed (a run under "
                       "--runs)")  # fmt: skip
    found.add_argument("--seed-candidates", type=Path, nargs="+", default=[], metavar="DIR",
                       help="skills (a skill's directory each) entering the next round as "
                       "candidates")  # fmt: skip
    return found


def checked(args: argparse.Namespace) -> argparse.Namespace:
    """`args` validated (paths absolute: a conda python is a link) or SystemExit saying what to
    fix."""
    if args.robotwin_root is None:
        raise SystemExit("give --robotwin-root or set ROBOTWIN_ROOT")
    if args.no_sandbox and not args.smoke:
        raise SystemExit("--no-sandbox is for --smoke only: a real build runs every job sandboxed")
    if (args.only or args.tasks) and not args.smoke:
        raise SystemExit("--only and --tasks are for --smoke: a real build covers every task")
    if args.slots_per_gpu < 1:
        raise SystemExit(f"--slots-per-gpu {args.slots_per_gpu}: at least 1")
    if args.sessions_at_once < 1:
        raise SystemExit(f"--sessions-at-once {args.sessions_at_once}: at least 1")
    if args.rounds < 0:
        raise SystemExit(f"--rounds {args.rounds}: 0 (no cap) or more")
    args.robotwin_root, args.python, args.recordings, args.runs, args.motion_config = (
        p.absolute()
        for p in (args.robotwin_root, args.python, args.recordings, args.runs, args.motion_config)
    )
    args.builder_config = args.builder_config.absolute()
    args.criteria = args.criteria.absolute() if args.criteria else None
    if args.runs.is_relative_to(REPO):
        raise SystemExit(f"--runs {args.runs} is inside the repository a session may read")
    args.tasks = args.tasks.split(",") if args.tasks else None
    seeds = [p.resolve() for p in args.seed_candidates]
    if bad := [str(p) for p in seeds if not (p / "skill.py").is_file()]:
        raise SystemExit(f"--seed-candidates: not a skill's directory (no skill.py): {bad}")
    if len({p.name for p in seeds}) < len(seeds):
        raise SystemExit("--seed-candidates: two seeds of one name")
    args.seed_candidates = seeds
    try:
        check_outside(args.outside)
    except ValueError as e:
        raise SystemExit(f"--outside: {e}") from None
    return args


def main(argv: list[str] | None = None) -> None:
    args = checked(parser().parse_args(argv))
    config = session_config(args)
    bench = RoboTwin(args, extra_reads=config.sandbox_read_roots)
    runs = not (args.dry_run or args.regate)  # only a run starts sessions
    reads = [*assets(args.robotwin_root), *config.sandbox_read_roots]
    session = make_session(config, read_roots=reads) if runs else None
    driver = Driver(
        bench.run_dir,
        bench,
        session=session,
        import_baseline=args.import_baseline.resolve() if args.import_baseline else None,
        initial_skills=None,  # RoboTwin's binding data alone (`bindings_only`): no written skill
        per_task=True,
        limits=SMOKE if args.smoke else Limits(),
        only=set(args.only.split(",")) if args.only else None,
        rounds=1 if args.smoke else (args.rounds or None),
        until_target=args.until_target,
        seeds=args.seed_candidates,
        variants=args.variants,
        task_variants=args.task_variants,
        sessions_at_once=args.sessions_at_once,
        outside=args.outside,
    )
    if args.dry_run:
        print(driver.plan())
        return
    exported = [t for t in bench.tasks()
                if any((args.recordings / t / "transfer").glob("*/truth.npz"))]  # fmt: skip
    untransferred = [t for t in bench.tasks() if t not in exported]
    if untransferred:  # without them every transfer case would be silently skipped
        raise SystemExit(f"{args.recordings}: no transfer demonstrations for {untransferred}: "
                         "run demo_export.py --transfer first")  # fmt: skip
    if args.regate:
        source = args.regate if args.regate.is_absolute() else args.runs / args.regate
        if source.resolve() == bench.run_dir.resolve():
            raise SystemExit("--regate writes a run of its own: give another --run")
    else:
        record_run(driver, bench.run_dir, bench.mode,
                   {"backend": config.backend, "smoke": args.smoke})  # fmt: skip

    async def run() -> None:
        async with motion(args, bench.gpus) as found:  # before any session asks for a case
            bench.planners = found
            if args.regate:
                rows = await driver.regate(source.resolve())
                print(f"re-gated {len(rows)} skills: {bench.run_dir / 'regate.md'}")
            else:
                marks = asyncio.create_task(marking(bench.run_dir, bench.mode))
                try:
                    await driver.run_all()
                finally:
                    marks.cancel()
                    await asyncio.to_thread(mark_summaries, bench.run_dir, bench.mode)
                print(f"done: {bench.run_dir}")

    until_stopped(run)


if __name__ == "__main__":
    main()
