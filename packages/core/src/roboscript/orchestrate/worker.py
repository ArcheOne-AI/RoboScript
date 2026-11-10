"""One job of a launch in a process of its own (docs/concepts/operations.md), so that a
crash, a kill for memory or a blocked call ends only that job: the launch runs `<worker> <directory>
<phase> <task> <unit> <gpu> <services JSON> [--modes <modes>] [--lane <lane>]` (`command`), whose
script calls `main` with the job bodies by phase (`JobKind`: the method's `Method.kinds` and the
evaluation's `EvaluationFactory.kinds`, `job_kinds`; a launch checks them against its stages
first, `check_kinds`) and the pipeline that runs them (`WorkerHost`,
made from the request; each kind's package declares what else it needs of it). The job records how
it ends in its state (`jobs.STATE`, `run_job`); the exit status is 0 whenever it did."""

import argparse
import asyncio
import contextlib
import inspect
import json
import logging
import signal
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

import anyio

from roboscript.contracts.jobs import JobKind, JobRequest
from roboscript.orchestrate.graph import one_line
from roboscript.orchestrate.jobs import HEARTBEAT, Failure, RunKey, RunState
from roboscript.records import write_atomic
from roboscript.settings import require_source_sha
from roboscript.stack import on_signals

log = logging.getLogger(__name__)


class WorkerHost(Protocol):
    """What the worker needs of the pipeline that runs a job: how often the job beats its
    heartbeat (`heartbeat_s`), a job's records (`job_dir`), what a job is made of as it first
    starts (`definition`, which a later launch must not change) and whether an error is the
    infrastructure's (`failure`: `infra`, respawned) or not (`error`)."""

    heartbeat_s: float

    def job_dir(self, key: RunKey) -> Path: ...

    def definition(self, key: RunKey) -> dict[str, Any]: ...

    def failure(self, error: BaseException) -> Failure: ...


def command(
    worker: Sequence[str],
    directory: Path,
    phase: str,
    key: RunKey,
    gpu: int,
    services: Mapping[str, str],
    modes: Sequence[str] | None = None,
) -> list[str]:
    """The command line that runs a job (its kind's default modes unless `modes`; a lane's,
    `RunKey.lane`), `worker` being the script's command."""
    where = [str(directory), phase, key.task, key.variant, str(gpu), json.dumps(services)]
    lane = ["--lane", key.lane] if key.lane is not None else []
    return [*worker, *where, *(["--modes", ",".join(modes)] if modes else []), *lane]


def job_kinds(*kinds: Mapping[str, JobKind]) -> dict[str, JobKind]:
    """The job bodies of a launch by phase, the method's then the evaluation's: a phase two give
    is an error."""
    found: dict[str, JobKind] = {}
    for given in kinds:
        if twice := sorted(set(found) & set(given)):
            raise ValueError(f"job kinds given twice: {', '.join(twice)}")
        found |= given
    return found


def check_kinds(kinds: Mapping[str, JobKind], phases: Sequence[str]) -> list[str]:
    """What is wrong with the job bodies `kinds` of `phases` (a method's or an evaluation's), one
    line each (empty: a worker runs each phase): a phase without a body, a body of no phase, a body
    without a stage or whose `run` is not a coroutine function."""
    lines = [f"phase {phase!r} has no job kind" for phase in phases if phase not in kinds]
    lines += [f"job kind {name!r} runs no phase" for name in kinds if name not in phases]
    for name, kind in kinds.items():
        if not isinstance(getattr(kind, "stage", None), str) or not kind.stage:
            lines.append(f"job kind {name!r}: no stage")
        if not inspect.iscoroutinefunction(getattr(kind, "run", None)):
            lines.append(f"job kind {name!r}: run is not async")
    return lines


def main(
    kinds: Mapping[str, JobKind],
    lanes: Sequence[str],
    host: Callable[[JobRequest], WorkerHost],
    argv: Sequence[str],
    *,
    prog: str,
    description: str | None = None,
) -> int:
    """Run the job `argv` names (`command`) by its phase's kind in `kinds`, a lane among `lanes`,
    on the pipeline `host` makes of the request (before anything runs: e.g. what its processes
    inherit)."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("directory", type=Path)
    parser.add_argument("phase", choices=list(kinds))
    parser.add_argument("task")
    parser.add_argument("variant")
    parser.add_argument("gpu", type=int)
    parser.add_argument("services", type=json.loads)
    parser.add_argument(
        "--modes",
        type=lambda text: text.split(","),
        help="evaluation modes to run, comma-separated",
    )
    parser.add_argument("--lane", choices=sorted(lanes), help="a unit's lane")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    request = JobRequest(
        directory=args.directory,
        phase=args.phase,
        task=args.task,
        unit=args.variant,
        lane=args.lane,
        gpu=args.gpu,
        services=args.services,
        modes=args.modes,
    )
    running = host(request)
    anyio.run(run_job, running, request, kinds[args.phase])
    return 0


async def run_job(host: WorkerHost, request: JobRequest, kind: JobKind) -> RunState:
    """Run (or continue) one job by its `kind` and record how it ends, beating its heartbeat
    (`HEARTBEAT`) meanwhile; SIGTERM, or SIGHUP unless ignored, stops it (`stopped`). Its records
    are in `host.job_dir` (a lane's apart)."""
    key = RunKey.of(request)
    run_dir = host.job_dir(key)
    state = RunState.load(run_dir, key)
    state.status, state.stage, state.gpu = "running", kind.stage, request.gpu
    state.failure = state.error = None
    state.launches += 1
    state.started = time.time()
    if state.definition is None:
        state.definition = host.definition(key)
    state.save(run_dir)
    received: list[int] = []
    beating = asyncio.get_running_loop().create_task(  # not a task group: they wrap errors
        _heartbeat(run_dir / HEARTBEAT, host.heartbeat_s)
    )
    try:
        with _signalled() as received:
            require_source_sha()  # its results name their code: none, it fails
            ended = await kind.run(host, request, state.definition)
    except Exception as error:
        if received:  # stopped, P3's launcher first (`Stopped`) or not
            state.end(run_dir, "stopped")
            return state
        log.exception("%s failed", key.name)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        state.failure = host.failure(error)
        state.end(run_dir, "failed", one_line(error))
        return state
    finally:
        beating.cancel()
    if received:
        state.end(run_dir, "stopped")
    else:
        state.stage = ended.stage or state.stage
        state.end(run_dir, ended.status, ended.note)  # type: ignore[arg-type]  # a `Status`
    return state


async def _heartbeat(path: Path, interval: float) -> None:
    """The time, every `interval` seconds: a worker whose loop stops beating is killed."""
    while True:
        write_atomic(path, f"{time.time():.3f}\n".encode())
        await anyio.sleep(interval)


@contextlib.contextmanager
def _signalled() -> Iterator[list[int]]:
    """SIGTERM, and SIGHUP unless ignored (`nohup`), cancel the body; the list holds the signals
    that came (`roboscript.kernel.signals`)."""
    loop = asyncio.get_running_loop()
    received: list[int] = []
    with anyio.CancelScope() as scope:

        def stop(sig: int, frame: FrameType | None) -> None:
            received.append(sig)
            loop.call_soon_threadsafe(scope.cancel)  # handlers run between bytecodes: defer

        with on_signals(stop, (signal.SIGTERM, signal.SIGHUP)):
            yield received
