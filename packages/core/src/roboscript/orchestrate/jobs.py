"""A run's jobs and their records (docs/concepts/operations.md, `orchestrate`): a job's
key (`RunKey`: a task's unit, e.g. its S1 or one of its variants, and a lane, a job of its own
beside the unit's) and its state file (`RunState`, `STATE` in the job's directory), the process
groups it started (`PidRecord`), and whether a launch takes it up (`pending`). `stage` names the
method's stage the job is at (a string the method chooses)."""

import json
import time
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt

from roboscript.contracts.ids import TaskName
from roboscript.contracts.jobs import JobRequest
from roboscript.records import write_atomic

STATE: Final = "pipeline.json"
PIDS: Final = ".roboscript/pids.json"
HEARTBEAT: Final = ".roboscript/heartbeat"
"""The worker's heartbeat, every tenth of the launch's watchdog."""
LAUNCHES: Final = "launches.jsonl"
"""A run's launches, a line each (in the run's directory)."""
Status = Literal[
    "queued",
    "running",
    "done",
    "not_passed",
    "not_converged",
    "blocked",
    "failed",
    "stopped",
    "preempted",
]
Failure = Literal["infra", "error"]


class RunKey(NamedTuple):
    """A job: a task's unit (`variant`: its shared S1 or one of its variants in the pipeline),
    with a `lane`, a job of its own beside the unit's (the pipeline's spare lane runs a variant's
    P3 blocks in another mode of the benchmark), whose records lie apart in the unit's directory."""

    task: TaskName
    variant: str
    lane: str | None = None

    @property
    def name(self) -> str:
        return f"{self.task}/{self.variant}" + (f"/{self.lane}" if self.lane else "")

    @property
    def main(self) -> "RunKey":
        """The variant's own job."""
        return RunKey(self.task, self.variant)

    @classmethod
    def of(cls, request: JobRequest) -> "RunKey":
        """The job a worker's request names."""
        return cls(TaskName(request.task), request.unit, request.lane)

    def unit_dir(self, directory: Path) -> Path:
        """The unit's directory in a run's `directory` (`<task>/<unit>`), its lane's too."""
        return directory / self.task / self.variant


class RunState(BaseModel):
    """`pipeline.json`: the job's status, the stage it is at, its GPU in its latest launch, its
    launches and respawns, when this launch started it (Unix time; None when not running), its
    running time in earlier launches, how it failed, and what it was started with."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task: TaskName
    variant: str
    status: Status = "queued"
    stage: str | None = None
    gpu: NonNegativeInt | None = None
    launches: NonNegativeInt = 0
    respawns: NonNegativeInt = 0
    started: float | None = None
    elapsed_s: NonNegativeFloat = 0.0
    failure: Failure | None = None
    error: str | None = None
    definition: dict[str, Any] | None = None

    @classmethod
    def load(cls, directory: Path, key: RunKey) -> "RunState":
        path = directory / STATE
        if not path.is_file():
            return cls(task=key.task, variant=key.variant)
        return cls.model_validate_json(path.read_bytes())

    def save(self, directory: Path) -> None:
        write_atomic(directory / STATE, (self.model_dump_json(indent=2) + "\n").encode())

    def running_s(self, now: float) -> float:
        return self.elapsed_s + (0.0 if self.started is None else now - self.started)

    def end(self, directory: Path, status: Status, error: str | None = None) -> None:
        self.elapsed_s, self.started = self.running_s(time.time()), None
        self.status, self.error = status, error
        self.save(directory)


class PidRecord:
    """The process groups a stack started, by service, in a file (`up(on_start=...)`), for as
    long as they may run: removed once the stack has stopped them."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.pids: dict[str, int] = {}

    def add(self, name: str, pid: int) -> None:
        self.pids[name] = pid
        write_atomic(self.path, (json.dumps(self.pids, indent=1) + "\n").encode())

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def pending(state: RunState, retry_failed: bool) -> bool:
    """Whether a launch takes the job up: queued, stopped, preempted, blocked, left running, or
    failed on infrastructure; failed otherwise only with `retry_failed`."""
    if state.status == "failed":
        return retry_failed or state.failure == "infra"
    return state.status in ("queued", "running", "stopped", "blocked", "preempted")
