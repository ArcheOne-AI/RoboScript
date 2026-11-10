"""A task's `record.json` (`RecordedTask`): how its demonstration was recorded, candidate after
candidate. Light: the simulator process that records writes it
(`roboscript_robotwin.sim.record`)."""

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from roboscript.contracts.ids import TaskName

RECORD: Final = "record.json"
Outcome = Literal[
    "success", "no success latch", "expert failed", "pre-run failed", "validation failed"
]


class Attempt(BaseModel):
    """One try at a candidate (the expert data's episode and seed): a replay strategy, or None
    for the expert re-run that should give its trajectory, and how it ended."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode: NonNegativeInt
    seed: NonNegativeInt
    strategy: str | None
    outcome: Outcome
    frames: NonNegativeInt | None = None
    seconds: float


class RecordedTask(BaseModel):
    """`record.json`: the candidate recorded now (`recorded` false: this one failed to record),
    every attempt so far and the candidates `rejected` (not recorded, or failed validation)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    category: str | None
    recorded: bool
    episode: NonNegativeInt
    seed: NonNegativeInt
    source: Literal["expert_data", "expert_rerun"]
    instruction: str
    strategy: str | None = None
    frames: NonNegativeInt | None = None
    success_frame: NonNegativeInt | None = None
    attempts: list[Attempt]
    notes: list[str]
    rejected: list[NonNegativeInt] = []


def read(out: Path) -> RecordedTask | None:
    """A task's `record.json`, once one was written."""
    path = out / RECORD
    return (
        RecordedTask.model_validate_json(path.read_text(encoding="utf-8"))
        if path.is_file()
        else None
    )


def write(out: Path, recorded: RecordedTask) -> RecordedTask:
    out.mkdir(parents=True, exist_ok=True)
    (out / RECORD).write_text(recorded.model_dump_json(indent=1) + "\n", "utf-8")
    return recorded
