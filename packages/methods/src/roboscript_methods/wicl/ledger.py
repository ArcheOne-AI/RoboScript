"""The S2 ledger (docs/concepts/learning.md): `ledger.jsonl`, one row per seed, written when the
seed is done, so a practice resumes after the last row. A counted seed's try 1 is written as soon
as it ends (`FirstTry`, `<seed>/first.json`), so a resumed practice continues that seed from its
recorded try 1 instead of running it again: a resumed seed cannot change its counted first try.
Caches, diffs, Reviser output and merge details lie beside it, under the seed's directory. Its
tries are the core's `TryRecord`s (`learning.ledger`)."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from roboscript.learning.episode import EndClass, EpisodeOutcome
from roboscript.learning.ledger import TryRecord, wilson
from roboscript.records import write_atomic
from roboscript_methods.wicl.extension import UNCONFIRMED
from roboscript_methods.wicl.memory.schedule import ScheduleStats

Outcome = Literal["pass", "merge", "jitter", "skip", "unsolvable"]
"""A seed's outcome: its first try passed; a cache made a later try pass and was merged; a
later try passed with the cache equal to M (a jitter: nothing merged); all three failed and the
cache was discarded (a skip); the env's setup check failed (not counted)."""


class RevisionRecord(BaseModel):
    """One Reviser run on a seed's cache: how it ended, its cost, and its edits (accepted,
    rejected); `directory` is relative to the task's directory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision: int
    end: str
    end_class: EndClass
    model_calls: NonNegativeInt
    tokens: NonNegativeInt
    wall_clock_s: float
    infra_reruns: NonNegativeInt
    accepted: NonNegativeInt
    rejected: NonNegativeInt
    directory: str

    @classmethod
    def of(
        cls,
        revision: int,
        outcome: EpisodeOutcome,
        reruns: int,
        edits: tuple[int, int],
        root: Path,
    ) -> "RevisionRecord":
        return cls(
            revision=revision,
            end=outcome.end.reason,
            end_class=outcome.end_class,
            model_calls=outcome.model_calls,
            tokens=outcome.tokens,
            wall_clock_s=outcome.wall_clock_s,
            infra_reruns=reruns,
            accepted=edits[0],
            rejected=edits[1],
            directory=outcome.directory.relative_to(root).as_posix(),
        )


class LedgerRow(BaseModel):
    """A seed: `k` counts solvable seeds (None: not counted); `memory_before` and
    `memory_after` are the Memory's version numbers around it; `budget_used` is the cache's;
    `revisions` are the Reviser's runs; `demo_context` is what its Actor read of the
    demonstration (`roles.DemoContext`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: NonNegativeInt
    k: int | None
    solvable: bool
    instruction: str | None
    tries: list[TryRecord]
    revisions: list[RevisionRecord] = []
    first_try_pass: bool | None
    streak_after: NonNegativeInt
    alpha_k: float | None = None
    budget_k: int | None = None
    budget_used: int | None = None
    outcome: Outcome
    memory_before: int
    memory_after: int
    failure_labels: list[str] = []
    note: str = ""
    demo_context: str = "none"


class FirstTry(BaseModel):
    """A counted seed's try 1 as it ended, with what its row needs and the try would change if
    run again: the seed's count `k`, its schedule (`alpha_k`, `budget_k`), the instruction every
    later try reuses, and the Memory version try 1 read. Written before anything else of the seed
    runs; the row, once written, supersedes it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: NonNegativeInt
    k: int
    instruction: str | None
    alpha_k: float
    budget_k: int
    memory_before: int
    first: TryRecord
    reopen_failures: NonNegativeInt = 0
    """Resumes in a row whose reopen of the seed (the env's setup check) failed."""


FIRST: Final = "first.json"
"""A seed's `FirstTry`, in its directory beside `try1/`."""


class Ledger:
    """`ledger.jsonl` of a task's S2, and the try 1 records of its seeds (`FirstTry`)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def first_path(self, seed: int) -> Path:
        return self.path.parent / str(seed) / FIRST

    def first(self, seed: int) -> FirstTry | None:
        """Seed `seed`'s recorded try 1, if it has one."""
        path = self.first_path(seed)
        return FirstTry.model_validate_json(path.read_bytes()) if path.is_file() else None

    def write_first(self, record: FirstTry) -> None:
        write_atomic(self.first_path(record.seed), (record.model_dump_json() + "\n").encode())

    def rows(self) -> list[LedgerRow]:
        """The complete rows: a reader may find the last one half appended while S2 runs."""
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").split("\n")[:-1]
        return [LedgerRow.model_validate_json(line) for line in lines if line.strip()]

    def repair(self) -> None:
        """Cut a row half appended when S2 stopped mid-write, so that the next row starts a
        line of its own."""
        if not self.path.is_file():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            write_atomic(self.path, data[: data.rfind(b"\n") + 1])

    def append(self, row: LedgerRow) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(row.model_dump_json() + "\n")


class Progress(BaseModel):
    """Where a practice stands after some rows: the counted seeds `k`, their first tries in
    order (`firsts`, passed or not: the early stop's window), the Memory version each read
    (`versions`) and whether each ended as an unconfirmed finish (`unconfirmed`), the first-try
    passes in a row, the schedule's statistics and the next seed."""

    model_config = ConfigDict(frozen=True)

    k: int = 0
    firsts: tuple[bool, ...] = ()
    versions: tuple[int, ...] = ()
    unconfirmed: tuple[bool, ...] = ()
    streak: int = 0
    stats: ScheduleStats = ScheduleStats()
    next_seed: int
    skipped_in_a_row: int = 0

    @classmethod
    def of(cls, rows: Sequence[LedgerRow], first_seed: int) -> "Progress":
        progress = cls(next_seed=first_seed)
        for row in rows:
            progress = progress.after(row)
        return progress

    def after(self, row: LedgerRow) -> "Progress":
        counts = self.stats.model_dump()
        field = {"pass": "passed", "merge": "merged", "jitter": "jitter", "skip": "discarded"}
        if row.outcome in field:
            counts[field[row.outcome]] += 1
        counted = row.k is not None
        first = row.tries[0] if row.tries else None
        unconfirmed = first is not None and first.end == UNCONFIRMED
        return Progress(
            k=self.k if row.k is None else row.k,
            firsts=(*self.firsts, bool(row.first_try_pass)) if counted else self.firsts,
            versions=(*self.versions, row.memory_before) if counted else self.versions,
            unconfirmed=(*self.unconfirmed, unconfirmed) if counted else self.unconfirmed,
            streak=row.streak_after,
            stats=ScheduleStats(**counts),
            next_seed=row.seed + 1,
            skipped_in_a_row=self.skipped_in_a_row + 1 if row.outcome == "unsolvable" else 0,
        )


def same_version_run(progress: Progress, version: int) -> tuple[int, int, bool]:
    """The counted seeds at the end whose first try read Memory `version` (the current one):
    how many, how many of their first tries passed, and whether any ended as an unconfirmed
    finish."""
    n = 0
    for read in reversed(progress.versions):
        if read != version:
            break
        n += 1
    if n == 0:
        return 0, 0, False
    return n, sum(progress.firsts[-n:]), any(progress.unconfirmed[-n:])


def failure_labels(outcomes: Sequence[EpisodeOutcome]) -> list[str]:
    """First labels of the failed tries, from how they ended, in words (the Reviser would read codes
    as stage names): a hard failure (`unreachable`, `empty grasp`, `lost hold`, `plan failed`)
    with its workflow stage (`in stage <stage>`), `step limit`, `finished without success` (S2:
    `finish unconfirmed`, the official check did not confirm an accepted finish), `no finish`
    (turn, call or time limits, format errors). Perception and placement labels need the
    ground-truth attribution."""
    labels = []
    for outcome in outcomes:
        if outcome.success:
            continue
        reason = outcome.end.reason
        if reason.startswith("fail_fast:"):
            failure, _, stage = reason.removeprefix("fail_fast:").partition("@")
            label = failure.replace("_", " ")
            labels.append(f"{label} in stage {stage}" if stage else label)
        elif reason == "step_limit":
            labels.append("step limit")
        elif reason == "finish":
            labels.append("finished without success")
        elif reason == UNCONFIRMED:
            labels.append("finish unconfirmed")
        else:
            labels.append("no finish")
    return labels


def version_records(rows: Sequence[LedgerRow]) -> dict[int, tuple[int, int]]:
    """Each Memory version's record over the counted seeds whose try 1 read it (`memory_before`):
    (first-try successes, seeds). Uncounted seeds (unsolvable) are left out; set-aside seeds
    never reach the ledger, and later tries (the seed's cache) are not first tries."""
    found: dict[int, tuple[int, int]] = {}
    for row in rows:
        if row.k is None:
            continue
        k, n = found.get(row.memory_before, (0, 0))
        found[row.memory_before] = (k + bool(row.first_try_pass), n + 1)
    return found


def best_version(rows: Sequence[LedgerRow]) -> dict[str, Any] | None:
    """The Memory version with the best first-try record (`version_records`): the highest Wilson
    95 % lower bound of its rate, then more successes, then the later version; one without a
    success is never picked. Its `version`, `successes`, `n` and `lower`; None when no counted
    first try succeeded."""
    ranked = [
        (wilson(k, n)[0], k, version, n)
        for version, (k, n) in version_records(rows).items()
        if k > 0
    ]
    if not ranked:
        return None
    lower, k, version, n = max(ranked)
    return {"version": version, "successes": k, "n": n, "lower": round(lower, 6)}


def summary(rows: Sequence[LedgerRow]) -> dict[str, Any]:
    """What a practice did: seeds by outcome, first tries with their Wilson interval, the cost of
    the tries and revisions, and how the tries' objects were identified (afresh or reused)."""
    counted = [row for row in rows if row.k is not None]
    outcomes = Counter(row.outcome for row in rows)
    first = sum(bool(row.first_try_pass) for row in counted)
    tries = [t for row in rows for t in row.tries]
    revisions = [r for row in rows for r in row.revisions]
    return {
        "seeds": len(rows),
        "counted": len(counted),
        **{name: outcomes[name] for name in ("pass", "merge", "jitter", "skip", "unsolvable")},
        "first_try": {
            "successes": first,
            "n": len(counted),
            "rate": first / len(counted) if counted else None,
            "wilson95": list(wilson(first, len(counted))),
        },
        "tries": len(tries),
        "revisions": len(revisions),
        "identified": {
            "fresh": sum(t.identification == "identified" for t in tries),
            "reused": sum((t.identification or "").startswith("reused:") for t in tries),
        },
        **{
            key: round(
                sum(getattr(t, key) for t in tries) + sum(getattr(r, key) for r in revisions), 3
            )
            for key in ("infra_reruns", "model_calls", "tokens", "wall_clock_s")
        },
    }
