"""wicl's part of a pipeline's status and summary (`roboscript.orchestrate.report`), read
from the jobs' records.

A task's S1 (`<task>/S1`): its latest run's attempts, referee passes and exit. A unit's S2: what
its Actor read of the demonstration (`demo_context`, from `start.json`), its ledger (seeds,
counted seeds, first-try passes, merges, streak) and how it stopped; the summary adds S2's
first-try curve and outcomes, Wilson intervals, and its failure labels (taxonomy `s2_labels`). A
task's S1 whose fallback let its units start shows the fallback version, not converged, with the
unhinted official successes and all official successes of the attempts that read it (under
`s1_fallback_rule` `success`; the unhinted passes and successes for one found under the
`progress` rule); one that stopped (no official success), why; one whose official successes all
came in hinted attempts is marked hinted only. The summary lists those kinds of task
(`s1_best_memory`, `s1_best_memory_hinted_only`, `s1_stopped`) and the units whose S2 stopped
at its seed cap (`s2_best_memory`, `s2_stopped`). A unit whose S2 stopped at its seed cap
without converging shows how it went on (`s2.exit`, the pipeline's `exit`: `best_memory` with the
version the evaluation reads and its first-try record, or `none`: no first-try success). A unit's
row in the summary table stands on its task's S1."""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from roboscript.learning import wilson
from roboscript.orchestrate.report import (
    LEARNING,
    Column,
    JobView,
    Lane,
    Row,
    Section,
    rate,
    read_json,
)
from roboscript_methods.wicl.imitate import EXIT, attempt_passed
from roboscript_methods.wicl.ledger import Ledger
from roboscript_methods.wicl.ops import FALLBACK, S1
from roboscript_methods.wicl.practice import LEDGER, START, STOP

_SEED: Final = re.compile(r"\d+")
_ATTEMPT: Final = re.compile(r"attempt\d+")

S2Exit = Callable[[Path, str], dict[str, Any] | None]
"""How a unit's S2 ended for its evaluation (`run_dir`, task): its `S2_EXIT` record or what
the pick would record now; None while S2 has not stopped (the pipeline's `run.s2_exit`)."""


class LearnReport:
    """wicl's `ReportPart`: a task's S1 and its units' S2, with `exit` (`S2Exit`)."""

    def __init__(self, exit: S2Exit) -> None:
        self.exit = exit
        self.readings = {"s1": LEARNING, "s2": LEARNING}
        self.taxonomy_labels = {"s2_labels": "S2 failure labels"}
        self.status_columns = (
            Column("S1 tries (pass)", _tries),
            Column("exit", _s1_exit),
            Column("demo", lambda row, _: _s2(row, lambda s2: s2["demo_context"])),
            Column("S2 seeds (k)", lambda row, _: _s2(row, _seeds)),
            Column("1st", lambda row, _: _s2(row, lambda s2: str(s2["first_try"]))),
            Column("merges", lambda row, _: _s2(row, lambda s2: str(s2["merges"]))),
            Column("streak", lambda row, _: _s2(row, lambda s2: str(s2["streak"]))),
            Column("S2 stop", lambda row, _: _s2(row, s2_stop)),
        )
        self.summary_columns = (
            Column("S1 attempts to exit", _attempts),
            Column("demo", lambda row, _: row["s2"]["demo_context"]),
            Column("S2 seeds (k)", lambda row, _: _seeds(row["s2"])),
            Column(
                "first-try rate [95%]",
                lambda row, _: rate(row["s2"]["first_try_rate"], row["s2"]["wilson95"]),
            ),
            Column("curve", lambda row, _: "".join(map(str, row["s2"]["curve"])) or "-"),
            Column("S2 stop", lambda row, _: s2_stop(row["s2"])),
        )

    def episodes(self, job: JobView) -> Iterable[tuple[str, Path]]:
        stage, folder = ("s1", "S1") if job.key.variant == S1 else ("s2", "S2")
        found = sorted((job.run_dir / job.key.task / folder).glob("**/config.json"))
        return [(stage, path.parent) for path in found]

    def section(self, job: JobView, deep: bool) -> Section:
        run_dir, task = job.run_dir, job.key.task
        if job.key.variant == S1:
            fallback = read_json(run_dir / FALLBACK)
            return Section({"s1": _s1_of(run_dir / task / "S1") | {"fallback": fallback}})
        s2 = _s2_of(run_dir / task / "S2", deep)
        labels = s2.pop("labels") if deep else None
        s2["exit"] = self.exit(run_dir, task)
        return Section({"s2": s2}, {"s2_labels": labels} if deep else {})

    def summary(self, rows: list[Row]) -> Row:
        return {**s1_ends(rows), **s2_ends(rows)}

    def flags(self, rows: list[Row]) -> list[str]:
        return []

    def headline(self, found: Mapping[str, Any]) -> list[str]:
        return []

    def notes(self, found: Mapping[str, Any]) -> list[str]:
        """The tasks whose S1 did not exit and the units whose S2 did not converge."""
        lines = []
        if best := found.get("s1_best_memory"):
            lines.append(f"- S1 did not exit, went on from its best Memory: {', '.join(best)}")
        if hinted := found.get("s1_best_memory_hinted_only"):
            lines.append(f"- of those, official successes only with the hint: {', '.join(hinted)}")
        if stopped := found.get("s1_stopped"):
            lines.append(f"- S1 stopped (no official success; P3 counted 0): {', '.join(stopped)}")
        if best := found.get("s2_best_memory"):
            lines.append(f"- S2 did not converge, P3 on its best Memory: {', '.join(best)}")
        if stopped := found.get("s2_stopped"):
            lines.append(f"- S2 stopped (no first-try success; P3 counted 0): {', '.join(stopped)}")
        return lines

    def lanes(self, row: Mapping[str, Any]) -> list[Lane]:
        return []

    def chain(self, row: Mapping[str, Any], rows: list[Row]) -> list[Row] | None:
        """A unit's row stands on its task's S1; an S1's is not a row of the summary table."""
        return None if _is_s1(row) else [_task_s1(row, rows)]

    def stopped(self, row: Mapping[str, Any], rows: list[Row]) -> bool:
        """Whether a unit's task stopped at S1 (`not_passed`: its evaluation counts 0)."""
        return bool(_task_s1(row, rows)["status"] == "not_passed")


def _is_s1(row: Mapping[str, Any]) -> bool:
    return str(row["run"]).endswith(f"/{S1}")


def _task_s1(row: Mapping[str, Any], rows: list[Row]) -> Row:
    task = row["run"].split("/")[0]
    s1 = {r["run"].split("/")[0]: r for r in rows if _is_s1(r)}
    return s1[task]


def s1_ends(rows: list[Row]) -> dict[str, list[str]]:
    """The tasks whose S1 did not exit: those whose variants went on from its best Memory
    (`s1_best_memory`: `s1_exit` `best_memory`, or a fallback recorded without the field),
    and those that stopped at S1 (`s1_stopped`: `not_passed`, counted 0 in P3)."""
    best, hinted, stopped = [], [], []
    for row in rows:
        if not _is_s1(row):
            continue
        task = row["run"].split("/")[0]
        fallback = row["s1"]["fallback"]
        if row["status"] == "not_passed":
            stopped.append(task)
        elif fallback is not None:
            best.append(task)
            if fallback.get("hinted_only"):
                hinted.append(task)
    return {"s1_best_memory": best, "s1_best_memory_hinted_only": hinted, "s1_stopped": stopped}


def s2_ends(rows: list[Row]) -> dict[str, list[str]]:
    """The variants whose S2 stopped at its seed cap: those P3 evaluates on the version of the
    best first-try record (`s2_best_memory`, with it as `v<n>`) and those stopped with no
    first-try success (`s2_stopped`, counted 0 in P3)."""
    best, stopped = [], []
    for row in rows:
        found = (row.get("s2") or {}).get("exit") or {}
        if found.get("s2_exit") == "best_memory":
            best.append(f"{row['run']} v{found['version']:03d}")
        elif found.get("s2_exit") == "none":
            stopped.append(row["run"])
    return {"s2_best_memory": best, "s2_stopped": stopped}


def _s1_of(folder: Path) -> dict[str, Any]:
    """The latest S1 run's attempts so far, their referee passes and its exit, if any; after an
    exit, only the attempts up to it (a resumed replay may exit before attempts that ran, e.g.
    when the linter no longer holds its exit: those stay on disk, counted as `after_exit`)."""
    runs = sorted((p for p in folder.glob("*") if _SEED.fullmatch(p.name)), key=_number)
    if not runs:
        return {"seed": None, "attempts": 0, "passes": 0, "after_exit": 0, "exit": None}
    current = runs[-1]
    attempts = [p for p in current.glob("attempt*") if _ATTEMPT.fullmatch(p.name)]
    exit = read_json(current / EXIT)
    later = []
    if exit is not None:
        later = [p for p in attempts if int(p.name.removeprefix("attempt")) > exit["attempts"]]
        attempts = [p for p in attempts if p not in later]
    keep = ("passed", "attempts", "hinted", "unhinted", "version", "infra_reruns", "demo_truth")
    keep += ("fallback", "converged", "s1_exit", "fallback_rule", "fallback_reason")
    return {
        "seed": int(current.name),
        "attempts": sum((p / "summary.json").is_file() for p in attempts),
        "passes": sum(bool(attempt_passed(p)) for p in attempts),
        "after_exit": len(later),
        "exit": None if exit is None else {k: exit[k] for k in keep if k in exit},
    }


def _s2_of(folder: Path, deep: bool) -> dict[str, Any]:
    rows = Ledger(folder / LEDGER).rows()
    counted = [row for row in rows if row.k is not None]
    first = sum(bool(row.first_try_pass) for row in counted)
    stop = read_json(folder / STOP)
    start = read_json(folder / START) or {}
    found: dict[str, Any] = {
        "demo_context": start.get("demo_context", "none"),
        "seeds": len(rows),
        "counted": len(counted),
        "first_try": first,
        "merges": sum(row.outcome == "merge" for row in rows),
        "streak": rows[-1].streak_after if rows else 0,
        "stop": None
        if stop is None
        else {k: stop[k] for k in ("reason", "k", "version", "frozen")},
    }
    if deep:
        found |= {
            "curve": [int(bool(row.first_try_pass)) for row in counted],
            "outcomes": dict(Counter(row.outcome for row in rows)),
            "budget_k": [row.budget_k for row in counted],
            "first_try_rate": first / len(counted) if counted else None,
            "wilson95": list(wilson(first, len(counted))) if counted else None,
            "labels": dict(Counter(label for row in rows for label in row.failure_labels)),
        }
    return found


def _tries(row: Mapping[str, Any], _: Mapping[str, Any]) -> str:
    s1 = row.get("s1")
    return "-" if s1 is None else f"{s1['attempts']} ({s1['passes']})"


def _s1_exit(row: Mapping[str, Any], _: Mapping[str, Any]) -> str:
    s1 = row.get("s1")
    return "-" if s1 is None else exit_cell(s1["exit"], s1["fallback"])


def _s2(row: Mapping[str, Any], cell: Callable[[Mapping[str, Any]], str]) -> str:
    s2 = row.get("s2")
    return "-" if s2 is None else cell(s2)


def _seeds(s2: Mapping[str, Any]) -> str:
    return f"{s2['seeds']} ({s2['counted']})"


def _attempts(row: Mapping[str, Any], found: Mapping[str, Any]) -> str:
    """A unit's task's S1: its attempts to exit, and the exit."""
    task_s1 = _task_s1(row, found["rows"])
    exit = task_s1["s1"]["exit"]
    shown = exit_cell(exit, task_s1["s1"]["fallback"])
    return "-" if exit is None else f"{exit['attempts']} ({shown})"


def s2_stop(s2: Mapping[str, Any]) -> str:
    """How S2 stopped, with the version P3 evaluates when it stopped at the cap."""
    if s2["stop"] is None:
        return "-"
    found = s2.get("exit") or {}
    if found.get("s2_exit") == "best_memory":
        p = found["picked"]
        return (
            f"{s2['stop']['reason']}: best_memory v{found['version']:03d} "
            f"({p['successes']}/{p['n']}, lower {p['lower']:.2f})"
        )
    if found.get("s2_exit") == "none":
        return f"{s2['stop']['reason']}: stopped (no first-try success)"
    return str(s2["stop"]["reason"])


def exit_cell(exit: Mapping[str, Any] | None, fallback: Mapping[str, Any] | None = None) -> str:
    """How a task's S1 exited (or how it went on from its fallback, `ops.FALLBACK`)."""
    if fallback is not None:
        evidence = ((exit or {}).get("fallback") or {}).get("evidence", {})
        score = evidence.get("score", [0, 0])
        tried = len(evidence.get("attempts", []))
        if evidence.get("rule") == "success":  # unhinted successes, successes in all
            shown = f"U {score[0]}s, all {score[1]}s of {tried}"
            hint = ", hinted only" if fallback.get("hinted_only") else ""
            return f"best_memory v{fallback['version']} (not converged, {shown}{hint})"
        shown = f"U {score[0]}p/{score[1]}s of {tried}"
        return f"fallback v{fallback['version']} (not converged, {shown})"
    if exit is None:
        return "-"
    if exit["passed"]:
        return f"v{exit['version']}"
    reason = exit.get("fallback_reason")
    return f"no exit, stopped ({reason})" if reason else "no exit"


def _number(path: Path) -> int:
    return int(path.name)
