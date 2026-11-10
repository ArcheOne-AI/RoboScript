"""Helpful and harmful counts from paired outcomes, and the entries' status
(docs/concepts/learning.md).

A round evaluates the RSI document beside a baseline without it on the same seeds. For each
(task, seed) pair, `PairedOutcome` holds both arms' official success and the entry ids the RSI
arm's Actor cited (named in its scratchpad). With the round's document, `update_counters`:

- RSI success, baseline failure: helpful +1 for each cited entry;
- RSI failure, baseline success: harmful +1 for each cited entry;
- equal outcomes: no count.

An episode's Actor was shown GLOBAL and its own task's part only: a cited id outside those is
ignored (`unseen`), one naming no readable entry too (`unknown`). An entry was shown this round
when it is in GLOBAL (with any outcome) or in the part of a task the outcomes hold; an entry not
shown is left as it was.

Then the status of every shown entry that is not retired: retire it when harmful > helpful with
at least `retire_min_counts` counts in all, or when no Actor cited it for `retire_idle_rounds`
consecutive rounds it was shown in; else an entry on probation becomes active after a round
whose net (helpful - harmful gained) is positive. Only these paired counts move status; the
reflectors' votes never do. Every count and change goes into the entry's history.
"""

from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from roboscript.contracts.ids import TaskName
from roboscript_methods.rsi.doc import Change, EntryId, RsiDoc, RsiEntry
from roboscript_methods.rsi.recipe import RsiRules


class PairedOutcome(BaseModel):
    """One seed of a task in both arms: the official success of each, and the entry ids the RSI
    arm's Actor cited in that episode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    seed: NonNegativeInt
    rsi_success: bool
    baseline_success: bool
    cited: tuple[EntryId, ...] = ()


class CountReport(BaseModel):
    """What a round's counting did: counts and status changes by entry, cited ids that name no
    readable entry of the document (`unknown`) and ids of entries the citing episode was not
    shown (`unseen`); neither is counted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round: PositiveInt
    helpful: dict[str, int]
    harmful: dict[str, int]
    cited: dict[str, int]
    promoted: list[str]
    retired: dict[str, str]
    unknown: list[str]
    unseen: list[str]


def update_counters(
    doc: RsiDoc, outcomes: Sequence[PairedOutcome], rules: RsiRules, round: PositiveInt
) -> tuple[RsiDoc, CountReport]:
    """`doc` (the document evaluated in round `round`) after counting `outcomes`, and the
    report. A pair given twice is an error."""
    pairs = Counter((o.task, o.seed) for o in outcomes)
    if twice := sorted(pair for pair, n in pairs.items() if n > 1):
        raise ValueError(f"paired outcomes given more than once: {twice}")
    readable = {entry.id for _, entry in doc.entries() if entry.readable}
    shown_tasks = {outcome.task for outcome in outcomes}
    helpful: Counter[str] = Counter()
    harmful: Counter[str] = Counter()
    cited: Counter[str] = Counter()
    unknown: set[str] = set()
    unseen: set[str] = set()
    for outcome in outcomes:
        visible = {e.id for e in (*doc.global_entries, *doc.section(outcome.task)) if e.readable}
        ids = set(outcome.cited)
        unknown |= ids - readable
        unseen |= (ids & readable) - visible
        ids &= visible
        cited.update(ids)
        if outcome.rsi_success and not outcome.baseline_success:
            helpful.update(ids)
        elif outcome.baseline_success and not outcome.rsi_success:
            harmful.update(ids)
    promoted: list[str] = []
    retired: dict[str, str] = {}

    def update(entry: RsiEntry, shown: bool) -> RsiEntry:
        if not entry.readable or not shown:
            return entry
        up, down = helpful[entry.id], harmful[entry.id]
        uncited = 0 if cited[entry.id] else entry.uncited + 1
        note = f"helpful +{up}, harmful +{down}, cited in {cited[entry.id]}"
        history = [*entry.history, Change(round=round, op="count", note=note)]
        total_up, total_down = entry.helpful + up, entry.harmful + down
        status, why = entry.status, ""
        if total_down > total_up and total_up + total_down >= rules.retire_min_counts:
            status, why = "retired", f"harmful {total_down} > helpful {total_up}"
        elif uncited >= rules.retire_idle_rounds:
            status, why = "retired", f"not cited for {uncited} rounds"
        elif status == "probation" and up - down > 0:
            status, why = "active", f"net +{up - down}"
        if status == "retired":
            history.append(Change(round=round, op="auto_retire", note=why))
            retired[entry.id] = why
        elif status != entry.status:
            history.append(Change(round=round, op="promote", note=why))
            promoted.append(entry.id)
        return entry.model_copy(
            update={
                "helpful": total_up,
                "harmful": total_down,
                "uncited": uncited,
                "status": status,
                "history": tuple(history),
            }
        )

    updated = doc.model_copy(
        update={
            "global_entries": tuple(update(e, bool(outcomes)) for e in doc.global_entries),
            "task_entries": {
                task: tuple(update(e, task in shown_tasks) for e in entries)
                for task, entries in doc.task_entries.items()
            },
        }
    )
    report = CountReport(
        round=round,
        helpful=dict(sorted(helpful.items())),
        harmful=dict(sorted(harmful.items())),
        cited=dict(sorted(cited.items())),
        promoted=promoted,
        retired=retired,
        unknown=sorted(unknown),
        unseen=sorted(unseen),
    )
    return updated, report
