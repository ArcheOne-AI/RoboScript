"""The per-task closure's task plans (README, "Per-task closure"): one task's calls as data,
written by a per-task builder session and judged by the driver. Never code and never handed to S1,
S2 or P3: the plan is P1's evidence that the skillset can do the task, kept under the run directory
(`<run>/plans/`), not in the skillset or a Memory. Skills written in those sessions enter the
skillset through the gate.

    {"task": "<task>",
     "segments": {"<segment name>": {"skill": "...", "form": "...",
                                     "adjust": {"<param>": delta, "<param>": [delta, ...]},
                                     "before": [{"skill": "...", "adjust": {...}}]}}}

Per segment (its name as the task's reports give it; one left out keeps the call it has): the
skill and form of one of the segment's options (the core's fit, reuse's alternatives, a written
skill's fit: the glue's chain options), whose params are fitted from the demonstration under
`demo_coverage.fitting.RELATIVE_RULE`; `adjust`, numbers added to a written skill's fitted params
(a delta to a relative number stays relative: aiming deeper into the success region than the
demonstration's end, which is often only just across it); `before`, transition calls run first,
from where the scene is (`fitting.Call.before`): a written skill, fitted on this segment, or a core
entry of `fitting.TRANSITION_CORE`. What cannot be applied (no such option, a fit refused) is
recorded, and that segment keeps its call.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

Delta = float | list[float]


class Transition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    skill: str
    adjust: dict[str, Delta] = {}


class Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    skill: str | None = None
    form: str | None = None
    adjust: dict[str, Delta] = {}
    before: list[Transition] = []


class TaskPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    segments: dict[str, Entry] = {}

    def key(self) -> str:
        """A hash of the plan as data (defaults left out, keys sorted): which chains a chained
        step may start from (`cover_demos.prior_fixes`: only those of the same plan)."""
        text = json.dumps(self.model_dump(mode="json", exclude_defaults=True), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def skills(self) -> set[str]:
        """Every skill the plan names, for segments and transitions."""
        named = {e.skill for e in self.segments.values() if e.skill}
        return named | {t.skill for e in self.segments.values() for t in e.before}


class PlanError(ValueError):
    """Why a plan's entry cannot be applied to the task's calls."""


def read(path: Path) -> TaskPlan:
    return TaskPlan.model_validate_json(path.read_text(encoding="utf-8"))


def write(path: Path, plan: TaskPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan.model_dump(mode="json", exclude_defaults=True), indent=1)
    path.write_text(text + "\n", encoding="utf-8")


def adjusted(params: dict[str, Any], adjust: dict[str, Delta]) -> dict[str, Any]:
    """`params` with `adjust`'s deltas added: a number to a number, a list to a list of numbers
    of its length, element by element."""
    found = dict(params)
    for key, delta in adjust.items():
        value = found.get(key)
        if isinstance(delta, list):
            if not (isinstance(value, list) and len(value) == len(delta)
                    and all(number(v) for v in value)):  # fmt: skip
                raise PlanError(f"adjust {key}: not a list of {len(delta)} numbers in params")
            found[key] = [float(v) + float(d) for v, d in zip(value, delta, strict=True)]
        elif number(value):
            found[key] = float(value) + float(delta)
        else:
            raise PlanError(f"adjust {key}: not a number in params")
    return found


def number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def prefilled(task: str, chain: dict[str, Any] | None, accepted: TaskPlan | None) -> TaskPlan:
    """A per-task session's starting plan: every segment's skill and form as the task's chained
    step left its calls (`chain["calls"]`, fixes included), and where the task's accepted plan has
    an entry for the same skill and form, that entry (its `adjust`, `before`). Applied, it is the
    task's current plan."""
    entries: dict[str, Entry] = {}
    for c in (chain or {}).get("calls", []):
        if not c.get("skill"):
            continue
        entry = Entry(skill=c["skill"], form=c.get("form") or "")
        kept = accepted.segments.get(c["segment"]) if accepted else None
        if kept is not None and (kept.skill, kept.form) == (entry.skill, entry.form):
            entry = kept
        entries[c["segment"]] = entry
    return TaskPlan(task=task, segments=entries)


# -- the ceiling's repeats and a task's target (protocol constants) ---------------------------

FIRST_REPEATS: Final = 4
"""The ceiling's first runs of a plan, every ceiling (coverage, reuse, chained step, plan trial):
the official success varies from run to run on unchanged skills."""
CONFIRM_FROM: Final = 2
"""First runs that latch and call for the confirmation round."""
CONFIRMED_REPEATS: Final = 8
CONFIRMED_SUCCESSES: Final = 4
"""A ceiling whose first runs latched `CONFIRM_FROM` runs on to `CONFIRMED_REPEATS` (the same
plan and episode protocol); a task is at its target only where `CONFIRMED_SUCCESSES` of them
latched: stricter than 2 of 4, never looser (half of twice the runs; 4 of 8 by a plan whose rate
is 1/4 happens about one time in nine, 2 of 4 about one in four)."""
if not 0 < CONFIRM_FROM <= FIRST_REPEATS < CONFIRMED_REPEATS:
    raise ValueError("the ceiling's repeats: 0 < CONFIRM_FROM <= FIRST_REPEATS < CONFIRMED_REPEATS")
if not CONFIRM_FROM <= CONFIRMED_SUCCESSES <= CONFIRMED_REPEATS:
    raise ValueError("a task's target: CONFIRM_FROM <= CONFIRMED_SUCCESSES <= CONFIRMED_REPEATS")
TARGET_TEXT: Final = (
    f"at least {CONFIRM_FROM} of the ceiling's first {FIRST_REPEATS} runs latch the official "
    f"success, then it runs {CONFIRMED_REPEATS - FIRST_REPEATS} more on the same plan, and at "
    f"least {CONFIRMED_SUCCESSES} of all {CONFIRMED_REPEATS} latch"
)
"""The target as a builder's brief and prompt state it."""


def latched(runs: Sequence[dict[str, Any]]) -> int:
    """The ceiling runs that latched the official success (one the infrastructure broke did not)."""
    return sum(bool(r.get("success")) and not r.get("infra") for r in runs)


def due(runs: Sequence[dict[str, Any]]) -> int:
    """The ceiling runs a plan's ceiling owes, given those it ran so far (in order):
    `FIRST_REPEATS`, or `CONFIRMED_REPEATS` once `CONFIRM_FROM` of the first latched."""
    first = latched(runs[:FIRST_REPEATS])
    return CONFIRMED_REPEATS if first >= CONFIRM_FROM else FIRST_REPEATS


def confirmed(runs: Sequence[dict[str, Any]]) -> bool:
    """Whether a ceiling ran its confirmation round (`due`): a report whose ceiling stopped
    at 4 runs did not."""
    return len(runs) >= CONFIRMED_REPEATS and due(runs) == CONFIRMED_REPEATS


def at_target(runs: Sequence[dict[str, Any]]) -> bool:
    """A task's ceiling at its target: confirmed, and `CONFIRMED_SUCCESSES` of its runs latched."""
    return confirmed(runs) and latched(runs[:CONFIRMED_REPEATS]) >= CONFIRMED_SUCCESSES


# -- comparing measured plans (the driver's trials, the chained step's seeds) ----------


def counted(state: dict[str, Any], key: str) -> int:
    """`key` (`latched`, `judged`, `stages_done`, `stages_of`) over the ceiling's first
    `FIRST_REPEATS` runs (`first_<key>`), the runs every ceiling makes: a ceiling runs its
    confirmation only where those latched `CONFIRM_FROM`, so a rate over all its runs would
    compare 8 runs with 4 (2 of 4 then 0 of 4 would tie 1 of 4).
    A state without those counts: all its runs, at most 4."""
    return int(state.get(f"first_{key}", state.get(key, 0)) or 0)


def rate(state: dict[str, Any]) -> float:
    """A task's latched successes over its judged ceiling repeats, of its first `FIRST_REPEATS`
    (`counted`): every comparison like with like; the confirmation decides only `at_target`."""
    judged = counted(state, "judged")
    return float(counted(state, "latched")) / judged if judged else 0.0


def share(state: dict[str, Any]) -> float:
    """The share of the benchmark's stages a task's judged ceiling runs did, of its first
    `FIRST_REPEATS` (`counted`)."""
    of = counted(state, "stages_of")
    return float(counted(state, "stages_done")) / of if of else 0.0


def first_counts(ran: Sequence[dict[str, Any]]) -> dict[str, int]:
    """`counted`'s keys of a ceiling's runs (in order): its first `FIRST_REPEATS`, judged."""
    judged = [r for r in ran[:FIRST_REPEATS] if not r.get("infra")]
    stages = [r["stages"] for r in judged if r.get("stages")]
    return {
        "first_latched": sum(bool(r.get("success")) for r in judged),
        "first_judged": len(judged),
        "first_stages_done": sum(int(x["of"]) - int(x["left"]) for x in stages),
        "first_stages_of": sum(int(x["of"]) for x in stages),
    }


def trial_state(report: dict[str, Any]) -> dict[str, Any]:
    """A plan trial's chain report as `state` counts a task: its rollout's segments explained,
    whether it reached the end, its ceiling's judged repeats, those that latched, the stages
    they did."""
    ran = list(report.get("ceiling") or [])
    judged = [r for r in ran if not r.get("infra")]
    stages = [r["stages"] for r in judged if r.get("stages")]
    return {
        "chained": int(report.get("explained", 0)),
        "to_end": bool(report.get("to_end")),
        "latched": sum(bool(r.get("success")) for r in judged),
        "judged": len(judged),
        "repeats": len(ran),
        "stages_done": sum(int(x["of"]) - int(x["left"]) for x in stages),
        "stages_of": sum(int(x["of"]) for x in stages),
        "confirmed": confirmed(ran),
        "met": at_target(ran),
    } | first_counts(ran)


def better(before: dict[str, Any], after: dict[str, Any]) -> str | None:
    """Why a plan trial (`trial_state`) beats the task's current plan's, paired, else None
    (`rises`)."""
    return rises(
        (before["chained"], after["chained"]),
        (rate(before), rate(after)),
        (share(before), share(after)),
    )


def rises(
    chained: tuple[float, float], success: tuple[float, float], stages: tuple[float, float]
) -> str | None:
    """Why the second of two measured plans beats the first (each pair: first, second), else
    None. Official success first (success is the only criterion): the share of
    judged ceiling repeats latched rose and the share of stages done did not fall, whatever the
    chain did (it is judged against the demonstration's segment ends, which lie just across the
    official boundary). Otherwise the chain went deeper or the successes or stages rose, and
    none of the three fell."""
    pairs = {"chained": chained, "success": success, "stages": stages}
    rose = [f"{k} {b:.2f} -> {a:.2f}" for k, (b, a) in pairs.items() if a > b]
    if success[1] > success[0] and stages[1] >= stages[0]:
        b, a = chained
        return "; ".join(rose) + (f" (chain {b:g} -> {a:g})" if a < b else "")
    if any(b > a for b, a in pairs.values()):
        return None
    return "; ".join(rose) or None
