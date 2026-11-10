"""What a P4 round did (docs/concepts/learning.md): the RSI arm against the baseline on the same
official seeds, and the document's entries.

`round_report` pairs the two arms' episodes of one evaluation block by task and seed (an episode
lost to the infrastructure or voided by the host, after an unanswered infrastructure error,
pairs with nothing and is listed as unpaired), and reports:

- the success rates, the discordant pairs, McNemar's exact test on them (two-sided binomial) and a
  bootstrap 95 % interval of the difference over seed pairs (fixed resampling seed, so a report
  rendered again is the same);
- per task: both arms' successes on the paired seeds and the pairs each arm alone won; the tasks
  where the RSI arm lost two or more pairs are listed on their own (where it harms);
- per entry of the document: cited / helpful / harmful in this round and in all, status, and
  promotions and retirements of the counting (`counters.CountReport`);
- the voided share and the infrastructure's share of each arm, tokens per episode, the
  injected length of the document, and the arms not run (GLOBAL-only and TASK-only).

`render` writes the report for people (`report.md`); `final_report` puts the rounds side by side.
"""

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from roboscript.contracts.ids import TaskName
from roboscript_methods.rsi.counters import CountReport
from roboscript_methods.rsi.doc import RsiDoc

HARM_PAIRS: Final = 2
"""A task the RSI arm lost at least this many pairs of (baseline success, RSI failure) is listed
as harmed."""
BOOTSTRAP: Final = 10000
BOOTSTRAP_SEED: Final = 0


class ArmEpisode(BaseModel):
    """One evaluated episode of an arm: its task, official index and seed, the official success
    (None: not an outcome of the strategy: lost to the infrastructure, or voided by the host),
    whether it was voided or lost, the entry ids its Actor cited and its tokens (None:
    not recorded)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    index: NonNegativeInt
    seed: NonNegativeInt
    success: bool | None
    voided: bool = False
    infra: bool = False
    cited: tuple[str, ...] = ()
    tokens: NonNegativeInt | None = None


class Pair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    seed: NonNegativeInt
    baseline: bool
    rsi: bool


def pair_up(
    baseline: Sequence[ArmEpisode], rsi: Sequence[ArmEpisode]
) -> tuple[list[Pair], dict[str, list[str]]]:
    """The pairs (task, seed) both arms have an outcome for, in task and seed order, and by arm
    the episodes left unpaired (`<task>@<seed>: <why>`)."""
    sides = {"baseline": baseline, "rsi": rsi}
    outcome: dict[str, dict[tuple[str, int], bool]] = {}
    unpaired: dict[str, list[str]] = {"baseline": [], "rsi": []}
    for arm, episodes in sides.items():
        found: dict[tuple[str, int], bool] = {}
        for e in episodes:
            if (e.task, e.seed) in found:
                raise ValueError(f"{arm}: {e.task} seed {e.seed} evaluated twice")
            if e.success is None:
                why = "voided (A-prime)" if e.voided else "lost to the infrastructure"
                unpaired[arm].append(f"{e.task}@{e.seed}: {why}")
            else:
                found[(e.task, e.seed)] = e.success
        outcome[arm] = found
    both = sorted(set(outcome["baseline"]) & set(outcome["rsi"]))
    for arm, other in (("baseline", "rsi"), ("rsi", "baseline")):
        for key in sorted(set(outcome[arm]) - set(outcome[other])):
            unpaired[arm].append(f"{key[0]}@{key[1]}: the other arm has no outcome")
    pairs = [
        Pair(
            task=TaskName(t),
            seed=s,
            baseline=outcome["baseline"][(t, s)],
            rsi=outcome["rsi"][(t, s)],
        )
        for t, s in both
    ]
    return pairs, unpaired


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value of `b` and `c` discordant pairs (binomial, p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_interval(pairs: Sequence[Pair]) -> tuple[float, float] | None:
    """The bootstrap 95 % interval of rsi - baseline success rate over seed pairs (None: no
    pairs)."""
    if not pairs:
        return None
    diffs = [int(p.rsi) - int(p.baseline) for p in pairs]
    draw = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    means = sorted(sum(draw.choices(diffs, k=n)) / n for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def _shares(episodes: Sequence[ArmEpisode]) -> dict[str, Any]:
    n = len(episodes)
    tokens = [e.tokens for e in episodes if e.tokens is not None]
    return {
        "episodes": n,
        "voided": sum(e.voided for e in episodes),
        "voided_share": None if not n else sum(e.voided for e in episodes) / n,
        "infra": sum(e.infra and not e.voided for e in episodes),
        "tokens_per_episode": None if not tokens else sum(tokens) / len(tokens),
    }


def round_report(
    *,
    round: int,
    block: str,
    mode: str,
    baseline: Sequence[ArmEpisode],
    rsi: Sequence[ArmEpisode],
    doc: RsiDoc,
    counted: RsiDoc | None = None,
    counts: CountReport | None = None,
) -> dict[str, Any]:
    """The report of round `round` on `block` in `mode`: `doc` is the document the RSI arm read,
    `counted` the same after this round's counting (`counts`; None for a block nothing is
    counted on, the final one)."""
    pairs, unpaired = pair_up(baseline, rsi)
    b = sum(p.baseline and not p.rsi for p in pairs)  # the RSI arm lost these
    c = sum(p.rsi and not p.baseline for p in pairs)  # and won these
    n = len(pairs)
    tasks: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(_TASK_COLUMNS, 0))
    for p in pairs:
        row = tasks[p.task]
        row["pairs"] += 1
        row["baseline"] += p.baseline
        row["rsi"] += p.rsi
        row["rsi_lost"] += p.baseline and not p.rsi
        row["rsi_won"] += p.rsi and not p.baseline
    harmed = sorted(t for t, row in tasks.items() if row["rsi_lost"] >= HARM_PAIRS)
    entries = _entries(doc, counted, counts)
    return {
        "round": round,
        "block": block,
        "mode": mode,
        "pairs": n,
        "baseline_rate": None if not n else sum(p.baseline for p in pairs) / n,
        "rsi_rate": None if not n else sum(p.rsi for p in pairs) / n,
        "rsi_lost": b,
        "rsi_won": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "diff_ci95": bootstrap_interval(pairs),
        "tasks": {t: tasks[t] for t in sorted(tasks)},
        "harmed": harmed,
        "unpaired": unpaired,
        "arms": {"baseline": _shares(baseline), "rsi": _shares(rsi)},
        "doc": {
            "entries": len(list(doc.entries())),
            "global_chars": len(doc.part(None)),
            "task_chars": {t: len(doc.part(TaskName(t))) for t in sorted(tasks)},
        },
        "entries": entries,
        "counts": None if counts is None else counts.model_dump(mode="json"),
        "not_run": ["GLOBAL-only arm", "TASK-only arm"],
    }


_TASK_COLUMNS: Final = ("pairs", "baseline", "rsi", "rsi_lost", "rsi_won")


def _entries(
    doc: RsiDoc, counted: RsiDoc | None, counts: CountReport | None
) -> list[dict[str, Any]]:
    after = {} if counted is None else {entry.id: entry for _, entry in counted.entries()}
    rows = []
    for part, entry in doc.entries():
        now = after.get(entry.id, entry)
        rows.append(
            {
                "id": entry.id,
                "part": "GLOBAL" if part is None else part,
                "cited": 0 if counts is None else counts.cited.get(entry.id, 0),
                "helpful": 0 if counts is None else counts.helpful.get(entry.id, 0),
                "harmful": 0 if counts is None else counts.harmful.get(entry.id, 0),
                "helpful_total": now.helpful,
                "harmful_total": now.harmful,
                "status": now.status,
            }
        )
    return rows


def _rate(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render(report: Mapping[str, Any]) -> str:
    """`report` (`round_report`) for people, harm first."""
    ci = report["diff_ci95"]
    shown_ci = "-" if ci is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] pp"
    lines = [
        f"# P4 round {report['round']}: {report['block']} ({report['mode']})",
        "",
        f"- pairs (task, seed) with an outcome in both arms: {report['pairs']}",
        f"- baseline {_rate(report['baseline_rate'])}, RSI {_rate(report['rsi_rate'])}; "
        f"RSI lost {report['rsi_lost']} pairs, won {report['rsi_won']}; "
        f"McNemar exact p = {report['mcnemar_p']:.3g}; bootstrap 95% CI of RSI - baseline "
        f"{shown_ci}",
        "",
        f"## Tasks where RSI lost {HARM_PAIRS} or more pairs",
        "",
    ]
    tasks = report["tasks"]
    if report["harmed"]:
        lines += [
            "| task | pairs | baseline | RSI | RSI lost | RSI won |",
            "|---|---|---|---|---|---|",
        ]
        for task in report["harmed"]:
            row = tasks[task]
            lines.append(
                f"| {task} | {row['pairs']} | {row['baseline']} | {row['rsi']} | "
                f"{row['rsi_lost']} | {row['rsi_won']} |"
            )
    else:
        lines.append("none")
    lines += [
        "",
        "## Per task (paired seeds)",
        "",
        "| task | pairs | baseline | RSI | RSI lost | RSI won |",
        "|---|---|---|---|---|---|",
    ]
    for task, row in tasks.items():
        lines.append(
            f"| {task} | {row['pairs']} | {row['baseline']} | {row['rsi']} | {row['rsi_lost']} | "
            f"{row['rsi_won']} |"
        )
    lines += [
        "",
        "## Entries",
        "",
        "| id | part | cited | helpful | harmful | helpful (all) | harmful (all) | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in report["entries"]:
        lines.append(
            f"| {e['id']} | {e['part']} | {e['cited']} | {e['helpful']} | {e['harmful']} | "
            f"{e['helpful_total']} | {e['harmful_total']} | {e['status']} |"
        )
    counts = report["counts"]
    if counts is not None:
        lines += [
            "",
            f"- promoted: {', '.join(counts['promoted']) or 'none'}",
            "- retired: "
            + (", ".join(f"{k} ({v})" for k, v in counts["retired"].items()) or "none"),
            f"- cited ids naming no readable entry: {', '.join(counts['unknown']) or 'none'}; "
            f"cited but not shown: {', '.join(counts['unseen']) or 'none'}",
        ]
    lines += ["", "## Arms", ""]
    for arm, shares in report["arms"].items():
        tokens = shares["tokens_per_episode"]
        lines.append(
            f"- {arm}: {shares['episodes']} episodes, A-prime voided {shares['voided']} "
            f"({_rate(shares['voided_share'])}), lost to the infrastructure {shares['infra']}, "
            f"tokens per episode {'-' if tokens is None else f'{tokens:.0f}'}"
        )
    for arm, rows in report["unpaired"].items():
        if rows:
            lines.append(f"- unpaired ({arm}): " + "; ".join(rows))
    doc = report["doc"]
    lines += [
        f"- document: {doc['entries']} entries, GLOBAL {doc['global_chars']} characters",
        f"- not run in this version: {', '.join(report['not_run'])}",
        "",
    ]
    return "\n".join(lines)


def final_report(reports: Sequence[Mapping[str, Any]]) -> str:
    """The rounds' and final blocks' reports side by side (`report.md` of the run)."""
    lines = [
        "# P4 run",
        "",
        "| round | block | mode | pairs | baseline | RSI | RSI lost | RSI won | McNemar p "
        "| 95% CI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        ci = r["diff_ci95"]
        shown = "-" if ci is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
        lines.append(
            f"| {r['round']} | {r['block']} | {r['mode']} | {r['pairs']} | "
            f"{_rate(r['baseline_rate'])} | {_rate(r['rsi_rate'])} | {r['rsi_lost']} | "
            f"{r['rsi_won']} | {r['mcnemar_p']:.3g} | {shown} |"
        )
    harmed = {f"{r['block']} {r['mode']}": r["harmed"] for r in reports if r["harmed"]}
    lines += ["", "Harmed tasks (RSI lost >= 2 pairs): " + (str(harmed) if harmed else "none"), ""]
    return "\n".join(lines)
