"""The gap clusters of the skill-writing rounds: the gaps still open after reuse, grouped so that
one new skill could plausibly close a cluster (README: one builder session per cluster).

A fixed rule, no task names: a gap's cluster is its segment's kind (`demo_coverage.segments`) times
its failure class, the class read from the receipts of every attempt at the segment (the coverage
step's fitted call and each reuse alternative): the first receipt that did not reach its target,
mapped by `CLASSES`; an attempt whose receipts all reached but that truth judged off the segment's
end is `accuracy`; a segment never attempted (no skill fits, none tried) is `unfit`. The gap's
class is its attempts' most frequent, ties broken in `CLASSES` order; but `fit` where the relative
rule refused the fit (`demo_coverage.fitting.RELATIVE_RULE`, the report's `refused`) of a skill
written for that gap or that closed it before (`rewrite`): a skill to write again, relative. Other
refusals are listed with the attempts and leave the class as it is (an absolute skill of a kind is
refused on every gap of it). A kind and class's gaps, taken round-robin over their tasks (tasks by
name, a task's gaps by start frame) so that one skill is written for the variety, make up to
`SPLITS` clusters of at most `CAP` gaps each (one session each, side by side: `<kind>-<class>`,
`<kind>-<class>-2`, whose skills' names must end in `_a`, `_b`, so the two never clash); the rest
are `deferred` to the next round, by when an accepted skill may have closed them.

Besides the gaps of segments run from their own start, the chained step's (`<task>.chain.json`,
`demo_coverage.run.chain_reuse`): where the task's plan, run end to end from the reset, breaks
(`<task>/<segment>#chained`: a segment explained from its own start that is not from the state
the calls before it left; one that is an open gap from its own start already is that gap), or,
where every segment chains but the official success is not reached, where the benchmark's stages
fell behind the demonstration's for good (`#stages`, class `stages`). They cluster by the same
rule, their attempts the chain's run and its alternatives' there. Where the chained step kept an
earlier round's deeper chain (`noisy`), the segment its plan's own rollout broke at is a gap too
(cause `noisy chain break`): a flaky segment, for a more robust skill; one round whose chain
holds there has none.

Apart from the clusters: gaps that are infrastructure (`not judged`: an error, a replay or a call
the benchmark ended; README: excluded) and tasks whose ceiling did not latch success in every
repeat although every segment is explained (no segment to write a skill for).

    python clusters.py <reports dir>            # e.g. <glue>/coverage/<run>
    python clusters.py <reports dir> --json
"""

import argparse
import collections
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from roboscript_methods.skill_builder.demo_coverage.report import gaps as coverage_gaps
from roboscript_methods.skill_builder.demo_coverage.report import why

RECEIPTS: Final = ("grasp", "lift", "place", "release", "move", "home")
"""A row's receipts, in the order its call ran them (`demo_coverage.run.explain`)."""
REACHED: Final = ("ok", "clamped")
CLASSES: Final[dict[str, tuple[str, ...]]] = {
    "reach": ("unreachable", "plan_failed"),
    "hold": ("empty_grasp", "lost_hold"),
    "path": ("stopped_short", "deviated"),
    "accuracy": (),
    "refused": ("invalid_args", "rejected_repeat"),
    "budget": ("budget_exhausted",),
    "fit": (),
    "unfit": (),
    "stages": (),
    "ended": (),
}
"""Failure classes, in tie-break order, and the receipt reasons each holds: `reach`, nothing
moved (no IK or no plan); `hold`, the grasp did not hold; `path`, it moved but did not get there;
`accuracy`, every receipt reached and the object ended off; `refused`, the call's arguments;
`budget`, the episode's steps; `fit`, a written skill's fit the relative rule refused (its
attempt carries why); `unfit`, never attempted; `stages`, a chain explained it but the
benchmark's stages stayed behind the demonstration's (`#stages` gaps only); `ended`, the
benchmark ended a chain's episode at it (`#chained` only: nothing was replayed, the plan's own
calls fired its failure query). A reason not listed is `other`."""
INFRA: Final = ("not judged", "replay ended", "the benchmark ended the episode")
"""Causes that are the infrastructure's or the benchmark's, not the skill's (`report.gaps`)."""
CHAIN_INFRA: Final = ("not judged", "replay ended")
"""A chain's causes that are the infrastructure's (`demo_coverage.run.CHAIN_INFRA`); the
benchmark's end is the plan's there (class `ended`)."""
ENDED: Final = "the benchmark ended the episode"
NOISY: Final = "noisy chain break"
"""A chained gap's cause where the task's chain kept an earlier round's deeper one (`noisy`):
the segment its plan's own rollout broke at, which that chain held (a flaky segment, for a more
robust skill), its class from that rollout's receipts and the tries there."""
CAP: Final = 4
SPLITS: Final = 2
"""Gaps per cluster, the cases one session writes for (its case-run budget is 12), and clusters
per kind and class (smaller sessions side by side keep a round short; as many gaps a round as
one cluster of 8)."""


@dataclass(frozen=True)
class Gap:
    """One open gap: `id` is `<task>/<segment>`, as a glue's case runner takes it (`mode`
    `chained` or `stages`: `<task>/<segment>#<mode>`, run from the plan's rollout)."""

    id: str
    task: str
    segment: str
    kind: str
    arm: str | None
    obj: str | None
    target: str | None
    frames: list[int] | None
    cause: str
    failure: str
    attempts: list[dict[str, Any]]
    """Each attempt: its form, skill, first failing receipt (`reason`, `note`) and `why`."""
    mode: str | None = None


@dataclass(frozen=True)
class Cluster:
    id: str
    kind: str
    failure: str
    gaps: list[Gap]
    deferred: list[Gap] = field(default_factory=list)
    suffix: str = ""
    """What the name of a skill its session writes must end in (`_a`, `_b`: a kind and class
    split into clusters side by side, whose briefs are alike), else empty."""
    variant: int = 0
    """Which of the competing sessions on these gaps this is (`driver.compete`: 1, 2, ..., each
    with its own approach and a name suffix ending `_v<n>`); 0: the only one."""

    @property
    def tasks(self) -> list[str]:
        return sorted({g.task for g in self.gaps + self.deferred})


@dataclass(frozen=True)
class Clusters:
    clusters: list[Cluster]
    infra: list[dict[str, Any]]
    """Gaps not judged: `{id, cause, why}`."""
    unattributed: list[dict[str, Any]]
    """Tasks with every segment explained whose ceiling did not latch in every repeat."""

    def record(self) -> dict[str, Any]:
        return {
            "clusters": [asdict(c) | {"tasks": c.tasks} for c in self.clusters],
            "infra": self.infra,
            "unattributed": self.unattributed,
        }


def load(directory: Path) -> dict[str, tuple[dict[str, Any], dict[str, Any] | None]]:
    """A reports directory's per-task coverage report and its reuse report (None: not run); the
    chained step's report, where there is one, under the coverage report's `chain`. The sweep's
    `missing.json` (tasks it could not run) is not a task, nor a name with a further suffix."""
    found = {}
    for path in sorted(directory.glob("*.json")):
        if "." in path.stem or path.name == "missing.json":
            continue
        reuse = path.with_name(f"{path.stem}.reuse.json")
        chain = path.with_name(f"{path.stem}.chain.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        if chain.is_file():
            report["chain"] = json.loads(chain.read_text(encoding="utf-8"))
        found[path.stem] = (
            report,
            json.loads(reuse.read_text(encoding="utf-8")) if reuse.is_file() else None,
        )
    return found


def runs(report: dict[str, Any], reuse: dict[str, Any] | None) -> list[dict[str, Any]]:
    """A task's ceiling runs on its final plan: the chained step's, else the reuse's, else the
    coverage's."""
    chain = report.get("chain")
    if chain is not None and chain.get("ceiling") is not None:
        return list(chain["ceiling"])
    return list((reuse or report)["ceiling"])


def failing(row: dict[str, Any]) -> dict[str, Any]:
    """An attempt's first receipt that did not reach its target, as `{reason, note, failure}`."""
    motions = [m for m in row.get("motions") or [] if isinstance(m, dict)]
    for receipt in [row.get(key) for key in RECEIPTS] + motions:
        if receipt and receipt.get("reason") and receipt["reason"] not in REACHED:
            reason = receipt["reason"]
            failure = next((c for c, held in CLASSES.items() if reason in held), "other")
            return {
                "receipt": next((k for k in RECEIPTS if row.get(k) is receipt), "motion"),
                "reason": reason,
                "note": str(receipt.get("note", ""))[:200],
                "failure": failure,
            }
    return {"receipt": None, "reason": None, "note": "", "failure": "accuracy"}


def failure_class(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "unfit"
    counts = collections.Counter(str(a["failure"]) for a in attempts)
    order = [*CLASSES, "other"]
    return min(counts, key=lambda c: (-counts[c], order.index(c)))


def task_gaps(
    task: str, report: dict[str, Any], reuse: dict[str, Any] | None
) -> tuple[list[Gap], list[dict[str, Any]], dict[str, Any] | None]:
    """A task's open gaps after reuse, its infrastructure gaps, and its unattributed ceiling (or
    None)."""
    closed = {c["segment"] for c in reuse["closed"]} if reuse else set()
    calls = {c["segment"]: c for c in report["calls"]}
    rows = {r["segment"]: r for r in report["explained"]}
    tries: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for t in reuse["tries"] if reuse else []:
        tries[t["segment"]].append(t)
    found, infra = [], []
    for gap in coverage_gaps(report):
        name = gap["segment"]
        if name == "the task" or name in closed:
            continue
        if gap["cause"] in INFRA:
            infra.append({"id": f"{task}/{name}", "cause": gap["cause"], "why": gap["why"]})
            continue
        call = calls[name]
        made = [rows[name]] if name in rows else []
        attempts = [
            {"form": r.get("form", call.get("form")), "skill": r.get("skill"), "why": why(r)}
            | failing(r)
            for r in made + tries.get(name, [])
            if "error" not in r and r.get("cause") not in INFRA
        ]
        refused = refusals((reuse or {}).get("refused", []), name)
        rewrite = [r for r in refused if r["rewrite"]]
        found.append(
            Gap(
                id=f"{task}/{name}",
                task=task,
                segment=name,
                kind=call["kind"],
                arm=call.get("arm"),
                obj=call.get("obj"),
                target=call.get("target"),
                frames=call.get("frames"),
                cause=gap["cause"],
                failure="fit" if rewrite else failure_class(attempts),
                attempts=attempts + refused,
            )
        )
    isolated = {g.segment for g in found}
    chained, broken = chain_gap(task, report, isolated)
    flaky, _ = chain_gap(task, report, isolated, noisy=True)
    found += [chained] if chained else []
    if flaky and (chained is None or flaky.id != chained.id):  # one gap per segment and mode
        found.append(flaky)
    infra += [broken] if broken else []
    ran = runs(report, reuse)
    unattributed = None
    if not found and not any(i["id"].startswith(f"{task}/") for i in infra):
        latched = sum(bool(r["success"]) for r in ran)
        if not ran or latched < len(ran):
            unattributed = {"task": task, "latched": latched, "repeats": len(ran)}
    return found, infra, unattributed


def chain_gap(
    task: str, report: dict[str, Any], isolated: set[str], noisy: bool = False
) -> tuple[Gap | None, dict[str, Any] | None]:
    """The task's chained gap (`report["chain"]["gap"]`, where its plan's rollout broke or its
    stages fell behind), or its infrastructure gap; none where the segment is an open gap from
    its own start (`isolated`) or no skill fits it (that segment's own gap, `unfit`). `noisy`:
    where the plan's own rollout broke when the chain kept an earlier one (`noisy["broke"]`,
    cause `NOISY`), none when it kept none or the infrastructure broke that rollout."""
    chain = report.get("chain")
    if chain is None or chain.get("infra"):
        return None, None
    gap = (chain.get("noisy") or {}).get("broke") if noisy else chain.get("gap")
    if not gap:
        return None, None
    name, mode, k = gap["segment"], gap["mode"], gap["index"]
    row = gap["row"] if noisy else chain["rows"][k]
    gid = f"{task}/{name}#{mode}"
    if mode == "chained" and (name in isolated or row.get("cause") == "no skill"):
        return None, None
    if mode == "chained" and row.get("cause") in CHAIN_INFRA:
        return None, None if noisy else {"id": gid, "cause": row["cause"], "why": why(row)}
    call = next(c for c in report["calls"] if c["segment"] == name)
    tried = [t for t in chain["tries"] if t["segment"] == name and t.get("mode") == mode]
    attempts = [
        {"form": r.get("form", call.get("form")), "skill": r.get("skill"),
         "why": r.get("why") or why(r)} | failing(r)
        for r in [row, *tried] if "error" not in r and r.get("cause") not in CHAIN_INFRA
    ]  # fmt: skip
    refused = refusals(chain.get("refused", []), name)
    rewrite = [r for r in refused if r["rewrite"]]
    cause = NOISY if noisy else "stages behind" if mode == "stages" else str(row.get("cause"))
    failure = ("stages" if mode == "stages" else "ended" if row.get("cause") == ENDED
               else "fit" if rewrite else None)  # fmt: skip
    return Gap(
        id=gid,
        task=task,
        segment=name,
        kind=call["kind"],
        arm=call.get("arm"),
        obj=call.get("obj"),
        target=call.get("target"),
        frames=call.get("frames"),
        cause=cause,
        failure=failure or failure_class(attempts),
        attempts=attempts + refused,
        mode=mode,
    ), None


def refusals(refused: list[dict[str, Any]], segment: str) -> list[dict[str, Any]]:
    """The attempts the relative rule refused on `segment` (a report's `refused`), as `fit`;
    `rewrite`, whether the gap is the skill's to write again (`cover_demos.marked`)."""
    return [
        {"form": f"written {r['skill']}", "skill": r["skill"], "why": r["why"], "receipt": None,
         "reason": "absolute_params", "note": r["why"][:200], "failure": "fit",
         "rewrite": bool(r.get("rewrite"))}
        for r in refused if r.get("segment") == segment
    ]  # fmt: skip


def margins(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Over a task's judged ceiling runs, the benchmark's success checks at the end (each run's
    `conditions`, `demo_coverage.run.ceiling`): the checks that did not hold (`failing`, the most
    runs first: how many, and per float argument the median distance to where the check flips,
    what the run missed it by) and those of the last stage held (`held`: the median margin it
    held by). A distance is in the argument's own units (`ConditionMargin`)."""
    judged = [r for r in runs if not r.get("infra")]
    tally: dict[tuple[str, bool], dict[str, Any]] = {}
    for r in judged:
        for c in r.get("conditions") or []:
            latched = str(c.get("stage", "")).startswith("latched")
            if c["holds"] and not latched:
                continue
            entry = tally.setdefault((c["check"], bool(c["holds"])),
                                     {"check": c["check"], "stage": c.get("stage"), "runs": 0,
                                      "by": collections.defaultdict(list)})  # fmt: skip
            entry["runs"] += 1
            for m in c.get("margins") or []:
                if m.get("flips_at") is not None:
                    entry["by"][m["arg"]].append(float(m["flips_at"]) - float(m["value"]))

    def shown(holds: bool) -> list[dict[str, Any]]:
        found = sorted((e for (_, h), e in tally.items() if h == holds), key=lambda e: -e["runs"])
        return [
            {"check": e["check"], "stage": e["stage"], "runs": e["runs"], "of": len(judged),
             "margins": {a: round(sorted(v)[len(v) // 2], 4) for a, v in sorted(e["by"].items())}}
            for e in found
        ]  # fmt: skip

    return {"failing": shown(False), "held": shown(True)}


def round_robin(gaps: list[Gap]) -> list[Gap]:
    """Gaps ordered one per task in turn (tasks by name, a task's by start frame)."""
    by_task: dict[str, list[Gap]] = collections.defaultdict(list)
    for g in sorted(gaps, key=lambda g: (g.task, (g.frames or [0])[0], g.segment)):
        by_task[g.task].append(g)
    order: list[Gap] = []
    queues = [by_task[t] for t in sorted(by_task)]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order


def cluster(reports: dict[str, tuple[dict[str, Any], dict[str, Any] | None]]) -> Clusters:
    """The clusters of the open gaps of every task in `reports` (`load`)."""
    open_, infra, unattributed = [], [], []
    for task, (report, reuse) in sorted(reports.items()):
        found, broken, ceiling = task_gaps(task, report, reuse)
        open_ += found
        infra += broken
        if ceiling:
            unattributed.append(ceiling)
    keyed: dict[tuple[str, str], list[Gap]] = collections.defaultdict(list)
    for gap in open_:
        keyed[(gap.kind, gap.failure)].append(gap)
    clusters = []
    for (kind, failure), members in sorted(keyed.items(), key=lambda x: (-len(x[1]), x[0])):
        ordered = round_robin(members)
        parts = [ordered[i : i + CAP] for i in range(0, min(len(ordered), CAP * SPLITS), CAP)]
        for n, part in enumerate(parts):
            rest = ordered[CAP * SPLITS :] if n == len(parts) - 1 else []
            cid = f"{kind}-{failure}" + (f"-{n + 1}" if n else "")
            suffix = f"_{chr(ord('a') + n)}" if len(parts) > 1 else ""
            clusters.append(Cluster(cid, kind, failure, part, rest, suffix))
    return Clusters(clusters, infra, unattributed)


def table(found: Clusters) -> str:
    lines = ["| cluster | kind | class | gaps | deferred | tasks |", "|---|---|---|---|---|---|"]
    for c in found.clusters:
        lines.append(
            f"| {c.id} | {c.kind} | {c.failure} | {len(c.gaps)} | {len(c.deferred)} "
            f"| {', '.join(c.tasks)} |"
        )
    lines += ["", f"Not judged (infrastructure, excluded): {len(found.infra)}"]
    lines += [f"- {i['id']}: {i['cause']}" for i in found.infra]
    lines += ["", f"Every segment explained, ceiling short: {len(found.unattributed)}"]
    lines += [f"- {u['task']}: {u['latched']}/{u['repeats']}" for u in found.unattributed]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", type=Path, help="a directory of <task>.json and .reuse.json")
    parser.add_argument("--json", action="store_true", help="the clusters as JSON")
    args = parser.parse_args()
    found = cluster(load(args.reports))
    print(json.dumps(found.record(), indent=1) if args.json else table(found))


if __name__ == "__main__":
    main()
