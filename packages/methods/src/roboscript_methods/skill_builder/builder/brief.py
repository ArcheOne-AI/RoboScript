"""A builder session's brief (README, "Pieces"): the gaps of one cluster as the reports left them,
rendered as the workspace's `BRIEF.md`. Only data: the instructions are the fixed prompt
(`prompts/builder.md`), the same for every cluster and benchmark.

Per gap: its id (`<task>/<segment>`, what `bin/case` takes; `#chained` or `#stages` after it: run
from the plan's rollout), kind, arm, objects, frames, the cause and failure class (`clusters.py`),
the call the coverage step fitted (skill, form, spec) with the receipts of its run (a chained
gap's: its run in the chain), and each alternative reuse tried, with its first failing receipt.
Built from a cluster (`from_cluster`), enriched from the coverage reports when their directory is
given; a cluster whose segments' ends decide the official checks also gets its tasks' official
success conditions and their ceiling runs' checks at the end (`conditions`).

A per-task session's (`from_task`, `prompts/task.md`): the task's segments with their current
calls, its chain row by row and where it broke, each ceiling run's official result with the
benchmark's success checks and their margins, its official success condition as the benchmark's
source states it, its open gaps, and its starting plan (`plan.json`).
"""

import json
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

import roboscript
from roboscript_methods.skill_builder.builder import plans
from roboscript_methods.skill_builder.builder.clusters import Cluster, Gap, margins, runs

REPO: Final = Path(__file__).resolve().parents[6]
"""The repository this file is in: the code the Agent reads (read-only)."""
METHOD: Final = Path(__file__).resolve().parents[1]
"""This method's package directory (`roboscript_methods/skill_builder`)."""
CORE: Final = Path(roboscript.__file__).resolve().parent
"""The core package's directory (`roboscript`)."""
RECEIPTS: Final = ("grasp", "lift", "place", "release", "move", "home")
MEASURED: Final = ("rose_m", "off_m", "dz_m", "turn_deg", "dropped_m", "pos_m", "axis_m", "deg")
MODES: Final = {
    "chained": "each earlier call of the task's plan runs first from the reset, then this one "
    "from where they left the scene; it was explained from its own start, not from there",
    "stages": "every segment was explained in that rollout but the benchmark's stages fell "
    "behind the demonstration's after this one for good; explained when they catch up",
}
NOTE: Final = 400
"""Characters of a receipt's note kept in the brief."""
READ: Final = (
    (METHOD / "builder" / "README.md", "the contract of a written skill and the gate"),
    (METHOD / "builder" / "plugins.py", "how a written skill is loaded and called"),
    (METHOD / "builder" / "static_check.py", "the gate's static checks"),
    (METHOD / "demo_coverage" / "segments.py", "segments: kinds, frames, objects"),
    (METHOD / "demo_coverage" / "fitting.py", "the core's fitters, `fit`'s model"),
    (METHOD / "demo_coverage" / "run.py", "how a segment is judged (`explain`)"),
    (METHOD / "demo_coverage" / "reuse.py", "the alternatives reuse tries"),
    (CORE / "manipulation" / "api.py", "`ActionApi`, `Built`: the handles"),
    (CORE / "manipulation" / "proxy.py", "`ObjectProxy`: an object's geometry"),
    (CORE / "manipulation" / "specs.py", "GraspSpec, PlaceSpec, PathSpec"),
    (CORE / "manipulation" / "fit.py", "fitting specs to a demonstration"),
    (CORE / "contracts" / "skills.py", "`SkillSpec`"),
    (CORE / "spatial", "pose and rotation helpers"),
)
"""What the Agent should read, found from where the packages are installed: shown relative to the
brief's `code_root` where they are inside it (`shown`)."""


class GapBrief(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    """`<task>/<segment>`: the argument of `bin/case` and an entry of `cases.yaml`."""
    task: str
    segment: str
    kind: str
    arm: str | None
    obj: str | None
    target: str | None
    frames: list[int] | None
    cause: str
    failure: str
    call: dict[str, Any] | None = None
    """The coverage step's fitted call (`fitting.Call.record`), when the reports were read."""
    row: dict[str, Any] | None = None
    """The run of that call (`run.explain`'s row)."""
    attempts: list[dict[str, Any]]
    """The coverage step's run and each reuse alternative: form, skill, first failing receipt,
    `why`."""
    mode: str | None = None
    """`chained` or `stages`: the gap is where the plan's rollout from the reset broke."""


class TaskBrief(BaseModel):
    """A per-task session's task: its current state, as the round's final reports
    left it."""

    model_config = ConfigDict(frozen=True)

    task: str
    plan: dict[str, Any]
    """The starting `plan.json` (`plans.prefilled`): the task's current plan."""
    condition: str = ""
    """The official success condition, as the benchmark's source states it."""
    center: str = ""
    """Where successful runs end inside the official checks (`Bench.success_center`)."""
    calls: list[dict[str, Any]] = []
    """The chain's calls (`fitting.Call.record`)."""
    chain: list[dict[str, Any]] = []
    """The chain's rows, one per segment (`run.chained`)."""
    gap: dict[str, Any] | None = None
    to_end: bool = False
    ceiling: list[dict[str, Any]] = []
    """The ceiling's runs on the current plan (`clusters.runs`)."""


class Brief(BaseModel):
    model_config = ConfigDict(frozen=True)

    cluster_id: str
    kind: str
    failure: str
    gaps: list[GapBrief]
    accepted: list[Path] = []
    """Skills accepted in earlier rounds (each a directory): readable examples, and names taken."""
    code_root: Path = REPO
    also_open: list[GapBrief] = []
    """The cluster's other gaps (`Cluster.deferred`, same kind and class): listed so the skill is
    designed for the class, not run by `bin/case` (the driver's rerun tries accepted skills on
    every gap)."""
    relay: str = ""
    """A relay session's notes on the session it continues (the driver's), else empty."""
    name_suffix: str = ""
    """What the skill's name must end in (`Cluster.suffix`), else empty."""
    conditions: dict[str, list[str]] = {}
    """Per task, its official success condition and its ceiling runs' checks at the end
    (`condition_lines`): a cluster's whose segments' ends decide them."""
    task: TaskBrief | None = None
    """A per-task session's task, else None."""
    strategy: str = ""
    """A competing session's approach (`driver.STRATEGIES`, `TASK_STRATEGIES`: other sessions
    work the same gaps or task beside it with another), else empty."""

    @classmethod
    def from_cluster(
        cls,
        cluster: Cluster,
        accepted: list[Path],
        reports: Path | None = None,
        conditions: dict[str, str] | None = None,
        centers: dict[str, str] | None = None,
    ) -> "Brief":
        """The cluster's gaps (not its deferred ones); with `reports` (a directory of
        `<task>.json` and `<task>.reuse.json`), each with its fitted call and the coverage row; with
        `conditions` (per task, its official success condition), those, the task's success
        center (`centers`, where it has one) and its ceiling runs' checks at the end."""
        loaded = load_reports(reports, {g.task for g in cluster.gaps + cluster.deferred})
        said = {
            task: [
                *condition_text(text),
                *center_text((centers or {}).get(task, "")),
                *condition_lines(ceiling_runs(reports, task)),
            ]
            for task, text in sorted((conditions or {}).items())
        }
        return cls(
            cluster_id=cluster.id,
            kind=cluster.kind,
            failure=cluster.failure,
            gaps=[gap_brief(g, loaded.get(g.task)) for g in cluster.gaps],
            accepted=[Path(p) for p in accepted],
            also_open=[gap_brief(g, loaded.get(g.task)) for g in cluster.deferred],
            name_suffix=cluster.suffix,
            conditions=said,
        )

    @classmethod
    def from_task(
        cls,
        cluster: Cluster,
        accepted: list[Path],
        reports: Path,
        plan: dict[str, Any],
        condition: str = "",
        name: str | None = None,
        center: str = "",
    ) -> "Brief":
        """A per-task session's brief: `cluster` holds the task's open gaps (`bin/case`'s), its
        state from `reports` (the round's final reports), its starting `plan`; `name`, the task
        (else the cluster's id after `task-`: a competing session's id carries more)."""
        task = name or cluster.id.removeprefix("task-")
        loaded = load_reports(reports, {task})
        report = loaded.get(task) or {}
        chain = report.get("chain") or {}
        return cls(
            cluster_id=cluster.id,
            kind=cluster.kind,
            failure=cluster.failure,
            gaps=[gap_brief(g, report or None) for g in cluster.gaps],
            accepted=[Path(p) for p in accepted],
            also_open=[gap_brief(g, report or None) for g in cluster.deferred],
            name_suffix=cluster.suffix,
            task=TaskBrief(
                task=task,
                plan=plan,
                condition=condition,
                center=center,
                calls=chain.get("calls") or report.get("calls", []),
                chain=chain.get("rows", []),
                gap=chain.get("gap"),
                to_end=bool(chain.get("to_end")),
                ceiling=ceiling_runs(reports, task),
            ),
        )

    @property
    def gap_ids(self) -> list[str]:
        return [g.id for g in self.gaps]


def load_reports(reports: Path | None, tasks: set[str]) -> dict[str, dict[str, Any]]:
    """Each task's coverage report, its chained step's under `chain`."""
    loaded: dict[str, dict[str, Any]] = {}
    for task in sorted(tasks) if reports is not None else []:
        assert reports is not None
        loaded[task] = json.loads((reports / f"{task}.json").read_text(encoding="utf-8"))
        if (chain := reports / f"{task}.chain.json").is_file():
            loaded[task]["chain"] = json.loads(chain.read_text(encoding="utf-8"))
    return loaded


def ceiling_runs(reports: Path | None, task: str) -> list[dict[str, Any]]:
    """The task's ceiling runs on its final plan (`clusters.runs`), [] without reports."""
    if reports is None or not (reports / f"{task}.json").is_file():
        return []
    report = load_reports(reports, {task})[task]
    reuse = reports / f"{task}.reuse.json"
    return runs(report, json.loads(reuse.read_text(encoding="utf-8")) if reuse.is_file() else None)


def condition_text(text: str) -> list[str]:
    return ["Official success condition (the benchmark's source, read only):", "", "```python",
            *text.splitlines(), "```", ""] if text else []  # fmt: skip


def center_text(text: str) -> list[str]:
    """The success center (`Bench.success_center`): the aim, not what the gate judges."""
    return ["Success center (where successful runs of this task end, against the "
            "demonstration's end; aim there, the gate still judges the official success first):",
            "", *text.splitlines(), ""] if text else []  # fmt: skip


def condition_lines(ceiling: list[dict[str, Any]]) -> list[str]:
    """Each ceiling run's official result and the benchmark's success checks at its end: which
    held, which did not, and per float argument the value at which the check would flip; then
    the checks most runs ended without (`clusters.margins`)."""
    lines = []
    for k, r in enumerate(ceiling, 1):
        if r.get("infra"):
            lines.append(f"- ceiling run {k}: not judged (infrastructure)")
            continue
        st = r.get("stages") or {}
        stages = f"; stages left {st.get('left')} of {st.get('of')}" if st else ""
        first = r.get("first_failure")
        failed = f"; first failure {first['segment']} ({first['reason']})" if first else ""
        lines.append(f"- ceiling run {k}: success {bool(r.get('success'))}{stages}{failed}")
        for c in r.get("conditions") or []:
            by = "; ".join(f"{m['arg']} {m['value']:g} -> flips at "
                           + ("none found" if m.get("flips_at") is None else f"{m['flips_at']:.4g}")
                           for m in c.get("margins") or [])  # fmt: skip
            said = "holds" if c["holds"] else "does not hold"
            lines.append(f"  - {c['stage']}: `{c['check']}` {said}" + (f"; {by}" if by else ""))
    found = margins(ceiling)["failing"]
    if found:
        lines += ["", "Checks the runs ended without (median distance to flipping, per argument):"]
        lines += [f"- `{f['check']}` ({f['stage']}): {f['runs']}/{f['of']} runs; "
                  + (", ".join(f"{a} {d:+.4g}" for a, d in f["margins"].items()) or "no margin")
                  for f in found]  # fmt: skip
    return lines


def gap_brief(gap: Gap, report: dict[str, Any] | None) -> GapBrief:
    call = row = None
    if report is not None:
        call = next((c for c in report["calls"] if c["segment"] == gap.segment), None)
        rows = report["explained"] if gap.mode is None else report.get("chain", {}).get("rows", [])
        row = next((r for r in rows if r["segment"] == gap.segment), None)
    return GapBrief(
        id=gap.id,
        task=gap.task,
        segment=gap.segment,
        kind=gap.kind,
        arm=gap.arm,
        obj=gap.obj,
        target=gap.target,
        frames=gap.frames,
        cause=gap.cause,
        failure=gap.failure,
        call=call,
        row=row,
        attempts=gap.attempts,
        mode=gap.mode,
    )


def receipts(row: dict[str, Any]) -> list[str]:
    lines = []
    for key in RECEIPTS:
        if (r := row.get(key)) is not None:
            lines.append(
                f"  - {key}: {r.get('reason')} ({r.get('steps')} steps): "
                f"{str(r.get('note', ''))[:NOTE]}"
            )
    measured = [f"{k} {row[k]}" for k in MEASURED if k in row]
    if measured:
        lines.append(f"  - truth measured: {', '.join(measured)}")
    return lines


def render(brief: Brief) -> str:
    """`BRIEF.md`: the cluster, its gaps with the reports' evidence, where to read the code (a
    per-task session's: `render_task`)."""
    if brief.task is not None:
        return render_task(brief, brief.task)
    out = [
        f"# Brief: cluster `{brief.cluster_id}`",
        "",
        f"Segment kind: `{brief.kind}`. Failure class: `{brief.failure}` (the first receipt that "
        "did not reach its target, across every attempt; `accuracy`: every receipt reached and "
        "truth found the object off the segment's end).",
        "",
        f"{len(brief.gaps)} gaps. Each id below is what `bin/case` and `cases.yaml` take.",
        "",
    ]
    if brief.name_suffix:
        out += [
            f"Your skill's name must end in `{brief.name_suffix}` (e.g. `guided_setdown"
            f"{brief.name_suffix}`): another session writes a skill for this kind and class "
            "beside you, and two skills cannot share a name. `bin/case` refuses any other.",
            "",
        ]
    if brief.relay:
        out += [brief.relay, ""]
    out += strategy_section(brief)
    for g in brief.gaps:
        out += gap_section(g)
    if brief.also_open:
        out += [
            "## Same class, also open",
            "",
            f"{len(brief.also_open)} more gaps of this cluster's kind and class. `bin/case` does "
            "not run them in this session; design the skill for them too: every accepted skill "
            "is tried on every gap of every task in the driver's rerun.",
            "",
        ]
        for g in brief.also_open:
            out += gap_section(g)
    if brief.conditions:
        out += [
            "## Official success conditions",
            "",
            "A demonstration's end may lie just across the official boundary: where a "
            "segment's end decides these checks, aim for the middle of the success region, not "
            "for the demonstration's end pose.",
            "",
        ]
        for task, lines in brief.conditions.items():
            out += [f"### `{task}`", "", *lines, ""]
    out += code_section(brief)
    return "\n".join(out)


def strategy_section(brief: Brief) -> list[str]:
    """A competing session's approach (`Brief.strategy`), else nothing."""
    if not brief.strategy:
        return []
    return ["## Your approach", "", brief.strategy, "",
            "Other sessions work the same gaps beside you, each with another approach. The driver "
            "gates every one and keeps each skill or plan that helps: follow your approach rather "
            "than guessing theirs.", ""]  # fmt: skip


def code_section(brief: Brief) -> list[str]:
    out = [
        "## Code to read (read-only)",
        "",
        f"Repository: `{brief.code_root}`.",
        "",
    ]
    out += [f"- `{shown(path, brief.code_root)}`: {what}" for path, what in READ]
    if brief.accepted:
        out += ["", "Skills accepted in earlier rounds (their names are taken):", ""]
        out += [f"- `{p}`" for p in brief.accepted]
    out.append("")
    return out


def shown(path: Path, root: Path) -> str:
    """`path` relative to `root` where it is inside it, else absolute."""
    root = root.resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def render_task(brief: Brief, task: TaskBrief) -> str:
    """A per-task session's `BRIEF.md`."""
    out = [
        f"# Brief: task `{task.task}`",
        "",
        f"Target: {plans.TARGET_TEXT}. Your skills' names must end in `{brief.name_suffix}`.",
        "",
        *([brief.relay, ""] if brief.relay else []),
        *strategy_section(brief),
        *condition_text(task.condition),
        *center_text(task.center),
        "## The demonstration's segments and the current calls",
        "",
        "In the order the chain runs them. `plan.json` starts as this plan.",
        "",
    ]
    for c in task.calls:
        before = ", ".join(b["skill"] for b in c.get("before", []))
        out.append(
            f"- `{c['segment']}`: kind `{c.get('kind')}`, arm `{c.get('arm')}`, object "
            f"`{c.get('obj')}`, target `{c.get('target')}`, frames {c.get('frames')}; call "
            f"`{c.get('skill')}` form `{c.get('form')}`"
            + (f", params {json.dumps(c['params'], sort_keys=True)}" if c.get("params") else "")
            + (f", before: {before}" if before else "")
        )
    gap = task.gap
    said = (
        "reaches the end" if task.to_end
        else f"breaks at `{gap['segment']}` ({gap['mode']})" if gap
        else "was not judged"
    )  # fmt: skip
    out += ["", f"## The chain: it {said}", ""]
    for r in task.chain:
        if r.get("applicable") is False:
            out.append(f"- `{r['segment']}`: not reached")
            continue
        why = ("explained" + (" (by the benchmark's stages)" if r.get("explained_by") else "")
               if r.get("explained") else f"not explained: {r.get('cause')} "
               f"({r.get('mechanism')})")  # fmt: skip
        out.append(f"- `{r['segment']}` with `{r.get('skill')}`: {why}")
        if r.get("stages"):
            st = r["stages"]
            out.append(f"  - stages left {st['before']} before, {st['after']} after, of {st['of']}")
        for m in r.get("transitions") or []:
            how = "ok" if m.get("ok") else m.get("reason") or m.get("act_error")
            out.append(f"  - transition `{m.get('transition')}`: {how}")
        out += receipts(r)
    out += ["", "## The ceiling: the current plan run from the reset", "",
            *condition_lines(task.ceiling), ""]  # fmt: skip
    if brief.gaps or brief.also_open:
        out += ["## Open gaps of this task", "",
                "`bin/case SKILL GAP` runs a skill of yours on these (as a cluster builder does).",
                ""]  # fmt: skip
        for g in brief.gaps + brief.also_open:
            out += gap_section(g)
    out += ["## The starting plan (`plan.json`)", "", "```json",
            json.dumps(task.plan, indent=1, sort_keys=True), "```", ""]  # fmt: skip
    out += code_section(brief)
    return "\n".join(out)


def gap_section(g: GapBrief) -> list[str]:
    out = [
        f"## `{g.id}`",
        "",
        f"- kind `{g.kind}`, arm `{g.arm}`, object `{g.obj}`, target `{g.target}`, "
        f"frames {g.frames}",
        f"- cause: {g.cause}; class: {g.failure}",
    ]
    if g.mode is not None:
        out.append(f"- run from the plan's rollout ({MODES[g.mode]})")
    if g.call is not None:
        spec = json.dumps(g.call.get("spec"), sort_keys=True)
        out.append(
            f"- fitted call: `{g.call.get('skill')}` form `{g.call.get('form')}` spec {spec}"
        )
    if g.row is not None:
        out.append("- its run:" if g.mode is None else "- its run in the chain:")
        out += receipts(g.row)
        if g.row.get("stages"):
            st = g.row["stages"]
            out.append(f"  - benchmark stages left: {st['before']} before, {st['after']} after, "
                       f"of {st['of']}")  # fmt: skip
    if g.attempts:
        out.append("- attempts (the first call, then each reuse alternative):")
        for a in g.attempts:
            failed = (
                f"the relative rule refused its fit: {str(a.get('note', ''))[:NOTE]}"
                if a.get("failure") == "fit"
                else f"{a.get('receipt')} {a.get('reason')}: {str(a.get('note', ''))[:NOTE]}"
                if a.get("reason")
                else "every receipt reached its target"
            )  # fmt: skip
            measured = [p for p in str(a.get("why", "")).split("; ") if p.startswith(MEASURED)]
            if measured:
                failed += f"; truth measured: {', '.join(measured)}"
            out.append(f"  - `{a.get('skill')}` form `{a.get('form')}`: {failed}")
    out.append("")
    return out
