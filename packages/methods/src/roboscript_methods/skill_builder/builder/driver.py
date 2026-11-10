"""The skill-writing driver (README: "The gate", "Budgets"): rounds of coverage + reuse, gap
clusters, one builder session per cluster in parallel, the gate on what the sessions wrote,
acceptance, rerun and report, with no human in the loop. Generic: a benchmark's glue supplies the
`Bench` (its sweeps and case runs, e.g. a `build.py`).

A run directory, resumable at every step (a step whose output exists is not redone):

    <run>/run.json                  tasks, the limits and the skills at start
    <run>/accepted/<name>/          every skill accepted so far (the skills each sweep runs with)
    <run>/round-<k>/coverage/       <task>.json + <task>.reuse.json with the accepted skills
                    clusters.json   the open gaps' clusters (`clusters.py`)
                    sessions/<id>/  the workspace (`work/`, Codex's beside it as `work.codex*`),
                                    per attempt `attempt-<a>.json` and `attempt-<a>/` (its case
                                    runs, the gate's runs of what it wrote)
                    gate.json       per skill written: static findings, its cases, verdict
                    seeds.json      the seeds this round took (`seeds/<name>`: their snapshots)
                    candidates/     the skills that passed the gate's steps 1-3
                    rerun/          coverage + reuse with the accepted and every candidate
                    rerun-confirm-{before,after}/  the tasks that lost, swept again, paired
                    kept/           the candidates no confirmed loss dropped (not yet accepted)
                    after[-<i>]/    ... with the accepted and the kept, when a candidate was
                                    dropped (and its own confirmation sweeps)
                    variants.json   the competing sessions per cluster and per task it ran with
                    tasks-early/    the per-task closure's early stage: accepted/ (the skills
                                    at the round's start), tasks.json, sessions/, <task>/ trials,
                                    stage.json
                    tasks/          its late stage (the same), seeds/, closure.json (both)
                    summary.json, summary.md

A cluster's session that ends at its budget (wall clock, tokens or case runs spent) with no skill
past the gate's steps 1-3 but with progress (`progressed`) is relayed: a new session in the same
workspace, told what the last one left, with `RELAY_SCALE` times the budget; at most
`Limits.relays` per cluster and round. A session whose model stream stalled is restarted with what
was left of its budget (`STALL_RETRIES`), not a relay. The driver never imports what an Agent
wrote: the bench validates a snapshot in a child process and runs every case and sweep job
sandboxed and bounded (`CASE_TIMEOUT_S`, `SWEEP_TASK_TIMEOUT_S`); the snapshot's content hash is
taken before it is validated and checked again before it is copied to `candidates/` and
`accepted/`.

Infrastructure never judges a skill: a gate whose case runs all broke on it is `hold` (not judged):
run again once after the round's sessions, then carried to the next round's gate; a task a sweep
left without a report is not a loss, unless the job was stopped inside a written skill's `act` (the
bench's `missing.json`: `stuck`), which blames that skill. A task that loses is swept again, both
arms paired (`rerun-confirm-*`), and blames only if it loses again (the planner's outcome can flip
run to run). A chained step starts from the deepest chain an earlier round ended with that still
fits the task's plan (`finals`) and keeps it where its own explains fewer (`noisy`, listed in the
summary), so one round's noisy break is not carried into the next; the regression compares each
arm's own chained rollout (`own_chain`), since a candidate that changes a task's plan leaves its
arm no earlier chain to keep. The sweep with the kept candidates is compared the same way before it
becomes the next round's baseline. Coverage comes first: a candidate past the gate no task used is
kept (one more alternative), and one a confirmed loss blames is dropped only where its losses
outweigh its gains (`ledger`, `score`: segments explained from their own start and chained, ceiling
repeats latched and stages done, per task, paired): a skill that raises any of them is kept unless
that. Skills another run's gate passed may be seeded into a round (`seeds`): validated again, then
the round's regression. On start it ends the process groups sessions of an earlier, stopped driver
left (`reap`). Round 0's coverage is imported from a finished sweep when told (`import_baseline`),
else swept; round k's is round k-1's final state, unless that was swept in another format or by
other code (`stale`): then it is swept again (`rebaselined.json`), so the round's before and after
are the same code.

What a round measures (`state`, `totals`; summary.md's table): the segments explained in the
chained rollout (each task's plan run end to end from the reset, `<task>.chain.json`), the tasks
whose chain reaches the end, the ceiling's official successes, the benchmark stages its runs did,
the segments explained from their own start. Gaps are both kinds (`clusters.py`): segments run from
their own start, and where a task's chain breaks or its stages fall behind
(`<task>/<segment>#chained`, `#stages`, whose case runs roll the plan out first); a chained gap's
transfer is its segment's own. The run stops after `ROUNDS` rounds, after a round in which none of
those grew (`GROWS`; unless it holds a skill over or accepted a task plan), after one that left
every task at its target (`met`), or before a round when `<run>/STOP` exists. Run to the target
(`until_target`): no round cap unless one is given, and only every task at its target or `STOP`
ends the run; a round in which nothing grew is logged (`no-growth.json`). It also ends after a
round that left every task at its target or paused outside skill writing (`blocked_outside`, before
and after the round) and opened no session of either kind (`all_paused`: the next round would do
the same), said in its summary and on the driver's output, unless the pauses no longer hold
(`Driver.resumes`: another sweep code, whose next round sweeps again, or other `--outside` causes);
a resumed run goes on only where the cause has cleared, and otherwise stops again at once.

Competing sessions: each gap cluster is worked by `variants` sessions side by side (`compete`: each
its own approach, `STRATEGIES`, and name suffix `_v<n>`), each gated, each skill that helps kept;
every session is seated (`Seats`, `sessions_at_once`) by rank: the early per-task sessions, then
the clusters holding a gap where a task's chain breaks, then the rest.

The per-task closure (`per_task`, README "Per-task closure"): `task_variants` builder sessions per
task below its target (`met`: `plans.TARGET_TEXT`, a ceiling confirmed), side by side; those that
chain to the end from the round's start (`early`, against a snapshot of the skills accepted then),
the rest after its acceptance (`closure`); each on the whole task: its demonstration's segments,
its chain and first break, its ceiling's official results with the benchmark's success checks and
their margins, its current plan, the skill library. It writes the task's plan (`builder.plans`: per
segment a skill, form and params fitted from the demonstration, deltas to them, generic transitions
between segments) and, where needed, new generic skills, which pass the same gate as a cluster's
(steps 1-3 where they name an open gap; then the next round's regression as seeds). The plan is
judged by two fresh plan trials (`Bench.plan_trial`: the task's own rollout and its ceiling's
repeats), the task's current plan and the session's, paired: accepted where the ceiling's official
successes rise and its stages do not fall, however deep the chain against the demonstration's
segment ends goes, else where the chain or the successes or stages rise and none falls (`better`; a
small rise, `marginal`, only once a second trial of each, pooled, confirms it); of a task's
competing accepted plans the best (`best_plan`), judged again by `better` when the stage closes
(`rejudged`). Plans are written after the round's regression. An accepted plan
(`<run>/plans/<task>.json`) is that task's plan in every later sweep's chained step; the next
round's coverage sweeps the chained step of the tasks whose plan changed (`replanned`). Plans are
P1's evidence only: never in the skillset, never handed to S1, S2 or P3. A task whose every ceiling
run stops first at a cause no skill or plan fixes (`OUTSIDE`: the executor's table-top refusal, the
episode's step budget, a kind no core entry serves) gets no session of either kind
(`blocked_outside`), judged again every round. Every number is a pipeline constant, the same for
every benchmark and task.
"""

import asyncio
import contextlib
import functools
import hashlib
import heapq
import itertools
import json
import os
import shutil
import signal
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Protocol

from roboscript.contracts.receipt import ReceiptReason
from roboscript.learning.ledger import wilson
from roboscript.settings.env import source_sha
from roboscript_methods.skill_builder.builder import plans
from roboscript_methods.skill_builder.builder.clusters import (
    Cluster,
    Clusters,
    cluster,
    load,
    margins,
    runs,
    table,
    task_gaps,
)
from roboscript_methods.skill_builder.builder.plans import (
    CONFIRMED_REPEATS,
    FIRST_REPEATS,
    TARGET_TEXT,
    at_target,
    better,
    confirmed,
    counted,
    first_counts,
    rate,
    rises,
    share,
    trial_state,
)
from roboscript_methods.skill_builder.builder.plugins import case_mode, own_gap
from roboscript_methods.skill_builder.demo_coverage.fitting import content_hash


@dataclass(frozen=True)
class Limits:
    """A session's budget (README, Budgets) and the relays a cluster gets."""

    case_runs: int = 12
    wall_s: float = 1.5 * 3600.0
    tokens: int = 5_000_000
    relays: int = 4

    def scaled(self, factor: int) -> "Limits":
        return replace(
            self,
            case_runs=self.case_runs * factor,
            wall_s=self.wall_s * factor,
            tokens=self.tokens * factor,
        )


LIMITS: Final = Limits()
ROUNDS: Final = 10
"""Rounds at most; a run stops earlier after a round in which none of `GROWS` grew
(`Driver.run_all`), or at a `STOP` file in the run directory. `Driver(rounds=None,
until_target=True)` runs until every task is at its target instead."""
GROWS: Final = ("chained", "to_end", "success", "stages_done", "explained")
"""A round's totals (`totals`) a next round is run for while one grows: the segments explained in
the chained rollout, the tasks whose chain reaches the end, the ceiling's official successes, the
benchmark stages its runs did, the segments explained from their own start."""
SWEEP_FORMAT: Final = 7
"""The current format of a sweep's reports. They hold the chained step's `<task>.chain.json`, with
its sweep code, the ceiling's stages, its success checks and its four repeats (4 more where the
first 4 latched 2, `plans.due`), and the fits the relative rule refused (`refused`). The chained
step takes the run's task plans (`plans`) and judges a segment by the benchmark's stages where it
ended off the demonstration's end (`run.stage_met`). The coverage step is given the earlier
coverages, whose place targets it keeps. The rerun's and the kept ones' chained steps take the
coverage's measurement of a seed it ran as it is (`reuse_before`).
A round whose last round's reports were swept in another format, or by other code, sweeps its
coverage again: before and after are the same code. "Other code" is the bench's `sweep_code`
where both rounds recorded it (a hash of the files that make a sweep's reports, not of the
driver's or the scheduler's), else `source_sha` where both are known. A change to how the bench
calls its sweep (its arguments, which inputs a step gets) raises this format."""
STOP: Final = "STOP"
"""A file of this name in the run directory halts the driver before its next round."""
RELAY_SCALE: Final = 1
"""A relay's budget, in times the first session's (each as long as the first keeps the round
short; a hard cluster gets more relays instead)."""
GRACE_S: Final = 600.0
"""Past a session's wall clock before the driver cancels it (the session keeps its own)."""
GATE_CASES: Final = 12
"""A candidate's `cases.yaml` gaps the driver runs, at most (the first, in its order)."""
REPEAT_INFRA: Final = 1
"""A gate case run the infrastructure broke (`infra` in its row) is run again this many times."""
CASE_TIMEOUT_S: Final = 1800.0
"""A case run longer than this is stopped and counts as infrastructure, not as a case run."""
CEILING_RUN_S: Final = 660.0
"""A bound on one ceiling run, a long task's plan trial included: what each run of a ceiling's
confirmation round adds to a job's bound."""
SWEEP_TASK_TIMEOUT_S: Final = 7200.0 + (CONFIRMED_REPEATS - FIRST_REPEATS) * CEILING_RUN_S
"""A sweep's job (one task's coverage, or its reuse) longer than this is stopped: the task is
missing from that sweep. 7200 s for 4 ceiling runs, and room for the confirmation's 4 more."""
STALL_RETRIES: Final = 2
"""A session whose model stream stalled (`stalled`) is restarted, with what is left of its
budget, this many times at most; not a relay."""
IMPROVES: Final[dict[str, int]] = {
    "off_m": -1,
    "dz_m": -1,
    "pos_m": -1,
    "axis_m": -1,
    "dropped_m": -1,
    "rose_m": 1,
    "deg": -1,
    "turn_deg": -1,
}
"""What truth measures on a case run (`demo_coverage.run.explain`), and which way is better
(`dz_m` by its magnitude)."""
PROGRESS_FRACTION: Final = 0.2
PROGRESS_M: Final = 0.005
PROGRESS_DEG: Final = 2.0
"""A measure improved when a later case run on the same gap beats the first by this fraction of
it and by at least this much (metres, degrees)."""
LAST_MESSAGE: Final = 3000
"""Characters of a session's last message a relay is told."""
TRANSFER_GAPS: Final = 2
"""Explained gate cases whose transfer the driver runs (`Bench.transfer`), at most."""
GATE_RERUN: Final = 1
"""A gate case (or a demonstration's transfer case) judged and not explained is run again this
many times, explained if any run explains it (both rows kept): the planner and the simulator's
contacts flip outcomes run to run."""
TRANSFER_REJECT: Final = 2
"""Judged transfer cases a skill must fail, all of them, to be rejected as bound to its
demonstration's scene; fewer, all failing: it passes `transfer_weak` (coverage first: one more
alternative is harmless)."""
NO_CASE: Final = "no case in cases.yaml is an open gap"
CONDITION_KINDS: Final = ("place", "carry", "release", "touch")
CONDITION_CLASSES: Final = ("accuracy", "stages")
"""The clusters whose briefs carry their tasks' official success conditions and the ceiling's
margins (a demonstration's end may lie just across the official boundary: aim for the middle
of the success region): where a segment's end decides the official checks."""
TRIAL: Final = "#plan"
"""A per-task session's plan trial, as its case channel names it (`<task>/#plan`)."""
STRATEGIES: Final = (
    "Build on the existing entry that came closest in the attempts above: wrap or extend it (its "
    "approach, its fit) and change only what its receipts and truth's measures show went wrong.",
    "Do not wrap the entry that failed: compose the motion yourself from the action API's "
    "primitives (approach, contact, retreat as waypoints of their own, relative to the object), "
    "shaped by what the receipts and truth measured.",
    "Aim for robustness over the demonstration's exact path: generous clearances, a slow final "
    "approach, waypoints whose receipts confirm each step, so that it holds from whatever state "
    "the calls before it left.",
)
"""The approaches of a gap cluster's competing sessions (`compete`, `Driver.variants`), one each,
in variant order (several sessions per cluster at once, every one gated, each skill that helps
kept)."""
TASK_STRATEGIES: Final = (
    "Change the plan first: deltas (`adjust`) toward the middle of the success region and "
    "generic transitions (`before`); write a new skill only where no plan change can do it.",
    "Fix the weakest segment first: where the chain breaks or the official check fails, write a "
    "new generic skill for that segment, then tune the plan around it.",
)
"""The approaches of a task's competing per-task sessions (`Driver.task_variants`)."""
EARLY: Final = 0
BLOCKING: Final = 1
OTHER: Final = 2
SPARE: Final = 3
"""Session ranks (`Seats`: a free seat goes to the lowest rank waiting, then the first to ask):
the first per-task session of each task that chains to the end below its
target (`Driver.early`), the clusters holding a gap where a task's chain breaks (`prioritized`),
the other clusters and the late per-task sessions, and last a task's further competing sessions
(a second plan session per task when seats allow)."""
DECISIVE: Final = 2
"""A plan trial's rise over the current plan's is decisive when its chain went this many segments
deeper, or this many more of the ceiling's repeats latched, or both rose; a smaller one is
confirmed by a second trial of each plan, the two pooled (`marginal`: one trial of 4 repeats per
side often accepts a plan no better than the current one)."""


@dataclass(frozen=True)
class Cost:
    slot_s: float = 0.0
    """GPU slot-seconds (a job holds one slot of one GPU)."""
    jobs: int = 0


class Bench(Protocol):
    """A benchmark's glue: what the driver runs on its simulator."""

    case_slots: int
    """Case runs at once: caps the concurrent sessions."""

    def tasks(self) -> list[str]: ...

    async def sweep(
        self,
        out: Path,
        skills: list[Path],
        tasks: list[str] | None = None,
        prior: Path | None = None,
        earlier: Sequence[Path] = (),
        plans: Path | None = None,
        reuse_before: bool = False,
    ) -> Cost:
        """Coverage + reuse + the chained step (and their ceilings) of every task (or of `tasks`)
        into `out` with the written skills under `skills` (`plugins.skill_dirs`); tasks already
        there are kept. Each task's job writes only a directory of its own; tasks left without a
        report are listed in `<out>/missing.json` as `{task: {"stuck": <skill> | null}}`, `stuck`
        the written skill whose `act` the job was stopped in; sweeps into one `out` run at once
        (`Driver.planned_sweep`), so another's entries are kept for tasks still missing. `prior`:
        reports whose chain fixes each task's chained step starts from (the round's coverage);
        `earlier`: the earlier rounds' final reports, newest first (`Driver.finals`), whose deepest
        chain still fitting a task's plan it starts from instead, and holds its chain to. `plans`:
        the run's accepted task plans (`<task>.json`, `builder.plans`), each applied in its task's
        chained step. `reuse_before` (the round's first sweeps against its coverage,
        `Driver.sweep_once`, not its confirmations): a task's chained step may take the coverage's
        measurement of a plan it would only run again unchanged, so the comparison reads no noise
        there. A bench may also give `sweep_code()`, a hash of the files that make these reports
        (`files_hash`): rounds are compared by it (`Driver.stale`), so that a change to the driver
        or the scheduling sweeps nothing again."""

    async def case(
        self, gap: str, skill: Path, out: Path, plan: Path | None = None
    ) -> dict[str, Any]:
        """One gap (`<task>/<segment>`) fitted and explained with the skill in `skill`, sandboxed
        (writing only a fresh directory of its own, never `skill`), within `CASE_TIMEOUT_S` (a
        gap with a mode, `<task>/<segment>#chained` or `#stages`: from the rollout of the plan
        `plan`'s reports hold, the round's coverage, with the accepted skills): its row, with
        `explained`; with `infra: true` and `error` when the infrastructure broke it or
        it timed out outside the skill's `act` (a timeout inside it is the skill's failure)."""

    async def transfer(
        self, gap: str, skill: Path, out: Path, demos: list[int] | None = None
    ) -> list[dict[str, Any]]:
        """The skill's `params`, fitted on the gap's segment of the task's demonstration, run on
        the corresponding segment of other demonstrations of the task whose objects lie
        elsewhere (sandboxed, like `case`; only those `demos` when given): one row per such
        demonstration (`demo`), `explained` and `infra` as `case`'s; [] when none differs enough
        to tell (README, The gate)."""

    async def core_transfer(self, gap: str, out: Path) -> list[dict[str, Any]]:
        """`transfer` of the core's own entry for the gap's segment (as the core fitted it,
        moved the same way): the reference a skill's transfer is held to."""

    async def validate(self, skill: Path) -> dict[str, Any]:
        """Gate step 1 on a skill directory, in a child process: `{ok, name, kinds, cases,
        findings}`. The driver never imports a written skill."""

    async def plan_trial(
        self, task: str, plan: Path, out: Path, reports: Path, skills: list[Path]
    ) -> dict[str, Any]:
        """The per-task closure's plan trial, sandboxed and bounded like a case: the task plan
        `plan` applied to the task's coverage and reuse in `reports`, its one rollout from the
        reset (no alternative tried, no earlier chain) and its ceiling's repeats, with the written
        skills under `skills`: the chain report (`rows`, `explained`, `to_end`, `ceiling`, `plan`);
        `infra: true` and `error` when the infrastructure broke it. A bench may also give
        `success_condition(task) -> str`, the task's official success condition as its source
        states it (the per-task brief's), and `success_center(task) -> str`, where successful runs
        of the task end inside the official checks against where the demonstration ends (the
        briefs' target is that center, not the demonstration's end), else empty; neither changes
        what a trial or the gate judges (official success first)."""

    def accept(self, skill: Path, provenance: dict[str, Any]) -> Path:
        """Copy an accepted skill into the benchmark's written skills; where it went."""

    def describe(self, out: Path, skills: list[Path]) -> list[str]:
        """The commands a sweep and a case run would be (dry run)."""


@dataclass
class SessionOutcome:
    status: str
    """`done`, `timeout`, `tokens`, `error: ...`."""
    skills: list[str] = field(default_factory=list)
    """The skill directories in the workspace when it ended (found by the driver)."""
    tokens: int = 0
    wall_s: float = 0.0
    case_runs: int = 0
    last_message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskContext:
    """A per-task session's task: its starting plan (`plans.prefilled`), its official
    success condition (`Bench.success_condition`)."""

    task: str
    plan: dict[str, Any]
    condition: str = ""
    center: str = ""
    """Where successful runs end inside the official checks (`Bench.success_center`), else
    empty."""


@dataclass(frozen=True)
class SessionContext:
    reports: Path
    """The round's coverage reports (the brief's evidence; a per-task session's: the round's
    final ones)."""
    limits: Limits
    relay: str = ""
    """A relay's notes on the session it continues; empty for a cluster's first."""
    task: TaskContext | None = None
    """A per-task session's task; None: a cluster's session."""
    conditions: dict[str, str] = field(default_factory=dict)
    """Per task of the cluster, its official success condition (`CONDITION_KINDS`)."""
    strategy: str = ""
    """A competing session's approach (`STRATEGIES`, `TASK_STRATEGIES`), else empty."""
    centers: dict[str, str] = field(default_factory=dict)
    """Per task of `conditions`, its success center (`Bench.success_center`), where it has one."""


@dataclass
class TaskRun:
    """A task's per-task stage (`Driver.closure`): what its sessions are told, the round's final
    reports they run from, its starting plan's hash and its trial (`Driver.start_trial`: the
    current plan's `trial_state`, None when the infrastructure broke it)."""

    context: TaskContext
    reports: Path
    start_key: str
    start: "asyncio.Future[dict[str, Any] | None]"
    home: Path | None = None
    """Where its trials go (`<stage>/<task>`, `start_trial`, `confirm_start`)."""
    accepted: Path | None = None
    """The skills its sessions read and its trials run with (`<run>/accepted`, or the snapshot an
    early stage took of it: `Driver.early`); None: `<run>/accepted`."""
    second: "asyncio.Future[dict[str, Any] | None] | None" = None
    """The current plan's second trial (`confirm_start`), shared by the task's sessions."""


CaseFn = Callable[[Any], Awaitable[dict[str, Any]]]
Session = Callable[[Cluster, Path, CaseFn, list[Path], SessionContext], Awaitable[SessionOutcome]]
"""`(cluster, workdir, case, accepted skill dirs, context) -> outcome`; a deployment's from
`skill_builder.session.make_session`."""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def files_hash(root: Path, paths: Sequence[str], skip: Sequence[str] = ()) -> str:
    """A hash of the files under `paths` (relative to `root`, files or directories), those under
    `skip` and caches and Markdown left out, by their paths and contents: what
    `Bench.sweep_code` returns."""
    digest = hashlib.sha256()
    for rel in sorted(paths):
        base = root / rel
        found = [base] if base.is_file() else sorted(base.rglob("*")) if base.is_dir() else []
        for path in found:
            name = path.relative_to(root).as_posix()
            if (not path.is_file() or "__pycache__" in path.parts
                    or path.suffix in (".pyc", ".md")
                    or any(name == x or name.startswith(x + "/") for x in skip)):  # fmt: skip
                continue
            digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:16]


def copy_skill(source: Path, target: Path) -> None:
    """A skill directory's regular files (links and caches left out: a link could point the
    snapshot at what the Agent may not read). Links are copied as links, never followed, and
    removed from the copy: a link made after a check could not pull a file in."""

    def skipped(directory: str, names: list[str]) -> set[str]:
        base = Path(directory)
        return {
            n
            for n in names
            if n == "__pycache__" or n.endswith(".pyc")
            or not ((base / n).is_symlink() or (base / n).is_file() or (base / n).is_dir())
        }  # fmt: skip

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=True, ignore=skipped)
    for directory, dirs, files in os.walk(target):  # never into a link
        for name in dirs + files:
            if (path := Path(directory) / name).is_symlink():
                path.unlink()


def written(work: Path) -> list[Path]:
    """The skill directories a workspace holds (a `skill.py` in it; hidden ones not)."""
    found = (
        p.parent
        for p in work.rglob("skill.py")
        if not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(work).parts)
    )
    return sorted(found)


# -- per-task state and the paired regression ------------------------------------------------


def state(reports: dict[str, tuple[dict[str, Any], dict[str, Any] | None]]) -> dict[str, Any]:
    """Per task after reuse and the chained step: segments explained from their own start of total;
    segments explained in the chained rollout (`chained`; None: no chained step or the
    infrastructure broke it; an earlier round's where the step kept it, `noisy`), its own rollout's
    (`chained_own`: what the regression compares, `lost` and `score`) and whether it reached the
    end (`to_end`); of the ceiling runs on the final plan (`clusters.runs`) those judged (runs the
    infrastructure broke are not), those that latched, and the benchmark stages they did of theirs;
    of its first `FIRST_REPEATS` (`plans.first_counts`: what totals sum and every comparison reads,
    the same runs whether or not the ceiling went on to its confirmation), and whether it is
    `confirmed` and `met`."""
    found = {}
    for task, (report, reuse) in sorted(reports.items()):
        ran = runs(report, reuse)
        judged = [r for r in ran if not r.get("infra")]
        chain = report.get("chain")
        chained = None if chain is None or chain.get("infra") else int(chain.get("explained", 0))
        kept = (chain or {}).get("noisy") or {}  # the step kept an earlier chain: its own depth
        own = None if chained is None else int(kept.get("explained", chained))
        stages = [r["stages"] for r in judged if r.get("stages")]
        found[task] = {
            "explained": sum(bool(r["explained"]) for r in report["explained"])
            + len(reuse["closed"] if reuse else []),
            "segments": len(report["calls"]),
            "chained": chained,
            "chained_own": own,
            "to_end": None if chained is None else bool(chain.get("to_end")),
            "latched": sum(bool(r["success"]) for r in judged),
            "judged": len(judged),
            "repeats": len(ran),
            "stages_done": sum(int(x["of"]) - int(x["left"]) for x in stages),
            "stages_of": sum(int(x["of"]) for x in stages),
            "kinds": sorted({c["kind"] for c in report["calls"]}),
            "confirmed": confirmed(ran),
            "met": at_target(ran),
        } | first_counts(ran)
    return found


def first(s: dict[str, Any], key: str) -> int:
    """A task's `key` (latched, judged, stages) among its ceiling's first `FIRST_REPEATS` runs
    (`plans.counted`): totals compare like with like whether or not a ceiling ran its
    confirmation (a `state` without those counts: all of them, at most 4)."""
    return counted(s, key)


def totals(states: dict[str, Any]) -> dict[str, int]:
    """A sweep's totals over its tasks (`state`), `GROWS` among them; `success` and `runs` over
    the ceilings' first runs (`first`), `met` the tasks at their target."""
    v = states.values()
    return {
        "chained": sum(s.get("chained") or 0 for s in v),
        "to_end": sum(bool(s.get("to_end")) for s in v),
        "success": sum(first(s, "latched") for s in v),
        "runs": sum(first(s, "judged") for s in v),
        "met": sum(met(s) for s in v),
        "stages_done": sum(first(s, "stages_done") for s in v),
        "stages_of": sum(first(s, "stages_of") for s in v),
        "explained": sum(s["explained"] for s in v),
        "segments": sum(s["segments"] for s in v),
        "tasks": len(states),
    }


def paired(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """`GROWS`' totals before and after over the tasks both sides judged (what the infrastructure
    broke on either side is no growth and no loss): the chain's over the tasks with a chained step
    judged on both (`chain_tasks`, their `chain_segments`), the ceiling's successes among its first
    runs (`first`) over those whose every repeat was judged on both (`runs`), its stages over those
    of them with stages on both, the segments explained from their own start over every task on
    both sides; `excluded`, the tasks each measure left out."""
    both = sorted(set(before) & set(after))

    def chain(t: str) -> bool:
        return before[t].get("chained") is not None and after[t].get("chained") is not None

    def whole(t: str) -> bool:
        return all(first(x[t], "judged") == min(x[t]["repeats"], FIRST_REPEATS) > 0
                   for x in (before, after))  # fmt: skip

    def staged(t: str) -> bool:
        return whole(t) and all(first(x[t], "stages_of") for x in (before, after))

    sides: dict[str, Any] = {}
    for side, st in (("before", before), ("after", after)):
        sides[side] = {
            "chained": sum(int(st[t]["chained"]) for t in both if chain(t)),
            "to_end": sum(bool(st[t]["to_end"]) for t in both if chain(t)),
            "chain_tasks": sum(chain(t) for t in both),
            "chain_segments": sum(st[t]["segments"] for t in both if chain(t)),
            "success": sum(first(st[t], "latched") for t in both if whole(t)),
            "runs": sum(first(st[t], "judged") for t in both if whole(t)),
            "stages_done": sum(first(st[t], "stages_done") for t in both if staged(t)),
            "stages_of": sum(first(st[t], "stages_of") for t in both if staged(t)),
            "explained": sum(st[t]["explained"] for t in both),
            "segments": sum(st[t]["segments"] for t in both),
            "tasks": len(both),
        }
    sides["excluded"] = {
        "unpaired": sorted(set(before) ^ set(after)),
        "chain": [t for t in both if not chain(t)],
        "ceiling": [t for t in both if not whole(t)],
        "stages": [t for t in both if whole(t) and not staged(t)],
    }
    return sides


def uses(report: dict[str, Any], reuse: dict[str, Any] | None) -> set[str]:
    """The skills a task's final calls run (what its ceiling ran and its segments' explanations)."""
    names = {c.get("skill") for c in (reuse or report)["calls"]}
    names |= {r.get("skill") for r in report["explained"] if r.get("explained")}
    names |= {c.get("skill") for c in (reuse["closed"] if reuse else [])}
    names |= {c.get("skill") for c in (report.get("chain") or {}).get("calls", [])}
    return {n for n in names if n}


def refused_fits(
    reports: dict[str, tuple[dict[str, Any], dict[str, Any] | None]],
) -> dict[str, list[str]]:
    """Per written skill, the gaps (`<task>/<segment>`) where the relative rule refused its fit
    (the reuse's and the chained step's `refused`), ` (rewrite)` after one marked `rewrite`:
    where such a gap stays open, the next round's builder writes the skill again (`clusters`,
    class `fit`); elsewhere the refusal is only recorded."""
    found: dict[str, set[str]] = {}
    for task, (report, reuse) in reports.items():
        for r in (reuse or {}).get("refused", []) + (report.get("chain") or {}).get("refused", []):
            mark = " (rewrite)" if r.get("rewrite") else ""
            found.setdefault(r["skill"], set()).add(f"{task}/{r['segment']}{mark}")
    for gaps in found.values():  # one marked and one not at a gap: it is a rewrite gap
        gaps -= {g[: -len(" (rewrite)")] for g in gaps if g.endswith(" (rewrite)")}
    return {skill: sorted(gaps) for skill, gaps in sorted(found.items())}


async def rerun(
    fetch: Callable[[str], Awaitable[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Transfer rows (`fetch("")`), each demonstration's judged miss run again (`GATE_RERUN`,
    `fetch("-again<i>")`): explained if either run explains it, both outcomes kept (`runs`)."""
    rows = await fetch("")
    for again in range(GATE_RERUN):
        if not any(not r.get("infra") and not r.get("explained") for r in rows):
            break
        more = {r.get("demo"): r for r in await fetch(f"-again{again}")}
        for i, r in enumerate(rows):
            other = more.get(r.get("demo"))
            if r.get("infra") or r.get("explained") or other is None:
                continue
            runs = [*r.get("runs", [outcome(r)]), outcome(other)]
            rows[i] = (other if other.get("explained") else r) | {"runs": runs}
    return rows


def outcome(row: dict[str, Any]) -> dict[str, bool]:
    return {"explained": bool(row.get("explained")), "infra": bool(row.get("infra"))}


def own_chain(s: dict[str, Any]) -> int | None:
    """A task's chained depth from its own rollout (`state`'s `chained_own`), not an earlier
    round's chain its step kept (`noisy`): only a sweep whose plan is an earlier chain's keeps
    one, and a candidate that changes the plan leaves its sweep none, so comparing kept depths
    would hold the arm without the candidate to a floor the arm with it lacks."""
    return s.get("chained_own", s.get("chained"))


def score(s: dict[str, Any]) -> float:
    """A task's state as one number: its segments explained from their own start and in its own
    chained rollout (`own_chain`), plus its ceiling repeats latched and the share of stages they
    did (each a share of the judged ones of its first runs, `plans.rate`, times `FIRST_REPEATS`:
    a confirmed ceiling's 8 runs weigh as much as another's 4)."""
    return (float(s["explained"]) + float(own_chain(s) or 0)
            + (rate(s) + share(s)) * FIRST_REPEATS)  # fmt: skip


def ledger(
    before: dict[str, Any],
    after: dict[str, tuple[dict[str, Any], dict[str, Any] | None]],
    losses: list[dict[str, Any]],
    candidates: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Per candidate: its gains, the tasks whose final calls run it that gained against `before`
    (`score`, paired), and its losses, the confirmed ones that blame it (`lost`: how much); `net`
    is gains less losses. A task two candidates share counts whole for each. Coverage first, so a
    candidate is dropped only where its losses outweigh its gains."""
    now = state(after)
    book: dict[str, dict[str, Any]] = {n: {"gains": {}, "losses": {}} for n in candidates}
    for task, reports in sorted(after.items()):
        if task in before and task in now and (gain := score(now[task]) - score(before[task])) > 0:
            for n in sorted(uses(*reports) & set(candidates)):
                book[n]["gains"][task] = round(gain, 2)
    for x in losses:
        for n in x["blamed"]:
            book[n]["losses"][x["task"]] = round(x["lost"], 2)
    for entry in book.values():
        entry["net"] = round(sum(entry["gains"].values()) - sum(entry["losses"].values()), 2)
    return book


def lost(was: dict[str, Any], is_: dict[str, Any]) -> bool:
    """A task's state lost against an earlier one: fewer segments explained from their own start
    or in its own chained rollout (`own_chain`, where both were judged), or a lower share of
    judged ceiling repeats latched or of the stages they did, both of the ceiling's first runs
    (`plans.rate`, `plans.share`: a confirmed ceiling's 8 against another's 4 is no pair)."""
    a, b = own_chain(was), own_chain(is_)
    return bool(
        is_["explained"] < was["explained"]
        or (a is not None and b is not None and b < a)
        or rate(is_) < rate(was)
        or (counted(was, "stages_of") and counted(is_, "stages_of") and share(is_) < share(was))
    )


def met(s: dict[str, Any]) -> bool:
    """Whether a task (`state`) is at its target (`plans.at_target`): its
    ceiling ran its confirmation round (2 of its first 4 runs latched the official success, then
    4 more ran) and at least `CONFIRMED_SUCCESSES` of the 8 latched. A state whose
    ceiling stopped at 4 runs is not confirmed: not met. Whether its chain reaches the end
    (`to_end`) is reported beside it, not part of it: the chain is judged against the
    demonstration's segment ends, which may lie just across the official boundary, and a plan aiming
    for the middle of the success region departs from them."""
    return bool(s.get("met"))


@dataclass(frozen=True)
class Outside:
    """A cause of a ceiling run's first failure that no skill or plan a session writes can fix:
    `reason` the failing call's receipt reason (`first_failure`), `note` a phrase
    its note holds ('': any note). The fix is in the core (its executor, its entries, the
    episode's step budget), which a builder session cannot change."""

    reason: str
    note: str = ""

    def label(self) -> str:
        return f"{self.reason}: {self.note}" if self.note else self.reason

    @classmethod
    def parse(cls, text: str) -> "Outside":
        """`reason` or `reason:note phrase` (a deployment's `--outside`)."""
        reason, _, note = text.partition(":")
        return cls(reason.strip(), note.strip())


OUTSIDE: Final = (
    Outside("invalid_args", "below the table top"),  # the executor's table-top refusal
    Outside("budget_exhausted"),  # the whole episode's step budget, not a segment's
    Outside("no_skill", "no entry presses or steadies"),  # no core entry for the segment's kind
)
"""The causes that pause a task's sessions (`blocked_outside`). IK-unreachable causes
(`unreachable`, `plan_failed`) stay out: they are not known to be outside skill writing."""
FIRST_FAILURES: Final = frozenset(
    {r.value for r in ReceiptReason} - {"ok", "clamped"} | {"no_skill", "transition"}
)
"""What a ceiling run's `first_failure` reason can be: a receipt's that did not reach, or the
chain's own (`demo_coverage.run`: a segment no skill fits, a transition that failed)."""


def check_outside(outside: Sequence[Outside]) -> tuple[Outside, ...]:
    """`outside` validated: known reasons (`FIRST_FAILURES`), no cause given twice."""
    found = tuple(outside)
    if bad := sorted({o.reason for o in found} - FIRST_FAILURES):
        raise ValueError(f"outside causes {bad}: a first failure's reason is one of "
                         f"{sorted(FIRST_FAILURES)}")  # fmt: skip
    if len(set(found)) < len(found):
        raise ValueError(f"outside causes given twice: {[o.label() for o in found]}")
    return found


def first_cause(run: dict[str, Any]) -> tuple[str, str, str] | None:
    """A ceiling run's first failure as (segment, reason, note): the note of its failing call at
    that segment; None where nothing failed (it latched, or every call reached and the official
    check did not latch)."""
    failure = run.get("first_failure")
    if not failure or not failure.get("reason"):
        return None
    segment, reason = str(failure.get("segment", "")), str(failure["reason"])
    note = next((str(c.get("note", "")) for c in run.get("calls") or []
                 if c.get("segment") == segment and c.get("reason") == reason
                 and not c.get("ok")), "")  # fmt: skip
    return segment, reason, note


def blocked_outside(
    reports: dict[str, tuple[dict[str, Any], dict[str, Any] | None]],
    outside: Sequence[Outside] = OUTSIDE,
) -> dict[str, dict[str, Any]]:
    """The tasks the driver pauses: below their target (`met`), every judged run of
    their ceiling (`clusters.runs`; at least one) unlatched, each stopped first by a cause in
    `outside`. Their gaps go to no cluster session and their task to no per-task session: what
    breaks them is not skill writing. Judged by the recorded cause, never by the task; each
    round from its own reports, so a task whose cause went away with a change to the sweep code (the
    next round's coverage is then swept again, `Driver.stale`) is picked up again. Per task: its
    causes (`Outside.label`), the segments they stopped at, the runs."""
    states = state(reports)
    found: dict[str, dict[str, Any]] = {}
    for task, (report, reuse) in sorted(reports.items()):
        judged = [r for r in runs(report, reuse) if not r.get("infra")]
        if not judged or met(states[task]) or any(r.get("success") for r in judged):
            continue
        causes, segments = set(), set()
        for r in judged:
            cause = first_cause(r)
            if cause is None:
                break
            hit = next((o for o in outside if o.reason == cause[1] and o.note in cause[2]), None)
            if hit is None:
                break
            causes.add(hit.label())
            segments.add(cause[0])
        else:
            found[task] = {"causes": sorted(causes), "segments": sorted(segments),
                           "runs": len(judged)}  # fmt: skip
    return found


def targets(reports: dict[str, tuple[dict[str, Any], dict[str, Any] | None]]) -> dict[str, Any]:
    """Per task (`state` of `reports`): segments explained from their own start and in the chain,
    whether it chains to the end, the ceiling's successes and stages, whether it is at its target
    (`met`), its open gaps with their causes (`clusters.task_gaps`) and its ceiling's success
    checks at the end (`clusters.margins`)."""
    found = {}
    for task, s in state(reports).items():
        report, reuse = reports[task]
        gaps, infra, _ = task_gaps(task, report, reuse)
        found[task] = {
            "isolated": f"{s['explained']}/{s['segments']}",
            "chained": None if s["chained"] is None else f"{s['chained']}/{s['segments']}",
            "to_end": bool(s.get("to_end")),
            "ceiling": f"{s['latched']}/{s['judged']}" + (" confirmed" if s["confirmed"] else ""),
            "stages": f"{s['stages_done']}/{s['stages_of']}" if s.get("stages_of") else None,
            "met": met(s),
            "gaps": [{"id": g.id, "cause": g.cause, "class": g.failure} for g in gaps]
            + [{"id": i["id"], "cause": i["cause"], "class": "infrastructure"} for i in infra],
            "margins": margins(runs(report, reuse)),
        }
    return found


def seed_verdicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A task's sessions' skills that become the next round's seeds (`Driver.closure`), one
    snapshot per name: a relay may change a skill an earlier attempt's gate passed, so a name the
    accepted plan ran is the snapshot it ran (that attempt's), any other its first pass."""
    chosen: dict[str, dict[str, Any]] = {}
    for r in records:
        plan = r.get("plan") or {}
        if plan.get("accepted"):
            used = set(plan.get("skills") or [])
            chosen |= {v["name"]: v for v in r["verdicts"] if v["name"] in used}
            break
    for v in (v for r in records for v in r["verdicts"]):
        if v["gate"] == "pass" and v["name"] not in chosen:
            chosen[v["name"]] = v
    return [chosen[name] for name in sorted(chosen)]


def marginal(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Whether a plan trial's rise (`better`) is too small to take from one trial each
    (`DECISIVE`). A chain that went less deep (a rise by success) counts as no depth: only
    `DECISIVE` more latched repeats take it from one trial, counted in a trial's first runs
    (`plans.rate`: a confirmed trial's 8 runs are no more decisive)."""
    chain = after["chained"] - before["chained"]
    runs = max(counted(before, "judged"), counted(after, "judged"))
    latched = round((rate(after) - rate(before)) * runs)
    return not (chain >= DECISIVE or latched >= DECISIVE or (chain > 0 and latched > 0))


def pooled(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Two trials of one plan (`trial_state`) as one: counts summed, `to_end` where both are."""
    summed = ("chained", "latched", "judged", "repeats", "stages_done", "stages_of")
    firsts = {f"first_{k}": counted(a, k) + counted(b, k)  # what `rate` and `share` read
              for k in ("latched", "judged", "stages_done", "stages_of")}  # fmt: skip
    return {"to_end": bool(a["to_end"] and b["to_end"])} | {k: a[k] + b[k] for k in summed} | firsts


def regress(
    before: dict[str, Any],
    after: dict[str, tuple[dict[str, Any], dict[str, Any] | None]],
    candidates: dict[str, list[str]],
    missing: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Per task, its state (`state`: explained segments, chained depth, the shares of judged
    ceiling repeats that latched and of stages done), after against before. A task that loses
    (`lost`) blames the candidates (name: kinds) its final calls run. A task missing from the
    rerun, or with no judged repeat left, is not judged, no loss (infrastructure), unless the
    sweep's `missing` says its job was stopped inside a candidate's `act` (`stuck`): that candidate
    is blamed. Returns the losses and the dropped (before confirmation: `Driver.confirmed`)."""
    now = state(after)
    losses, dropped = [], set()
    for task, was in sorted(before.items()):
        is_ = now.get(task)
        if is_ is None or (was["judged"] and not is_["judged"]):
            stuck = (missing or {}).get(task, {}).get("stuck")
            if stuck in candidates:  # loses the task whole
                losses.append({"task": task, "stuck": stuck, "before": was, "blamed": [stuck],
                               "lost": score(was)})  # fmt: skip
                dropped.add(stuck)
            else:
                losses.append({"task": task, "not_judged": True, "before": was, "blamed": []})
            continue
        if lost(was, is_):
            blamed = sorted(uses(*after[task]) & set(candidates))
            dropped |= set(blamed)
            losses.append({"task": task, "before": was, "after": is_, "blamed": blamed})
    return losses, dropped


# -- a session's case runs and its progress -----------------------------------------------------


def measured(row: dict[str, Any]) -> dict[str, float]:
    return {k: float(row[k]) for k in IMPROVES if isinstance(row.get(k), (int, float))}


class CaseRunner:
    """A session's `case` callable (`builder.codex.CaseRequest` in, the row out): each run
    logged in `directory`. The budget is the session's to count (`codex.Channel`); a row with
    `infra` is not a case run there."""

    def __init__(
        self,
        bench: Bench,
        directory: Path,
        explained: Callable[[str], None] | None = None,
        plan: Path | None = None,
        trial: Callable[[str, Path, Path], Awaitable[dict[str, Any]]] | None = None,
    ):
        self.bench, self.directory, self.explained = bench, directory, explained
        """`explained`: told each gap a run explained (the driver starts the gate's work)."""
        self.plan = plan
        """The round's reports, a chained gap's plan (`Bench.case`)."""
        self.trial = trial
        """A per-task session's plan trials (`<task>/#plan`): `(task, snapshot, out)` -> the
        trial's report (`Driver.trial`); None: a cluster's session, which has none."""
        self.pending: set[asyncio.Future[dict[str, Any]]] = set()

    async def __call__(self, request: Any) -> dict[str, Any]:
        run = asyncio.ensure_future(self.run(request.n, request.gap, Path(request.skill_dir)))
        self.pending.add(run)
        run.add_done_callback(self.pending.discard)
        # a session that ends mid-run cancels its await, not the run: its row is still logged
        return await asyncio.shield(run)

    async def run(self, n: int, gap: str, skill: Path) -> dict[str, Any]:
        started = time.monotonic()
        out = self.directory / "cases" / f"{n:02d}.json"
        task, _, segment = gap.partition("/")
        if segment == TRIAL and self.trial is not None:
            row = await self.trial(task, skill, out)
        else:
            row = await case_of(self.bench, gap, skill, out, self.plan)
        line = {
            "n": n,
            "gap": gap,
            "skill": skill.name,
            "explained": bool(row.get("explained")),
            "infra": bool(row.get("infra")),
            "error": row.get("error"),
            "cause": row.get("cause"),
            "measured": measured(row),
            "seconds": round(time.monotonic() - started, 1),
        }
        if segment == TRIAL and not row.get("infra"):
            line["trial"] = trial_state(row)
        await asyncio.to_thread(append_line, self.directory / "cases.jsonl", line)
        if self.explained is not None and line["explained"] and not line["infra"]:
            self.explained(gap)
        return row

    async def drain(self) -> None:
        """Wait for the runs still going (their session has ended), each bounded by the bench."""
        if self.pending:
            await asyncio.wait(self.pending, timeout=CASE_TIMEOUT_S + GRACE_S)


async def case_of(
    bench: Bench, gap: str, skill: Path, out: Path, plan: Path | None
) -> dict[str, Any]:
    """`bench.case`, given the round's reports where the gap runs from the plan's rollout."""
    if case_mode(gap.partition("/")[2])[1] is None:
        return await bench.case(gap, skill, out)
    return await bench.case(gap, skill, out, plan)


def append_line(path: Path, line: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(line) + "\n")


def best(values: list[float], key: str) -> float:
    found = [abs(v) if key == "dz_m" else v for v in values]
    return max(found) if IMPROVES[key] > 0 else min(found)


def baseline(
    reports: dict[str, tuple[dict[str, Any], dict[str, Any] | None]],
) -> dict[str, dict[str, float]]:
    """Per gap (`<task>/<segment>`), the best of each measure the round's coverage and reuse
    attempts reached (before any session): what a session's progress is measured against."""
    found: dict[str, dict[str, list[float]]] = {}
    for task, (report, reuse) in reports.items():
        chain = report.get("chain") or {}
        tagged = [(r, "") for r in report["explained"] + (reuse["tries"] if reuse else [])]
        tagged += [(r, f"#{r.get('mode', 'chained')}") for r in chain.get("tries", [])]
        if (gap := chain.get("gap")) and gap["index"] < len(chain.get("rows", [])):
            tagged.append((chain["rows"][gap["index"]], f"#{gap['mode']}"))
        for r, mode in tagged:
            for key, value in measured(r).items():
                gid = f"{task}/{r['segment']}{mode}"
                found.setdefault(gid, {}).setdefault(key, []).append(value)
    return {gap: {k: best(v, k) for k, v in keys.items()} for gap, keys in found.items()}


def progressed(
    runs: list[dict[str, Any]],
    before: dict[str, dict[str, float]],
    earlier: list[dict[str, Any]] | None = None,
) -> str | None:
    """Why a session's case runs show progress (a gap explained, or a measure some run on a gap
    took past the best the round's core attempts reached on it, `baseline`, by the margins of
    `IMPROVES`), else None; a relay's beyond its cluster's `earlier` sessions' runs too (a gap
    they explained, a measure they reached, claims nothing again: a stuck cluster is not relayed
    on its first session's progress). A measure the core never took on a gap claims nothing:
    against the session's own first run, a deliberately bad first run would earn a relay."""
    judged = [r for r in runs if not r.get("infra")]
    past = [r for r in earlier or [] if not r.get("infra")]
    done = {r["gap"] for r in past if r.get("explained")}
    explained = sorted({r["gap"] for r in judged if r.get("explained")} - done)
    if explained:
        return f"explained {', '.join(explained)}"
    for gap in sorted({r["gap"] for r in judged}):
        on_gap = [r["measured"] for r in judged if r["gap"] == gap and r.get("measured")]
        reached = [r["measured"] for r in past if r["gap"] == gap and r.get("measured")]
        for key, sign in IMPROVES.items():
            values = [m[key] for m in on_gap if key in m]
            if not values or key not in before.get(gap, {}):
                continue
            first = best([before[gap][key]] + [m[key] for m in reached if key in m], key)
            now = best(values, key)
            floor = PROGRESS_DEG if key.endswith("deg") else PROGRESS_M
            if (now - first) * sign >= max(PROGRESS_FRACTION * abs(first), floor):
                said = "core" if not reached else "core and earlier sessions"
                return f"{gap}: {key} {first:.3f} ({said}) -> {now:.3f}"
    return None


def trial_progress(
    runs: list[dict[str, Any]],
    start: dict[str, Any] | None,
    earlier: list[dict[str, Any]] | None = None,
) -> str | None:
    """Why a per-task session's plan trials show progress (`trial` in its case runs): one beat
    (`better`) the task's current plan's trial (`start`) and every trial of the task's earlier
    sessions, else None."""
    if start is None:
        return None
    best = start
    for r in earlier or []:
        if r.get("trial") and better(best, r["trial"]):
            best = r["trial"]
    for r in runs:
        if r.get("trial") and (why := better(best, r["trial"])):
            return f"plan trial {r['n']}: {why}"
    return None


def relay_notes(attempt: int, last: dict[str, Any], limits: Limits) -> str:
    """What a relay is told about the session it continues: how it ended, its case runs, the
    gate on what it left and its last message."""
    o = last["outcome"]
    lines = [
        f"## Relay {attempt}: you continue an earlier session on this cluster",
        "",
        f"The session before you ended at its budget ({o['status']}, {o['case_runs']} case runs, "
        f"{o['tokens']:,} tokens) with no skill passing the driver's gate, but with progress "
        f"({last['progress']}). Its workspace is yours as it left it: `skill/` and any notes. "
        f"Your budget: {limits.case_runs} case runs, {limits.wall_s / 3600:g} h, "
        f"{limits.tokens / 1e6:g} M tokens.",
        "",
        "Its case runs:",
    ]
    for r in last["cases"]:
        what = "explained" if r["explained"] else (r.get("error") or r.get("cause") or "not")
        lines.append(f"- {r['n']}. `{r['gap']}` with `{r['skill']}`: {what}; {r['measured']}")
    lines += ["", "The gate on what it left in `skill/`:"]
    for v in last["verdicts"]:
        lines.append(f"- `{v['name']}`: {v['reason']}; " + "; ".join(v.get("static", [])[:5]))
    if last.get("plan"):  # a per-task session's: the driver's trial of its plan.json
        plan = last["plan"]
        lines += ["", f"The driver's trial of its `plan.json`: {plan['reason']} (the task's "
                  f"current plan {plan.get('before')}; its plan {plan.get('after')})."]  # fmt: skip
    lines += ["", "Its last message:", ""]
    lines += ["> " + x for x in o["last_message"][-LAST_MESSAGE:].splitlines()]
    return "\n".join(lines)


def retry_notes(attempt: int, last: dict[str, Any], limits: Limits) -> str:
    o = last["outcome"]
    return "\n".join([
        f"## Restart {attempt}: the model's connection stalled",
        "",
        f"The session before you on this cluster stopped because the model's stream stalled "
        f"({o['status']}), not because of its work. Its workspace is yours as it left it. You "
        f"have what was left of its budget: {limits.case_runs} case runs, "
        f"{limits.wall_s / 3600:.2g} h, {limits.tokens / 1e6:.2g} M tokens.",
        "",
        "Its last message:",
        "",
        *("> " + x for x in o["last_message"][-LAST_MESSAGE:].splitlines()),
    ])  # fmt: skip


def left(limits: Limits, outcome: dict[str, Any]) -> Limits:
    """What a stalled session left of its budget."""
    return replace(
        limits,
        case_runs=limits.case_runs - outcome["case_runs"],
        wall_s=limits.wall_s - outcome["wall_s"],
        tokens=limits.tokens - outcome["tokens"],
    )


def reap(run: Path) -> list[dict[str, Any]]:
    """Process groups that sessions of `run` recorded (`<work>.codex*/pid`) and that still run
    as this user: a driver stopped mid-session left them. Terminated, then killed."""
    found = []
    found_pids = [
        *run.glob("round-*/sessions/*/*.codex*/pid"),
        *run.glob("round-*/tasks*/sessions/*/*.codex*/pid"),
    ]  # the per-task closure's stages too
    for pid_file in sorted(found_pids):
        pgid = int(pid_file.read_text(encoding="utf-8").split()[0])
        members = group(pgid)
        if not members:
            continue
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)
            deadline = time.monotonic() + 10
            while group(pgid) and time.monotonic() < deadline:
                time.sleep(0.5)
        found.append({"pid_file": str(pid_file), "pgid": pgid, "processes": members})
    return found


def group(pgid: int) -> list[int]:
    """The live processes of process group `pgid` owned by this user (Linux `/proc`)."""
    members = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(OSError, ValueError, IndexError):
            fields = stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and stat.parent.stat().st_uid == os.getuid():
                members.append(int(stat.parent.name))
    return members


def set_aside(log: Path, work: Path, attempt: int) -> None:
    """What a driver stopped during an attempt left (its case log; a first attempt's workspace),
    renamed aside so the attempt is redone; then the workspace, made."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stale = [log] if log.exists() else []
    if attempt == 0 and work.exists() and any(work.iterdir()):
        stale += [work, *work.parent.glob(f"{work.name}.codex*")]
    for path in stale:
        path.rename(path.with_name(f"{path.name}.interrupted-{stamp}"))
    work.mkdir(parents=True, exist_ok=True)


def cut_short(work: Path, started: float) -> SessionOutcome:
    """A session the driver cut at its wall clock (`GRACE_S` past the session's own): what the
    session left in `<work>.codex*/result.json` since `started` (a Codex session writes it even
    when cancelled: its tokens), else nothing but the status (the driver counts the case runs)."""
    since = time.time() - (time.monotonic() - started) - 2.0  # file times run a tick behind
    results = [p for p in work.parent.glob(f"{work.name}.codex*/result.json")
               if p.stat().st_mtime >= since]  # fmt: skip
    outcome = SessionOutcome(status="timeout")
    if results:
        found = read_json(max(results, key=lambda p: p.stat().st_mtime))
        outcome.tokens = int(found.get("tokens") or 0)
        outcome.case_runs = int(found.get("case_runs") or 0)
        outcome.wall_s = float(found.get("wall_s") or 0.0)
        outcome.last_message = str(found.get("last_message") or "")
    return outcome


def noisy(directory: Path) -> list[str]:
    """The tasks whose chained step in a sweep's `directory` kept an earlier round's chain (its
    `noisy`: the chain broke where that one held and explained fewer, `run.chain_reuse`). Their
    chained depth is the earlier one's, and the next round starts from that chain; the regression
    compares each arm's own rollout (`own_chain`), as only an arm whose plan is that chain's keeps
    one."""
    return sorted(p.name.removesuffix(".chain.json") for p in directory.glob("*.chain.json")
                  if json.loads(p.read_text(encoding="utf-8")).get("noisy"))  # fmt: skip


def complete(final: Path, coverage: Path, tasks: list[str]) -> list[str]:
    """`final`'s missing tasks (a sweep could not judge them) filled from the round's coverage, so
    the next round still has their gaps: the tasks filled."""
    filled = []
    for task in tasks:
        if not (final / f"{task}.json").is_file() and (coverage / f"{task}.json").is_file():
            for name in (f"{task}.json", f"{task}.reuse.json", f"{task}.chain.json"):
                if (coverage / name).is_file():
                    shutil.copy2(coverage / name, final / name)
            filled.append(task)
    return filled


# -- competing sessions and their order -----------------------------------------------------


def compete(found: Clusters, variants: int) -> Clusters:
    """Each cluster as `variants` competing sessions side by side: variant `n`
    keeps the cluster's id (`<id>-v<n>` from the second on), takes `STRATEGIES[n - 1]` and names
    its skills ending `<suffix>_v<n>`, so that no two clash; one variant leaves `found` as it
    is."""
    if variants <= 1:
        return found
    clusters = [
        replace(c, id=c.id if n == 1 else f"{c.id}-v{n}", suffix=f"{c.suffix}_v{n}", variant=n)
        for c in found.clusters
        for n in range(1, variants + 1)
    ]
    return Clusters(clusters, found.infra, found.unattributed)


def breaks(reports: dict[str, tuple[dict[str, Any], dict[str, Any] | None]]) -> set[str]:
    """The gap ids at which a task's chained rollout broke (its chain's `gap` in mode `chained`,
    or where a kept earlier chain's own rollout broke, `noisy`): as the segment's own gap and as
    its chained one (`clusters.chain_gap`: whichever it is)."""
    found = set()
    for task, (report, _) in reports.items():
        chain = report.get("chain") or {}
        for gap in (chain.get("gap"), (chain.get("noisy") or {}).get("broke")):
            if gap and gap.get("mode") == "chained":
                found |= {f"{task}/{gap['segment']}", f"{task}/{gap['segment']}#chained"}
    return found


def prioritized(found: Clusters, blocked: set[str]) -> list[tuple[int, Cluster]]:
    """The clusters with their session ranks: those holding a gap where a task's chain breaks
    (`breaks`) first (`BLOCKING`, the most such gaps first), the rest after (`OTHER`), each in
    `found`'s order (its variants together)."""
    ranked = [(BLOCKING if sum(g.id in blocked for g in c.gaps) else OTHER,
               -sum(g.id in blocked for g in c.gaps), i, c)
              for i, c in enumerate(found.clusters)]  # fmt: skip
    return [(rank, c) for rank, _, _, c in sorted(ranked, key=lambda x: x[:3])]


class Seats:
    """Builder sessions at once (`Driver.sessions_at_once`): a free seat goes to the lowest rank
    waiting, then to the first that asked. A session keeps its seat through its relays."""

    def __init__(self, seats: int):
        self.free = seats
        self.waiting: list[tuple[int, int]] = []
        self.seq = itertools.count()
        self.ready = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def take(self, rank: int) -> AsyncIterator[None]:
        me = (rank, next(self.seq))
        async with self.ready:
            heapq.heappush(self.waiting, me)
            try:
                await self.ready.wait_for(lambda: self.free > 0 and self.waiting[0] == me)
            except BaseException:  # cancelled while waiting: the next one may take the seat
                self.waiting.remove(me)
                heapq.heapify(self.waiting)
                self.ready.notify_all()
                raise
            heapq.heappop(self.waiting)
            self.free -= 1
            self.ready.notify_all()
        try:
            yield
        finally:
            async with self.ready:
                self.free += 1
                self.ready.notify_all()


def trial_key(verdict: dict[str, Any]) -> tuple[float, float, float]:
    """An accepted plan's trials (`plan_verdict`: its first, pooled with its confirmation where
    it had one) as `rises` weighs them: the share of judged repeats latched, the share of stages
    done, segments chained per trial."""
    after = verdict["after"]
    trials = 1
    if (verdict.get("confirm") or {}).get("after"):
        after, trials = pooled(after, verdict["confirm"]["after"]), 2
    return (rate(after), share(after), after["chained"] / trials)


def best_plan(verdicts: list[tuple[str, dict[str, Any]]]) -> tuple[str, dict[str, Any]] | None:
    """Of a task's competing sessions' accepted plans (`(label, verdict)`, variant order), each
    taken over the one chosen so far where it beats it as `better` judges a trial (`rises` on
    `trial_key`: more latched and no fewer stages, else no worse in each and better in one);
    where neither of two beats the other, the earlier."""
    chosen = None
    for label, v in verdicts:
        if chosen is None:
            chosen = (label, v)
            continue
        (s, g, c), (s0, g0, c0) = trial_key(v), trial_key(chosen[1])
        if rises((c0, c), (s0, s), (g0, g)):
            chosen = (label, v)
    return chosen


def rejudged(plan: dict[str, Any]) -> dict[str, Any]:
    """A session's plan verdict (`plan_verdict`) judged again from its stored trials by this
    code's `better` and `marginal`: a stage closed by newer code than judged its sessions takes
    their plans by its own rule. A marginal rise its record holds no confirmation for is not
    taken (its second trials were never run); one the infrastructure kept from being judged
    (`held`), or with no trials, stays as it was."""
    before, after = plan.get("before"), plan.get("after")
    if plan.get("held") or before is None or after is None:
        return plan
    why = better(before, after)
    if why is not None and marginal(before, after):
        confirm = plan.get("confirm") or {}
        if confirm.get("before") is None or confirm.get("after") is None:
            why = None
        else:
            why = better(pooled(before, confirm["before"]), pooled(after, confirm["after"]))
            why = why and f"confirmed over two trials: {why}"
    if (why is not None) == bool(plan.get("accepted")):
        return plan
    reason = why or "no rise over the current plan's trial"
    return plan | {"accepted": why is not None, "reason": reason, "was": plan.get("reason")}


# -- the driver ----------------------------------------------------------------------------


class Driver:
    def __init__(
        self,
        run: Path,
        bench: Bench,
        session: Session | None = None,
        import_baseline: Path | None = None,
        rounds: int | None = ROUNDS,
        until_target: bool = False,
        initial_skills: Path | None = None,
        limits: Limits = LIMITS,
        only: set[str] | None = None,
        seeds: list[Path] | None = None,
        per_task: bool = False,
        variants: int = 1,
        task_variants: int = 1,
        sessions_at_once: int | None = None,
        outside: Sequence[Outside] = OUTSIDE,
    ):
        if not 1 <= variants <= len(STRATEGIES):
            raise ValueError(f"variants {variants}: 1 to {len(STRATEGIES)} (one approach each)")
        if not 1 <= task_variants <= len(TASK_STRATEGIES):
            raise ValueError(f"task variants {task_variants}: 1 to {len(TASK_STRATEGIES)}")
        if sessions_at_once is not None and sessions_at_once < 1:
            raise ValueError(f"sessions at once {sessions_at_once}: at least 1")
        if rounds is not None and rounds < 1:
            raise ValueError(f"rounds {rounds}: at least 1, or None for no cap")
        self.run, self.bench, self.session = run, bench, session
        self.per_task = per_task
        """Run the per-task closure: for the tasks that chain to the end below their target
        from the round's start, beside the clusters (`early`); for the rest after the round's
        acceptance (`closure`)."""
        self.variants, self.task_variants = variants, task_variants
        """Competing sessions per gap cluster (`compete`) and per task, each with its
        own approach; a round keeps the counts it started its sessions with
        (`round-<k>/variants.json`)."""
        self.sessions_at_once = sessions_at_once or bench.case_slots
        """Builder sessions at once, every kind together (`Seats`); default the bench's case
        slots."""
        self.outside = check_outside(outside)
        """The causes a task's sessions are paused on (`blocked_outside`)."""
        self.seats: Seats | None = None
        self.early_stage: asyncio.Future[dict[str, Any]] | None = None
        """The running round's early per-task stage (`early`), awaited by `closure`."""
        self.plans = run / "plans"
        """The accepted task plans (`<task>.json`), each its task's chained step's."""
        self.import_baseline, self.rounds = import_baseline, rounds
        self.until_target = until_target
        """Stop only when every task is at its target (or at `STOP`), not when nothing grew."""
        self.initial_skills, self.limits, self.only = initial_skills, limits, only
        self.seeds = list(seeds or [])
        self.accepted = run / "accepted"
        self.references: dict[Path, asyncio.Future[list[int] | None]] = {}
        self.prefetching: set[asyncio.Future[Any]] = set()

    def start(self) -> dict[str, Any]:
        """`run.json`, the run's frame, written once."""
        meta = self.run / "run.json"
        if meta.is_file():
            return dict(read_json(meta))
        self.accepted.mkdir(parents=True, exist_ok=True)
        if self.initial_skills and self.initial_skills.is_dir():
            for d in written(self.initial_skills):
                copy_skill(d, self.accepted / d.name)
        started_with = sorted(p.name for p in self.accepted.iterdir() if p.is_dir())
        if self.import_baseline and started_with:
            raise SystemExit(
                f"{self.import_baseline} was swept without written skills; the run starts with "
                f"{started_with}: sweep round 0 instead of importing"
            )
        record = {
            "tasks": self.bench.tasks(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "skills_at_start": started_with,
            "baseline": str(self.import_baseline) if self.import_baseline else None,
            "limits": asdict(self.limits),
            "rounds": self.rounds,
            "until_target": self.until_target,
            "only": sorted(self.only) if self.only else None,
        }
        write_json(meta, record)
        return record

    async def run_all(self) -> None:
        if self.session is None:  # a dry run (`plan`) and `regate` need none
            raise ValueError("the driver runs sessions: give one (`session.make_session`)")
        self.start()
        reaped = await asyncio.to_thread(reap, self.run)
        for entry in reaped:
            await asyncio.to_thread(append_line, self.run / "reaped.jsonl", entry)
        for k in itertools.count() if self.rounds is None else range(self.rounds):
            summary = self.run / f"round-{k}" / "summary.json"
            fresh = not summary.is_file()  # run by this driver, not read back on a restart
            if fresh:
                if (self.run / STOP).is_file():  # halted between rounds; resumable
                    print(f"{self.run / STOP}: stopped before round {k}", flush=True)
                    return
                try:
                    await self.round(k)
                finally:  # a round that raised leaves no early stage running
                    if self.early_stage is not None and not self.early_stage.done():
                        self.early_stage.cancel()
                        await asyncio.gather(self.early_stage, return_exceptions=True)
            done = read_json(summary)
            if self.until_target:
                if stopped(done, until_target=True):
                    break
                if (spun := all_paused(done)) is not None:
                    if (why := self.resumes(k)) is None:
                        again = "" if fresh else " (still so on this restart)"
                        print(f"round {k}: {spun['line']}{again}", flush=True)
                        done["stopped_paused"] = spun | {"line": spun["line"] + again}
                        write_json(summary, done)
                        (summary.parent / "summary.md").write_text(render(done) + "\n",
                                                                   encoding="utf-8")  # fmt: skip
                        break
                    print(f"round {k}: every task below its target paused, no session; going "
                          f"on: {why}", flush=True)  # fmt: skip
                if stopped(done):  # the default rule would have ended the run here
                    self.no_growth(k, done)
            elif stopped(done):
                break

    def no_growth(self, k: int, summary: dict[str, Any]) -> None:
        """Log a round in which nothing grew while the run goes on to the target
        (`round-<k>/no-growth.json`, written once; the line printed at every pass)."""
        found = summary.get("targets") or {}
        met = sum(bool(x["met"]) for x in found.values())
        line = (f"round {k}: no growth this round (none of {', '.join(GROWS)} grew, nothing held "
                f"over or accepted); {met}/{len(found)} tasks at their target; going on "
                "(until_target)")  # fmt: skip
        print(line, flush=True)
        path = self.run / f"round-{k}" / "no-growth.json"
        if not path.is_file():
            unmet = sorted(t for t, x in found.items() if not x["met"])
            write_json(path, {"round": k, "met": met, "tasks": len(found), "unmet": unmet,
                              "line": line})  # fmt: skip

    def resumes(self, k: int) -> str | None:
        """Why the run goes on after round `k`, which left every task below its target paused
        (`all_paused`): its next round was started already, or will sweep again (`stale`: the
        sweep code changed), or a task below its target is no longer paused (other `--outside`
        causes, judged on the round's final reports); None: the pauses still hold, it stops."""
        if (self.run / f"round-{k + 1}").exists():
            return f"round {k + 1} was started"
        if (why := self.stale(k + 1)) is not None:
            return f"round {k + 1} sweeps again: {why}"
        here = self.run / f"round-{k}"
        reports = load(here / read_json(here / "summary.json")["state"])
        paused = blocked_outside(reports, self.outside)
        if free := sorted(t for t, s in state(reports).items() if not met(s) and t not in paused):
            return f"no longer paused outside skill writing: {', '.join(free)}"
        return None

    def accepted_dirs(self) -> list[Path]:
        return written(self.accepted)

    def stale(self, k: int) -> str | None:
        """Why round `k` cannot take its last round's final reports as its coverage (they were
        swept in another `SWEEP_FORMAT`, or by other code), else None."""
        if k == 0:
            return None
        path = self.run / f"round-{k - 1}" / "code.json"
        was = read_json(path) if path.is_file() else {}
        if was.get("format") != SWEEP_FORMAT:
            return f"round {k - 1} swept in format {was.get('format')}, not {SWEEP_FORMAT}"
        code = self.sweep_code()
        if was.get("sweep_code") and code:  # the files that make the reports, not the driver's
            if was["sweep_code"] != code:
                return f"round {k - 1}'s sweep code {was['sweep_code']}, this is {code}"
            return None
        now = source_sha()
        if was.get("sha") and now and was["sha"] != now:
            return f"round {k - 1} swept by {was['sha']}, this is {now}"
        return None

    def finals(self, here: Path) -> list[Path]:
        """The final reports (`summary.json`'s `state`) of the rounds before `here`, newest first:
        a chained step starts from the deepest chain among them that still fits its plan, so a
        chain that noise broke in one round is not carried into the next (`Bench.sweep`)."""
        k = int(here.name.removeprefix("round-"))
        found = []
        for j in range(k - 1, -1, -1):
            summary = self.run / f"round-{j}" / "summary.json"
            if summary.is_file():
                found.append(summary.parent / read_json(summary)["state"])
        return found

    def sweep_code(self) -> str | None:
        """The bench's `sweep_code` (None: it has none)."""
        code = getattr(self.bench, "sweep_code", None)
        return code() if callable(code) else None

    def mixed(self, here: Path) -> str | None:
        """Why a round already started (its coverage done) cannot be resumed by this code: its
        coverage was swept in another `SWEEP_FORMAT` (an import excepted, as it was swept), so its
        before and after would not be measured alike (ceiling repeats); else None."""
        if not (here / "coverage.done").is_file():
            return None
        code = read_json(here / "code.json") if (here / "code.json").is_file() else {}
        if code.get("imported") or code.get("format") == SWEEP_FORMAT:
            return None
        return (f"{here}: its coverage was swept in format {code.get('format')}, this code's is "
                f"{SWEEP_FORMAT}: resume it with the code that started it, or move {here.name} "
                "aside to start it again")  # fmt: skip

    async def coverage(self, k: int, here: Path) -> Cost:
        out = here / "coverage"
        if (here / "coverage.done").is_file():
            return Cost(**read_json(here / "coverage.done"))
        cost = Cost()
        why = self.stale(k)
        if why is not None:  # re-baselined: the round's before and after are this code's
            write_json(here / "rebaselined.json", {"why": why})
            previous = self.run / f"round-{k - 1}"  # its chain fixes the chain starts from
            final = previous / read_json(previous / "summary.json")["state"]
            cost = await self.planned_sweep(here, out, self.bench.tasks(), final)
        elif k == 0 and self.import_baseline:
            out.mkdir(parents=True, exist_ok=True)
            for task in self.bench.tasks():
                for name in (f"{task}.json", f"{task}.reuse.json", f"{task}.chain.json"):
                    if (self.import_baseline / name).is_file():
                        shutil.copy2(self.import_baseline / name, out / name)
        elif k > 0:
            previous = self.run / f"round-{k - 1}"
            final = previous / read_json(previous / "summary.json")["state"]
            out.mkdir(parents=True, exist_ok=True)
            for path in final.glob("*.json"):
                shutil.copy2(path, out / path.name)
            if changed := self.replanned(out):  # a task plan accepted since: its chain again
                for task in changed:
                    (out / f"{task}.chain.json").unlink()
                write_json(here / "replanned.json", changed)
                cost = await self.planned_sweep(here, out, changed, final)
        else:
            cost = await self.bench.sweep(out, [self.accepted], plans=self.plans)
        write_json(here / "coverage.done", asdict(cost))
        return cost

    async def planned_sweep(self, here: Path, out: Path, tasks: list[str], final: Path) -> Cost:
        """The coverage sweep of `tasks` with the accepted skills; a task whose accepted plan
        names a skill not accepted yet (`plan_skills`) swept apart, with those skills too."""
        given, pending = self.plan_skills(here)
        groups = [([self.accepted], [t for t in tasks if t not in pending])]
        groups += [([self.accepted, given / t], [t]) for t in tasks if t in pending]  # its own only
        found = await asyncio.gather(*(
            self.bench.sweep(out, skills, part, final, earlier=self.finals(here),
                             plans=self.plans)
            for skills, part in groups if part
        ))  # fmt: skip
        return Cost(slot_s=sum(c.slot_s for c in found), jobs=sum(c.jobs for c in found))

    def plan_skills(self, here: Path) -> tuple[Path, dict[str, list[str]]]:
        """The skills the accepted task plans name that are not accepted yet: the last round's
        per-task seeds (`tasks/seeds/`), which this round's regression judges for the skillset.
        A plan's trials ran with them, so its task's sweeps and its current plan's trial run with
        them too; else the plan's segments fall back to the core's calls and the task loses
        what its plan won. Copied once to
        `<round>/plan-skills/<task>/`, a task's apart (another task's sweep is not given them),
        written as `plan-skills.json`: per task, the names its plan takes there."""
        directory, path = here / "plan-skills", here / "plan-skills.json"
        if path.is_file():
            return directory, dict(read_json(path))
        k = int(here.name.removeprefix("round-"))
        seeds = self.run / f"round-{k - 1}" / "tasks" / "seeds"
        found: dict[str, list[str]] = {}
        for plan_path in sorted(self.plans.glob("*.json")) if k and seeds.is_dir() else []:
            named = plans.read(plan_path).skills()
            names = sorted(n for n in named if not (self.accepted / n).exists()
                           and (seeds / n).is_dir())  # fmt: skip
            for name in names:  # again whole: a stop before the record leaves no half copy
                copy_skill(seeds / name, directory / plan_path.stem / name)
            if names:
                found[plan_path.stem] = names
        write_json(path, found)
        return directory, found

    def replanned(self, reports: Path) -> list[str]:
        """The tasks whose accepted plan (`plans`) is not the one their chained step in `reports`
        applied (by its hash): their chain is swept again at the round's start, so the round's
        before is its plan's."""
        found = []
        for path in sorted(self.plans.glob("*.json")):
            chain = reports / f"{path.stem}.chain.json"
            if not chain.is_file():
                continue
            ran = (read_json(chain).get("plan") or {}).get("hash")
            if ran != plans.read(path).key():
                found.append(path.stem)
        return found

    async def sessions(
        self,
        here: Path,
        found: Clusters,
        open_ids: set[str],
        before: dict[str, Any],
        blocked: set[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Every cluster's sessions (a first and its relays), seated (`Seats`) by rank
        (`prioritized`: those blocking a task's chain first): per cluster, its attempts' records.
        `before`: the round's `baseline` measures; `blocked`: the gaps where a task's chain breaks
        (`breaks`)."""
        seats = self.seated()

        async def one(rank: int, c: Cluster) -> tuple[str, list[dict[str, Any]]]:
            async with seats.take(rank):
                records = await self.attempts(here, c, open_ids, before)
            return c.id, records

        ranked = prioritized(found, blocked or set())
        return dict(await asyncio.gather(*(one(rank, c) for rank, c in ranked)))

    def seated(self) -> Seats:
        """The run's seats (`sessions_at_once`), made in the running loop."""
        if self.seats is None:
            self.seats = Seats(self.sessions_at_once)
        return self.seats

    async def attempts(
        self,
        here: Path,
        c: Cluster,
        open_ids: set[str],
        before: dict[str, Any],
        task: TaskRun | None = None,
    ) -> list[dict[str, Any]]:
        """The cluster's sessions (`task`: a task's): a first, then relays (`RELAYS`) and
        restarts after a stall (`STALL_RETRIES`) as the last one's decision asks."""
        directory = here / "sessions" / c.id
        work = directory / "work"
        records: list[dict[str, Any]] = []
        while True:
            a = len(records)
            path = directory / f"attempt-{a}.json"
            if path.is_file():
                record = read_json(path)
            else:
                limits, notes = self.limits, ""
                if records and records[-1]["decision"] == "relay":
                    limits = self.limits.scaled(RELAY_SCALE)
                    notes = relay_notes(a, records[-1], limits)
                elif records:  # a restart after a stall
                    limits = left(Limits(**records[-1]["limits"]), records[-1]["outcome"])
                    notes = retry_notes(a, records[-1], limits)
                relays = sum(r["decision"] == "relay" for r in records)
                retries = sum(r["decision"] == "retry" for r in records)
                earlier = [x for r in records for x in r["cases"]]
                record = await self.attempt(
                    here,
                    c,
                    a,
                    work,
                    limits,
                    notes,
                    open_ids,
                    relays,
                    retries,
                    before,
                    earlier,
                    task,
                )
                write_json(path, record)
            records.append(record)
            if record["decision"] not in ("relay", "retry"):
                return records

    async def attempt(
        self,
        here: Path,
        c: Cluster,
        a: int,
        work: Path,
        limits: Limits,
        notes: str,
        open_ids: set[str],
        relays: int,
        retries: int,
        before: dict[str, Any],
        earlier: list[dict[str, Any]] | None = None,
        task: TaskRun | None = None,
    ) -> dict[str, Any]:
        """One session on the cluster, then the gate's steps 1-3 on what it left; `earlier`, the
        cluster's earlier sessions' case runs (a relay's progress is beyond them). A task's
        session (`task`) runs from the round's final reports, may run plan trials, and
        its plan is judged (`plan_verdict`): accepted, the task's sessions end."""
        log = work.parent / f"attempt-{a}"
        set_aside(log, work, a)
        reports = here / "coverage" if task is None else task.reports
        trial = None if task is None else functools.partial(self.trial, task)
        case = CaseRunner(self.bench, log, lambda gap: self.prefetch(log, gap), reports, trial)
        approaches = STRATEGIES if task is None else TASK_STRATEGIES
        said = self.conditions(c) if task is None else {}
        context = SessionContext(reports, limits, notes, task and task.context, said,
                                 approaches[c.variant - 1] if c.variant else "",
                                 {t: x for t in said if (x := self.center(t))})  # fmt: skip
        accepted = written(task.accepted) if task and task.accepted else self.accepted_dirs()
        started = time.monotonic()
        assert self.session is not None  # `run_all` refuses a driver without one
        try:
            outcome = await asyncio.wait_for(
                self.session(c, work, case, accepted, context),
                limits.wall_s + GRACE_S,
            )
        except asyncio.TimeoutError:
            outcome = cut_short(work, started)
        except Exception as error:  # a session's failure is its cluster's, not the round's
            outcome = SessionOutcome(status=f"error: {type(error).__name__}: {error}")
        await case.drain()
        cases = read_lines(log / "cases.jsonl")
        if outcome.status == "timeout" and not outcome.case_runs:  # cut: the driver's count
            outcome.case_runs = sum(not r["infra"] for r in cases)
        outcome.wall_s = outcome.wall_s or round(time.monotonic() - started, 1)
        outcome.skills = [str(p) for p in written(work)]
        verdicts = [await self.gate(log / "gate", d, open_ids, c.suffix, reports)
                    for d in map(Path, outcome.skills)]  # fmt: skip
        progress = progressed(cases, before, earlier)
        plan = None
        if task is not None:
            progress = progress or trial_progress(cases, await task.start, earlier)
            plan = await self.plan_verdict(task, work, log, verdicts)
        spent = outcome.status in ("timeout", "tokens") or outcome.case_runs >= limits.case_runs
        if plan is not None and plan["accepted"]:
            decision = "plan accepted"
        elif any(v["gate"] == "pass" for v in verdicts):
            decision = "candidate"
        elif any(v["gate"] == "hold" for v in verdicts):
            decision = "held: not judged"
        elif outcome.status == "stalled" and retries < STALL_RETRIES:
            decision = "retry"
        elif outcome.status == "stalled":
            decision = "open: stalled, restarts spent"
        elif spent and progress and relays < self.limits.relays:
            decision = "relay"
        elif spent and progress:
            decision = "open: relays spent"
        elif spent:
            decision = "open: budget spent without progress"
        else:
            decision = f"open: session ended ({outcome.status}) without a passing skill"
        return {
            "attempt": a,
            "limits": asdict(limits),
            "outcome": asdict(outcome),
            "cases": cases,
            "verdicts": verdicts,
            "progress": progress,
            "decision": decision,
        } | ({"plan": plan} if task is not None else {})

    def conditions(self, c: Cluster) -> dict[str, str]:
        """Per task of a cluster whose segments' ends decide the official checks
        (`CONDITION_KINDS`, `CONDITION_CLASSES`), its official success condition as the bench
        states it (`success_condition`), else none."""
        said = getattr(self.bench, "success_condition", None)
        if not callable(said) or (c.kind not in CONDITION_KINDS
                                  and c.failure not in CONDITION_CLASSES):  # fmt: skip
            return {}
        return {t: str(said(t)) for t in c.tasks}

    def center(self, task: str) -> str:
        """The task's success center as the bench gives it (`success_center`), else empty."""
        said = getattr(self.bench, "success_center", None)
        return str(said(task)) if callable(said) else ""

    async def trial(self, run: TaskRun, task: str, snapshot: Path, out: Path) -> dict[str, Any]:
        """A per-task session's plan trial (`<task>/#plan`): the plan it asked with
        (`snapshot/plan.json`), the accepted skills and its own (`snapshot/skill/`), from the
        round's final reports."""
        skills = [run.accepted or self.accepted, snapshot / "skill"]
        return await self.bench.plan_trial(task, snapshot / "plan.json", out, run.reports, skills)

    async def judged_trial(
        self, task: str, plan: Path, out: Path, reports: Path, skills: list[Path]
    ) -> dict[str, Any] | None:
        """The driver's own plan trial (`trial_state`), run again once when the infrastructure
        broke it (`REPEAT_INFRA`); None when it broke every run or left a repeat unjudged. Its
        report in `out` (`-again<i>` after a retry)."""
        for again in range(1 + REPEAT_INFRA):
            path = out if not again else out.with_name(f"{out.stem}-again{again}{out.suffix}")
            found = await self.bench.plan_trial(task, plan, path, reports, skills)
            if not found.get("infra"):
                now = trial_state(found)
                return now if now["judged"] == now["repeats"] > 0 else None
        return None

    async def start_trial(self, stage: Path, task: str, plan: plans.TaskPlan, reports: Path,
                          accepted: Path) -> dict[str, Any] | None:  # fmt: skip
        """The task's current plan's trial (`plans.prefilled`, the skills `accepted`): what its
        sessions' plans are paired against. Kept (`<stage>/<task>/start.json`): a resumed stage
        does not run it again."""
        here = stage / task
        kept = here / "start.json"
        if kept.is_file():
            return read_json(kept)["state"]
        plans.write(here / "start-plan.json", plan)
        found = await self.judged_trial(task, here / "start-plan.json", here / "trial.json",
                                        reports, [accepted])  # fmt: skip
        if found is not None:
            write_json(kept, {"state": found})
        return found

    async def plan_verdict(
        self, run: TaskRun, work: Path, log: Path, verdicts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The session's task plan (`work/plan.json`) judged: a trial of it by the driver, with
        the accepted skills and those of the session's that may run (its gate passed them, or it
        named no open gap and the plan names it: `NO_CASE`), paired against the task's current
        plan's trial (`better`). Accepted where it rises; not judged (`held`) where the
        infrastructure broke either trial."""
        path = work / "plan.json"
        if not path.is_file():
            return {"accepted": False, "reason": "no plan.json"}
        try:
            plan = await asyncio.to_thread(plans.read, path)
        except ValueError as error:  # the Agent's file: a boundary
            return {"accepted": False, "reason": f"plan.json: {str(error)[:300]}"}
        if plan.task != run.context.task:
            return {"accepted": False, "reason": f"plan.json is task {plan.task!r}'s"}
        if plan.key() == run.start_key:
            return {"accepted": False, "reason": "plan.json is the task's current plan"}
        here = log / "plan"
        plans.write(here / "plan.json", plan)
        usable = {v["name"]: Path(v["dir"]) for v in verdicts if v["gate"] == "pass"
                  or (v.get("reason") == NO_CASE and v["name"] in plan.skills())}  # fmt: skip
        for name, source in usable.items():
            copy_skill(source, here / "skills" / name)
        skills = [run.accepted or self.accepted, here / "skills"]
        found = await self.judged_trial(run.context.task, here / "plan.json", here / "trial.json",
                                        run.reports, skills)  # fmt: skip
        start = await run.start
        record: dict[str, Any] = {"plan": str(here / "plan.json"), "key": plan.key(),
                                  "before": start, "after": found,
                                  "skills": sorted(set(usable) & plan.skills())}  # fmt: skip
        if start is None or found is None:
            return record | {"accepted": False, "held": True,
                             "reason": "not judged: the infrastructure broke a trial"}  # fmt: skip
        why = better(start, found)
        if why is not None and marginal(start, found):  # one more trial each, pooled
            again, mine = await asyncio.gather(
                self.confirm_start(run),
                self.judged_trial(run.context.task, here / "plan.json", here / "trial-2.json",
                                  run.reports, skills),
            )  # fmt: skip
            record["confirm"] = {"before": again, "after": mine}
            if again is None or mine is None:
                return record | {"accepted": False, "held": True, "reason": "not judged: the "
                                 "infrastructure broke a confirmation trial"}  # fmt: skip
            why = better(pooled(start, again), pooled(found, mine))
            why = why and f"confirmed over two trials: {why}"
            if why is None:
                why_not = "a marginal rise the second trial did not confirm"
                return record | {"accepted": False, "reason": why_not}
        return record | {"accepted": why is not None,
                         "reason": why or "no rise over the current plan's trial"}  # fmt: skip

    async def confirm_start(self, run: TaskRun) -> dict[str, Any] | None:
        """The task's current plan's second trial (`start_trial`'s plan), which confirms a
        marginal rise (`marginal`): run once per task and stage, shared by its competing
        sessions, kept (`start-2.json`)."""
        if run.second is None:
            run.second = asyncio.ensure_future(self.second_start(run))
        return await asyncio.shield(run.second)

    async def second_start(self, run: TaskRun) -> dict[str, Any] | None:
        assert run.home is not None
        kept = run.home / "start-2.json"
        if kept.is_file():
            return dict(read_json(kept)["state"])
        found = await self.judged_trial(run.context.task, run.home / "start-plan.json",
                                        run.home / "trial-2.json", run.reports,
                                        [run.accepted or self.accepted])  # fmt: skip
        if found is not None:
            write_json(kept, {"state": found})
        return found

    async def gate(
        self,
        directory: Path,
        skill: Path,
        open_ids: set[str],
        suffix: str = "",
        plan: Path | None = None,
    ) -> dict[str, Any]:
        """Gate steps 1 and 2 on one written skill: a snapshot first (what is judged, whatever a
        relay changes later; `skills/`, outside every directory a case writes), its content hash,
        then the bench's validator in a child process; then `judge`. A name already accepted
        fails here, as one without the cluster's `suffix` (`Cluster.suffix`); two candidates of
        one name are settled by `candidates`."""
        snapshot = directory / "skills" / skill.name
        await asyncio.to_thread(copy_skill, skill, snapshot)
        digest = await asyncio.to_thread(content_hash, snapshot)
        checked = await self.bench.validate(snapshot)
        name = checked.get("name") or skill.name
        v: dict[str, Any] = {
            "name": name,
            "dir": str(snapshot),
            "sha256": digest,
            "kinds": list(checked.get("kinds") or []),
            "cases_yaml": list(checked.get("cases") or []),
            "static": list(checked.get("findings") or []),
        }
        if not checked.get("ok"):
            return v | {"gate": "fail", "reason": "static check"}
        if (self.accepted / name).exists():
            return v | {"gate": "fail", "reason": f"name {name!r} taken"}
        if not name.endswith(suffix):
            return v | {"gate": "fail", "reason": f"name {name!r} does not end in {suffix!r}"}
        return await self.judge(directory, v, open_ids, plan)

    async def judge(
        self, directory: Path, v: dict[str, Any], open_ids: set[str], plan: Path | None = None
    ) -> dict[str, Any]:
        """Gate step 2 on a validated snapshot (`v`): the driver's own run of its `cases.yaml`
        gaps that are open gaps of the round (others ignored), each in a directory of its own
        under `directory/cases/`; then the transfer of the first explained ones. `pass`: a case
        explained and, where some other demonstration tells, a transfer case explained (none of
        fewer than `TRANSFER_REJECT` judged: `transfer_weak`); `fail`: cases judged and none
        explained, or `TRANSFER_REJECT` transfer cases judged and none explained; `hold`: none
        judged. A chained gap's case runs from `plan`'s reports (default: the round's
        coverage)."""
        snapshot = Path(v["dir"])
        cases = [g for g in v["cases_yaml"] if g in open_ids][:GATE_CASES]
        if plan is None:
            here = next((p for p in [directory, *directory.parents]
                         if p.name.startswith("round-")), None)  # fmt: skip
            plan = here / "coverage" if here is not None and (here / "coverage").is_dir() else None
        if not cases:
            return v | {"gate": "fail", "reason": NO_CASE}
        stamp = time.strftime("%Y%m%d-%H%M%S")

        async def one(n: int, gap: str) -> dict[str, Any]:
            row: dict[str, Any] = {}
            seconds, attempt, rows = 0.0, 0, []
            for _ in range(1 + GATE_RERUN):  # judged, not explained: once more (`GATE_RERUN`)
                for _ in range(1 + REPEAT_INFRA):
                    started = time.monotonic()
                    out = directory / "cases" / f"{stamp}-{n:02d}-{attempt}.json"
                    row = await case_of(self.bench, gap, snapshot, out, plan)
                    seconds += time.monotonic() - started
                    attempt += 1
                    if not row.get("infra"):
                        break
                rows.append(row)
                if row.get("infra") or row.get("explained"):
                    break
            best = next((r for r in rows if r.get("explained")), row)
            return {
                "gap": gap,
                "explained": bool(best.get("explained")),
                "infra": bool(best.get("infra")),
                "error": best.get("error"),
                "measured": measured(best),
                "runs": [outcome(r) for r in rows],
                "seconds": round(seconds, 1),
            }

        runs = list(await asyncio.gather(*(one(n, g) for n, g in enumerate(cases))))
        v = v | {"cases": runs, "case_s": round(sum(r["seconds"] for r in runs), 1)}
        judged = [r for r in runs if not r["infra"]]
        explained = [r["gap"] for r in judged if r["explained"]]
        if not judged:
            return v | {"gate": "hold", "reason": f"not judged: {len(runs)} cases broke on infra"}
        if not explained:
            return v | {"gate": "fail", "reason": f"0/{len(judged)} open-gap cases explained"}
        said = f"{len(explained)}/{len(judged)} open-gap cases explained"

        async def moved(n: int, gap: str) -> list[dict[str, Any]]:
            """The gap's transfer rows, on the demonstrations the core's own entry transfers
            to (`reference`; none: [], cannot tell)."""
            base = own_gap(gap)  # a chained gap transfers as its segment's own
            demos = await self.reference(directory, base)
            if demos is None:
                return [{"gap": gap, "explained": False, "infra": True,
                         "error": "the core's transfer reference was not judged"}]  # fmt: skip
            if not demos:
                return []

            async def fetch(tag: str) -> list[dict[str, Any]]:
                out = directory / "transfer" / f"{stamp}-{n:02d}{tag}.json"
                return await self.bench.transfer(base, snapshot, out, demos)

            return [r | {"gap": gap} for r in await rerun(fetch)]

        found = await asyncio.gather(
            *(moved(n, g) for n, g in enumerate(explained[:TRANSFER_GAPS]))
        )
        transfer = [r for rows in found for r in rows]
        v |= {"transfer": transfer}
        judged = [r for r in transfer if not r.get("infra")]
        if not transfer:
            return v | {"gate": "pass", "reason": f"{said}; no other demonstration tells "
                        "(none the core transfers to)"}  # fmt: skip
        if not judged:
            return v | {"gate": "hold", "reason": f"{said}; transfer not judged (infra)"}
        if not any(r.get("explained") for r in judged):
            if len(judged) >= TRANSFER_REJECT:
                return v | {"gate": "fail", "reason": f"{said}, but 0/{len(judged)} transfer "
                            "cases: it reproduces its demonstration's scene only"}  # fmt: skip
            return v | {"gate": "pass", "transfer_weak": True,
                        "reason": f"{said}; 0/{len(judged)} transfer cases (weak)"}  # fmt: skip
        hits = sum(bool(r.get("explained")) for r in judged)
        return v | {"gate": "pass", "reason": f"{said}; {hits}/{len(judged)} transfer cases"}

    async def reference(self, directory: Path, gap: str) -> list[int] | None:
        """The other demonstrations (layouts) on which the core's own entry for `gap` transfers
        (`Bench.core_transfer`, a judged miss run again as a skill's): a skill's transfer case
        counts only there, since a layout the core cannot do either is no evidence that a skill is
        bound to its scene. Cached per round (`round-<k>/transfer-core/`, one run per gap however
        many skills ask); None when the infrastructure broke every run (not cached)."""
        here = next((p for p in [directory, *directory.parents] if p.name.startswith("round-")),
                    self.run)  # fmt: skip
        key = hashlib.sha256(gap.encode()).hexdigest()[:16]
        path = here / "transfer-core" / f"{key}.json"
        if path not in self.references:

            async def run() -> list[int] | None:
                if path.is_file():
                    return list(read_json(path)["demos"])

                async def fetch(tag: str) -> list[dict[str, Any]]:
                    out = here / "transfer-core" / key / f"core{tag}.json"
                    return await self.bench.core_transfer(gap, out)

                rows = await rerun(fetch)
                if rows and all(r.get("infra") for r in rows):
                    return None
                demos = sorted(r["demo"] for r in rows if r.get("explained") and "demo" in r)
                write_json(path, {"gap": gap, "demos": demos, "rows": rows})
                return demos

            self.references[path] = asyncio.ensure_future(run())
        future = self.references[path]
        found = await future
        if found is None and self.references.get(path) is future:
            del self.references[path]  # the infrastructure's: asked again next time
        return found

    def prefetch(self, directory: Path, gap: str) -> None:
        """A session's case run explained `gap`: the core's transfer reference for it (what the
        gate's transfer step asks first, `reference`) started now, while the session still runs,
        so the GPUs work while the model thinks. The same cached run the gate would make, so
        nothing the gate decides changes; a round waits for these before its sweeps."""

        async def one() -> None:
            with contextlib.suppress(Exception):  # the gate asks again and sees the error
                await self.reference(directory, own_gap(gap))

        task = asyncio.ensure_future(one())
        self.prefetching.add(task)
        task.add_done_callback(self.prefetching.discard)

    def candidates(
        self, here: Path, sessions: dict[str, list[dict[str, Any]]], carried: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """The round's gate verdicts, `<cluster>/attempt-<a>/<name>` (`held/<name>` for one
        carried from the last round); those that passed are copied to `candidates/` (a name passed
        by two clusters: the first in cluster order keeps it; a snapshot whose content changed
        since it was validated fails)."""
        verdicts: dict[str, dict[str, Any]] = {}
        names: set[str] = set()
        found = [
            (f"{cid}/attempt-{record['attempt']}", dict(v) | {"cluster": cid,
                                                             "attempt": record["attempt"]})
            for cid in sorted(sessions) for record in sessions[cid] for v in record["verdicts"]
        ] + [(f"{v['cluster']}/{v['name']}", v) for v in carried]  # fmt: skip
        for key, v in found:
            if v["gate"] == "pass" and v["name"] in names:
                v |= {"gate": "fail", "reason": f"name {v['name']!r} taken this round"}
            if v["gate"] == "pass" and content_hash(Path(v["dir"])) != v["sha256"]:
                v |= {"gate": "fail", "reason": "its snapshot changed after it was validated"}
            if v["gate"] == "pass":
                names.add(v["name"])
                target = here / "candidates" / v["name"]
                if not target.exists():
                    copy_skill(Path(v["dir"]), target)
                if content_hash(target) != v["sha256"]:
                    raise RuntimeError(f"{target}: not the snapshot validated ({v['sha256']})")
            verdicts[f"{key}/{v['name']}"] = v
        return verdicts

    async def sweep_once(
        self,
        here: Path,
        name: str,
        skills: list[Path],
        tasks: list[str] | None = None,
        reuse_before: bool = False,
    ) -> dict[str, Any]:
        """A sweep of the round (rerun, confirmations, the kept ones'), its chained steps
        starting from the chain fixes of the round's coverage; of `tasks` only. `reuse_before`
        (`Bench.sweep`): the rerun and the kept ones' sweeps, compared with the coverage first;
        a confirmation's two arms measure afresh, so neither leans on the coverage. Tasks already
        in `here/name` are kept (`Bench.sweep`), and only the rest are swept."""
        done = here / f"{name}.done"
        if not done.is_file():
            cost = await self.bench.sweep(here / name, skills, tasks, here / "coverage",
                                          earlier=self.finals(here), plans=self.plans,
                                          reuse_before=reuse_before)  # fmt: skip
            write_json(done, asdict(cost))
        return dict(read_json(done))

    async def confirmed(
        self,
        here: Path,
        name: str,
        sweeps: tuple[list[Path], list[Path]],
        before: dict[str, Any],
        candidates: dict[str, list[str]],
        costs: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], set[str], dict[str, dict[str, Any]]]:
        """`regress` of the sweep `here/name` (with `sweeps[1]`) against `before`, each loss
        confirmed: the losing tasks swept again with `sweeps[0]` (the round's
        accepted) and `sweeps[1]`, paired; a loss stands only if the second pair loses too (a
        task the second pair could not judge: no loss), by the second pair's `score` drop. A
        `stuck` loss needs no confirmation. Dropped: the candidates whose `ledger` net is below
        zero; the ledger is returned too."""
        missing = here / name / "missing.json"
        losses, _ = regress(
            before, load(here / name), candidates,
            read_json(missing) if missing.is_file() else None,
        )  # fmt: skip
        suspects = sorted(x["task"] for x in losses if "after" in x and x["blamed"])
        if suspects:  # both arms at once: each sweep's jobs share the slots
            keys = [f"{name}-confirm-{arm}" for arm in ("before", "after")]
            arms = zip(keys, sweeps, strict=True)
            found = await asyncio.gather(*(self.sweep_once(here, x, s, suspects) for x, s in arms))
            costs.update(zip(keys, found, strict=True))
            again = state(load(here / f"{name}-confirm-before"))
            now = state(load(here / f"{name}-confirm-after"))
            for x in losses:
                if x["task"] in suspects:
                    t = x["task"]
                    x["confirmed"] = t in again and t in now and lost(again[t], now[t])
                    if x["confirmed"]:
                        x["lost"] = max(score(again[t]) - score(now[t]), 0.0)
                    else:
                        x["noise"], x["blamed"] = x["blamed"], []
        book = ledger(before, load(here / name), losses, candidates)
        dropped = {n for n, entry in book.items() if entry["net"] < 0}
        return losses, dropped, book

    async def round(self, k: int) -> None:
        here = self.run / f"round-{k}"
        if (why := self.mixed(here)) is not None:
            raise SystemExit(why)
        here.mkdir(parents=True, exist_ok=True)
        timing_path = here / "timing.json"
        timing = read_json(timing_path) if timing_path.is_file() else {"started": time.time()}
        write_json(timing_path, timing)
        costs: dict[str, Any] = {"coverage": asdict(await self.coverage(k, here))}
        if not (here / "code.json").is_file():  # what swept this round's reports (`stale`)
            code = {"format": SWEEP_FORMAT, "sha": source_sha(), "sweep_code": self.sweep_code()}
            if k == 0 and self.import_baseline:  # imported: as they were swept, code unknown
                whole = all((here / "coverage" / f"{t}.chain.json").is_file()
                            for t in self.bench.tasks())  # fmt: skip
                code = {"format": SWEEP_FORMAT if whole else None, "sha": None,
                        "imported": str(self.import_baseline)}  # fmt: skip
            write_json(here / "code.json", code)
        before_reports = load(here / "coverage")
        before = state(before_reports)
        paused = blocked_outside(before_reports, self.outside)
        found = cluster({t: r for t, r in before_reports.items() if t not in paused})
        if self.only is not None:
            found = Clusters(
                [c for c in found.clusters if c.id in self.only], found.infra, found.unattributed
            )
        counts = self.counts(here)
        found = compete(found, counts["clusters"])
        write_json(here / "clusters.json", found.record())
        # the last round's plans and seeds would never be swept: no closure there
        closing = self.per_task and (self.rounds is None or k + 1 < self.rounds)
        self.early_stage = self.early(here, before, set(paused)) if closing else None

        open_ids = {g.id for c in cluster(before_reports).clusters for g in c.gaps + c.deferred}
        # the last round's held skills gated while the sessions run (their gate needs no session)
        held = asyncio.ensure_future(self.carried(k, here, open_ids))
        seeds = asyncio.ensure_future(self.seeded(here))
        try:
            sessions = await self.sessions(here, found, open_ids, baseline(before_reports),
                                           breaks(before_reports))  # fmt: skip
            await self.recheck(here, sessions, open_ids)
            carried = await held + await seeds
        except BaseException:
            held.cancel()
            seeds.cancel()
            raise
        if self.prefetching:  # references no gate asked for: done before the sweeps start
            await asyncio.gather(*self.prefetching, return_exceptions=True)
        verdicts = self.candidates(here, sessions, carried)
        passed = {v["name"]: v["kinds"] for v in verdicts.values() if v["gate"] == "pass"}
        candidates = set(passed)

        losses: list[dict[str, Any]] = []
        dropped: set[str] = set()
        books: dict[str, dict[str, dict[str, Any]]] = {}
        final = here / "coverage"
        start = [self.accepted]
        if candidates:  # one no task used is one more alternative, kept
            costs["rerun"] = await self.sweep_once(here, "rerun", [*start, here / "candidates"],
                                                   reuse_before=True)  # fmt: skip
            losses, dropped, books["rerun"] = await self.confirmed(
                here, "rerun", (start, [*start, here / "candidates"]), before, passed, costs
            )
            final = here / "rerun"
        kept = sorted(candidates - dropped)
        step = 0
        while dropped and kept:  # the kept ones' own sweep, compared the same way
            stage = here / "kept"
            if stage.exists():
                shutil.rmtree(stage)
            for name in kept:
                copy_skill(here / "candidates" / name, stage / name)
            name = "after" if step == 0 else f"after-{step}"
            costs[name] = await self.sweep_once(here, name, [*start, stage], reuse_before=True)
            more, dropped, books[name] = await self.confirmed(
                here, name, (start, [*start, stage]), before, {n: passed[n] for n in kept}, costs
            )
            losses += [x | {"sweep": name} for x in more]
            kept = sorted(set(kept) - dropped)
            final, step = here / name, step + 1
        if candidates and not kept:
            final = here / "coverage"
        for v in verdicts.values():
            if v["gate"] == "fail":
                v["verdict"] = "rejected"
            elif v["gate"] == "hold":
                v["verdict"] = "not judged"
            elif v["name"] not in kept:
                entry = next(b[v["name"]] for b in reversed(books.values()) if v["name"] in b)
                lost_in, gained = ", ".join(entry["losses"]), ", ".join(entry["gains"]) or "-"
                why = f"regression: net {entry['net']} (lost {lost_in}; gained {gained})"
                v |= {"verdict": "rejected", "ledger": entry, "reason": why}
            else:
                v["verdict"] = "accepted"
        for name in kept:
            v = next(
                v for v in verdicts.values() if v["name"] == name and v["verdict"] == "accepted"
            )
            records = sessions.get(v["cluster"], [])
            copy_skill(here / "candidates" / name, self.accepted / name)
            if content_hash(self.accepted / name) != v["sha256"]:
                raise RuntimeError(f"{self.accepted / name}: not the snapshot validated")
            provenance = {
                "name": name,
                "run": self.run.name,
                "round": k,
                "cluster": v["cluster"],
                "sha256": v["sha256"],
                "sessions": [
                    {key: r["outcome"][key] for key in ("status", "tokens", "wall_s", "case_runs")}
                    | {"attempt": r["attempt"], "decision": r["decision"]}
                    for r in records
                ],
                "tokens": sum(r["outcome"]["tokens"] for r in records),
                "wall_s": round(sum(r["outcome"]["wall_s"] for r in records), 1),
                "case_runs": sum(r["outcome"]["case_runs"] for r in records),
                "gate_cases": v["cases"],
                "transfer": v.get("transfer", []),
            }
            v["written_to"] = str(self.bench.accept(self.accepted / name, provenance))
        write_json(here / "gate.json", verdicts)
        carried_over = complete(final, here / "coverage", self.bench.tasks())
        after_reports = load(final)
        closed = await self.closure(here, final, self.early_stage) if closing else None
        timing["ended"] = time.time()
        write_json(timing_path, timing)
        summary = self.summarize(
            k,
            found,
            before,
            after_reports,
            sessions,
            verdicts,
            losses,
            kept,
            costs,
            timing,
            final,
        )
        summary["carried_tasks"] = carried_over
        rebaselined = (here / "rebaselined.json").is_file()
        copied = (k > 0 and not rebaselined) or (k == 0 and self.import_baseline is not None)
        summary["noisy_chains"] = {  # the coverage copied from the last round's: listed there
            name: found
            for name in costs
            if not (name == "coverage" and copied)
            and (here / name).is_dir()
            and (found := noisy(here / name))
        }
        summary["rebaselined"] = (
            read_json(here / "rebaselined.json")["why"] if (here / "rebaselined.json").is_file()
            else None
        )  # fmt: skip
        summary["ledger"] = books
        summary["transfer_weak"] = sorted(
            v["name"] for v in verdicts.values() if v.get("transfer_weak") and v["name"] in kept
        )
        summary["not_judged"] = [
            {"name": v["name"], "dir": v["dir"], "sha256": v["sha256"], "reason": v["reason"]}
            for v in verdicts.values()
            if v["gate"] == "hold"
        ]
        summary["targets"] = targets(after_reports)
        summary["blocked_outside"] = {  # paused this round, and still where it ended
            "before": paused,
            "after": blocked_outside(after_reports, self.outside),
        }
        if closed is not None:
            summary["closure"] = closed
            summary["plans_accepted"] = sorted(closed["plans"])
            summary["cost"]["tokens"] += sum(s["tokens"] for ss in closed["sessions"].values()
                                             for s in ss)  # fmt: skip
        if (here / "replanned.json").is_file():
            summary["replanned"] = read_json(here / "replanned.json")
        write_json(here / "summary.json", summary)
        (here / "summary.md").write_text(render(summary) + "\n", encoding="utf-8")

    def counts(self, here: Path) -> dict[str, int]:
        """The round's competing sessions per cluster and per task, written when it first runs
        its sessions (`variants.json`): a resumed round keeps them, whatever the driver's now."""
        path = here / "variants.json"
        if not path.is_file():
            write_json(path, {"clusters": self.variants, "tasks": self.task_variants})
        return {k: int(v) for k, v in read_json(path).items()}

    def early(
        self, here: Path, before: dict[str, Any], paused: set[str]
    ) -> "asyncio.Future[dict[str, Any]] | None":
        """The per-task closure's early stage: the tasks that chain to the end below their
        target in the round's coverage (`before`, `state`) get their per-task sessions from the
        round's start, beside the clusters and ahead of them (`EARLY`), judged against the
        round's coverage and a snapshot of the skills accepted at its start
        (`tasks-early/accepted/`: the round's acceptance changes `<run>/accepted` under them,
        and a plan is paired against one fixed skill set). Their plans are written with the late
        stage's (`closure`), after the round's regression, so the round's sweeps never apply them.
        None where there are none, or where the round's acceptance ran before any early stage
        did (a resumed round: the late stage takes them)."""
        stage = here / "tasks-early"
        if not stage.is_dir():
            tasks = sorted(t for t, s in before.items()
                           if s.get("to_end") and not met(s) and t not in paused)  # fmt: skip
            if not tasks or (here / "gate.json").is_file():
                return None
            made = stage.with_name(stage.name + ".tmp")
            if made.exists():
                shutil.rmtree(made)
            (made / "accepted").mkdir(parents=True)
            given, pending = self.plan_skills(here)  # the plans' trials ran with them
            names = {n: given / t / n for t in sorted(tasks) for n in pending.get(t, [])}
            for d in self.accepted_dirs() + [names[n] for n in sorted(names)]:
                copy_skill(d, made / "accepted" / d.name)
            write_json(made / "tasks.json", tasks)
            made.rename(stage)
        tasks = list(read_json(stage / "tasks.json"))
        return asyncio.ensure_future(self.closure_stage(here, stage, here / "coverage", tasks,
                                                        stage / "accepted", EARLY))  # fmt: skip

    async def closure(
        self, here: Path, final: Path, early: "asyncio.Future[dict[str, Any]] | None" = None
    ) -> dict[str, Any]:
        """The per-task closure after the round's acceptance: the late stage (`closure_stage`), one
        session (`task_variants` competing) per task below its target in the round's final reports
        that the early stage (`early`) did not take, against the accepted skills; then the early
        stage awaited. Each task's best accepted plan (`best_plan`) becomes its plan
        (`<run>/plans/<task>.json`); the stages' seeds are the next round's
        (`tasks/seeds/`). Written once, as `tasks/closure.json`; a resumed stage keeps what its
        sessions did."""
        done = here / "tasks" / "closure.json"
        if done.is_file():
            return dict(read_json(done))
        taken = here / "tasks-early" / "tasks.json"
        took = set(read_json(taken)) if taken.is_file() else set()
        reports = load(final)
        paused = blocked_outside(reports, self.outside)
        below = sorted(t for t, s in state(reports).items()
                       if not met(s) and t not in took and t not in paused)  # fmt: skip
        late = await self.closure_stage(here, here / "tasks", final, below, self.accepted, OTHER)
        first = await early if early is not None else None
        parts = [x for x in (first, late) if x is not None]
        chosen = {task: v for x in parts for task, v in x["plans"].items()}
        for task, v in sorted(chosen.items()):
            plans.write(self.plans / f"{task}.json", plans.read(Path(v["plan"])))
        record = {
            "tasks": sorted(t for x in parts for t in x["tasks"]),
            "early": first["tasks"] if first else [],
            "use": "P1 evidence only: plans stay under the run, not in the skillset export, not "
            "handed to S1, S2 or P3",
            "sessions": {label: ss for x in parts for label, ss in x["sessions"].items()},
            "plans": {t: {k: v[k] for k in ("reason", "before", "after", "skills", "key",
                                             "session")} for t, v in sorted(chosen.items())},
            "seeds": sorted({n for x in parts for n in x["seeds"]}),
        }  # fmt: skip
        write_json(done, record)
        return record

    async def closure_stage(
        self, here: Path, stage: Path, final: Path, tasks: list[str], accepted: Path, rank: int
    ) -> dict[str, Any]:
        """One stage of the per-task closure: per task of `tasks`, the round's `counts` of competing
        sessions side by side (seated at `rank`; a first, then relays on progress, as a cluster's),
        each with its task's starting plan (`plans.prefilled`) from the reports in `final`, its
        official success condition and its own approach (`TASK_STRATEGIES`); each plan judged
        (`plan_verdict`) against the task's current plan's trial, with the skills `accepted`.
        Per task, the best accepted plan (`best_plan`, not yet written); the skills of its
        sessions the gate passed, and those the chosen plan runs, copied to `tasks/seeds/`
        (`seed_verdicts`). Written once, as `<stage>/stage.json`."""
        done = stage / "stage.json"
        if done.is_file():
            return dict(read_json(done))
        reports = load(final)
        open_ids = {g.id for c in cluster(reports).clusters for g in c.gaps + c.deferred}
        said = getattr(self.bench, "success_condition", None)
        n = self.counts(here)["tasks"]
        index = {t: i for i, t in enumerate(self.bench.tasks())}
        runs_: dict[str, TaskRun] = {}
        for task in tasks:
            kept = self.plans / f"{task}.json"
            current = plans.read(kept) if kept.is_file() else None
            start = plans.prefilled(task, reports[task][0].get("chain"), current)
            context = TaskContext(task, start.model_dump(mode="json", exclude_defaults=True),
                                  str(said(task)) if callable(said) else "",
                                  center=self.center(task))  # fmt: skip
            trial = asyncio.ensure_future(self.start_trial(stage, task, start, final, accepted))
            runs_[task] = TaskRun(context, final, start.key(), trial, stage / task, accepted)
        seats = self.seated()
        measures = baseline(reports)

        async def one(task: str, v: int) -> tuple[str, str, list[dict[str, Any]]]:
            gaps = task_gaps(task, *reports[task])[0]
            number = index[task] if task in index else len(index) + tasks.index(task)
            suffix = f"_t{number}" + (f"_v{v}" if n > 1 else "")
            c = Cluster(f"task-{task}" + (f"-v{v}" if v > 1 else ""), "task", "target", gaps, [],
                        suffix, v if n > 1 else 0)  # fmt: skip
            async with seats.take(rank if v == 1 else SPARE):
                found = await self.attempts(stage, c, open_ids, measures, runs_[task])
            return task, c.id.removeprefix("task-"), found

        try:
            ran = await asyncio.gather(*(one(t, v) for t in tasks for v in range(1, n + 1)))
        finally:
            for x in runs_.values():
                for future in (x.start, x.second):
                    if future is not None:
                        future.cancel()
        chosen: dict[str, Any] = {}
        seeds: list[str] = []
        for _, _, records in ran:
            for r in records:
                if r.get("plan") is not None:
                    r["plan"] = rejudged(r["plan"])
        for task in tasks:
            mine = [(label, records) for t, label, records in ran if t == task]
            verdicts = [(label, r["plan"]) for label, records in mine for r in records
                        if (r.get("plan") or {}).get("accepted")]  # fmt: skip
            best = best_plan(verdicts)
            if best is not None:
                label = best[0]
                chosen[task] = best[1] | {"session": label, "competing": len(verdicts)}
                mine.sort(key=lambda x: x[0] != label)  # its records first (`seed_verdicts`)
            for v in seed_verdicts([r for _, records in mine for r in records]):
                target = here / "tasks" / "seeds" / v["name"]
                if not target.exists():
                    copy_skill(Path(v["dir"]), target)
                if content_hash(target) != v["sha256"]:
                    raise RuntimeError(f"{target}: not the snapshot validated ({v['sha256']})")
                seeds.append(v["name"])
        keys = ("status", "tokens", "wall_s", "case_runs")
        record = {
            "tasks": tasks,
            "sessions": {
                label: [
                    {key: r["outcome"][key] for key in keys}
                    | {
                        "attempt": r["attempt"],
                        "decision": r["decision"],
                        "progress": r["progress"],
                        "plan": r.get("plan"),
                    }
                    for r in records
                ]
                for _, label, records in sorted(ran, key=lambda x: x[1])
            },
            "plans": chosen,
            "seeds": sorted(set(seeds)),
        }
        write_json(done, record)
        return record

    async def recheck(
        self, here: Path, sessions: dict[str, list[dict[str, Any]]], open_ids: set[str]
    ) -> None:
        """The held verdicts (`hold`: no case judged) judged once more, after the sessions
        (the infrastructure may have recovered); their attempt records rewritten."""
        for cid, records in sessions.items():
            for record in records:
                held = [i for i, v in enumerate(record["verdicts"]) if v["gate"] == "hold"]
                if not held or record.get("rechecked"):
                    continue
                for i in held:
                    v = record["verdicts"][i]
                    directory = Path(v["dir"]).parent.parent
                    record["verdicts"][i] = await self.judge(directory, v, open_ids) | {
                        "rechecked": True
                    }
                if any(v["gate"] == "pass" for v in record["verdicts"]):
                    record["decision"] = "candidate"
                record["rechecked"] = True
                write_json(here / "sessions" / cid / f"attempt-{record['attempt']}.json", record)

    async def carried(self, k: int, here: Path, open_ids: set[str]) -> list[dict[str, Any]]:
        """The last round's skills no case could judge (its summary's `not_judged`), gated again
        on this round's open gaps (`held/<name>`); written once, as `held.json`."""
        path = here / "held.json"
        if path.is_file():
            return list(read_json(path))
        previous = self.run / f"round-{k - 1}" / "summary.json"
        found = []
        for h in read_json(previous).get("not_judged", []) if k and previous.is_file() else []:
            source = Path(h["dir"])
            if content_hash(source) != h["sha256"]:
                continue  # changed since validated: judged no more
            v = await self.gate(here / "held", source, open_ids)
            found.append(v | {"cluster": "held", "attempt": -1})
        write_json(path, found)
        return found

    async def seeded(self, here: Path) -> list[dict[str, Any]]:
        """The skills given to the run (`seeds`, each a skill's directory: another run's gate
        passed it) entering this round as candidates: a snapshot (`seeds/<name>`) and its content
        hash, the bench's validator again (static check, contract), then the round's regression
        like any candidate, with no case runs. A seed (its source and hash) an earlier round took
        is not taken again; a name accepted already fails. Written once, as `seeds.json`."""
        path = here / "seeds.json"
        if path.is_file():
            return list(read_json(path))
        k = int(here.name.removeprefix("round-"))
        last = self.run / f"round-{k - 1}" / "tasks" / "seeds"  # the per-task closure's
        sources = self.seeds + (sorted(p for p in last.iterdir() if p.is_dir())
                                if k and last.is_dir() else [])  # fmt: skip
        taken = {
            (v["seed"], v["sha256"])
            for p in self.run.glob("round-*/seeds.json")
            for v in read_json(p)
        }
        found = []
        for source in sources:
            digest = await asyncio.to_thread(content_hash, source)
            if (str(source), digest) in taken:
                continue
            snapshot = here / "seeds" / source.name
            await asyncio.to_thread(copy_skill, source, snapshot)
            if await asyncio.to_thread(content_hash, snapshot) != digest:
                raise RuntimeError(f"{snapshot}: not the copy of {source}")
            checked = await self.bench.validate(snapshot)
            name = checked.get("name") or source.name
            v: dict[str, Any] = {
                "name": name, "dir": str(snapshot), "sha256": digest, "seed": str(source),
                "kinds": list(checked.get("kinds") or []),
                "cases_yaml": list(checked.get("cases") or []),
                "static": list(checked.get("findings") or []),
                "cases": [], "transfer": [], "cluster": "seed", "attempt": -1,
            }  # fmt: skip
            if not checked.get("ok"):
                v |= {"gate": "fail", "reason": "static check"}
            elif (self.accepted / name).exists():
                v |= {"gate": "fail", "reason": f"name {name!r} taken"}
            else:
                v |= {"gate": "pass", "reason": f"seeded from {source}"}
            found.append(v)
        write_json(path, found)
        return found

    def summarize(
        self,
        k: int,
        found: Clusters,
        before: dict[str, Any],
        after_reports: dict[str, Any],
        sessions: dict[str, list[dict[str, Any]]],
        verdicts: dict[str, Any],
        losses: list[dict[str, Any]],
        kept: list[str],
        costs: dict[str, Any],
        timing: dict[str, Any],
        final: Path,
    ) -> dict[str, Any]:
        after = state(after_reports)
        causes: dict[str, int] = {}
        for task, (report, reuse) in after_reports.items():
            open_, infra, ceiling = task_gaps(task, report, reuse)
            for g in open_:
                causes[g.cause] = causes.get(g.cause, 0) + 1
            if infra:
                causes["not judged"] = causes.get("not judged", 0) + len(infra)
            if ceiling:
                causes["success not reached"] = causes.get("success not reached", 0) + 1
        records = [r for rs in sessions.values() for r in rs]
        slot_s = sum(c.get("slot_s", 0.0) for c in costs.values())
        slot_s += sum(c.get("seconds", 0.0) for r in records for c in r["cases"])
        slot_s += sum(v.get("case_s", 0.0) for r in records for v in r["verdicts"])
        return {
            "round": k,
            "state": final.name,
            "metrics": paired(before, after),
            "totals_unpaired": {"before": totals(before), "after": totals(after)},
            "tasks": {t: {"before": before.get(t), "after": after.get(t)} for t in sorted(after)},
            "totals": {
                "explained_before": sum(v["explained"] for v in before.values()),
                "explained_after": sum(v["explained"] for v in after.values()),
                "segments": sum(v["segments"] for v in after.values()),
                "ceiling_all_before": sum(
                    v["latched"] == v["repeats"] > 0 for v in before.values()
                ),
                "ceiling_all_after": sum(v["latched"] == v["repeats"] > 0 for v in after.values()),
                "tasks": len(after),
            },
            "open_gaps_by_cause": dict(sorted(causes.items())),
            "refused_fits": refused_fits(after_reports),
            "clusters": [
                {
                    "id": c.id,
                    "gaps": len(c.gaps),
                    "deferred": len(c.deferred),
                    "tasks": c.tasks,
                    "sessions": [
                        {
                            key: r["outcome"][key]
                            for key in ("status", "tokens", "wall_s", "case_runs")
                        }
                        | {
                            "attempt": r["attempt"],
                            "progress": r["progress"],
                            "decision": r["decision"],
                        }
                        for r in sessions.get(c.id, [])
                    ],
                }
                for c in found.clusters
            ],
            "relays": [
                {
                    "cluster": cid,
                    "attempt": r["attempt"] + 1,
                    "reason": r["progress"],
                    "after": r["outcome"]["status"],
                }
                for cid, rs in sorted(sessions.items())
                for r in rs
                if r["decision"] == "relay"
            ],
            "restarts": [
                {"cluster": cid, "attempt": r["attempt"] + 1, "after": r["outcome"]["status"]}
                for cid, rs in sorted(sessions.items())
                for r in rs
                if r["decision"] == "retry"
            ],
            "accepted": kept,
            "rejected": [
                {
                    "name": v["name"],
                    "cluster": v["cluster"],
                    "attempt": v["attempt"],
                    "reason": v["reason"],
                }
                for v in verdicts.values()
                if v["verdict"] == "rejected"
            ],
            "regression": losses,
            "cost": {
                "tokens": sum(r["outcome"]["tokens"] for r in records),
                "gpu_slot_hours": round(slot_s / 3600, 2),
                "session_wall_h": round(sum(r["outcome"]["wall_s"] for r in records) / 3600, 2),
                "wall_h": round((timing["ended"] - timing["started"]) / 3600, 2),
            },
        }

    async def regate(self, source: Path) -> list[dict[str, Any]]:
        """Gate steps 1-3 (transfer) again on every skill that passed them in a finished
        run `source` (its rounds' `candidates/`: accepted or dropped by the regression), with this
        gate, against the open gaps of the round that judged it. No session, no sweep; `source`
        is only read. The skills are gated side by side (the bench's slots bound the runs), each
        in a directory of its own (`round-<k>/gate/<skill>/`). Writes `<run>/regate.json` and
        `regate.md`; returns the rows, in `source`'s order."""

        async def one(k: int, skill: Path, v: dict[str, Any], open_ids: set[str]) -> dict[str, Any]:
            now = await self.gate(self.run / f"round-{k}" / "gate" / skill.name, skill, open_ids)
            return {
                "round": k, "name": v["name"],
                "was": {key: v.get(key) for key in ("gate", "verdict", "reason")},
                "now": {key: now.get(key) for key in ("gate", "reason", "sha256")},
                "cases": now.get("cases", []), "transfer": now.get("transfer", []),
            }  # fmt: skip

        found = await asyncio.to_thread(passed, source)
        rows = list(await asyncio.gather(*(one(*x) for x in found)))
        write_json(self.run / "regate.json", rows)
        lines = [
            f"# Re-gate of {source.name}",
            "",
            "| round | skill | was | now |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {r['round']} | {r['name']} | {r['was']['verdict'] or r['was']['gate']}: "
            f"{r['was']['reason']} | {r['now']['gate']}: {r['now']['reason']} |"
            for r in rows
        ]
        (self.run / "regate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return rows

    def plan(self) -> str:
        """What a run would do from here (`--dry-run`): nothing is written or launched."""
        tasks = self.bench.tasks()
        lines = [f"run: {self.run}", f"tasks ({len(tasks)}): {', '.join(tasks)}"]
        k = 0
        while (self.run / f"round-{k}" / "summary.json").is_file():
            k += 1
        cap = "no cap" if self.rounds is None else f"of {self.rounds}"
        if self.until_target:
            lines.append(
                f"next round: {k} ({cap}; stop when every task is at its target, or at "
                f"{self.run / STOP}; a round in which none of {', '.join(GROWS)} grew is logged, "
                "not a stop)"
            )
        else:
            lines.append(
                f"next round: {k} {cap} (stop early after a round in which none of "
                f"{', '.join(GROWS)} grew, or at {self.run / STOP})"
            )
        if self.seeds:
            lines.append("seeds (candidates of the next round): " + ", ".join(map(str, self.seeds)))
        here = self.run / f"round-{k}"
        source: Path | None
        if (here / "coverage.done").is_file():
            source = here / "coverage"
            lines.append(f"coverage: done ({source})")
        elif k == 0 and self.import_baseline:
            source = self.import_baseline
            have = [t for t in tasks if (source / f"{t}.json").is_file()]
            lines.append(f"coverage: import {source} ({len(have)}/{len(tasks)} tasks)")
        else:
            source = None
            lines.append("coverage: sweep, as:")
            lines += [f"  {c}" for c in self.bench.describe(here / "coverage", [self.accepted])]
        if source is not None:
            found = cluster({t: r for t, r in load(source).items() if t in tasks})
            if self.only is not None:
                found = Clusters([c for c in found.clusters if c.id in self.only], [], [])
            lines += ["", "clusters (one session each, relays as needed):", table(found), ""]
            n = len(found.clusters) * self.variants
            lines.append(f"sessions: {n} ({self.variants} competing per cluster), at most "
                         f"{self.sessions_at_once} sessions of every kind at once")  # fmt: skip
        r = self.limits.scaled(RELAY_SCALE)
        lines += [
            f"session budget: {self.limits.case_runs} case runs, {self.limits.wall_s / 3600:g} h, "
            f"{self.limits.tokens:,} tokens; up to {self.limits.relays} relays per cluster, each "
            f"{r.case_runs} case runs, {r.wall_s / 3600:g} h, {r.tokens:,} tokens",
            "a case run, as:",
        ]
        out = here / "sessions/<id>/attempt-0/cases/01.json"
        lines += [f"  {c}" for c in self.bench.describe(out, [Path("<skill snapshot>")])[-1:]]
        lines += [
            "gate: a snapshot of each skill (content hash kept), validated in a sandboxed child "
            "process (static check, then contract); the driver runs its cases.yaml gaps that are "
            f"open gaps of the round (at most {GATE_CASES}, >=1 explained, none judged: held), "
            f"then the transfer of up to {TRANSFER_GAPS} explained ones to other demonstrations "
            "(>=1 explained where one tells); rerun of every task with accepted + candidates, "
            "per task explained segments and judged ceiling repeats paired against the round's "
            "coverage, each loss confirmed by a second paired sweep: a candidate whose tasks "
            "lose (or that a job was stopped inside) is dropped, one no task uses is rejected; "
            "the kept ones' sweep is compared the same way",
            f"stalled sessions restart with what is left ({STALL_RETRIES} at most); cases and "
            f"sweep steps bounded ({CASE_TIMEOUT_S:.0f} s, {SWEEP_TASK_TIMEOUT_S:.0f} s)",
            f"accepted -> {self.accepted}/<name>/ and the benchmark's written skills",
        ]
        return "\n".join(lines)


def grew(summary: dict[str, Any]) -> list[str]:
    """The `GROWS` totals a round's final state has more of than its coverage had."""
    m = summary["metrics"]
    return [key for key in GROWS if m["after"].get(key, 0) > m["before"].get(key, 0)]


def stopped(summary: dict[str, Any], until_target: bool = False) -> bool:
    """Whether the run stops after this round: every task is at its target (`targets`, `met`),
    or none of `GROWS` grew, no skill is held over to be judged next round, and the per-task closure
    neither accepted a task plan (its rise is measured from the next round on) nor left seeds
    (skills the next round's regression judges). `until_target`: only the first. A round
    summarized before the chained step (no `metrics`) measured none of them: the run goes on."""
    if "metrics" not in summary:
        return False
    found = summary.get("targets") or {}
    if found and all(x["met"] for x in found.values()):
        return True
    if until_target:
        return False
    closed = summary.get("closure") or {}
    return not (grew(summary) or summary.get("not_judged") or summary.get("plans_accepted")
                or closed.get("seeds"))  # fmt: skip


def all_paused(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Why a run to the target (`until_target`) stops after this round though a task is below
    its target: every such task was paused outside skill writing (`blocked_outside`) when the
    round began and still is where it ended, and the round opened no session of either kind,
    held no skill over, accepted no task plan and left no seeds, so the next round would do the
    same. None where a task below its target is not paused (never stopped then), every task is
    at its target (`stopped`), or the round measured nothing (no `metrics`)."""
    found = summary.get("targets") or {}
    unmet = sorted(t for t, x in found.items() if not x["met"])
    if "metrics" not in summary or not unmet:
        return None
    paused = summary.get("blocked_outside") or {}
    before, after = paused.get("before") or {}, paused.get("after") or {}
    if any(t not in before or t not in after for t in unmet):
        return None
    closed = summary.get("closure") or {}
    opened = sum(len(c["sessions"]) for c in summary.get("clusters", []))
    opened += sum(len(ss) for ss in (closed.get("sessions") or {}).values())
    if opened or summary.get("not_judged") or summary.get("plans_accepted") or closed.get("seeds"):
        return None
    causes = sorted({c for t in unmet for c in after[t]["causes"]})
    line = (
        f"stopped: every task below its target ({len(unmet)} of {len(found)}: "
        f"{', '.join(unmet)}) is paused outside skill writing ({'; '.join(causes)}) and no "
        "session could open; restart after a core fix or with other --outside causes"
    )
    return {"unmet": unmet, "causes": {t: after[t]["causes"] for t in unmet}, "line": line}


def passed(source: Path) -> list[tuple[int, Path, dict[str, Any], set[str]]]:
    """Per skill a finished run's gate passed (its round's `candidates/`): the round, the skill,
    its verdict, the round's open gaps."""
    found = []
    for summary in sorted(source.glob("round-*/summary.json")):
        there = summary.parent
        if not (there / "gate.json").is_file():
            continue
        clusters = cluster(load(there / "coverage")).clusters
        open_ids = {g.id for c in clusters for g in c.gaps + c.deferred}
        for v in read_json(there / "gate.json").values():
            skill = there / "candidates" / v["name"]
            if v["gate"] == "pass" and skill.is_dir():
                found.append((int(there.name.split("-")[1]), skill, v, open_ids))
    return found


def ci(side: dict[str, Any], key: str, of: str) -> str:
    """The official success's Wilson 95% interval (the ceiling repeats are samples), else ""."""
    if key != "success" or not side.get(of):
        return ""
    low, high = wilson(int(side[key]), int(side[of]))
    return f" ({low:.2f}-{high:.2f})"


def render(summary: dict[str, Any]) -> str:
    t = summary["totals"]
    lines = [f"# P1-4 round {summary['round']}", ""]
    if summary.get("stopped_paused"):
        lines += [f"Run {summary['stopped_paused']['line']}.", ""]
    if summary.get("rebaselined"):
        lines += [f"Coverage swept again (re-baselined): {summary['rebaselined']}.", ""]
    if "metrics" in summary:
        b, a = summary["metrics"]["before"], summary["metrics"]["after"]
        rows = [
            ("chained segments explained", "chained", "chain_segments"),
            ("tasks that chain to the end", "to_end", "chain_tasks"),
            ("official success (ceiling runs)", "success", "runs"),
            ("official stages done", "stages_done", "stages_of"),
            ("segments explained from their own start", "explained", "segments"),
        ]
        lines += ["| metric | before | after |", "|---|---|---|"]
        lines += [f"| {name} | {b[key]}/{b[of]}{ci(b, key, of)} "
                  f"| {a[key]}/{a[of]}{ci(a, key, of)} |" for name, key, of in rows]  # fmt: skip
        lines += ["", f"Grew: {', '.join(grew(summary)) or 'none'} (paired by task: what the "
                  "infrastructure broke on either side counted on neither)."]  # fmt: skip
        left = {k: v for k, v in summary["metrics"].get("excluded", {}).items() if v}
        said = "; ".join(k + " " + ", ".join(v) for k, v in left.items())
        lines += [f"Left out: {said}."] if left else []
        lines.append("")
    lines.append(
        f"Segments explained {t['explained_before']} -> {t['explained_after']} of "
        f"{t['segments']}; tasks whose ceiling latched every repeat {t['ceiling_all_before']} -> "
        f"{t['ceiling_all_after']} of {t['tasks']}."
    )
    c = summary["cost"]
    lines += [
        "",
        f"Cost: {c['tokens']:,} tokens, {c['gpu_slot_hours']} GPU slot-hours, "
        f"sessions {c['session_wall_h']} h, round wall {c['wall_h']} h.",
        "",
        "| task | explained | chained | ceiling | stages |",
        "|---|---|---|---|---|",
    ]
    for task, v in summary["tasks"].items():
        b, a = v["before"] or {}, v["after"]
        lines.append(
            f"| {task} | {b.get('explained', '-')} -> {a['explained']}/{a['segments']} "
            f"| {b.get('chained', '-')} -> {a.get('chained', '-')}"
            f"{' (end)' if a.get('to_end') else ''} "
            f"| {b.get('latched', '-')} -> {a['latched']}/{a['repeats']} "
            f"| {b.get('stages_done', '-')} -> {a.get('stages_done', '-')}"
            f"/{a.get('stages_of', '-')} |"
        )
    lines += ["", "## Open gaps by cause", ""]
    lines += [f"- {k}: {n}" for k, n in summary["open_gaps_by_cause"].items()] or ["- none"]
    paused = summary.get("blocked_outside") or {}
    for when, said in (("before", "paused this round"), ("after", "still so where it ended")):
        tasks = paused.get(when) or {}
        lines += ["", f"## Outside skill writing, {said} ({len(tasks)})", ""]
        lines += [f"- {t}: {', '.join(x['causes'])} at {', '.join(x['segments'])} "
                  f"({x['runs']} runs)" for t, x in tasks.items()] or ["- none"]  # fmt: skip
    lines += [
        "",
        "## Clusters and sessions",
        "",
        "| cluster | gaps | attempt | status | tokens | cases | decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for cl in summary["clusters"]:
        for s in cl["sessions"]:
            lines.append(
                f"| {cl['id']} | {cl['gaps']}+{cl['deferred']} | {s['attempt']} "
                f"| {s['status'][:50]} | {s['tokens']:,} | {s['case_runs']} | {s['decision']} |"
            )
    lines += ["", f"## Relays ({len(summary['relays'])})", ""]
    lines += [f"- {r['cluster']} relay {r['attempt']}: {r['reason']}" for r in summary["relays"]]
    lines += ["", f"## Restarts after a stall ({len(summary['restarts'])})", ""]
    lines += [f"- {r['cluster']} restart {r['attempt']}" for r in summary["restarts"]]
    lines += ["", f"## Accepted ({len(summary['accepted'])})", ""]
    lines += [f"- {n}" for n in summary["accepted"]] or ["- none"]
    lines += ["", f"## Rejected ({len(summary['rejected'])})", ""]
    lines += [
        f"- {r['name']} ({r['cluster']}, attempt {r['attempt']}): {r['reason']}"
        for r in summary["rejected"]
    ] or ["- none"]
    refused = summary.get("refused_fits") or {}
    lines += ["", f"## Fits the relative rule refused ({len(refused)} skills)", "",
              "(rewrite): the skill's own gap, which its builder writes again while it is open;"
              " the others are only recorded.", ""]  # fmt: skip
    lines += [f"- {n}: {', '.join(g)}" for n, g in refused.items()] or ["- none"]
    held = summary.get("not_judged", [])
    lines += ["", f"## Not judged ({len(held)}): gated again next round", ""]
    lines += [f"- {h['name']}: {h['reason']}" for h in held] or ["- none"]
    if summary.get("carried_tasks"):
        lines += ["", "Tasks no sweep judged, carried from the round's coverage: "
                  + ", ".join(summary["carried_tasks"])]  # fmt: skip
    for sweep, tasks in summary.get("noisy_chains", {}).items():
        lines += ["", f"Chains that broke where an earlier round's held, kept from it ({sweep}, "
                  f"{len(tasks)}): " + ", ".join(tasks)]  # fmt: skip
    if summary["regression"]:
        lines += ["", "## Regression (paired, per task; a loss is confirmed by a second pair)", ""]
        for x in summary["regression"]:
            what = (
                "not judged (infrastructure)" if x.get("not_judged")
                else f"stopped inside {x['stuck']}'s act" if x.get("stuck")
                else f"{x.get('before')} -> {x.get('after')}"
            )  # fmt: skip
            noise = f"not confirmed (noise; would blame {x['noise']})" if x.get("noise") else ""
            lines.append(f"- {x['task']}: {what}; blamed {x['blamed'] or noise or 'none'}")
    for sweep, book in summary.get("ledger", {}).items():
        lines += ["", f"## Gains and confirmed losses per candidate ({sweep})", "",
                  "| skill | gains | losses | net |", "|---|---|---|---|"]  # fmt: skip
        for name, e in sorted(book.items()):
            gains = ", ".join(f"{k} +{n}" for k, n in e["gains"].items()) or "-"
            losses = ", ".join(f"{k} -{n}" for k, n in e["losses"].items()) or "-"
            lines.append(f"| {name} | {gains} | {losses} | {e['net']} |")
    if summary.get("transfer_weak"):
        lines += ["", "Accepted with one transfer case, not explained (`transfer_weak`): "
                  + ", ".join(summary["transfer_weak"])]  # fmt: skip
    lines += render_targets(summary)
    return "\n".join(lines)


def render_targets(summary: dict[str, Any]) -> list[str]:
    """summary.md's per-task targets (`targets`) and the per-task closure's sessions and plans."""
    found = summary.get("targets") or {}
    if not found:
        return []
    met_ = sum(x["met"] for x in found.values())
    lines = ["", f"## Per-task targets: {met_}/{len(found)} met", "",
             f"Target: {TARGET_TEXT} (to end: reported, not part of it). Failing check: the "
             "benchmark's success check that most runs ended without, and per float argument "
             "the median distance to where it would hold.",
             "", "| task | isolated | chained | to end | ceiling | stages | met | remaining gaps "
             "| failing check |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for task, x in sorted(found.items()):
        gaps = "; ".join(f"`{g['id'].partition('/')[2]}` {g['cause']}" for g in x["gaps"]) or "-"
        failing = x["margins"]["failing"][:1]
        check = "-"
        if failing:
            f = failing[0]
            by = ", ".join(f"{a} {d:+g}" for a, d in f["margins"].items())
            check = f"{f['stage']}: `{f['check'][:120]}` {f['runs']}/{f['of']}"
            check += f" ({by})" if by else ""
        lines.append(f"| {task} | {x['isolated']} | {x['chained'] or '-'} "
                     f"| {'yes' if x['to_end'] else 'no'} | {x['ceiling']} | {x['stages'] or '-'} "
                     f"| {'yes' if x['met'] else 'no'} | {gaps} | {check} |")  # fmt: skip
    closed = summary.get("closure")
    if closed is None:
        return lines
    lines += ["", f"## Per-task closure (P1-4b): {len(closed['tasks'])} sessions, "
              f"{len(closed['plans'])} plans accepted", "",
              "Task plans are P1's evidence only (`<run>/plans/`): not in the skillset export, "
              "not handed to S1, S2 or P3. An accepted plan is its task's plan from the next "
              "round on; the skills a session wrote enter the next round as seeds.", "",
              "| task | attempt | status | tokens | cases | decision | plan |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for task, sessions in closed["sessions"].items():
        for s in sessions:
            plan = s.get("plan") or {}
            lines.append(f"| {task} | {s['attempt']} | {s['status'][:40]} | {s['tokens']:,} "
                         f"| {s['case_runs']} | {s['decision']} "
                         f"| {plan.get('reason', '-')[:120]} |")  # fmt: skip
    lines += ["", "Seeds for the next round: " + (", ".join(closed["seeds"]) or "none")]
    return lines
