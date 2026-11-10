"""S2 `practice` (docs/concepts/learning.md): distil the task's Memory over new seeds.

    M = the start version (v_S1); k = 0; firsts = []      (the counted seeds' try 1, in order)
    for seed = first_seed, first_seed + 1, ...:
        stop, freezing M, if >= stop_passes (3) of the last stop_window (5) firsts passed
            ("converged"; with freeze `wilson`: if the last >= freeze_seeds counted seeds all
            read M, their first-try rate's Wilson 95% lower bound is >= freeze_lower and none
            ended as an unconfirmed finish) or k == max_seeds (60: "not converged")
        the env's setup check fails -> skip, not counted
        k += 1; (alpha_k, B_k) = schedule(k, stats); one instruction for every try
        try 1 with M: success -> pass; firsts += [pass]
                      failure -> firsts += [fail]; C = copy(M); the Reviser edits C within B_k
        try 2 with C: success -> merge C into M, a jitter if C == M; next seed
                      failure -> the Reviser edits C within what is left
        try 3 with C: success -> merge (or jitter); failure -> discard C (skip), keeping only
                      its done criteria that refuse its false finishes (`keep_done`)

Only try 1 counts, and M changes at most once per seed, never after a first-try pass. The env's
setup check (an optional capability an env declares, such as an expert pre-run that tells a
solvable seed) runs once per seed, at try 1's open: tries 2 and 3 reopen the seed without it and
with try 1's instruction (the env's reopen contract: only the seed whose check passed last). The
Reviser reads the tries' logs as the agent saw them with their failure labels, in mode (b) the
ground-truth diagnosis, the cache's diff, the budget and the
task's rejected edits, and with `demo_truth` the demonstration's ground-truth layer as guidance
(of the task, at the seed its S1 ran on; the Actor never reads it, and a practice resumes only as
it started). The Actor reads the demonstration at the level `demo_context` sets (none by default:
a clean one-shot; `goal`, `ops`, `full`: `roles.demo_context_brief`) from the package `demo`,
the same one S1 read, in its brief; with `full` its episode also holds the recording's frames as
`demo:<n>`. The level goes into every try's `config.json`, `start.json` and the summary, and a
practice resumes only at the level it started at. The Reviser's episode holds every try's frames,
aliased `log:<try>:<turn>`, and each try's scene registry as it ended (read-only), under the
object aliases the logs use; its prompt and brief are made for each run, and a voided run's edits
are undone. The objects are identified once per seed and roles: tries 2 and 3 start from try 1's
identification (its registry and outcome, logged as a reuse naming the try) unless the Reviser
changed OBJECTS or the scene drifted, which makes the try identify again; each try's ledger
record says which it did. Under done criteria
(`tools.done_gate`) a try's `finish` the official check did not confirm fails it
(`finish_unconfirmed`), and the Reviser sees the criteria at its end to tighten them, plus the
confirmed successes since the seed's last revision the criteria would have refused, to loosen
them (`roles.criteria_misses`): a try ends where its success check first holds, as the
demonstration's last frame, the criteria's target, does. Infrastructure outcomes are re-run
without using a try, and a seed whose setup check keeps faulting is unsolvable.

S2 starts from the version a passed S1 left (`S1/<seed>/exit.json`) unless `start` names one,
and records it in `start.json` (with `s1_converged`: whether an S1 passed, false when S2 starts
from the version S1's fallback named). A counted seed's try 1 is recorded as soon
as it
ends (`FirstTry`, `<seed>/first.json`), and a ledger row, with the seed's rejected edits, is
written when its seed is done (a row torn by a stop is cut first); `resume` continues after the
last row from its Memory (or the start), taking a newer version only if it is edited (committed
during a `pause_after` pause) and ignoring a merge the interrupted seed committed before its row.
An interrupted seed with a recorded try 1 keeps it (`try1/`, `first.json`) and continues from it:
try 1 never runs again and its verdict is the one counted; what came after it (the Reviser's
runs, tries 2 and 3, the cache) is set aside and runs again, the first reopen with the env's
setup check (a restarted env has not checked the seed) and try 1's instruction. Its Memory must be
the version try 1 read. Any other interrupted seed's directory is set aside and the seed run
again. A stopped practice
(`stop.json`, or a freeze record whose `stop.json` was not written yet) returns its result again.

    <task>/S2/start.json  ledger.jsonl  rejected.jsonl  stop.json
    <task>/S2/<seed>/first.json              try 1's record, as soon as it ended
    <task>/S2/<seed>/try1/ try2/ try3/       the episodes (roboscript.learning.episode)
    <task>/S2/<seed>/cache/c1/ c2/           the cache after each revision, SCRATCH included
    <task>/S2/<seed>/cache/diff_c1.json ...  the cache against M, with the budget used
    <task>/S2/<seed>/cache/reviser/r1/ r2/   the Reviser's episodes and edits.jsonl
    <task>/S2/<seed>/merge.json              the merge record, or the jitter or skip verdict
"""

import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Final, Literal

import anyio
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from roboscript.agent import read_lines
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.env import EpisodeOpened, EpisodeSpec
from roboscript.contracts.events import EventKind
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.ids import TaskName, ToolName
from roboscript.contracts.robot import Arm
from roboscript.contracts.tools import ContentPart
from roboscript.demos.manifest import content_sha256
from roboscript.hub import Hub
from roboscript.learning.context import TaskContext
from roboscript.learning.demo import (
    PACKAGE_FILE,
    TRUTH_FILE,
    DemoPackage,
    DemoTruth,
    import_blobs,
    load_demo_truth,
    load_package,
)
from roboscript.learning.episode import (
    SUMMARY,
    Episode,
    EpisodeDriver,
    EpisodeOutcome,
    InfraExhausted,
    set_aside,
    tries_frames,
)
from roboscript.learning.ledger import TryRecord, wilson
from roboscript.learning.opening import IDENTIFY, Reuse, load
from roboscript.memory import (
    DoneSpecParam,
    Editor,
    EditRecord,
    MemoryDoc,
    Version,
    VersionSource,
    carried_roles,
    content_hash,
    diff,
    merge,
    object_roles,
    open_cache,
    read_doc,
    write_doc,
)
from roboscript.memory.params import jsonable
from roboscript.perception.criteria import EndTarget
from roboscript.records import write_atomic
from roboscript.settings import repo_root
from roboscript.tools import DEMO
from roboscript.tools.done import carried_ends, end_for, load_end_targets
from roboscript.toolsets.scene import SceneConfig
from roboscript_methods.wicl.extension import (
    END_TARGETS,
    UNCONFIRMED,
    WiclEpisode,
    demo_end_targets,
)
from roboscript_methods.wicl.finish import done_criteria
from roboscript_methods.wicl.goal import (
    DemoGoal,
    end_observation,
    end_views,
    ends_pressed,
    holds_known,
    latch_hint,
    latch_motion,
    latched_lifts,
)
from roboscript_methods.wicl.imitate import EXIT
from roboscript_methods.wicl.ledger import (
    FIRST,
    FirstTry,
    Ledger,
    LedgerRow,
    Outcome,
    Progress,
    RevisionRecord,
    failure_labels,
    same_version_run,
    summary,
)
from roboscript_methods.wicl.memory import validators
from roboscript_methods.wicl.memory.arms import unbind_arms
from roboscript_methods.wicl.memory.faces import box_faces
from roboscript_methods.wicl.memory.schedule import checked_schedule, recipe_schedule
from roboscript_methods.wicl.opening import orders_of, reuse_key, roles_of, towards_of
from roboscript_methods.wicl.recurring import recurring
from roboscript_methods.wicl.roles import (
    DemoContext,
    criteria_misses,
    criteria_reads,
    demo_context_brief,
    demo_seed,
    grasped_objects,
    read_relations,
    reviser_brief,
    role_prompts,
    truth_brief,
    try_log,
)

log = logging.getLogger(__name__)

START: Final = "start.json"
STOP: Final = "stop.json"
LEDGER: Final = "ledger.jsonl"
MAX_REOPENS: Final = 3
"""How many resumes in a row may fail to reopen a seed (the env's setup check) before the seed
stops the practice (`ReopenRefused`)."""
REJECTED: Final = "rejected.jsonl"
Stop = Literal["converged", "not converged", "paused"]


class PracticeConfig(BaseModel):
    """`first_seed`: the first practice seed (the seed blocks keep S2, `evaluate` and P3 seeds
    disjoint); `max_seeds` counted seeds at most; `stop_passes` first-try passes among the last
    `stop_window` counted seeds stop early (an early stop as in training: a rate, not an
    unbroken streak, so that one odd seed does not reset it); `prerun_attempts`: tries of the
    env's setup check per seed, for an env that declares one (`EpisodeSpec.prerun_attempts`);
    `diagnosis`: the Reviser sees the ground-truth diagnosis (mode (b)); false is the
    success-only ablation; `budget_factor` multiplies the schedule's cache budget B_k (2: the
    doubled-budget ablation); `images`: key images in its brief; `start`: the version to start
    from (a number, `v003` or a hash; None: the one S1 passed with); `resume`: continue a
    practice that started; `pause_after`: stop without freezing once exactly that many seeds
    counted (an edited version committed meanwhile is where the resumed practice continues);
    `max_unsolvable`: unsolvable seeds in a row that stop it with an error; `demo_truth`: a demo
    package whose ground-truth layer the Reviser reads as guidance (relative to the repository
    root unless absolute; None, the default: off), of the task at the seed S1 ran on (its
    `exit.json`). The Actor never reads it. `demo_context`: what the Actor reads of the
    demonstration (`roles.DemoContext`; none, the default, is the clean one-shot), from the
    package `demo` (as `demo_truth`, of the task at S1's seed; needed unless none).
    `done_goal`: a demo package (as `demo`) whose final state (last keyframe, every camera)
    gates the tries' `finish` through `check_done`, whatever the Actor reads
    (`tools.done_gate`; None: `demo`'s, shown or not, else none).
    `freeze`: the early stop's rule. `window` (the default) is `stop_passes` of the last
    `stop_window` firsts, whichever versions they read: optional stopping on a short window, which
    certifies little. `wilson` freezes a version only on its own record: the last
    `freeze_seeds` or more counted seeds all read the current M, the Wilson 95% lower bound of
    their first-try rate is at least `freeze_lower`, and none of them ended as an unconfirmed
    finish (a done check that let a failure through is not a converged one); it costs seeds.
    `false_finish_edit`: after a try whose `finish` the official check did not confirm, the
    Reviser's `finish` is refused once while its run has not changed the done criteria (tightened
    a tolerance, moved a value or added a criterion): a false finish says the criteria let a
    failed scene through, so they are what to correct (off: the prompt only asks).
    `end_targets`: a file of the demonstration's end as targets (`read_end_targets`' report, as
    S1's attempts write it, `end_targets.json`), which the tries' `check_done` also measures
    (`tools.done.end_for` keeps those that carry over to the try's Memory). Without one,
    `read_end_targets` reads them from the demonstration's package (`done_goal`'s, else
    `demo`'s) on its recording alone (`extension.demo_end_targets`: perception identifies
    the Memory's roles in its first frame, no env, no brain, nothing privileged) into
    `S2/end_targets.json`, once under the start Memory's roles and again at the stop when the
    frozen Memory's roles differ, which is the file the pipeline gives P3. The start's reading
    runs beside the seeds (the tries opened after it landed carry the targets), in at most
    `end_targets_attempts` attempts of at most `end_targets_timeout_s` each (it identifies the
    roles with several VLM calls); one that fails leaves
    the tries without targets, and a resume reads them again.
    `stalled_calls`: a try ends (`stalled`, a policy failure) after this many model calls in a row
    with no env step used (`agent.session.Limits.max_stalled_calls`; None: off), as in P3.
    `latch_finish` (on): an Actor's `finish` while the simulator's success latch has not fired is
    refused (neither credited nor a hard failure) up to `finish_refusals` times, with S1's latch
    hint (`goal.latch_hint`: keep moving the way the demonstration's last motion went);
    the `finish` after those ends the try as `finish_unconfirmed`. In P3 the benchmark's own loop
    ends an episode only at the latch (or the step limit), so an episode still running is what an
    Agent knows there too (`agent.host.HostConfig.latch_finish`); only the latched flag is read,
    never the scene's ground truth. Off: the first unconfirmed `finish` ends the try.
    `base_targets_positional`: the tries' base targets (the demonstration's end in the robot's
    frame, for roles placed at a robot-frame constant) also cover a role a placement names without
    `obj=` (`tools.done.end_for`); off by default.
    `recurring_merge`: when a seed is skipped, a parameter change its cache shares with at least
    `recurring_seeds` other skipped seeds since M's version, which no skipped seed made another
    way and whose calls went through more often than with M's value beyond noise (each side over
    at least `recurring_calls` calls), is merged into M although no try of those seeds succeeded
    (`learning.recurring`). Off by default: it departs from merging only a cache that made its
    seed succeed. The candidates are written to the skipped seed's `merge.json` (`recurring`) either
    way."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first_seed: NonNegativeInt = 10000
    max_seeds: PositiveInt = 60
    stop_passes: PositiveInt = 3
    stop_window: PositiveInt = 5
    prerun_attempts: PositiveInt = 3
    diagnosis: bool = True
    budget_factor: PositiveInt = 1
    images: NonNegativeInt = 3
    start: int | str | None = None
    resume: bool = False
    pause_after: PositiveInt | None = None
    max_unsolvable: PositiveInt = 20
    demo_truth: Path | None = None
    demo_context: DemoContext = "none"
    demo: Path | None = None
    done_goal: Path | None = None
    false_finish_edit: bool = True
    end_targets: Path | None = None
    read_end_targets: bool = True
    end_targets_attempts: PositiveInt = 2
    end_targets_timeout_s: PositiveFloat = 2700.0
    stalled_calls: PositiveInt | None = 30
    latch_finish: bool = True
    finish_refusals: PositiveInt = 3
    freeze: Literal["window", "wilson"] = "window"
    freeze_seeds: PositiveInt = 10
    freeze_lower: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.45
    base_targets_positional: bool = False
    recurring_merge: bool = False
    recurring_seeds: Annotated[int, Field(ge=2)] = 3
    recurring_calls: PositiveInt = 3

    @field_validator("demo_truth", "demo", "done_goal", "end_targets")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        return path if path is None or path.is_absolute() else repo_root() / path

    @model_validator(mode="after")
    def _window(self) -> Self:
        if self.stop_passes > self.stop_window:
            raise ValueError(
                f"stop_passes {self.stop_passes} cannot pass within stop_window {self.stop_window}"
            )
        return self

    @model_validator(mode="after")
    def _demo(self) -> Self:
        if (self.demo_context == "none") != (self.demo is None):
            raise ValueError("demo names the package demo_context shows, and only then")
        return self


class PracticeResult(BaseModel):
    """How a practice stopped: `converged` (the early stop), `not converged` (the cap) or `paused`;
    the counted seeds `k`, the Memory at the end, whether it is frozen, and the ledger's
    summary."""

    model_config = ConfigDict(frozen=True)

    task: TaskName
    reason: Stop
    k: int
    version: int
    hash: str
    frozen: bool
    summary: dict[str, Any]


class ReopenRefused(Exception):
    """A resumed seed's setup check failed `MAX_REOPENS` resumes in a row: the practice stops.
    Not infrastructure: no scheduler respawns it."""


class Practice:
    name = "practice"
    Config = PracticeConfig

    async def run(self, ctx: TaskContext, config: PracticeConfig) -> PracticeResult:
        return await practice(ctx, config)


async def practice(ctx: TaskContext, config: PracticeConfig) -> PracticeResult:
    checked_schedule(ctx.memory.schedule)  # a bad schedule stops S2 before any episode
    folder = ctx.directory / "S2"
    if (folder / STOP).is_file():
        return PracticeResult.model_validate_json((folder / STOP).read_bytes())
    ledger = Ledger(folder / LEDGER)
    ledger.repair()  # a row torn by a stop would corrupt the next one
    rows = ledger.rows()
    begun = bool(rows) or (folder / START).is_file()
    if begun and not config.resume:
        raise ValueError(
            f"{folder}: this practice has started ({len(rows)} seeds in its ledger): resume it "
            "(resume: true) or use a new run"
        )
    run = _Practice(ctx, config, folder, ledger)
    await run.load_demo()
    progress = Progress.of(rows, config.first_seed)
    pending = ledger.first(progress.next_seed) if begun else None
    memory = await run.resume(rows, progress.next_seed, pending) if begun else await run.start()
    failed: Exception | None = None
    async with anyio.create_task_group() as background:
        # the reading takes minutes under a slow gateway: the seeds start meanwhile, and the
        # tries opened after it landed carry the targets
        background.start_soon(run.read_end_in_background, memory.doc)
        try:
            result = await _seeds(run, progress, memory)
        except Exception as error:  # raised as it is, not inside the group's exception group
            failed = error
        background.cancel_scope.cancel()
    if failed is not None:
        raise failed
    return result


async def _seeds(run: "_Practice", progress: Progress, memory: Version) -> PracticeResult:
    ctx, config, folder = run.ctx, run.config, run.folder
    while True:
        frozen = await ctx.store.frozen(ctx.task)
        if frozen is not None:  # frozen before stop.json was written
            return await run.stop(frozen.reason, progress, memory)  # type: ignore[arg-type]
        if converged(config, progress, memory.meta.version):
            return await run.stop("converged", progress, memory)
        if progress.k >= config.max_seeds:
            return await run.stop("not converged", progress, memory)
        if progress.skipped_in_a_row >= config.max_unsolvable:
            raise RuntimeError(
                f"{ctx.task}: {progress.skipped_in_a_row} unsolvable seeds in a row up to "
                f"{progress.next_seed - 1}; check the task's env server"
            )
        seed = progress.next_seed
        recorded = run.ledger.first(seed)
        if recorded is None:
            set_aside(folder / str(seed))
            row, memory = await run.seed(seed, progress, memory)
        else:  # interrupted after its try 1: continued from it, never run again
            _set_aside_after_first(folder / str(seed))
            row, memory = await run.continue_seed(recorded, progress, memory)
        run.commit(row)
        progress = progress.after(row)
        if row.k is not None and progress.k == config.pause_after:
            return await run.stop("paused", progress, memory)


def converged(config: PracticeConfig, progress: Progress, version: int) -> bool:
    """Whether the early stop holds for current Memory `version` (`PracticeConfig.freeze`)."""
    if config.freeze == "window":
        return sum(progress.firsts[-config.stop_window :]) >= config.stop_passes
    n, passes, unconfirmed = same_version_run(progress, version)
    if n < config.freeze_seeds or unconfirmed:
        return False
    return wilson(passes, n)[0] >= config.freeze_lower


class _Practice:
    def __init__(
        self, ctx: TaskContext, config: PracticeConfig, folder: Path, ledger: Ledger
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.folder = folder
        self.ledger = ledger
        rejected = folder / REJECTED
        lines = read_lines(rejected) if rejected.is_file() else []
        self.rejected = [EditRecord.model_validate(line) for line in lines]
        self.pending: list[EditRecord] = []  # this seed's, written with its row
        # A resumed seed the env has not checked since: its next reopen runs the setup check again.
        self.reopen_prerun: int | None = None
        self.truth: list[ContentPart] = []  # the Reviser's guidance, when configured
        # Labels the Reviser's config.json, start.json and the summary while the layer is on.
        self.shown: dict[str, str] = {}
        self.layer: DemoTruth | None = None  # the diagnosis's demonstration end tilts
        if (path := config.demo_truth) is not None:
            if (truth := load_demo_truth(path)) is None:
                raise ValueError(f"{path} has no ground-truth layer (demo_truth)")
            seeds = sorted({exit["seed"] for exit in _s1_exits(ctx.directory)})
            if truth.task != ctx.task or truth.seed not in seeds:
                raise ValueError(
                    f"{path}: its ground-truth layer is of {truth.task} seed {truth.seed}, and S1 "
                    f"of {ctx.task} ran on seeds {seeds} ({EXIT}): name that demonstration's"
                )
            digest = content_sha256(path / TRUTH_FILE)
            self.truth = truth_brief(truth)
            self.layer = truth
            self.shown = {"demo_truth": f"on for the Reviser: {path} (sha256 {digest})"}
        # The Actor's view of the demonstration: its brief and, with full, its frames.
        # Labels start.json and the summary (not the Reviser, which it does not change).
        self.package: DemoPackage | None = None
        self.demo_brief: list[ContentPart] = []
        self.demo_seed = None
        self.goal: dict[str, BlobRef] | None = None  # its final state, for check_done
        self.goal_press = False  # it pushed with empty fingers, for check_done
        self.goal_frame: BlobRef | None = None  # its final frame, for check_done's lifts
        self.goal_latched: list[str] = []  # roles whose lift latched its success: minimums
        self.goal_latch: dict[Arm, Vec3] = {}  # arms still moving into its latch, and how
        self.finish_hint = ""  # which way to keep moving at an unlatched finish (`latch_hint`)
        self.end: list[EndTarget] = []  # its end as targets, for check_done (`end_targets`)
        self.given_end = False  # from `end_targets`, not read here
        self.background: anyio.CancelScope | None = None  # the start's reading, while it runs
        self.context: dict[str, str] = {"demo_context": config.demo_context}
        self.goal_package: DemoPackage | None = None
        if (path := config.done_goal) is not None:
            self.goal_package = load_package(path)
            if self.goal_package.task != ctx.task:
                raise ValueError(
                    f"{path}: its package is of {self.goal_package.task}, not {ctx.task}"
                )
            digest = content_sha256(path / PACKAGE_FILE)
            self.context["done_goal"] = f"{path} (sha256 {digest})"
        if (path := config.demo) is not None:
            self.package = load_package(path)
            seeds = sorted({exit["seed"] for exit in _s1_exits(ctx.directory)})
            if self.package.task != ctx.task or self.package.seed not in seeds:
                raise ValueError(
                    f"{path}: its package is of {self.package.task} seed {self.package.seed}, and "
                    f"S1 of {ctx.task} ran on seeds {seeds} ({EXIT}): name that demonstration's"
                )
            self.demo_brief = demo_context_brief(self.package, config.demo_context)
            self.demo_seed = demo_seed(self.package, config.demo_context)
            digest = content_sha256(path / PACKAGE_FILE)
            self.context["demo_package"] = f"{path} (sha256 {digest})"

    async def load_demo(self) -> None:
        """Put the package's blobs into the run's store, where the Actor's brief and frames
        resolve, and take what `check_done` gates on, the final-state views and whether it
        pushed with empty fingers: `done_goal`'s, else the demonstration's own, shown in the
        Actor's context or not (the gate's criteria come from the package)."""
        if self.config.demo is not None:
            await import_blobs(self.config.demo, self.ctx.driver.blobs)
        if self.goal_package is not None:
            assert self.config.done_goal is not None
            await import_blobs(self.config.done_goal, self.ctx.driver.blobs)
        package = self.goal_package or self.package
        if package is not None:
            self.goal = await end_views(package, self.ctx.driver.blobs)
            self.goal_press = await ends_pressed(package, self.ctx.driver.blobs, self.ctx.profile)
            self.goal_frame = end_observation(package)
            self.goal_latched = latched_lifts(package, [])  # until its end targets say which
            self.goal_latch = latch_motion(package.keypoints)
            self.finish_hint = latch_hint(package.keypoints)
        if self.config.end_targets is not None:
            self.end = load_end_targets(self.config.end_targets)
            if package is not None:
                self.end = holds_known(self.end, package, self.ctx.profile)
                self.goal_latched = latched_lifts(package, self.end)
            self.given_end = True

    async def start(self) -> Version:
        """The version to start from, recorded in `start.json`."""
        ctx, start = self.ctx, self.config.start
        if start is None:
            passed = [exit for exit in _s1_exits(ctx.directory) if exit["passed"]]
            if len(passed) != 1:
                raise ValueError(
                    f"{ctx.task}: S2 starts from the version a passed S1 left, and "
                    f"{ctx.directory / 'S1'} holds {len(passed)} passed runs; name the version "
                    "to start from (start)"
                )
            version = await ctx.store.load(ctx.task, passed[0]["version"])
            if version.meta.hash != passed[0]["hash"]:
                raise ValueError(f"{ctx.task} v{version.meta.version:03d} is not the one S1 left")
        else:
            version = await ctx.store.load(ctx.task, start)
        version = await self.unbind(version)
        meta = version.meta
        criteria = len(done_criteria(version.doc))
        # whether S1 passed, or S2 starts from the version S1's fallback named
        converged = any(exit["passed"] for exit in _s1_exits(ctx.directory))
        started = {"version": meta.version, "hash": meta.hash, "s1_converged": converged}
        started |= {**self.shown, **self.context}
        _write(self.folder / START, {**started, "done_criteria": criteria})
        if not criteria and self.goal is None:  # neither criteria nor an end state to compare
            ctx.emit("done_gate", gated=False, reason="no done criteria and no demonstration end")
        return version

    async def resume(
        self, rows: list[LedgerRow], next_seed: int, pending: FirstTry | None = None
    ) -> Version:
        """The Memory after the last row (or the start), a newer edited version, if any, instead;
        a merge the interrupted seed committed before its row is ignored. With `pending`, the
        interrupted seed's recorded try 1, it must be the version that try read."""
        ctx = self.ctx
        started = json.loads((self.folder / START).read_bytes())
        shown = {**self.shown, **self.context}
        for key in ("demo_truth", "demo_context", "demo_package", "done_goal"):
            was = started.get(key, "none" if key == "demo_context" else None)
            if was != (now := shown.get(key)):
                raise ValueError(
                    f"{self.folder}: this practice started with {key} {was or 'off'}, not "
                    f"{now or 'off'}: resume it as it started"
                )
        base = rows[-1].memory_after if rows else started["version"]
        newer = [meta for meta in await ctx.store.versions(ctx.task) if meta.version > base]
        edited = [meta for meta in newer if meta.status == "edited"]
        orphans = [m for m in newer if m.status == "merged" and m.source.seed == next_seed]
        if other := [m for m in newer if m not in edited and m not in orphans]:
            shown = ", ".join(f"v{meta.version:03d}" for meta in other)
            raise ValueError(
                f"{ctx.task}: {shown} came after the ledger's Memory v{base:03d} and are neither "
                "edited nor the interrupted seed's merge; resume from a consistent run"
            )
        for meta in orphans:
            log.warning(
                "%s: ignoring v%03d, merged before seed %d's row", ctx.task, meta.version, next_seed
            )
        version = await ctx.store.load(ctx.task, edited[-1].version if edited else base)
        version = await self.unbind(version)
        if pending is not None and version.meta.version != pending.memory_before:
            raise ValueError(
                f"{ctx.task}: seed {next_seed}'s try 1 ran with Memory "
                f"v{pending.memory_before:03d}, and the practice would resume with "
                f"v{version.meta.version:03d}; the seed's later tries must start from the version "
                f"its try 1 read: remove {self.ledger.first_path(next_seed)} to run the seed "
                "again, or the newer version"
            )
        ctx.emit(
            "resume",
            seeds=len(rows),
            version=version.meta.version,
            hash=version.meta.hash,
            ignored=[meta.version for meta in orphans],
        )
        return version

    async def unbind(self, version: Version) -> Version:
        """`version`, or where it binds the left or the right arm (the `fixed_arm` lint), its
        rewrite by role (`unbind_arms`) as an edited version:
        caches could not open over it, and practice must not learn the demonstration's arm."""
        ctx = self.ctx
        rewrite = unbind_arms(version.doc)
        if not rewrite.changes or await ctx.store.frozen(ctx.task) is not None:
            return version  # frozen: the practice stops without another version
        source = VersionSource(kind="edited", note=f"unbind_arms: {len(rewrite.changes)} changes")
        unbound = await ctx.store.commit(
            rewrite.doc, status="edited", source=source, parent=version.meta.version
        )
        self.folder.mkdir(parents=True, exist_ok=True)
        _write(self.folder / f"unbind_arms-v{unbound.meta.version:03d}.json", rewrite.record())
        ctx.emit("unbind_arms", version=unbound.meta.version, changes=len(rewrite.changes))
        return unbound

    def commit(self, row: LedgerRow) -> None:
        """Write a seed's row, then its rejected edits."""
        self.ledger.append(row.model_copy(update={"demo_context": self.config.demo_context}))
        if self.pending:
            _lines(self.folder / REJECTED, self.pending, append=True)
        self.rejected += self.pending
        self.pending = []

    async def seed(
        self, seed: int, progress: Progress, memory: Version
    ) -> tuple[LedgerRow, Version]:
        ctx = self.ctx
        await ctx.hooks.on_seed_start(ctx.task, seed, memory.meta)
        ctx.emit("seed_start", seed=seed, memory=memory.meta.version)
        label = f"v{memory.meta.version:03d}"
        try:
            first, reruns = await self.attempt(
                seed, 1, memory.doc, label, None, stored=memory.meta.hash, announce=False
            )
        except InfraExhausted as error:  # the scene could not be set up: unsolvable
            if any(o.end.reason != "open_failed:sim_fault" for o in error.outcomes):
                raise
            return await self.unsolvable(seed, progress, memory, str(error)), memory
        if first.end_class == "unsolvable":
            return await self.unsolvable(seed, progress, memory, first.end.note), memory
        k = progress.k + 1
        alpha, budget = recipe_schedule(ctx.memory.schedule)(k, progress.stats)
        budget *= self.config.budget_factor
        recorded = FirstTry(
            seed=seed,
            k=k,
            instruction=first.instruction,
            alpha_k=alpha,
            budget_k=budget,
            memory_before=memory.meta.version,
            first=TryRecord.of(1, label, first, reruns, ctx.directory),
        )
        self.ledger.write_first(recorded)  # before anything else of the seed runs
        await self._announce(seed, 1, first, reruns)
        return await self._after_first(recorded, first, progress, memory)

    async def continue_seed(
        self, recorded: FirstTry, progress: Progress, memory: Version
    ) -> tuple[LedgerRow, Version]:
        """An interrupted seed from its recorded try 1 (its outcome read back from `try1/`):
        the rest of the seed, its first reopen with the env's setup check."""
        ctx, seed = self.ctx, recorded.seed
        if recorded.k != progress.k + 1 or recorded.memory_before != memory.meta.version:
            raise ValueError(
                f"{ctx.task}: seed {seed}'s recorded try 1 ({self.ledger.first_path(seed)}) "
                f"counts k={recorded.k} with v{recorded.memory_before:03d}; the ledger's next "
                f"is k={progress.k + 1} with v{memory.meta.version:03d}"
            )
        first = _read_outcome(ctx.directory / recorded.first.directory)
        if first.success != recorded.first.success:
            raise ValueError(f"{first.directory}: its summary disagrees with {FIRST}")
        ctx.emit("seed_continue", seed=seed, memory=memory.meta.version, first=first.success)
        self.reopen_prerun = seed
        return await self._after_first(recorded, first, progress, memory)

    async def _after_first(
        self, recorded: FirstTry, first: EpisodeOutcome, progress: Progress, memory: Version
    ) -> tuple[LedgerRow, Version]:
        ctx, seed = self.ctx, recorded.seed
        alpha, budget = recorded.alpha_k, recorded.budget_k
        tries = [recorded.first]
        common: dict[str, Any] = {
            "seed": seed,
            "k": recorded.k,
            "solvable": True,
            "instruction": recorded.instruction,
            "alpha_k": alpha,
            "budget_k": budget,
            "memory_before": recorded.memory_before,
        }
        if first.success:
            row = LedgerRow(
                **common,
                tries=tries,
                first_try_pass=True,
                streak_after=progress.streak + 1,
                outcome="pass",
                memory_after=memory.meta.version,
            )
            return row, memory
        assert first.instruction is not None
        # the instructions seen so far: a role description naming an object one of them varies
        # names an instance (`instance_name`)
        seen = [memory.doc.instruction, *(r.instruction for r in self.ledger.rows())]
        known = [i for i in dict.fromkeys([*seen, first.instruction]) if i]
        lint = ctx.memory.lint.model_copy(
            update={"instructions": known, "carried_ends": carried_ends(self.end)}
        )
        cache = open_cache(
            memory.doc,
            budget=budget,
            seed=seed,
            rules=ctx.memory.rules("s2"),
            lint=lint,
            validators=validators(),
        )
        done: list[tuple[int, str, EpisodeOutcome]] = [(1, recorded.first.memory, first)]
        docs = {1: memory.doc}  # what each try read
        revisions: list[RevisionRecord] = []
        outcome: Outcome = "skip"
        after = memory
        note = ""
        identified = _kept({}, first)
        for attempt in (2, 3):
            if attempt == 3:
                cache.revise()
            revision = attempt - 1
            revisions.append(
                await self.revise(seed, revision, cache, done, first.instruction, memory.doc, docs)
            )
            docs[attempt] = cache.doc
            cached = f"c{revision}"
            reuse = _reuse(identified, cache.doc, first.instruction)
            tried, reruns = await self.attempt(
                seed, attempt, cache.doc, cached, first.instruction, reuse
            )
            identified = _kept(identified, tried)
            tries.append(TryRecord.of(attempt, cached, tried, reruns, ctx.directory))
            done.append((attempt, cached, tried))
            if tried.success:
                outcome, after = await self.merge(seed, attempt, cache, alpha, budget, memory)
                break
        else:
            after, kept = await self.keep_done(seed, cache.doc, done, memory)
            verdict: dict[str, Any] = {"verdict": "skip"}
            if kept is not None:
                verdict["done_kept"] = kept
                note = f"skipped; its done criteria kept as v{after.meta.version:03d}"
            kept_as = after
            after, found = await self.recur(seed, memory, after, cache.doc, docs, done)
            if found:
                verdict["recurring"] = found
            if after is not kept_as:
                merged_as = f"recurring edits merged as v{after.meta.version:03d}"
                note = f"{note or 'skipped'}; {merged_as}"
            _write(self.folder / str(seed) / "merge.json", verdict)
            await ctx.hooks.on_skip(ctx.task, seed, "discarded")
            ctx.emit("skip", seed=seed, reason="discarded")
        row = LedgerRow(
            **common,
            tries=tries,
            revisions=revisions,
            first_try_pass=False,
            streak_after=0,
            budget_used=cache.used,
            outcome=outcome,
            memory_after=after.meta.version,
            failure_labels=failure_labels([o for _, _, o in done]),
            note=note,
        )
        return row, after

    async def keep_done(
        self,
        seed: int,
        cache: MemoryDoc,
        done: Sequence[tuple[int, str, EpisodeOutcome]],
        memory: Version,
    ) -> tuple[Version, dict[str, Any] | None]:
        """A skipped seed's done criteria that would have caught its false finishes: when a try
        ended on a finish the official check did not confirm and the cache adds or changes done
        criteria (tightened only, floors kept: the Editor's rules), each such criterion that
        holds at every confirmed success's end read so far is kept, and if together they refuse
        every false finish's end on a measured value, they alone become a new version (the rest
        of the cache goes with the seed). Without the ends' relations nothing is kept: a
        criterion is never kept unchecked."""
        false = [o for _, _, o in done if not o.success and o.end.reason == UNCONFIRMED]
        ends = [read_relations(o.directory) for o in false]
        if not ends or any(end is None for end in ends):
            return memory, None
        before = memory.doc.params
        changed = {
            name: param
            for name, param in cache.params.items()
            if isinstance(param, DoneSpecParam)
            and (name not in before or jsonable(before[name].value) != jsonable(param.value))
        }
        successes = [relations for _, relations in self.successes()]
        if not changed or not successes:
            return memory, None
        kept = {
            name: param
            for name, param in changed.items()
            if all(passed for s in successes for passed, _, _ in criteria_reads([param.value], s))
        }
        criteria = [param.value for param in kept.values()]
        refused = [
            any(not passed and measured is not None for passed, measured, _ in reads)
            for reads in (criteria_reads(criteria, end) for end in ends if end is not None)
        ]
        if not kept or not all(refused):
            return memory, None
        merged = merge(
            memory.doc,
            memory.doc.replace(params={**before, **kept}),
            1.0,  # the values the ends were checked with, not an interpolation of them
            seed=seed,
            base=memory.meta.hash,
        )
        assert merged.doc is not None
        version = await self.ctx.store.commit(
            merged.doc,
            status="merged",
            source=VersionSource(
                kind="merge",
                seed=seed,
                ref=f"S2/{seed}/merge.json",
                note="done criteria of a skipped seed that refuse its false finishes",
            ),
            parent=memory.meta.version,
            alpha=1.0,
            merge=merged.record,
        )
        record = merged.record.model_dump(mode="json")
        self.ctx.events.emit(
            EventKind.MEMORY_MERGE,
            {
                "task": self.ctx.task,
                "seed": seed,
                "version": version.meta.version,
                "record": record,
            },
        )
        self.ctx.emit(
            "merge",
            seed=seed,
            attempt=None,
            jitter=False,
            done_only=True,
            version=version.meta.version,
        )
        return version, {"version": version.meta.version, "params": sorted(kept), "record": record}

    async def recur(
        self,
        seed: int,
        memory: Version,
        after: Version,
        cache: MemoryDoc,
        docs: Mapping[int, MemoryDoc],
        done: Sequence[tuple[int, str, EpisodeOutcome]],
    ) -> tuple[Version, list[dict[str, Any]]]:
        """The changes of this skipped seed's cache that earlier skipped seeds of M's version made
        too (`learning.recurring`), and, under `recurring_merge`, those that merge, merged into
        `after` (M, or the version `keep_done` left) as a new version."""
        version = memory.meta.version
        rows = [r for r in self.ledger.rows() if r.memory_before == version and r.k is not None]
        skipped = [
            (r.seed, read_doc(self.folder / str(r.seed) / "cache" / f"c{len(r.revisions)}"))
            for r in rows
            if r.outcome == "skip" and r.revisions
        ]
        tries = [
            (
                memory.doc
                if t.attempt == 1
                else read_doc(self.folder / str(r.seed) / "cache" / f"c{t.attempt - 1}"),
                self.ctx.directory / t.directory,
            )
            for r in rows
            for t in r.tries
        ]
        tries += [(docs[attempt], outcome.directory) for attempt, _, outcome in done]
        found = recurring(
            memory.doc,
            [*skipped, (seed, cache)],
            tries,
            seeds=self.config.recurring_seeds,
            calls=self.config.recurring_calls,
        )
        found = [r for r in found if seed in r.seeds]
        shown = [r.model_dump(mode="json") for r in found]
        chosen = [r for r in found if r.merge]
        if not self.config.recurring_merge or not chosen:
            return after, shown
        params = {**after.doc.params}
        for r in chosen:
            params[r.param] = cache.params[r.param]
        merged = merge(
            after.doc, after.doc.replace(params=params), 1.0, seed=seed, base=after.meta.hash
        )
        assert merged.doc is not None
        new = await self.ctx.store.commit(
            merged.doc,
            status="merged",
            source=VersionSource(
                kind="merge",
                seed=seed,
                ref=f"S2/{seed}/merge.json",
                note="edits re-derived in skipped seeds whose calls went through more often",
            ),
            parent=after.meta.version,
            alpha=1.0,
            merge=merged.record,
        )
        record = merged.record.model_dump(mode="json")
        self.ctx.events.emit(
            EventKind.MEMORY_MERGE,
            {"task": self.ctx.task, "seed": seed, "version": new.meta.version, "record": record},
        )
        self.ctx.emit("merge", seed=seed, attempt=None, recurring=True, version=new.meta.version)
        return new, shown

    def successes(self) -> list[tuple[str, dict[str, Any]]]:
        """Every confirmed success of this practice so far with the relations read at its end
        (a try under no done criteria read none)."""
        ends = []
        for row in self.ledger.rows():
            for record in row.tries:
                if record.success:
                    found = read_relations(self.ctx.directory / record.directory)
                    if found is not None:
                        ends.append((f"seed {row.seed} try {record.attempt}", found))
        return ends

    async def attempt(
        self,
        seed: int,
        attempt: int,
        doc: MemoryDoc,
        label: str,
        instruction: str | None,
        reuse: Reuse | None = None,
        stored: str | None = None,
        announce: bool = True,
    ) -> tuple[EpisodeOutcome, int]:
        """A try of the Actor with `doc` (tries 2 and 3 reuse try 1's instruction, skip the
        env's setup check and start from `reuse`, an earlier try's identification, when it applies);
        `stored`: the version's hash when `doc` is one; `announce`: tell the hooks and the event
        log it ended (try 1 is told once its record is written, `seed`)."""
        ctx, first = self.ctx, instruction is None
        prerun = first or self.reopen_prerun == seed  # a resumed seed's first reopen checks
        spec = EpisodeSpec(
            task=ctx.task,
            seed=seed,
            stage="s2",
            instruction=instruction,
            repeat=not prerun,
            prerun_attempts=self.config.prerun_attempts,
        )

        def prompt(opened: EpisodeOpened | None) -> tuple[str, list[ContentPart] | None]:
            assert opened is not None and opened.instruction is not None
            text = role_prompts(ctx).actor(
                instruction=opened.instruction, memory=doc, rules="s2", profile=ctx.profile
            )
            return text, self.demo_brief or None

        episode = Episode(
            task=ctx.task,
            stage="s2",
            role="actor",
            directory=self.folder / str(seed) / f"try{attempt}",
            prompt=prompt,
            spec=spec,
            diagnose=True,
            memory=doc,
            memory_label=label,
            memory_hash=stored,
            reuse=reuse,
            frames=self.demo_seed,
            goal=DemoGoal(
                views=self.goal,
                press=self.goal_press,
                reference=self.goal_frame,
                end=end_for(doc, self.end, self.config.base_targets_positional),
                latched=self.goal_latched,
                latch=self.goal_latch,
                hint=self.finish_hint,
            ),
            stalled_calls=self.config.stalled_calls,
            finish_refusals=self.config.finish_refusals,
            method=WiclEpisode(
                official_finish=self.config.latch_finish,  # the latch flag alone: no target
                finish_hint=self.finish_hint,
                relations=bool(done_criteria(doc)),
                demo_end=self.layer,
            ),
            extra={
                "protocol": "practice",
                "seed": seed,
                "attempt": attempt,
                "demo_context": self.config.demo_context,
            },
        )
        outcome, reruns = await ctx.driver.run_counted(episode)
        if prerun and not first:
            self.reopen_prerun = None
            if not outcome.solvable:  # counted by its recorded try 1: no verdict to change
                self._reopen_failed(seed, attempt, outcome)
        if announce:
            await self._announce(seed, attempt, outcome, reruns)
        return outcome, reruns

    async def _announce(
        self, seed: int, attempt: int, outcome: EpisodeOutcome, reruns: int
    ) -> None:
        """Tell the hooks and the event log that a try ended (one the env could not set up is
        not a try)."""
        if outcome.solvable:
            await self.ctx.hooks.on_try_end(self.ctx.task, seed, attempt, outcome)
            self.ctx.emit("try_end", seed=seed, attempt=attempt, **_brief(outcome, reruns))

    def _reopen_failed(self, seed: int, attempt: int, outcome: EpisodeOutcome) -> None:
        """A resumed seed's setup check failed as it reopened: counted in its try 1's record;
        a resume tries again, up to `MAX_REOPENS` failures, after which the seed stops the
        practice (`ReopenRefused`)."""
        ctx, recorded = self.ctx, self.ledger.first(seed)
        assert recorded is not None
        failures = recorded.reopen_failures + 1
        self.ledger.write_first(recorded.model_copy(update={"reopen_failures": failures}))
        why = f"{ctx.task}: seed {seed}'s setup check failed as the resumed seed reopened for "
        why += f"try {attempt} ({outcome.end.note})"
        if failures >= MAX_REOPENS:
            raise ReopenRefused(
                f"{why}, {failures} resumes in a row: look at the seed's scene; remove "
                f"{self.ledger.first_path(seed)} to run the seed again from its try 1"
            )
        raise RuntimeError(f"{why} ({failures} of {MAX_REOPENS}); resume again to retry it")

    async def revise(
        self,
        seed: int,
        revision: int,
        cache: Editor,
        done: Sequence[tuple[int, str, EpisodeOutcome]],
        instruction: str,
        memory: MemoryDoc,
        docs: Mapping[int, MemoryDoc],
    ) -> RevisionRecord:
        """One Reviser run on the seed's cache, with every try's frames; its prompt and brief
        are made for each run (a voided run's edits are undone first). `docs`: the Memory each
        try read, so that a try's log names the calls its revision's edits left as they ran
        (`roles.unchanged_lines`)."""
        ctx = self.ctx
        folder = self.folder / str(seed) / "cache"
        start, before = cache.doc, len(cache.records)
        for _, _, outcome in done:
            box_faces(cache).grasped.update(grasped_objects(outcome.directory))
        frames, aliases = tries_frames([(attempt, outcome) for attempt, _, outcome in done])
        logs = [
            try_log(
                f"try {attempt}",
                label,
                outcome,
                diagnosis=self.config.diagnosis,
                aliases=aliases[attempt],
                criteria=done_criteria(memory if attempt == 1 else cache.doc),
                earlier=[(f"try {k}", o.directory) for k, _, o in done if k < attempt],
                edits=None
                if attempt == 1
                else _edited(
                    [r for r in cache.records if r.revision == attempt - 1],
                    docs[attempt - 1],
                    docs[attempt],
                ),
            )
            for attempt, label, outcome in done
        ]
        misses = criteria_misses(self.passes(), done_criteria(memory))

        def prompt(opened: EpisodeOpened | None) -> tuple[str, list[ContentPart]]:
            text = role_prompts(ctx).reviser(
                instruction=instruction, editor=cache, profile=ctx.profile
            )
            brief = reviser_brief(
                logs,
                editor=cache,
                diff=diff(memory, cache.doc),
                rejected=[*self.rejected, *self.pending],
                images=self.config.images,
                misses=misses,
            )
            return text, [*brief, *self.truth]

        episode = Episode(
            task=ctx.task,
            stage="s2",
            role="reviser",
            directory=folder / "reviser" / f"r{revision}",
            prompt=prompt,
            frames=frames,
            editor=cache,
            memory=cache.doc,
            memory_label=f"c{revision - 1}",  # c0: the copy of M the first revision starts from
            step_limit=max((o.step_limit or 0 for _, _, o in done), default=0) or None,
            extra={"protocol": "practice", "seed": seed, "revision": revision, **self.shown},
            finish_guard=self._false_finish_guard(cache, start, done),
        )
        outcome, reruns = await ctx.driver.run_counted(episode)
        records = cache.records[before:]
        rejected = [record for record in records if not record.accepted]
        _lines(episode.directory / "edits.jsonl", records)
        self.pending += rejected
        write_doc(folder / f"c{revision}", cache.doc)
        changes = diff(memory, cache.doc)
        _write(
            folder / f"diff_c{revision}.json",
            {"diff": changes.model_dump(mode="json"), "budget": cache.budget, "used": cache.used},
        )
        await ctx.hooks.on_revise(ctx.task, seed, revision, records, diff(start, cache.doc))
        ctx.emit(
            "revise",
            seed=seed,
            revision=revision,
            end=outcome.end.reason,
            accepted=len(records) - len(rejected),
            rejected=len(rejected),
            used=cache.used,
            budget=cache.budget,
        )
        edits = (len(records) - len(rejected), len(rejected))
        return RevisionRecord.of(revision, outcome, reruns, edits, ctx.directory)

    def _false_finish_guard(
        self, cache: Editor, start: MemoryDoc, done: Sequence[tuple[int, str, EpisodeOutcome]]
    ) -> Callable[[str], Awaitable[str | None]] | None:
        """The Reviser's finish guard after a false finish (`PracticeConfig.false_finish_edit`):
        a refusal while the run has left the done criteria as they were; None without one."""
        if not self.config.false_finish_edit:
            return None
        false = [a for a, _, o in done if not o.success and o.end.reason == UNCONFIRMED]
        if not false:
            return None
        before = done_criteria(start)

        async def guard(_: str) -> str | None:
            if done_criteria(cache.doc) != before:
                return None
            tries = ", ".join(f"try {a}" for a in false)
            return (
                f"{tries} finished although the official check did not confirm success: the "
                "done criteria let a failed scene pass. Before finishing, correct them with "
                "memory_edit: tighten the tolerance or move the value of the criterion that "
                "should have caught it, or define a criterion on what its end got wrong (where "
                "an object it moved ended relative to the object it was placed by, or relative "
                "to another object it moved, on the axis that differs from the demonstration's "
                "end)."
            )

        return guard

    def passes(self) -> list[tuple[str, dict[str, Any]]]:
        """The successes since the last Reviser run (a seed's first-try passes, and the try that
        passed after a seed's last revision), with the relations read at their ends: no Reviser
        runs after a success, so the next one reads which of them the Memory's done criteria
        would have refused (`criteria_misses`)."""
        rows = self.ledger.rows()
        last = next((i for i in range(len(rows) - 1, -1, -1) if rows[i].revisions), None)
        ends = []
        for row in rows if last is None else rows[last:]:
            for record in row.tries:
                if record.success:
                    found = read_relations(self.ctx.directory / record.directory)
                    if found is not None:
                        ends.append((f"seed {row.seed} try {record.attempt}", found))
        return ends

    async def merge(
        self, seed: int, attempt: int, cache: Editor, alpha: float, budget: int, memory: Version
    ) -> tuple[Outcome, Version]:
        """Merge the cache that made try `attempt` pass: a new version, or a jitter."""
        ctx = self.ctx
        merged = merge(
            memory.doc, cache.doc, alpha, seed=seed, attempt=attempt, base=memory.meta.hash
        )
        path = self.folder / str(seed) / "merge.json"
        record = merged.record.model_dump(mode="json")
        if merged.doc is None:
            _write(path, {"verdict": "jitter", "record": record})
            await ctx.hooks.on_merge(ctx.task, seed, merged.record, None)
            ctx.emit("merge", seed=seed, attempt=attempt, jitter=True)
            return "jitter", memory
        version = await ctx.store.commit(
            merged.doc,
            status="merged",
            source=VersionSource(
                kind="merge", seed=seed, attempt=attempt, ref=f"S2/{seed}/merge.json"
            ),
            parent=memory.meta.version,
            alpha=alpha,
            budget=budget,
            merge=merged.record,
        )
        meta = version.meta
        _write(
            path,
            {"verdict": "merge", "version": meta.version, "hash": meta.hash, "record": record},
        )
        ctx.events.emit(
            EventKind.MEMORY_MERGE,
            {"task": ctx.task, "seed": seed, "version": meta.version, "record": record},
        )
        await ctx.hooks.on_merge(ctx.task, seed, merged.record, meta)
        ctx.emit("merge", seed=seed, attempt=attempt, jitter=False, version=meta.version)
        return "merge", version

    async def unsolvable(
        self, seed: int, progress: Progress, memory: Version, note: str
    ) -> LedgerRow:
        ctx = self.ctx
        await ctx.hooks.on_skip(ctx.task, seed, "unsolvable")
        ctx.emit("skip", seed=seed, reason="unsolvable")
        return LedgerRow(
            seed=seed,
            k=None,
            solvable=False,
            instruction=None,
            tries=[],
            first_try_pass=None,
            streak_after=progress.streak,
            outcome="unsolvable",
            memory_before=memory.meta.version,
            memory_after=memory.meta.version,
            note=note,
        )

    async def read_end_in_background(self, doc: MemoryDoc) -> None:
        """`ensure_end` for the start Memory, cancelled by the stop (which reads for the frozen
        Memory itself) or the practice's end."""
        with anyio.CancelScope() as scope:
            self.background = scope
            await self.ensure_end(doc)
        self.background = None

    async def ensure_end(self, doc: MemoryDoc) -> None:
        """The demonstration's end targets under `doc`'s roles (`PracticeConfig.read_end_targets`,
        `read_end_targets`) as the tries' `end`."""
        if self.given_end or not self.config.read_end_targets:
            return
        package = self.goal_package or self.package
        if package is None or not package.frames:
            return
        found, why = await read_end_targets(
            self.ctx.driver,
            self.folder,
            package,
            doc,
            attempts=self.config.end_targets_attempts,
            timeout_s=self.config.end_targets_timeout_s,
        )
        if found is None:  # the tries run without; a resume (or read_end_targets.py) reads them
            log.warning("%s: the demonstration's end targets were not read: %s", self.ctx.task, why)
            self.ctx.emit("end_targets", read=False, reason=why)
            return
        self.end = holds_known(found, package, self.ctx.profile)
        self.goal_latched = latched_lifts(package, self.end)
        if why != KEPT:
            roles = sorted(object_roles(doc))
            self.ctx.emit("end_targets", read=True, targets=len(found), roles=roles)

    async def stop(self, reason: Stop, progress: Progress, memory: Version) -> PracticeResult:
        """Freeze M (unless paused) and write `stop.json`; a freeze of M made already stands. The
        end targets are read again first when M's roles are not theirs (`ensure_end`)."""
        ctx = self.ctx
        if self.background is not None:  # read again here, for the frozen Memory
            self.background.cancel()
        await self.ensure_end(memory.doc)
        shown = {**summary(self.ledger.rows()), **self.shown, **self.context}
        frozen = None
        if reason != "paused":
            frozen = await ctx.store.freeze(
                ctx.task,
                memory.meta.version,
                reason=reason,
                summary=shown,
                vision=cross_check(ctx.driver.hub),
            )
        result = PracticeResult(
            task=ctx.task,
            reason=reason,
            k=progress.k,
            version=memory.meta.version,
            hash=memory.meta.hash,
            frozen=frozen is not None,
            summary=shown,
        )
        if frozen is not None:
            _write(self.folder / STOP, result.model_dump(mode="json"))
        await ctx.hooks.on_stop(ctx.task, reason, frozen)
        ctx.emit("stop", reason=reason, version=memory.meta.version, k=progress.k)
        return result


KEPT: Final = "kept"
"""`read_end_targets`' word for targets taken from the file as it stood."""


async def read_end_targets(
    driver: EpisodeDriver,
    folder: Path,
    package: DemoPackage,
    doc: MemoryDoc,
    *,
    attempts: int = 2,
    timeout_s: float = 2700.0,
    overwrite: bool = False,
) -> tuple[list[EndTarget] | None, str]:
    """The demonstration's end targets under `doc`'s roles in `folder/end_targets.json` (an S2
    folder): the file's when its roles are `doc`'s (unless `overwrite`), else read on the
    package's recording (`extension.demo_end_targets`) in at most `attempts` attempts of at
    most `timeout_s` each and written there; None and why when no reading came (nothing is
    written then); `KEPT` for the file's."""
    path = folder / END_TARGETS
    roles = sorted(object_roles(doc))
    if path.is_file() and not overwrite:
        kept = json.loads(path.read_bytes())
        if kept.get("roles") == roles:
            return [EndTarget.model_validate(t) for t in kept["targets"]], KEPT
    keypoints = package.keypoints.model_dump(mode="json")
    seed = EpisodeSeed(frames=package.frames, facts={DEMO: keypoints})
    found, why = None, ""
    for attempt in range(1, attempts + 1):
        why = ""
        with anyio.move_on_after(timeout_s) as bounded:
            try:
                found = await demo_end_targets(
                    driver,
                    seed,
                    package.instruction,
                    roles_of(doc) or None,
                    orders_of(doc),
                    carried_roles(doc),
                    package.frames[0].alias,
                    phase="s2",
                )
            except RuntimeError as error:
                why = str(error)
        if found is not None:
            break
        if bounded.cancelled_caught:
            why = f"no reading within {timeout_s:g} s"
        elif not why:
            return None, "no perception toolset mounted"
        log.warning("the demonstration's end targets, attempt %d of %d: %s", attempt, attempts, why)
    if found is None:
        return None, why
    targets = [EndTarget.model_validate(t) for t in found["targets"]]
    _write(path, {"roles": roles, "memory": content_hash(doc), **found})
    return targets, ""


def cross_check(hub: Hub) -> bool | None:
    """Whether the one-time identification runs GroundingDINO's cross-check: the perception
    toolset has the vision service (None: no perception toolset in this process)."""
    tool = hub.mounted.get(ToolName(IDENTIFY))
    config = None if tool is None else tool.provider.config
    return config.vision is not None if isinstance(config, SceneConfig) else None


def _read_outcome(directory: Path) -> EpisodeOutcome:
    """A finished episode's outcome as its `summary.json` recorded it, where it lies now."""
    outcome = EpisodeOutcome.model_validate_json((directory / SUMMARY).read_bytes())
    return outcome.model_copy(update={"directory": directory})


def _set_aside_after_first(directory: Path) -> None:
    """Set aside what an interrupted seed left after its recorded try 1 (`set_aside`): its
    `first.json`, `try1/` and try 1's voided runs (`try1-infra<n>/`) stay."""
    aside = set_aside(directory)
    if aside is None:
        return
    directory.mkdir()
    kept = [c for c in aside.iterdir() if c.name == "try1" or c.name.startswith("try1-infra")]
    # the record last: cut in between, the seed's directory holds no record without its try 1
    for child in [*sorted(kept), *([aside / FIRST] if (aside / FIRST).is_file() else [])]:
        child.rename(directory / child.name)
    if not any(aside.iterdir()):
        aside.rmdir()


def _kept(identified: dict[str, Reuse], outcome: EpisodeOutcome) -> dict[str, Reuse]:
    """The seed's identifications by what they identified, with the one a try left."""
    made = load(outcome.directory)
    return identified if made is None else {**identified, made.key: made}


def _reuse(identified: dict[str, Reuse], doc: MemoryDoc, instruction: str) -> Reuse | None:
    """The seed's identification of `doc`'s roles under the instruction, if there is one (a
    malformed OBJECTS section has none: the try's episode reports it)."""
    try:
        key = reuse_key(roles_of(doc), instruction, towards_of(doc), orders_of(doc))
        return identified.get(key)
    except ValueError:
        return None


def _edited(records: Sequence[EditRecord], before: MemoryDoc, after: MemoryDoc) -> dict[str, str]:
    """The parameters that differ from `before` to `after`, each with the accepted edits of
    `records` (a revision's) that named it, as `op key=value ...` without the why."""
    ops: dict[str, list[str]] = {c.param: [] for c in diff(before, after).params}
    for record in records:
        name = getattr(record.op, "param", None)
        name = name if isinstance(name, str) else getattr(record.op, "name", None)
        if record.accepted and name in ops:
            shown = record.op.model_dump(mode="json")
            kind = shown.pop("op")
            rest = {
                k: v
                for k, v in shown.items()
                if k not in ("why", "param", "name") and (v is not None or k == "value")
            }
            ops[name].append(" ".join([kind, *(f"{k}={json.dumps(v)}" for k, v in rest.items())]))
    return {name: "; ".join(shown) or "changed" for name, shown in sorted(ops.items())}


def _brief(outcome: EpisodeOutcome, reruns: int) -> dict[str, Any]:
    return {
        "success": outcome.success,
        "end": outcome.end.reason,
        "end_class": outcome.end_class,
        "steps": outcome.steps,
        "turns": outcome.turns,
        "infra_reruns": reruns,
        "directory": outcome.directory.name,
        "identified": None if outcome.identification is None else outcome.identification.status,
    }


def _s1_exits(directory: Path) -> list[dict[str, Any]]:
    """The `exit.json` of every S1 run in a task's run directory."""
    return [json.loads(path.read_bytes()) for path in sorted((directory / "S1").glob(f"*/{EXIT}"))]


def _write(path: Path, data: dict[str, Any]) -> None:
    write_atomic(path, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())


def _lines(path: Path, records: Sequence[EditRecord], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")
