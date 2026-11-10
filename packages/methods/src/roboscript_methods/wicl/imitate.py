"""S1 `imitate` (docs/concepts/learning.md): learn the task's workflow from one demonstration, on
its seed.

1. The Demo Reader studies the demonstration package (its brief, with the package's
   ground-truth layer, which must be there and of the package's task and seed unless
   `demo_truth` is off) on the demo seed's live scene, with the recording's frames and the
   keypoints (the `DEMO` fact `demo_relation` reads) in its frame store (the package's blobs are
   checked and put into the runner's store first), and writes Memory v0 with typed edits under
   S1's rules (no budget); v0 becomes the first S1 draft.
2. Attempts reopen the demo seed with the demo's instruction; the env's setup check (an optional env
   capability, such as an expert pre-run) runs once per S1 run, at its first open (the env's reopen
   contract skips it only for the seed whose check passed last). The first attempt is hinted: the
   Actor reads the Memory and the package (with its ground-truth layer; its frame store seeded as
   the Demo Reader's), and after each action the keypoint progress. Whatever a hinted attempt
   scores, unhinted attempts follow (Memory only, no `demo` toolset, no ground-truth layer, at the
   same `tier`, fail-fast); `hinted_after` unhinted failures in a row insert one hinted attempt,
   after which they go on. Only unhinted passes count. Every attempt's `finish` is refused while the
   official success check is false, up to `finish_refusals` times (a hinted one with the
   ground-truth layer is told how the scene differs from the demonstration's end); a finish after
   those ends the attempt, failed.
3. The keypoint referee (`roboscript.demos`) judges each attempt from its path (the TCP along
   the joint line between the recorded steps, by FK through the task's planner, with the
   gripper command; `roboscript.demos.paths`) and the success latch: a pass is a latched
   success with every required keypoint passed in order. In-air steps are
   judged by the held object (`demos.referee.air_steps`), and a latched success that fails as
   it ran is judged again on the swapped arms (`demos.referee.swapped`: the Memory binds no
   hand), which its record names (`arms`). A latched success is judged by
   whether each step happened, not where, and a hold the
   demonstration lifted must lift its object off its support (`demos.referee.held_lifts`).
4. After a failure the Reviser edits the Memory itself (S1 rules) from each arm's first failed
   keypoint with its deviation and the turn it was closest in, the official result, the
   ground-truth diagnosis, the package's ground-truth layer and the attempt's log, with the
   attempt's frames aliased `log:<attempt>:<turn>`, and the privileged review of the attempt
   against the expert's state (`learning.expert_diff`, `review`): its first diverging stage with
   a verdict and the Memory parameter behind it; the Reviser may not set back what the previous
   revision changed unless the review names that parameter. A changed Memory becomes the next S1
   draft.
5. `exit_passes` passes among the last `exit_window` unhinted attempts (hinted ones are left
   out of the window, not counted against it) within `max_attempts` attempts exit, with the
   Memory the pass completing the window read (the latest: a pass is not revised; the window's
   earlier passes may have read older versions). On the exit a Reviser writes the done criteria (S2
   and P3 refuse `finish` until `check_done` passes them): aimed at the demonstration's final state,
   its last frame (the goal keyframe), as
   perception reads it (`read_relations` at `demo_end`, `relations_demo.json`), with its views;
   every attempt's end read the same way (`relations.json`), where there are any, only widens
   the tolerance. It edits nothing else, and criteria that all hold at a try's start are not
   kept (they would pass before the task is done). Then the latest draft is v_S1 (`exit.json`, which
   also records whether the ground-truth layer was on and the criteria step's record: how many
   criteria, or why none, and the attempts whose ends
   were not read), where `practice` starts. Otherwise S1 stops, or, with
   `fallback`, yields the best Memory it tried all the same (`fallback_ranking`). Under
   `fallback_rule` `success` only if an attempt had an official success: among the versions such
   attempts read, the one with the most unhinted successes, then the most successes in all (hinted
   too), then the later version; with no official success
   S1 stops. (`progress`, the other rule: every version, by unhinted referee passes, unhinted
   official successes, the longest prefix of required keypoints an arm passed in order, hinted
   passes and hinted successes, the later version on a tie.) The pick must be lint-clean (the
   best one the linter fails gets one lint-fix pass), with its done criteria written as on a
   pass; `exit.json` stays `passed: false` and records it as `fallback` with `converged: false`,
   `fallback_rule` and `s1_exit` (`passed`, `best_memory`, or `none` for an S1 that stops).

With `resume`, an interrupted S1 continues where it stopped: the Demo Reader's version, the
finished attempts (their verdicts and hinted flags, as they ran, replayed through the schedule to
rebuild the exit window, the failure streak and whether the next attempt is hinted) and the
Reviser's versions are kept; the attempt, revision or criteria step cut short is set aside as
`<name>-aborted<n>` and runs again (a finished criteria step is reused), and the seed's first open
makes the env's setup check again. An S1 that passed whose `exit.json` has
no `criteria` record gets them when run again, from the demonstration's final state read on its
attempts' frames (no attempt runs again), and `exit.json` names the new version.
Each attempt's `config.json` records the referee config it was judged with; a resume whose
referee has other values is refused (a finished attempt keeps its stored verdict).
The referee reads the motion starts (`since`) from the package's `segments.json` when there is
one (its hash is `referee_segments` in the config, which a resume compares as it compares the
referee), so a live run's packages keep their bytes and hash; the
agents see the package's keypoints unchanged.

The Demo Reader fits every spec on the demonstration (`demo_relation`; its `finish` is refused
once while one is not): a draft with an unfitted spec fails S1 before any attempt.

Every S1 role runs at tier P (privileged tools such as `true_objects`) unless `tier`,
`reader_tier` or `reviser_tier` say otherwise; the specs stay fitted in the coordinates S2
uses (`demo_relation`, on perceived geometry unless the demo toolset fits on the truth).

    <task>/S1/<seed>/reader/  attempt<n>/ (referee/keypoints.json, relations.json)  reviser/r<n>/
                     criteria/  exit.json
"""

import functools
import hashlib
import itertools
import json
import logging
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Final, Literal

import anyio
import numpy as np
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

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.env import EpisodeOpened, EpisodeSpec
from roboscript.contracts.errors import InfraError
from roboscript.contracts.events import EventKind
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.ids import TaskName
from roboscript.contracts.motion import Planner
from roboscript.contracts.phase import PhaseRules
from roboscript.contracts.receipt import HARD_FAILURES, ReceiptReason
from roboscript.contracts.robot import Arm, RobotState
from roboscript.contracts.tools import ContentPart, TextPart, Tier
from roboscript.demos import (
    ArmPath,
    ArmProgress,
    Keypoint,
    KeypointSet,
    ObjectShape,
    RefereeConfig,
    Verdict,
    arm_progress,
    held_placements,
    judge,
    keypoint_objects,
    keypoint_sizes,
    symmetry_axes,
)
from roboscript.demos.paths import ArmPaths
from roboscript.demos.referee import (
    Regrouped,
    Touches,
    acted_on,
    air_steps,
    attempt_holds,
    attempt_touches,
    describe,
    held_lifts,
    permute,
    regrouped,
    swapped,
)
from roboscript.hub import ActionNotes
from roboscript.learning.context import TaskContext
from roboscript.learning.demo import (
    SEGMENTS_FILE,
    TRUTH_FILE,
    DemoPackage,
    DemoTruth,
    demo_robots,
    import_blobs,
    load_demo_truth,
    load_package,
    load_segments,
    with_segments,
)
from roboscript.learning.episode import (
    EVENTS,
    PHYSICS,
    REFEREE,
    SHAPES,
    SUMMARY,
    Episode,
    EpisodeOutcome,
    Trajectory,
    set_aside,
    tries_frames,
)
from roboscript.manipulation import PlaceSpec, robot_base
from roboscript.memory import (
    DoneSpecParam,
    Editor,
    EditRecord,
    Finding,
    FitSpec,
    LintConfig,
    MemoryDiff,
    MemoryDoc,
    NumberParam,
    PlaceSpecParam,
    SetCategory,
    Version,
    VersionMeta,
    VersionSource,
    call_arguments,
    changed_targets,
    diff,
    draft_advice,
    errors,
    lint,
    render_diff,
    spec_arm_text,
    spec_arms,
    targets,
    template_refs,
    unfitted,
)
from roboscript.memory.edits import EditValidator
from roboscript.memory.params import FITTED_SOURCE, jsonable
from roboscript.perception.criteria import EndTarget
from roboscript.perception.registry import SceneRegistry
from roboscript.perception.table import Table
from roboscript.records import write_atomic
from roboscript.settings import repo_root
from roboscript.tools import DEMO, DEMO_ARMS, DEMO_FITS, DEMO_RELEASES, DEMO_TRACKS, TABLE
from roboscript.tools.done import carried_ends
from roboscript_methods.wicl.expert_diff import (
    EXPERT_DIFF,
    NOT_MEMORY,
    REFUSED,
    Call,
    ExpertDiff,
    ReviewConfig,
    attempt_calls,
    memory_specs,
    plan_check,
    progress_line,
    render,
    review,
    settled,
    spec_changes,
    stage_line,
    written_specs,
)
from roboscript_methods.wicl.expert_diff import load as load_review
from roboscript_methods.wicl.extension import (
    DEMO_RELATIONS,
    END_TARGETS,
    RELATIONS,
    UNCONFIRMED,
    WiclEpisode,
    demo_relations,
)
from roboscript_methods.wicl.finish import done_criteria
from roboscript_methods.wicl.goal import (
    RISING_M,
    DemoGoal,
    end_observation,
    end_views,
    ends_pressed,
    holds_known,
    latch_hint,
    latch_motion,
    latched_lifts,
    moving_at_latch,
)
from roboscript_methods.wicl.memory import validators
from roboscript_methods.wicl.memory.arms import unbind_arms
from roboscript_methods.wicl.memory.faces import required_fixes
from roboscript_methods.wicl.memory.review import AfterFirst, EditStreak, target_of
from roboscript_methods.wicl.opening import split_refusal, split_votes
from roboscript_methods.wicl.perception.done import lifts as lift_criterion
from roboscript_methods.wicl.perception.endstate import MOVED_M
from roboscript_methods.wicl.relation import contacts, released_poses, with_contacts
from roboscript_methods.wicl.roles import (
    cleanup_brief,
    demo_brief,
    end_brief,
    grasped_objects,
    held_at_start,
    lint_brief,
    relations_brief,
    review_brief,
    reviser_brief,
    role_prompts,
    stuck_failure,
    tipped_closes,
    truth_brief,
    try_log,
)
from roboscript_methods.wicl.spec_check import (
    Fit,
    SpecGap,
    demo_fits,
    dry_run_failed,
    dry_run_gate,
    first_registry,
    grasp_objects,
    last_fact,
    last_registry,
    object_fits,
    placement_effect,
    planned,
    render_specs,
    spec_check,
    unreproducible,
)
from roboscript_methods.wicl.toolsets.done import covers

log = logging.getLogger(__name__)
EXIT: Final = "exit.json"
VERDICT: Final = "keypoints.json"
DRY_RUN: Final = "dryrun"
"""The spec dry run before new attempt n (`dryrun<n>/`: `specs.json`, read by Reviser n)."""
AFTER: Final = "memory_after.json"
"""What the Demo Reader or a revision left the Memory as (its hash), written once it is stored:
the step is done."""
_ATTEMPT: Final = re.compile(r"attempt(\d+)")
CHANGES: Final = "changes.json"
"""What a revision changed in the Memory (`MemoryDiff`), beside its edits: the S1 Reviser's
ledger of earlier revisions reads it (`revision_ledger`)."""
LEDGER_TRIES: Final = 8
"""The earlier attempts of a seed (the latest ones) the S1 Reviser's ledger lists."""
_LEDGER_CHANGES: Final = 10  # changes listed per revision, the first ones


def segments_key(demo: Path) -> str | None:
    """What a package's motion starts are (`segments.json`, which the referee judges by): its
    bytes' hash, None without one."""
    path = demo / SEGMENTS_FILE
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None


LINT_PASSES = 2
"""How many Reviser episodes a lint-fix pass may take: a second one goes on from what the first
left when it ended with errors still on the Memory."""
LINT_FIX = "lintfix"
FALLBACK_FIX: Final = "lintfix-best"
"""Where the best version a fallback picks is fixed when the linter fails it."""
"""The lint-fix pass on a Memory the current linter fails: `lintfix<n>/` before new attempt n of
a resumed S1, `lintfix-exit<n>/` after an exit at new attempt n held because its Memory fails
the linter (no S1 exits on such a Memory: the window starts again on the fixed one)."""
LINT_RETRIES = 2
"""How often an S1 Demo Reader's or Reviser's `finish` is refused while the Memory still fails the
linter; one that still does after that is not stored."""


def _listed(findings: Sequence[Finding]) -> str:
    return "; ".join(f"{f.where}: {f.rule} {f.match!r} ({f.message})" for f in findings[:5])


def unlintable(memory: Version, ctx: TaskContext) -> list[Finding]:
    """The errors a stored Memory has once its arm bindings are rewritten (what `unbind` does
    before any edit): what a lint-fix revision must remove."""
    return errors(lint(unbind_arms(memory.doc).doc, ctx.memory.lint))


def lint_refusal(editor: Editor) -> str | None:
    """Why a `finish` is refused: the linter's errors still on the Memory (an Editor opened over
    a Memory that fails it only lets edits fix), or None."""
    left = editor.errors_left()
    if not left:
        return None
    return "the Memory still fails the linter: fix these, then finish: " + _listed(left)


async def _refuse_errors(editor: Editor, note: str) -> str | None:
    return lint_refusal(editor)


async def _refuse_errors_and_crossed_specs(editor: Editor, note: str) -> str | None:
    """The S1 Reviser's finish: refused while the linter's errors remain, and while a stage
    passes a spec fitted at another arm's keypoint (`memory.spec_arms`), and while workflow or
    lesson text takes the acting arm from a spec or skips plan_arms (`memory.spec_arm_text`)."""
    if refusal := lint_refusal(editor):
        return refusal
    if crossed := spec_arms(editor.doc) + spec_arm_text(editor.doc):
        return "before finishing, fix: " + _listed(crossed)
    return None


def _fits_used(doc: MemoryDoc) -> dict[str, str]:
    """The demonstration's fits (specs fitted on it) that a stage's call uses: name -> its value."""
    refs = {name for stage in doc.workflow for name in template_refs(stage.call, doc.params)}
    return {
        name: doc.params[name].value.model_dump_json()
        for name in sorted(refs)
        if doc.params[name].kind.endswith("_spec")
        and doc.params[name].source.startswith(f"{FITTED_SOURCE} ")
    }


async def _refuse_errors_and_dropped_fits(
    editor: Editor, original: MemoryDoc, note: str
) -> str | None:
    """A cleanup's finish is refused while errors remain, and while a fit a call used before is
    used by none now (by value: a rename keeps it): that step would have lost its spec."""
    if refusal := lint_refusal(editor):
        return refusal
    now = set(_fits_used(editor.doc).values())
    dropped = [n for n, value in _fits_used(original).items() if value not in now]
    if not dropped:
        return None
    return (
        f"the fits {', '.join(dropped)} were used by a call before and are used by none now: each "
        "step keeps the spec that carried it (rename a spec, pass it to the argument that takes "
        "it); restore it where you dropped it, then finish"
    )


def _same_segments(recorded: object, current: str | None) -> bool:
    """Whether a verdict's recorded motion starts are the current ones: a hash; a record without
    the hash, whether the package had any (True, False; absent: none)."""
    if isinstance(recorded, bool) or recorded is None:
        return bool(recorded) == (current is not None)
    return recorded == current


FREED_ARGUMENTS: Final = (
    "approach_axis",
    "constrain",
    "align_axis",
    "object_axis",
    "object_axis_type",
)
"""The approach arguments an unhinted S1 attempt may add to a stage's Memory call once that call
was rejected unreachable or plan_failed before anything moved (`ImitateConfig.freed_arguments`)."""


OWN_ARGUMENTS: Final = {"plan_arms": ("assign",)}
"""Arguments a Memory never writes but an unhinted S1 attempt may pass to a tool its stage's
call names: the Actor's own arm assignment, whose reason `plan_arms` logs."""


def stage_arguments(doc: MemoryDoc) -> dict[str, dict[str, list[str]]]:
    """The arguments each stage's Memory call names, by tool, with the tools' `OWN_ARGUMENTS`
    (`PhaseRules.call_arguments`)."""
    return {
        stage.name: {
            tool: [*args, *(a for a in OWN_ARGUMENTS.get(tool, ()) if a not in args)]
            for tool, args in call_arguments(stage.call).items()
        }
        for stage in doc.workflow
    }


FallbackRule = Literal["success", "progress"]
"""How S1's fallback picks its best Memory (`fallback_ranking`): `success` among the versions with
an official success only, `progress` (before it) among every version, by referee passes and keypoint
progress first."""


class ImitateConfig(BaseModel):
    """`demo`: the package directory (relative to the repository root unless absolute);
    `max_attempts` attempts at most, `exit_passes` passes among the last `exit_window` unhinted
    attempts exit (learning stops as training does, on a recent
    pass rate, not on an unbroken streak), `hinted_after` unhinted failures in a row insert one
    hinted attempt (the first is hinted too; unhinted ones follow each, whatever it scored);
    `tier`: the attempts' tier, `reader_tier` and `reviser_tier` the Demo Reader's and the
    Reviser's (P by default: S1 may use privileged data to act correctly; D is
    the ablation); the referee's `grip_threshold` on the gripper command, its path's spacing
    (`path_step_rad` of joint motion, `path_step_m` of TCP travel) and its tolerances
    (`referee`); `images`: key images in the Reviser's brief; `demo_truth`:
    the package's ground-truth layer, which it must then have, reaches the Demo Reader, hinted
    attempts and the Reviser (the default; false is the ablation without it; unhinted
    attempts never get it); `done_criteria`: on a pass, a Reviser writes the Memory's done
    criteria from the relations perception measured at the attempts' ends (false: S2 and P3 run
    without criteria, gated by the demonstration's end alone); `finish_refusals`: how often an
    attempt's `finish` is refused while the official success check is false (a hinted attempt
    with the ground-truth layer is also told how the scene differs from the demonstration's
    end; every attempt, when the demonstration was still moving what it held as its success
    latched, which way to keep moving it: `goal.latch_hint`);
    `end_targets`: once an attempt has read the demonstration's end as targets
    (`end_targets.json`, at a hinted attempt's end), the later attempts' `finish` also waits for
    `check_done` to pass them, which measures each gap with its correction (false: the
    demonstration's last-keyframe views alone, which gate every attempt's `finish` from the
    first on, beside the official refusals); `resume`: continue an S1 of this run that stopped
    before it exited; `continue_unmoved`:
    in an unhinted attempt a rejection that moved nothing does not end it (the demonstration's
    path is reachable, so the call was at fault), and each of the first `unmoved_events` adds
    `unmoved_turns` turns (`PhaseRules`; an unhinted pass needs every Memory spec so refused
    carried out as stored after it, `replaced_specs`); `review`: the
    privileged failure review against the expert's state (`learning.expert_diff`), for the
    Reviser after a failure and the hinted Actor after each action, and with it the Reviser's
    guard against undoing the previous revision's changes without new evidence (S1 at tier P,
    with the ground-truth layer; off with `review.enabled` false); `fallback`: an S1 that runs
    out of attempts yields its best Memory all the same (see the module docstring), picked under
    `fallback_rule` (`fallback_ranking`; `success`: only a
    version an attempt with an official success read, else S1 stops);
    `named_arguments`: in an unhinted attempt a call passing an optional argument its
    stage's Memory call does not name (at other than its default) is refused
    (`PhaseRules.call_arguments`), but for `freed_arguments` (the approach arguments,
    `FREED_ARGUMENTS`) once a call of that tool at that stage was rejected as unreachable or
    plan_failed before anything moved (the stage text tells the Actor to
    change the call's approach then; the referee still judges the result); `hinted_max_turns`
    (off by default: hinted attempts run on the Actor's own turn budget): when set, a hinted
    attempt ends at the smaller of its own budget and this many turns (`Episode.max_turns`);
    unhinted attempts are not capped. `hinted_failure_turns`: a hinted attempt has no fail-fast,
    so after its first hard failure that moved anything (what ends an unhinted one) it gets at
    most this many more turns (`PhaseRules.failure_turns`; None: its whole budget)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    demo: Path
    max_attempts: PositiveInt = 60
    exit_passes: PositiveInt = 3
    exit_window: PositiveInt = 5
    hinted_after: PositiveInt = 2
    prerun_attempts: PositiveInt = 3
    tier: Tier = "P"
    reader_tier: Tier = "P"
    reviser_tier: Tier = "P"
    grip_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.5
    path_step_rad: PositiveFloat = 0.01
    path_step_m: PositiveFloat = 0.005
    referee: RefereeConfig = RefereeConfig()
    images: NonNegativeInt = 3
    demo_truth: bool = True
    done_criteria: bool = True
    finish_refusals: PositiveInt = 3
    end_targets: bool = True
    resume: bool = False
    continue_unmoved: bool = True
    unmoved_turns: NonNegativeInt = 5
    unmoved_events: NonNegativeInt = 3
    named_arguments: bool = True
    freed_arguments: list[str] = Field(default_factory=lambda: list(FREED_ARGUMENTS))
    review: ReviewConfig = ReviewConfig()
    fallback: bool = False
    fallback_rule: FallbackRule = "success"
    hinted_max_turns: PositiveInt | None = None
    hinted_failure_turns: PositiveInt | None = 40

    @model_validator(mode="after")
    def _window(self) -> "ImitateConfig":
        if self.exit_passes > self.exit_window:
            raise ValueError(
                f"exit_passes {self.exit_passes} > exit_window {self.exit_window}: S1 never exits"
            )
        return self

    @field_validator("demo")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        return path if path.is_absolute() else repo_root() / path


def exits(window: list[bool], config: ImitateConfig) -> bool:
    """Whether the unhinted verdicts so far (`window`, in order) exit S1."""
    return sum(window[-config.exit_window :]) >= config.exit_passes


S1Exit = Literal["passed", "best_memory", "none"]
"""How an S1 ended for what follows it: it exited (`passed`), it did not but its fallback named
the Memory its variants start from (`best_memory`), or it stops there (`none`)."""
NO_SUCCESS: Final = "no official success"
NO_LINT_CLEAN: Final = "no lint-clean version"


class ImitateResult(BaseModel):
    """`version` is v_S1 when `passed`, else the latest draft (a fallback's version is
    `exit.json`'s `fallback`). `s1_exit`: `S1Exit` (None: not recorded)."""

    model_config = ConfigDict(frozen=True)

    task: TaskName
    seed: int
    passed: bool
    reason: Literal["passed", "attempts exhausted", "unfitted specs", "memory fails the linter"]
    attempts: int
    hinted: int
    unhinted: int
    version: int
    hash: str
    s1_exit: S1Exit | None = None


class Imitate:
    name = "imitate"
    Config = ImitateConfig

    async def run(self, ctx: TaskContext, config: ImitateConfig) -> ImitateResult:
        return await imitate(ctx, config)


async def imitate(ctx: TaskContext, config: ImitateConfig) -> ImitateResult:
    if ctx.planner is None:
        raise ValueError("imitate's referee needs FK: give the ProtocolRunner a planner")
    package = load_package(config.demo)
    if package.task != ctx.task:
        raise ValueError(f"{config.demo} demonstrates {package.task}, not {ctx.task}")
    layer = load_demo_truth(config.demo)  # the referee reads it whether or not a role does
    truth = None
    if config.demo_truth:
        truth = layer
        if truth is None:
            raise ValueError(
                f"{config.demo} has no ground-truth layer ({TRUTH_FILE}): rebuild the package "
                "with it, or run without it (demo_truth: false)"
            )
        if (truth.task, truth.seed) != (package.task, package.seed):
            raise ValueError(
                f"{config.demo}: its ground-truth layer is of {truth.task} seed {truth.seed}, the "
                f"package of {package.task} seed {package.seed}"
            )
    await import_blobs(config.demo, ctx.driver.blobs)
    folder = ctx.directory / "S1" / str(package.seed)
    if (folder / EXIT).is_file():
        exited = json.loads((folder / EXIT).read_bytes())
        result = ImitateResult.model_validate(exited)
        if not (result.passed and config.done_criteria) or "criteria" in exited:
            return result
        # Passed with no done criteria recorded: write them now, from the demonstration.
        end = await end_views(package, ctx.driver.blobs)
        run = _Imitate(ctx, ctx.planner, config, package, folder, truth, layer, end)
        before = await ctx.store.load(ctx.task, result.version)
        memory, written = await run.criteria(before, result.attempts)
        meta = memory.meta
        result = result.model_copy(update={"version": meta.version, "hash": meta.hash})
        _write(folder / EXIT, {**exited, **result.model_dump(mode="json"), "criteria": written})
        ctx.emit("criteria_after_exit", version=meta.version, **written)
        return result
    if folder.exists() and not config.resume:
        raise ValueError(
            f"{folder} exists: S1 of {ctx.task} ran in this run; resume it (resume: true) or "
            "use a new run"
        )
    end = await end_views(package, ctx.driver.blobs)
    run = _Imitate(ctx, ctx.planner, config, package, folder, truth, layer, end)
    memory = await run.start()
    if findings := unfitted(memory.doc):  # the draft fails its lint
        ctx.emit("draft_lint", findings=[finding.model_dump(mode="json") for finding in findings])
        return await run.exit(memory, 0, 0, passed=False, reason="unfitted specs")
    done = await run.resumed(memory)
    if done or run.aborted:  # the Memory its finished attempts left; the cut attempt's is not kept
        after = done[-1][3] if done else memory
        meta = after.meta
        ctx.emit(
            "resume", attempts=len(done), version=meta.version, hash=meta.hash, aborted=run.aborted
        )
    shown = True  # the first attempt is hinted
    window: list[bool] = []  # the unhinted verdicts, in order
    failures = hinted = 0
    attempts = 0
    for attempts in range(1, config.max_attempts + 1):
        if attempts <= len(done):  # finished before the interruption, as it ran
            shown, passed, read, memory = done[attempts - 1]
        else:
            if attempts == len(done) + 1 > 1:  # resumed: a stored Memory the linter now fails
                memory = await run.fix_on_load(f"{LINT_FIX}{attempts}", memory)
            memory = await run.dry_run(attempts, memory)
            read = memory  # the Memory the attempt reads
            outcome, verdict, report = await run.attempt(attempts, memory, shown)
            passed = verdict.passed
            if not passed:
                memory = await run.revise(attempts, memory, outcome, report)
        hinted += shown
        if shown:  # one hinted attempt, whatever it scored; unhinted ones follow
            shown, failures = False, 0
            continue
        window.append(passed)
        if passed:
            failures = 0
            if exits(window, config):  # with the Memory the completing pass read (`read`)
                if left := unlintable(read, ctx):  # never an exit on a Memory the linter fails:
                    # held, and the window starts again on the fixed one (a replayed exit's
                    # Memory is fixed before the next new attempt, `lintfix<n>`)
                    ctx.emit(
                        "exit_held", attempt=attempts, version=read.meta.version, errors=len(left)
                    )
                    window = []
                    if attempts > len(done):
                        memory = await run.fix_on_load(f"{LINT_FIX}-exit{attempts}", read)
                    continue
                written = None
                if config.done_criteria:
                    read, written = await run.criteria(read, attempts)
                return await run.exit(read, attempts, hinted, passed=True, criteria=written)
        else:
            failures += 1
            if failures == config.hinted_after:
                shown, failures = True, 0
    fallback = await run.fall_back(attempts) if config.fallback else None
    return await run.exit(memory, attempts, hinted, passed=False, fallback=fallback)


_Geometry = tuple[
    dict[Arm, list[tuple[float, float] | None]],
    dict[Arm, list[tuple[float, float] | None]],
    dict[Arm, list[Touches]] | None,
]


class _Imitate:
    def __init__(
        self,
        ctx: TaskContext,
        planner: Planner,
        config: ImitateConfig,
        package: DemoPackage,
        folder: Path,
        truth: DemoTruth | None,
        layer: DemoTruth | None = None,
        end: dict[str, BlobRef] | None = None,
    ) -> None:
        self.ctx = ctx
        self.end = end or {}  # the demonstration's final state by camera (`end_views`)
        self.layer = layer  # the ground-truth layer for the referee (symmetric objects)
        self.geometry: _Geometry | None = None  # the scene's, once shapes with sizes were found
        self.planner = planner
        self.config = config
        self.package = package
        # What the referee judges by: the package's keypoints, with the motion starts of its
        # `segments.json` where it has one (the agents see the package's as they are).
        segments = load_segments(config.demo)
        self.segments = segments_key(config.demo)  # what the attempts record they were judged by
        self.keypoints = package.keypoints
        if segments is not None:
            self.keypoints = with_segments(package.keypoints, segments)
        self._contacts: dict[Arm, Keypoint] | None = None  # `contact_keypoints`, once read
        self._pressed: bool | None = None  # `ends_pressed`, once read
        self._fits: dict[str, Fit | str] | None = None  # `demo_fits`, once made
        self._object_fits: dict[str, PlaceSpec] = {}  # `object_fits`, with them
        self.folder = folder
        # The package's ground-truth layer, for the Demo Reader, hinted attempts and the Reviser.
        self.truth: list[ContentPart] = [] if truth is None else truth_brief(truth)
        self.prerun = True  # the run's first open makes the expert pre-run
        self.rejected: list[EditRecord] = []
        self.aborted: list[str] = []  # what a resume set aside
        self.reruns = 0
        self.reader_checks = 0  # the Demo Reader's finishes the dry run checked (`reader_dry_run`)
        drop = "demo" in ctx.toolsets
        self.unhinted = [name for name in ctx.toolsets if name != "demo"] if drop else None
        self.unhinted_rules = PhaseRules(
            fail_fast=True,
            continue_unmoved=config.continue_unmoved,
            unmoved_turns=config.unmoved_turns,
            unmoved_events=config.unmoved_events,
        )

    def hinted_rules(self) -> PhaseRules | None:
        """A hinted attempt's rules: the phase's, but for `hinted_failure_turns`."""
        turns = self.config.hinted_failure_turns
        return None if turns is None else PhaseRules(failure_turns=turns)

    def rules_for(self, doc: MemoryDoc) -> PhaseRules:
        """An unhinted attempt's rules on `doc`: with `named_arguments`, a call passing an
        argument its stage's Memory call does not name is refused (`PhaseRules.call_arguments`:
        the Actor transcribes the Memory, it does not add to it)."""
        if not self.config.named_arguments:
            return self.unhinted_rules
        named = stage_arguments(doc)
        freed = self.config.freed_arguments
        return self.unhinted_rules.model_copy(
            update={"call_arguments": named, "freed_arguments": freed}
        )

    async def contact_keypoints(self) -> dict[Arm, Keypoint]:
        """The demonstration's contacts past an arm's last keypoint (`demos.relation.contacts`,
        from the recorded robot at the success frame): what the Demo Reader and hinted attempts
        are told `demo_relation` fits there, and the stages the review compares there."""
        if self._contacts is None:
            final = self.package.keypoints.success_frame
            frames = [] if final is None else [final]
            robots = await demo_robots(self.package, self.ctx.driver.blobs, frames)
            self._contacts = contacts(self.package.keypoints, robots)
        return self._contacts

    def shown(self, allowed: bool) -> dict[str, bool]:
        """What `config.json` records of the ground-truth layer: whether this episode got it."""
        return {"demo_truth": allowed and bool(self.truth)}

    def seed(self) -> EpisodeSeed:
        """The frame store of the Demo Reader and hinted attempts: the recording's frames, the
        demonstration's keypoints (`DEMO`) and its grasps' arms (`DEMO_ARMS`)."""
        keypoints = self.package.keypoints.model_dump(mode="json")
        facts = {DEMO: keypoints, DEMO_ARMS: demo_arms(self.package.keypoints)}
        if self.package.tracks is not None:  # for fitting written skills (`toolsets.skill_fit`)
            facts[DEMO_TRACKS] = {"blob": self.package.tracks.model_dump(mode="json")}
        return EpisodeSeed(frames=self.package.frames, facts=facts)

    def arms_seed(self) -> EpisodeSeed:
        """An unhinted attempt's: only the demonstration's grasps' arms (`DEMO_ARMS`), which
        `plan_arms` gives on this scene, the demonstration's."""
        return EpisodeSeed(facts={DEMO_ARMS: demo_arms(self.package.keypoints)})

    def spec(self) -> EpisodeSpec:
        return EpisodeSpec(
            task=self.ctx.task,
            seed=self.package.seed,
            stage="s1",
            instruction=self.package.instruction,
            repeat=not self.prerun,
            prerun_attempts=self.config.prerun_attempts,
        )

    def paths(self) -> ArmPaths:
        config = self.config
        return ArmPaths(self.planner, self.ctx.profile, config.path_step_rad, config.path_step_m)

    def holding(self) -> dict[Arm, float]:
        """Per arm, the embodiment's `Gripper.holding_threshold`."""
        return {arm: spec.gripper.holding_threshold for arm, spec in self.ctx.profile.arms.items()}

    async def run(self, episode: Episode) -> tuple[EpisodeOutcome, int]:
        outcome, reruns = await self.ctx.driver.run_counted(episode)
        self.reruns += reruns
        if not outcome.solvable:
            raise RuntimeError(
                f"the demo seed {self.package.seed} of {self.ctx.task} is unsolvable "
                f"({outcome.end.note}): replace the demonstration"
            )
        if outcome.env_episode is not None:
            self.prerun = False
        return outcome, reruns

    async def start(self) -> Version:
        """The Memory the first attempt reads: the Demo Reader's (again if it was cut short)."""
        directory = self.folder / "reader"
        if (found := await self._after(directory, None, self._read_by(1))) is not None:
            return found
        set_aside(directory)
        return await self.read()

    async def resumed(self, memory: Version) -> list[tuple[bool, bool, Version, Version]]:
        """The attempts an interrupted S1 finished, from `memory` on (the first attempt's), each
        with whether it was hinted, whether it passed, the Memory it read and the Memory after it
        (a revision cut short runs again); the attempt cut short is set aside, with its revision.
        An attempt a revision followed failed, as the run saw it then; an attempt without a
        verdict that a later one followed is history, counted as failed."""
        done = []
        for n in itertools.count(1):
            attempt = self.folder / f"attempt{n}"
            judged = attempt / REFEREE / VERDICT
            for fixed in (
                self.folder / f"{LINT_FIX}{n}",
                self.folder / f"{LINT_FIX}-exit{n - 1}",
                self.folder / f"{DRY_RUN}{n}",
            ):
                if (fixed / AFTER).is_file() and self._read_by(n) is not None:
                    # the Memory was fixed on load (on a resume, or after an exit held for the
                    # linter) before this attempt, which read the fix
                    memory = await self._after(fixed, memory, None) or memory
            if not judged.is_file() and (following := self._read_by(n + 1)) is not None:
                # a later attempt went on from this unjudged one: it counts as failed, and the
                # Memory is the one the next attempt read
                config = json.loads((attempt / "config.json").read_bytes())
                read = memory
                if following != memory.meta.hash:
                    memory = await self.ctx.store.load(self.ctx.task, following)
                done.append((bool(config["protocol"]["hinted"]), False, read, memory))
                continue
            if not judged.is_file():
                aside = [set_aside(attempt), set_aside(self.folder / "reviser" / f"r{n}")]
                self.aborted = [path.name for path in aside if path is not None]
                return done
            if (read := self._read_by(n)) != memory.meta.hash:
                raise ValueError(
                    f"{attempt} read the Memory {read}, not v{memory.meta.version:03d} "
                    f"({memory.meta.hash[:12]}): this S1 cannot resume; use a new run"
                )
            config = json.loads((attempt / "config.json").read_bytes())
            passed = self.verdict(attempt, config["protocol"])
            before = memory
            revision = self.folder / "reviser" / f"r{n}"
            if (after := await self._after(revision, memory, self._read_by(n + 1))) is not None:
                # a revision ran, so the attempt counted as failed then
                passed = False
                self.rejected += [
                    record for record in _records(revision / "edits.jsonl") if not record.accepted
                ]
                memory = after
            elif (following := self._kept(n + 1)) is not None:
                # history: it passed as it ran (no revision) and the next attempt went on with
                # the Memory it read; a revision mid-history would leave a Memory no later
                # attempt read
                set_aside(revision)
                if following != memory.meta.hash:
                    memory = await self.ctx.store.load(self.ctx.task, following)
            elif not passed:
                set_aside(revision)
                outcome = EpisodeOutcome.model_validate_json((attempt / SUMMARY).read_bytes())
                _, report = await self.judge(outcome, None)  # the verdict stays as judged
                memory = await self.revise(n, memory, outcome, report)
            else:  # a revision cut short after a pass is stale
                set_aside(revision)
            done.append((bool(config["protocol"]["hinted"]), passed, before, memory))
        raise AssertionError("unreachable")

    def verdict(self, attempt: Path, protocol: Mapping[str, Any]) -> bool:
        """Whether a finished attempt passed, as it was judged (`kept_verdict`, which says what is
        refused)."""
        return kept_verdict(attempt, protocol, self.config.referee, self.segments)

    def _kept(self, n: int) -> str | None:
        """The hash of the Memory attempt `n` read, if it is kept on resuming: judged, or history
        (a later attempt started); an attempt cut short runs again."""
        if (read := self._read_by(n)) is None:
            return None
        judged = (self.folder / f"attempt{n}" / REFEREE / VERDICT).is_file()
        return read if judged or self._read_by(n + 1) is not None else None

    def _read_by(self, n: int) -> str | None:
        """The hash of the Memory attempt `n` read, if it started."""
        path = self.folder / f"attempt{n}" / "config.json"
        if not path.is_file():
            return None
        return str(json.loads(path.read_bytes())["memory"]["hash"])

    async def _after(
        self, directory: Path, before: Version | None, following: str | None
    ) -> Version | None:
        """The Memory the step in `directory` (the Demo Reader's, a revision) left, if it is
        done: its `AFTER` record says, or the next attempt read it (runs without the
        record); None if it was cut short."""
        marker = directory / AFTER
        if marker.is_file():
            found = str(json.loads(marker.read_bytes())["hash"])
        elif following is not None and (directory / SUMMARY).is_file():
            found = following
        else:
            return None
        if before is not None and found == before.meta.hash:
            return before
        return await self.ctx.store.load(self.ctx.task, found)

    async def read(self) -> Version:
        """The Demo Reader writes Memory v0."""
        ctx, package = self.ctx, self.package
        doc = MemoryDoc(task=ctx.task, instruction=package.instruction)
        editor = Editor(
            doc, rules=ctx.memory.rules("s1"), lint=ctx.memory.lint, validators=validators()
        )
        found = await self.contact_keypoints()

        def prompt(opened: EpisodeOpened | None) -> tuple[str, list[ContentPart]]:
            text = role_prompts(ctx).demo_reader(
                instruction=package.instruction, memory=editor.doc, profile=ctx.profile
            )
            return text, [*demo_brief(package, self.end, found), *self.truth]

        async def fitted(note: str) -> str | None:
            if refusal := lint_refusal(editor):
                return refusal
            missing = [finding.match for finding in unfitted(editor.doc)]
            if missing:
                return (
                    f"the specs {', '.join(missing)} are not fitted on the demonstration: "
                    "redefine each from demo_relation's result (with fitted), then finish"
                )
            if grasps := self.unfitted_grasps(editor.doc):
                return (
                    f"the grasps at {', '.join(grasps)} have no spec fitted there: each arm's "
                    "grasp is its own (a hand-over's receiving grasp, each arm's handle), so call "
                    "demo_relation at each and define its spec with fitted naming it, have the "
                    "stage that grasps there pass it, then finish"
                )
            if crossed := spec_arms(editor.doc):
                return "before finishing: " + _listed(crossed)
            if advice := draft_advice(editor.doc):
                return "before finishing: " + "; ".join(
                    f"{finding.where} ({finding.match}): {finding.message}" for finding in advice
                )
            if split := self.split_guard(editor, directory):
                return split
            return await self.reader_dry_run(editor.doc)

        directory = self.folder / "reader"
        outcome, _ = await self.run(
            Episode(
                task=ctx.task,
                stage="s1",
                role="demo_reader",
                directory=directory,
                prompt=prompt,
                spec=self.spec(),
                tier=self.config.reader_tier,
                frames=self.seed(),
                editor=editor,
                memory=editor.doc,
                memory_label="draft",
                extra={"protocol": "imitate", "seed": package.seed, **self.shown(True)},
                finish_guard=fitted,
                finish_refusals=LINT_RETRIES,
            )
        )
        _lines(directory / "edits.jsonl", editor.records)
        if left := editor.errors_left():
            raise ValueError(f"the Demo Reader's draft fails the linter: {_listed(left)}")
        version = await ctx.store.commit(
            editor.doc,
            status="s1_draft",
            source=VersionSource(kind="s1", ref=self._ref(directory), note="demo reader"),
        )
        _write(directory / AFTER, {"version": version.meta.version, "hash": version.meta.hash})
        accepted = sum(record.accepted for record in editor.records)
        ctx.emit("read_demo", version=version.meta.version, end=outcome.end.reason, edits=accepted)
        return version

    def unfitted_grasps(self, doc: MemoryDoc) -> list[str]:
        """The demonstration's grasps (a close on an object, per arm) that no spec of `doc` was
        fitted at; none without the ground-truth layer, which tells what a close acts on."""
        if self.layer is None:
            return []
        shapes = self.shapes(self.folder / "reader")
        fitted = {
            param.source.removeprefix(f"{FITTED_SOURCE} ")
            for param in doc.params.values()
            if param.kind.endswith("_spec") and param.source.startswith(f"{FITTED_SOURCE} ")
        }
        return [
            kp for kp in grasp_keypoints(self.keypoints, self.layer, shapes) if kp not in fitted
        ]

    async def demo_fits(self) -> dict[str, Fit | str] | None:
        """The spec the demonstration fits at each keypoint, on the Demo Reader's registry
        (`spec_check.demo_fits`), for the Reviser's `fit_spec`; made once; None without the
        ground-truth layer or the Reader's registry."""
        if self._fits is not None or self.layer is None:
            return self._fits
        reader = self.folder / "reader"
        source = reader if (reader / EVENTS).is_file() else None
        ref = None if source is None else last_registry(source)
        if source is None or ref is None:
            return None
        blobs = self.ctx.driver.blobs
        registry = SceneRegistry.model_validate_json(await blobs.aget_bytes(ref))
        table = last_fact(source, TABLE)
        robots = await demo_robots(self.package, blobs)
        released = released_poses(last_fact(source, DEMO_RELEASES), registry)
        made = functools.partial(
            demo_fits,
            self.package.keypoints,
            self.layer,
            self.shapes(source),
            registry,
            robots,
            self.ctx.profile,
            None if table is None else Table.model_validate(table),
            self.config.review.fit_approach,
            released,
        )
        given = last_fact(source, DEMO_FITS) or {}
        relative = functools.partial(
            object_fits,
            self.package.keypoints,
            registry,
            robots,
            self.ctx.profile,
            None if table is None else Table.model_validate(table),
            given,
            self.config.review.fit_approach,
            released,
        )
        try:
            self._fits = await anyio.to_thread.run_sync(made)
            self._object_fits = await anyio.to_thread.run_sync(relative)
        except Exception as error:  # a help for the Reviser: S1 goes on without it
            log.exception("the demonstration's fits of %s could not be made", self.ctx.task)
            self.ctx.emit("demo_fits_error", error=f"{type(error).__name__}: {error}")
            return None
        return self._fits

    async def fix_on_load(self, name: str, memory: Version) -> Version:
        """`memory` as a resume finds it, or, where the current linter fails it (`unlintable`),
        the lint-fix pass's result (`cleanup`, in `<name>/`; reused once done, run again when cut
        short), stored as the next version when it is lint-clean."""
        directory = self.folder / name
        if (found := await self._after(directory, memory, None)) is not None:
            return found
        if not (left := unlintable(memory, self.ctx)):
            return memory
        set_aside(directory)
        for n in range(1, LINT_PASSES):
            set_aside(self.folder / f"{name}-retry{n}")
        self.ctx.emit("lint_on_load", version=memory.meta.version, errors=len(left), at=name)
        return await self.cleanup(memory, directory)

    async def cleanup(self, base: Version, directory: Path) -> Version:
        """A Reviser fixes what the linter finds on `base` (a lint-fix on load, in `directory`):
        it reads the linter's findings on it, bound arms rewritten first, and edits it under S1's
        rules; a change is the Memory the next attempt reads."""
        ctx = self.ctx
        memory = await self.unbind(base)
        rules = ctx.memory.rules("s1").model_copy(update={"scope": "cleanup"})
        config = self.lint_config()
        editor = Editor(memory.doc, rules=rules, lint=config, repair=True, validators=validators())
        findings = lint(memory.doc, config)
        for n in range(1, LINT_PASSES + 1):
            if n > 1:  # a pass that ended with errors left (it stopped, or was cut off): the next
                # one reads the findings still on what it edited, and goes on from there
                editor = Editor(
                    editor.doc, rules=rules, lint=config, repair=True, validators=validators()
                )
                findings = lint(editor.doc, config)
            where = directory if n == 1 else directory.with_name(f"{directory.name}-retry{n - 1}")
            await self._cleanup_pass(where, editor, base, findings, memory.doc)
            if not editor.errors_left():
                break
        changes = diff(memory.doc, editor.doc)
        ctx.emit(
            "cleanup",
            start=base.meta.version,
            findings=len(findings),
            changes=len(changes.params) + len(changes.text),
        )
        if not changes.empty and self._storable(editor):
            memory = await ctx.store.commit(
                editor.doc,
                status="s1_draft",
                source=VersionSource(kind="s1", ref=self._ref(directory), note=directory.name),
                parent=memory.meta.version,
            )
        _write(directory / AFTER, {"version": memory.meta.version, "hash": memory.meta.hash})
        return memory

    async def _cleanup_pass(
        self,
        directory: Path,
        editor: Editor,
        base: Version,
        findings: list[Finding],
        original: MemoryDoc,
    ) -> None:
        """One Reviser episode over `editor`'s document (a cleanup pass)."""
        ctx = self.ctx

        def prompt(opened: EpisodeOpened | None) -> tuple[str, list[ContentPart]]:
            text = role_prompts(ctx).cleanup(
                instruction=self.package.instruction, editor=editor, profile=ctx.profile
            )
            return text, cleanup_brief(findings, editor=editor)

        await self.run(
            Episode(
                task=ctx.task,
                stage="s1",
                role="reviser",
                directory=directory,
                prompt=prompt,
                tier=self.config.reviser_tier,
                editor=editor,
                memory=editor.doc,
                memory_label=f"v{base.meta.version:03d}",
                finish_guard=functools.partial(_refuse_errors_and_dropped_fits, editor, original),
                finish_refusals=LINT_RETRIES,
                extra={
                    "protocol": "imitate",
                    "seed": self.package.seed,
                    "cleanup": True,
                    "start": base.meta.hash,
                    "findings": len(findings),
                },
            )
        )
        _lines(directory / "edits.jsonl", editor.records)
        self.rejected += [record for record in editor.records if not record.accepted]

    async def attempt(
        self, n: int, memory: Version, hinted: bool
    ) -> tuple[EpisodeOutcome, Verdict, list[str]]:
        ctx, package, doc = self.ctx, self.package, memory.doc

        found = await self.contact_keypoints() if hinted else {}
        # The done gate from the first attempt on, anchored to the demonstration's last
        # keyframe: its views (and press, final frame, latched lifts) before any attempt has
        # read its end as targets, those targets too once one has.
        end, source = chosen_end_targets(self.folder) if self.config.end_targets else ([], None)
        if source is not None:
            ctx.emit("end_targets_from", attempt=n, source=source, targets=len(end))
        if self._pressed is None:
            self._pressed = await ends_pressed(package, ctx.driver.blobs, ctx.profile)
        known = holds_known(end, package, ctx.profile)
        latched = latched_lifts(package, known)
        # the motion into the success latch only (one-sided along it); the holding arms stay
        # unannotated in S1
        end = [t.model_copy(update={"motion": k.motion}) for t, k in zip(end, known, strict=True)]
        hint = latch_hint(package.keypoints)  # the official refusals' and the done gate's

        def prompt(opened: EpisodeOpened | None) -> tuple[str, list[ContentPart] | None]:
            assert opened is not None and opened.instruction is not None
            text = role_prompts(ctx).actor(
                instruction=opened.instruction,
                memory=doc,
                rules="s1_hinted" if hinted else "s1",
                profile=ctx.profile,
            )
            return text, [*demo_brief(package, contacts=found), *self.truth] if hinted else None

        outcome, _ = await self.run(
            Episode(
                task=ctx.task,
                stage="s1",
                role="actor",
                directory=self.folder / f"attempt{n}",
                prompt=prompt,
                spec=self.spec(),
                tier=self.config.tier,
                rules=self.hinted_rules() if hinted else self.rules_for(doc),
                toolsets=None if hinted else self.unhinted,
                frames=self.seed() if hinted else self.arms_seed(),
                diagnose=True,
                notes=functools.partial(self.notes, self.folder / f"attempt{n}")
                if hinted
                else None,
                memory=doc,
                memory_label=f"v{memory.meta.version:03d}",
                memory_hash=memory.meta.hash,
                extra={
                    "protocol": "imitate",
                    "seed": package.seed,
                    "attempt": n,
                    "hinted": hinted,
                    "referee": self.config.referee.model_dump(mode="json"),
                    "referee_segments": self.segments,
                    "finish_refusals": self.config.finish_refusals,
                    **self.shown(hinted),
                },
                finish_refusals=self.config.finish_refusals,
                method=WiclEpisode(
                    official_finish=True,  # S1 is privileged: a finish short of success is refused
                    finish_target=self.layer if hinted and self.truth else None,  # as the brief
                    finish_hint=hint,
                    relations=True,
                    demo_end=self.layer,
                ),
                shapes=True,
                goal=DemoGoal(
                    views=self.end or None,
                    press=self._pressed,
                    reference=end_observation(package),
                    end=end,
                    latched=latched,
                    latch=latch_motion(package.keypoints),
                    hint=hint,
                ),
                max_turns=self.config.hinted_max_turns if hinted else None,
            )
        )
        verdict, report = await self.judge(outcome, outcome.directory / REFEREE / VERDICT)
        await ctx.hooks.on_try_end(ctx.task, package.seed, n, outcome)
        ctx.emit(
            "attempt",
            attempt=n,
            hinted=hinted,
            passed=verdict.passed,
            success=outcome.success,
            end=outcome.end.reason,
        )
        return outcome, verdict, report

    def lint_config(self) -> LintConfig:
        """The task's lint config with the roles the demonstration ends holding where an arm
        carried them, over no anchor (`relative_carry`), once an attempt has read its end."""
        ends = carried_ends(holds_known(self.end_targets(), self.package, self.ctx.profile))
        return self.ctx.memory.lint.model_copy(update={"carried_ends": ends})

    def end_targets(self) -> list[EndTarget]:
        """The demonstration's end as targets, as the attempts that read it wrote them
        (`chosen_end_targets`); none before one has."""
        return chosen_end_targets(self.folder)[0]

    def notes(self, directory: Path, trajectory: Trajectory) -> ActionNotes:
        """The keypoint progress after each action of a hinted attempt (in `directory`)."""
        paths = self.paths()

        async def progress() -> list[ContentPart]:
            arrays = trajectory.arrays()
            await paths.extend(arrays)
            state = self.progress(paths.paths(), directory)
            shown = progress_text(self.package.keypoints, state)
            if self.reviewing() and self.layer is not None:
                diff = await review(
                    self.keypoints,
                    self.layer,
                    self.shapes(directory),
                    arrays,
                    [],
                    state,
                    False,
                    None,
                    referee=self.config.referee,
                    config=self.config.review,
                    grip_threshold=self.config.grip_threshold,
                    holding=self.holding(),
                )
                shown += "\n" + progress_line(diff)
            return [TextPart(text=shown)]

        return progress

    def progress(
        self,
        paths: Mapping[Arm, ArmPath],
        directory: Path,
        holds: Mapping[Arm, Mapping[int, frozenset[str]]] | None = None,
        truth: Mapping[str, np.ndarray] | None = None,
        sources: Mapping[Arm, np.ndarray] | None = None,
        latched: bool = False,
        layout: Regrouped | None = None,
    ) -> dict[Arm, ArmProgress]:
        """Each arm's keypoint progress (`paths`, `holds`, `sources`: by the demonstration's arm,
        whose keypoints they are judged against; with `layout`, by the attempt's arm its
        regrouped keypoints are judged on, `demos.referee.regrouped`); `holds` (`attempt_holds`,
        from the attempt's ground truth) judges its grasps and releases by their objects, by
        position without; with them, `truth` (the attempt's ground truth) and `sources` judge its
        in-air steps by their objects (`air_steps`) and, with the official check `latched`, its
        holds by whether they lifted what the demonstration lifted (`held_lifts`)."""
        keypoints, threshold, referee = (
            self.keypoints,
            self.config.grip_threshold,
            self.config.referee,
        )
        arms = keypoints.arms if layout is None else layout.arms
        axes, sizes, _ = self.symmetry(directory)
        if layout is not None:
            axes = permute(axes, layout.origin) if axes else {}
            sizes = permute(sizes, layout.origin) if sizes else {}
        objects = {}
        air: dict[Arm, dict[int, np.ndarray]] = {}
        lifts = None  # what `held_lifts` judges the holds by, with the ground truth
        touched: dict[Arm, dict[int, frozenset[str]]] = {}
        if holds is not None and self.layer is not None and (shapes := self.shapes(directory)):
            poses = {frame.frame: frame.poses for frame in self.layer.frames}
            objects = {arm: acted_on(kps, poses, shapes) for arm, kps in keypoints.arms.items()}
            if layout is not None:
                objects = permute(objects, layout.origin)
            if truth is not None and sources is not None:
                air = {
                    arm: air_steps(
                        kps,
                        objects[arm],
                        poses,
                        shapes,
                        truth,
                        sources[arm],
                        keypoints.success_frame,
                        _pointing(keypoints, arm),
                        referee,
                        paths[arm],
                        holds[arm],
                        threshold,
                        latched,
                    )
                    for arm, kps in arms.items()
                }
                lifts = poses, shapes, truth, sources, holds
                if latched:
                    touched = {
                        arm: attempt_touches(paths[arm], sources[arm], truth, shapes, threshold)
                        for arm in arms
                    }
        progress = {
            arm: arm_progress(
                kps,
                paths[arm],
                threshold,
                _pointing(keypoints, arm),
                axes.get(arm),
                keypoints.success_frame,
                self.config.referee,
                sizes.get(arm),
                objects.get(arm),
                None if holds is None else holds[arm],
                air.get(arm),
                latched,
                touched.get(arm),
            )
            for arm, kps in arms.items()
        }
        if lifts is None:
            return progress
        poses, shapes, truth, sources, holds = lifts
        return {
            arm: held_lifts(
                kps,
                progress[arm],
                objects[arm],
                poses,
                shapes,
                truth,
                sources[arm],
                paths[arm],
                holds[arm],
                threshold,
                referee,
                latched,
                keypoints.success_frame,
            )
            for arm, kps in arms.items()
        }

    def holds(
        self,
        paths: Mapping[Arm, ArmPath],
        recorded: Mapping[str, np.ndarray],
        sources: Mapping[Arm, np.ndarray],
        on: Mapping[Arm, Arm] | None = None,
    ) -> dict[Arm, dict[int, frozenset[str]]] | None:
        """What each arm's gripper held at its gripper changes (`attempt_holds`; `on`: the
        attempt's arm whose recorded TCP each of `paths` is, itself by default); None without
        the ground truth the attempt recorded."""
        if "truth.sample" not in recorded:
            return None
        truth = {key: value for key, value in recorded.items() if key.startswith("truth.")}
        return {
            arm: attempt_holds(
                path.gripper,
                sources[arm],
                np.asarray(recorded[f"{arm if on is None else on[arm]}.tcp"], dtype=float),
                truth,
                self.config.grip_threshold,
            )
            for arm, path in paths.items()
        }

    def held(
        self,
        progress: dict[Arm, ArmProgress],
        recorded: Mapping[str, np.ndarray],
        outcome: EpisodeOutcome,
        sources: Mapping[Arm, np.ndarray],
        layout: Regrouped | None = None,
    ) -> dict[Arm, ArmProgress]:
        """Place and release keypoints judged by the held object relative to its target, from
        the ground truth the attempt recorded (`demos.referee.held_placements`; with the official
        success, where it landed is information only); as they are without it
        or the ground-truth layer. `layout`: the keypoints regrouped (`progress`)."""
        if self.layer is None or "truth.sample" not in recorded:
            return progress
        shapes = self.shapes(outcome.directory)
        poses = {frame.frame: frame.poses for frame in self.layer.frames}
        truth = {key: value for key, value in recorded.items() if key.startswith("truth.")}
        return {
            arm: held_placements(
                (self.keypoints.arms if layout is None else layout.arms)[arm],
                p,
                poses,
                shapes,
                truth,
                sources[arm],
                self.config.referee,
                outcome.success,
            )
            for arm, p in progress.items()
        }

    def symmetry(self, directory: Path) -> _Geometry:
        """Per arm and keypoint, the symmetric object's axis the keypoint may turn about, the
        size of the object it acts on (`keypoint_sizes`) and the objects it acts on
        (`keypoint_objects`, which bind the arms' order): from the attempt's object
        shapes (`referee/shapes.json`) and the ground-truth layer's poses at the keypoints; none
        without either (the referee compares strictly and orders every keypoint). The demo
        seed's scene is the same in every attempt, so the first shapes found with sizes hold for
        all; shapes recorded without sizes take them from another attempt's, if one
        has them."""
        if self.geometry is not None:
            return self.geometry
        if self.layer is None or not (shapes := self.shapes(directory)):
            return {}, {}, None
        poses = {frame.frame: frame.poses for frame in self.layer.frames}
        arms = self.keypoints.arms.items()
        geometry = (
            {arm: symmetry_axes(kps, poses, shapes) for arm, kps in arms},
            {arm: keypoint_sizes(kps, poses, shapes) for arm, kps in arms},
            {arm: keypoint_objects(kps, poses, shapes) for arm, kps in arms},
        )
        if all(shape.size is not None for shape in shapes.values()):
            self.geometry = geometry
            symmetric = sorted(name for name, shape in shapes.items() if shape.symmetric)
            self.ctx.emit("referee_symmetry", symmetric=symmetric)
        return geometry

    def shapes(self, directory: Path) -> dict[str, ObjectShape]:
        """The object shapes an attempt in `directory` recorded as it opened; shapes recorded
        without sizes take them from another attempt's of this S1 (the same
        scene), if one has them."""
        found = directory / REFEREE / SHAPES
        if not found.is_file():
            return {}
        shapes = _shapes(found)
        if all(shape.size is not None for shape in shapes.values()):
            return shapes
        for other in sorted(self.folder.glob(f"attempt*/{REFEREE}/{SHAPES}")):
            sized = _shapes(other)
            if sized.keys() == shapes.keys() and all(s.size is not None for s in sized.values()):
                return {n: s.model_copy(update={"size": sized[n].size}) for n, s in shapes.items()}
        return shapes

    def verdict_on(
        self,
        on: Mapping[Arm, Arm],
        paths: ArmPaths,
        recorded: Mapping[str, np.ndarray],
        outcome: EpisodeOutcome,
    ) -> tuple[Verdict, dict[Arm, ArmProgress]]:
        """The verdict with each demonstration arm's keypoints judged on the attempt's arm `on`
        it maps to (itself, or the other: `swapped`), and the progress it rests on."""
        every, everywhere = paths.paths(), paths.sources()
        mapped = {arm: every[on[arm]] for arm in on}
        sources = {arm: everywhere[on[arm]] for arm in on}
        holds = self.holds(mapped, recorded, sources, on)
        truth = {key: value for key, value in recorded.items() if key.startswith("truth.")}
        progress = self.progress(
            mapped, outcome.directory, holds, truth or None, sources, outcome.success
        )
        progress = self.held(progress, recorded, outcome, sources)
        objects = self.symmetry(outcome.directory)[2]
        referee = self.config.referee
        return judge(self.keypoints, progress, outcome.success, sources, referee, objects), progress

    def verdict_regrouped(
        self, paths: ArmPaths, recorded: Mapping[str, np.ndarray], outcome: EpisodeOutcome
    ) -> tuple[Verdict, dict[Arm, ArmProgress], Regrouped] | None:
        """With the official success: the verdict with
        the demonstration's keypoints regrouped by object onto the arms that handled each object,
        in the order the attempt did them (`demos.referee.regrouped`), judged with only one
        object's keypoints binding each other's order (`judge`'s `free`); None where it does not
        apply (no ground truth, no official success, a block of no known object, or nothing
        moved)."""
        referee = self.config.referee
        if not outcome.success or self.layer is None:
            return None
        if not (shapes := self.shapes(outcome.directory)) or "truth.sample" not in recorded:
            return None
        every, sources = paths.paths(), paths.sources()
        holds = self.holds(every, recorded, sources)
        if holds is None:
            return None
        poses = {frame.frame: frame.poses for frame in self.layer.frames}
        demo = {arm: acted_on(kps, poses, shapes) for arm, kps in self.keypoints.arms.items()}
        threshold = self.config.grip_threshold
        layout = regrouped(self.keypoints, demo, every, sources, holds, threshold)
        if layout is None:
            return None
        mapped = {arm: every[arm] for arm in layout.arms}
        clocks = {arm: sources[arm] for arm in layout.arms}
        truth = {key: value for key, value in recorded.items() if key.startswith("truth.")}
        progress = self.progress(
            mapped, outcome.directory, holds, truth, clocks, outcome.success, layout
        )
        progress = self.held(progress, recorded, outcome, clocks, layout)
        objects = self.symmetry(outcome.directory)[2]
        touches = None if objects is None else permute(objects, layout.origin)
        verdict = judge(
            self.keypoints,
            progress,
            outcome.success,
            clocks,
            referee,
            touches,
            arms=layout.arms,
            free=True,
        )
        return verdict, progress, layout

    async def judge(
        self, outcome: EpisodeOutcome, record: Path | None
    ) -> tuple[Verdict, list[str]]:
        """The referee's verdict and the report the Reviser reads: each arm's first failed
        keypoint with its closest approach and the turn it happened in; `record` is where the
        verdict goes (`referee/keypoints.json`, which a resume never rewrites)."""
        paths = self.paths()
        with np.load(outcome.directory / PHYSICS) as physics:
            recorded = dict(physics)
        await paths.extend(recorded)
        keypoints = self.keypoints
        on: dict[Arm, Arm] = {arm: arm for arm in keypoints.arms}
        verdict, progress = self.verdict_on(on, paths, recorded, outcome)
        layout: Regrouped | None = None  # the keypoints regrouped by object, if they passed so
        swap = swapped(keypoints, list(paths.paths()))
        if outcome.success and not verdict.passed and swap is not None:
            # The Memory never binds a hand: an official success made with the other arm.
            other, moved = self.verdict_on(swap, paths, recorded, outcome)
            if other.passed:
                line = ", ".join(f"the {swap[arm]} arm did the demonstration's {arm}" for arm in on)
                verdict, progress, on = (
                    other.model_copy(update={"informational": [*other.informational, line]}),
                    moved,
                    swap,
                )
        if not verdict.passed and (found := self.verdict_regrouped(paths, recorded, outcome)):
            # Different objects in any order, by either arm (each object's steps in
            # order): an official success that did the objects in another order or arm passes.
            other, moved, regroup = found
            if other.passed:
                lines = [*other.informational, *regroup.moved]
                verdict, progress, layout = (
                    other.model_copy(update={"informational": lines}),
                    moved,
                    regroup,
                )
                on = {arm: arm for arm in regroup.arms}
        if (
            verdict.passed
            and outcome.memory is not None
            and not _hinted(outcome.directory)  # what S1's exit counts
        ):
            read = await self.ctx.store.load(self.ctx.task, outcome.memory)
            if replaced := replaced_specs(read.doc, outcome.directory):
                lines = [
                    f"{name}: the Memory's spec was refused before anything moved ({why}) and "
                    "never carried out as stored: the pass was made with a spec the Actor wrote "
                    "in its place, which S2 and P3 do not have"
                    for name, why in replaced.items()
                ]
                verdict = Verdict(
                    passed=False,
                    reasons=lines,
                    informational=[*verdict.informational, *verdict.reasons],
                )
        turns = paths.turns()
        actions = _actions(outcome.directory / "turns.jsonl")
        closest = {}  # a keypoint's line in the verdict -> with the turn it came closest in
        for arm, kps in (keypoints.arms if layout is None else layout.arms).items():
            for kp, p in zip(kps, progress[arm].keypoints, strict=True):
                if p.nearest is not None:
                    turn = int(turns[on[arm]][p.nearest.sample])
                    action = f" ({actions[turn]})" if turn in actions else ""
                    line = describe(arm, kp, p)
                    dx, dy, dz = (round(-100 * v, 1) + 0.0 for v in p.nearest.offset_m)
                    fix = f"to reach it, the TCP needed ({dx:+.1f}, {dy:+.1f}, {dz:+.1f}) cm"
                    if p.kind == "air":  # judged by the held object, wherever the TCP was
                        fix = "the held object had to rise (and turn) the demonstration's way"
                    if p.lifted_m is not None:  # a grasp whose hold never lifted it
                        fix = "the held object had to be lifted off its support, not slid along it"
                    closest[line] = f"{line}; closest in turn {turn}{action}; {fix}"
        official = "latched" if outcome.success else "never latched"
        report = [f"official success check: {official}"]
        report += [
            closest.get(reason, reason) for reason in verdict.reasons if "latched" not in reason
        ]
        if record is not None:
            saved: dict[str, Any] = {}
            saved["verdict"] = verdict.model_dump(mode="json")
            if any(arm != judged for arm, judged in on.items()):
                saved["arms"] = on  # demonstration arm -> the attempt's arm it was judged on
            if layout is not None:  # per attempt arm, the demonstration's (arm, keypoint) judged
                saved["regrouped"] = {
                    arm: [[demo_arm, keypoints.arms[demo_arm][j].index] for demo_arm, j in at]
                    for arm, at in layout.origin.items()
                }
            saved["progress"] = {arm: p.model_dump(mode="json") for arm, p in progress.items()}
            _write(record, saved)
        return verdict, report

    async def revise(
        self,
        n: int,
        memory: Version,
        outcome: EpisodeOutcome,
        report: list[str],
    ) -> Version:
        """The Reviser edits the Memory after a failed attempt; a change is the next draft."""
        ctx = self.ctx
        expert = await self.expert_review(n, memory, outcome)
        reverts = await self.reverts(n, memory) if expert is not None else {}
        memory = await self.unbind(memory)
        config = self.lint_config()
        fits = await self.demo_fits() if self.reviewing() else None
        grasped = grasped_objects(outcome.directory)
        tipped = [(c.spec, c.obj) for c in tipped_closes(outcome.directory)]
        editor = Editor(
            memory.doc,
            rules=ctx.memory.rules("s1").model_copy(
                update={"keep_fits": True, "strict_done": True}
            ),
            lint=config,
            repair=True,
            reverts=reverts,
            evidence=_named(expert),
            fits=fits,
            object_fits=self._object_fits if fits is not None else None,
            validators=[
                *validators(
                    grasped, required_fixes(memory.doc, grasped, tipped, refit=fits is not None)
                ),
                *self.record_guards(n, expert, memory.doc),
            ],
            first=_first_named(expert) if self.config.review.fit_reverts else None,
            **await self.guards(n, memory),
        )
        label = f"v{memory.meta.version:03d}"
        history = revision_ledger(self.folder, n)
        ledger = "\n".join(history)
        ctx.events.emit(  # model-visible input: logged as shown
            EventKind.REVISER_LEDGER,
            {
                "task": ctx.task,
                "attempt": n,
                "lines": history,
                "sha256": hashlib.sha256(ledger.encode()).hexdigest(),
            },
        )
        frames, aliases = tries_frames([(n, outcome)])
        log = try_log(
            f"attempt {n}",
            label,
            outcome,
            diagnosis=True,
            aliases=aliases[n],
            keypoints=report,
            expert=[] if expert is None else render(expert, outcome.end.reason == UNCONFIRMED),
            earlier=[
                (f"attempt {k}", self.folder / f"attempt{k}")
                for k in range(1, n)
                if (self.folder / f"attempt{k}" / SUMMARY).is_file()
            ],
        )

        def prompt(opened: EpisodeOpened | None) -> tuple[str, list[ContentPart]]:
            text = role_prompts(ctx).reviser(
                instruction=self.package.instruction, editor=editor, profile=ctx.profile
            )
            brief = reviser_brief(
                [log],
                editor=editor,
                diff=None,
                rejected=self.rejected,
                images=self.config.images,
                history=history,
            )
            checked = [TextPart(text="\n".join(dry))] if dry else []
            checked += [TextPart(text="\n".join(stuck))] if stuck else []
            carries = [f for f in lint(editor.doc, config) if f.rule == "relative_carry"]
            return text, [
                *brief,
                *checked,
                *lint_brief(editor.errors_left()),
                *review_brief(carries),
                *self.truth,
            ]

        dry = self.dry_run_lines(n)
        stuck = self.stuck_lines(n, expert)
        directory = self.folder / "reviser" / f"r{n}"
        await self.run(
            Episode(
                task=ctx.task,
                stage="s1",
                role="reviser",
                directory=directory,
                prompt=prompt,
                tier=self.config.reviser_tier,
                frames=frames,
                editor=editor,
                memory=editor.doc,
                memory_label=label,
                finish_guard=functools.partial(self._reviser_finish, editor, outcome.directory),
                finish_refusals=LINT_RETRIES,
                step_limit=outcome.step_limit,
                extra={
                    "protocol": "imitate",
                    "seed": self.package.seed,
                    "attempt": n,
                    **self.shown(True),
                },
            )
        )
        _lines(directory / "edits.jsonl", editor.records)
        self.rejected += [record for record in editor.records if not record.accepted]
        changes = diff(memory.doc, editor.doc)
        _write(directory / CHANGES, changes.model_dump(mode="json"))
        await ctx.hooks.on_revise(ctx.task, self.package.seed, n, editor.records, changes)
        ctx.emit("revise", attempt=n, changes=len(changes.params) + len(changes.text))
        if not changes.empty and self._storable(editor):
            memory = await ctx.store.commit(
                editor.doc,
                status="s1_draft",
                source=VersionSource(kind="s1", attempt=n, ref=self._ref(directory)),
                parent=memory.meta.version,
            )
        _write(directory / AFTER, {"version": memory.meta.version, "hash": memory.meta.hash})
        return memory

    def split_guard(
        self, editor: Editor, directory: Path, before: MemoryDoc | None = None
    ) -> str | None:
        """Why a finish waits for OBJECTS to say which is which: roles the identification in
        `directory` (the Demo Reader's own, an attempt's) split on (`review.split_votes`)."""
        if not self.config.review.split_votes:
            return None
        return split_refusal(editor.doc, split_votes(directory), before)

    async def _reviser_finish(self, editor: Editor, attempt: Path, note: str) -> str | None:
        """The S1 Reviser's finish: refused as `_refuse_errors_and_crossed_specs` says, and
        while the attempt's identification split on roles OBJECTS does not order."""
        if refusal := await _refuse_errors_and_crossed_specs(editor, note):
            return refusal
        if "objects" in editor.locked:  # it passed as it is: nothing to say which is which
            return None
        return self.split_guard(editor, attempt, editor.base)

    def reviewing(self) -> bool:
        """Whether S1 reviews attempts against the expert: on, the Reviser privileged, and the
        ground-truth layer there (the referee's)."""
        review = self.config.review
        return review.enabled and self.config.reviser_tier == "P" and self.layer is not None

    async def expert_review(
        self, n: int, memory: Version, outcome: EpisodeOutcome
    ) -> ExpertDiff | None:
        """Attempt `n` (which read `memory`) against the expert's state
        (`learning.expert_diff`), in `referee/expert_diff.json` (kept: a resume reuses it), with an
        `expert_diff` event; None when S1 does not review (`reviewing`)."""
        if not self.reviewing():
            return None
        assert self.layer is not None
        directory = outcome.directory / REFEREE
        if (found := load_review(directory)) is not None:
            return found
        judged = next(
            (directory / name for name in (VERDICT,) if (directory / name).is_file()),
            None,
        )
        progress = None
        if judged is not None:
            saved = json.loads(judged.read_bytes()).get("progress", {})
            progress = {arm: ArmProgress.model_validate(p) for arm, p in saved.items()}
        with np.load(outcome.directory / PHYSICS) as physics:
            recorded = dict(physics)
        check = plan_check(self.planner, self.ctx.profile, np.asarray(recorded["qpos_cmd"]))
        ref = first_registry(outcome.directory)
        registry = None
        if ref is not None:
            data = await self.ctx.driver.blobs.aget_bytes(ref)
            registry = SceneRegistry.model_validate_json(data)
        try:
            diff = await review(
                with_contacts(self.keypoints, await self.contact_keypoints()),
                self.layer,
                self.shapes(outcome.directory),
                recorded,
                attempt_calls(outcome.directory),
                progress,
                outcome.success,
                memory.doc,
                referee=self.config.referee,
                config=self.config.review,
                grip_threshold=self.config.grip_threshold,
                holding=self.holding(),
                plan=check,
                unreproducible=unreproducible(self.dry_run_gaps(n), self.config.review),
                registry=registry,
            )
        except InfraError:  # the planner's: the job's infra reruns handle it
            raise
        except Exception as error:  # a review of what the attempt did, not the attempt
            log.exception("the expert review of attempt %d of %s failed", n, self.ctx.task)
            self.ctx.emit("expert_diff_error", attempt=n, error=f"{type(error).__name__}: {error}")
            return None
        _write(directory / EXPERT_DIFF, diff.model_dump(mode="json"))
        first = None if diff.first is None else diff.stages[diff.first]
        self.ctx.emit(
            "expert_diff",
            attempt=n,
            verdict=diff.verdict,
            stage=None if first is None else stage_line(first),
            parameters=[] if first is None else first.parameters,
            plan=None if first is None else first.plan,
        )
        return diff

    async def reverts(self, n: int, memory: Version) -> dict[tuple[str, str | None], Any]:
        """What the previous revision changed: the targets the Memory attempt `n` read
        (`memory`) has changed from the one attempt `n - 1` read, with their values before."""
        if not self.config.review.revert_guard:
            return {}
        previous = self._read_by(n - 1) if n > 1 else None
        if previous is None or previous == memory.meta.hash:
            return {}
        before = await self.ctx.store.load(self.ctx.task, previous)
        return changed_targets(before.doc, memory.doc)

    async def dry_run(self, n: int, memory: Version) -> Version:
        """Before new attempt `n`, when the Memory is new to attempts (the first, or a revision
        changed it): its grasp, placement and held-path specs against the expert (placements and
        paths as their calls would command them, planned from the recorded robot) on
        perception's registry of the demonstration's start (`learning.spec_check`; the Demo
        Reader's, which every fit was made
        on; never a later attempt's, which may have measured the objects otherwise), in
        `dryrun<n>/specs.json`. It only reports: the specs off the expert go into the brief of
        the Reviser after attempt `n` (`dry_run_lines`), which corrects them with the attempt's
        own evidence; nothing here rewrites a fit. Done once (its `AFTER` record; a resume
        reuses it, and runs one cut short again)."""
        if not (self.reviewing() and self.config.review.dry_run):
            return memory
        if n > 1 and self._read_by(n - 1) == memory.meta.hash:
            return memory  # the previous attempt ran this Memory as it is
        directory = self.folder / f"{DRY_RUN}{n}"
        if (found := await self._after(directory, memory, None)) is not None:
            return found
        set_aside(directory)
        checked = await self.dry_gaps(memory.doc, f"before attempt {n}")
        if checked is None:
            return memory
        source, gaps, failed = checked
        if failed is not None:
            self.ctx.emit("spec_check_error", attempt=n, error=failed)
        off = [gap.parameter for gap in gaps if gap.status == "off"]
        self.ctx.emit("spec_check", attempt=n, source=source.name, off=off, checked=len(gaps),
                      error=failed)  # fmt: skip
        directory.mkdir(parents=True, exist_ok=True)
        specs = [gap.model_dump(mode="json") for gap in gaps]
        found = {"source": source.name, "specs": specs}
        _write(directory / "specs.json", found if failed is None else {**found, "error": failed})
        _write(directory / AFTER, {"version": memory.meta.version, "hash": memory.meta.hash})
        return memory

    async def dry_gaps(
        self, doc: MemoryDoc, when: str
    ) -> tuple[Path, list[SpecGap], str | None] | None:
        """The spec dry run of `doc` (`learning.spec_check`, planned from the expert's state) on
        the Demo Reader's registry as it stands (its events so far: during its episode too),
        with the episode it read and why it failed (None: it ran; a check, not the attempt: a
        failure leaves no gaps and is said, never a silent "checked 0"); None without the scene
        or the ground-truth layer. `when`: for the log."""
        inputs = await self.dry_inputs()
        if inputs is None or self.layer is None:
            return None
        source, registry, robots, shapes = inputs
        tcps = {t: {a: s.tcp.p for a, s in robot.arms.items()} for t, robot in robots.items()}
        try:
            gaps = spec_check(
                self.package.keypoints,
                self.layer,
                shapes,
                registry,
                doc,
                self.ctx.profile,
                self.config.review,
                tcps=tcps,
                robots=robots,
                given=last_fact(source, DEMO_FITS),
                released=released_poses(last_fact(source, DEMO_RELEASES), registry),
            )
            gaps = await planned(gaps, self.planner, self.ctx.profile, robots)
        except Exception as error:  # a check, not the attempt: the job goes on without it
            failed = f"{type(error).__name__}: {error}"
            log.warning("the spec dry run %s of %s failed: %s", when, self.ctx.task, failed,
                        exc_info=True)  # fmt: skip
            return source, [], failed
        return source, gaps, None

    async def dry_inputs(
        self,
    ) -> tuple[Path, SceneRegistry, dict[int, RobotState], dict[str, ObjectShape]] | None:
        """What the spec dry run checks on (`_dry_scene`) and the object shapes it reads the
        expert's grasps with (the Reader's, else an attempt's); None without the scene."""
        scene = await self._dry_scene()
        if scene is None:
            return None
        shapes = next(
            (
                self.shapes(path.parent.parent)
                for path in [
                    self.folder / "reader" / REFEREE / SHAPES,
                    *sorted(self.folder.glob(f"attempt*/{REFEREE}/{SHAPES}")),
                ]
                if path.is_file()
            ),
            {},
        )
        return (*scene, shapes)

    async def reader_dry_run(self, doc: MemoryDoc) -> str | None:
        """The Demo Reader's finish refused while the spec dry run of its draft finds a spec off
        the expert on the demonstration's state (`ReviewConfig.reader_dry_run` times at most): the
        gaps the Reviser after the first attempt would read, shown before any attempt, so that a
        fit `place_object` cannot reach or a form that turns the object off the expert's is
        corrected by the reader that fitted it; None when nothing is off (or the check is off,
        or did not run)."""
        review = self.config.review
        if not (self.reviewing() and review.dry_run) or self.reader_checks >= review.reader_dry_run:
            return None
        checked = await self.dry_gaps(doc, "of the Demo Reader's draft")
        if checked is None or checked[2] is not None:
            return None
        off = [gap for gap in checked[1] if gap.status == "off"]
        self.reader_checks += 1
        self.ctx.emit(
            "reader_dry_run", check=self.reader_checks, off=[gap.parameter for gap in off]
        )
        if not off:
            return None
        return (
            "before finishing: the spec dry run of this draft (simulator truth, S1 only) finds "
            "these specs off what the expert did: each grasp spec synthesized on this episode's "
            "registry against the expert's grasp, each placement and held path as its stage's "
            "call would command it on the demonstration's state, against the expert's TCP and "
            "planned from the expert's own state:\n"
            + "\n".join(render_specs(off)[1:])  # the gaps, without the Reviser's heading
            + "\nCorrect each (refit it with demo_relation, another of its forms, the call's "
            "constrain or distances, or a target the planner reaches), then finish."
        )

    async def _dry_scene(self) -> tuple[Path, SceneRegistry, dict[int, RobotState]] | None:
        """What the spec dry run checks on: the Demo Reader's episode, its
        last registry, and the recorded robot at every keypoint's frame; None without them."""
        reader = self.folder / "reader"
        source = reader if (reader / EVENTS).is_file() else None
        ref = None if source is None else last_registry(source)
        if source is None or ref is None:
            return None
        registry = SceneRegistry.model_validate_json(await self.ctx.driver.blobs.aget_bytes(ref))
        return source, registry, await demo_robots(self.package, self.ctx.driver.blobs)

    async def guards(self, n: int, memory: Version) -> dict[str, Any]:
        """The S1 Reviser's Editor guards after attempt `n` (which read `memory`): OBJECTS locked
        while it is as it was when an attempt passed on this seed (`lock_passed_objects`), and
        placement edits that cannot change behaviour refused (`noop_edits`)."""
        review = self.config.review
        found: dict[str, Any] = {}
        if review.lock_passed_objects:
            for k in range(1, n):
                attempt = self.folder / f"attempt{k}"
                hash_ = self._read_by(k)
                if hash_ is None or not attempt_passed(attempt):
                    continue
                passed = await self.ctx.store.load(self.ctx.task, hash_)
                if passed.doc.objects == memory.doc.objects:
                    found["locked"] = {
                        "objects": f"OBJECTS is as it was when attempt {k} passed on this seed "
                        f"(Memory v{passed.meta.version:03d}): the identification it gives "
                        "worked, so a later failure is no evidence against it; leave it, and "
                        "change what the expert comparison names"
                    }
                    break
        diff = load_review(self.folder / f"attempt{n}" / REFEREE) if self.reviewing() else None
        if diff is not None:
            # where the review matched each stage's release: a refit for it may be made there
            found["matched"] = {
                row.stage: f"{row.arm} {row.keypoint}"
                for row in diff.stages
                if row.event == "open" and row.stage is not None
            }
        if review.repeat_guard:
            found["failed"] = await self.failed_values(n)
        effects = []
        if review.identity_guard and diff is not None and diff.verdict == "identification":
            refused = target_refusals(self.folder / f"attempt{n}", memory.doc.params)
            effects.append(functools.partial(identity_guard, targets=refused))
        ended = self.folder / f"attempt{n}" / SUMMARY
        if review.settled_guard and diff is not None and ended.is_file():
            outcome = EpisodeOutcome.model_validate_json(ended.read_bytes())
            if found_settled := settled(diff, outcome.end.reason == UNCONFIRMED):
                effects.append(functools.partial(settled_guard, frozenset(found_settled)))
        motions = moving_at_latch(self.package.keypoints)
        if review.latch_guard and motions and ended.is_file():
            outcome = EpisodeOutcome.model_validate_json(ended.read_bytes())
            if outcome.end.reason == UNCONFIRMED:
                base = robot_base(self.ctx.profile).matrix()
                effects.append(functools.partial(latch_guard, motions, base))
        if review.noop_edits and self.reviewing():
            scene = await self._dry_scene()
            if scene is not None:
                _, registry, robots = scene
                effects.append(
                    placement_effect(
                        self.package.keypoints,
                        registry,
                        robots,
                        self.ctx.profile,
                        review,
                        last_fact(scene[0], DEMO_FITS),
                        released_poses(last_fact(scene[0], DEMO_RELEASES), registry),
                    )
                )
        if review.dry_run_gate and (gate := await self.dry_run_gate(n)) is not None:
            effects.append(gate)
        if effects:
            found["effect"] = _first(effects)
        return found

    async def dry_run_gate(
        self, n: int
    ) -> Callable[[MemoryDoc, MemoryDoc, Any], str | None] | None:
        """The Reviser's gate after attempt `n` (`spec_check.dry_run_gate`): the specs the dry run
        it ran under found reproducing the expert stay so; None without that dry run or its
        scene."""
        passing = [gap.parameter for gap in self.dry_run_gaps(n) if gap.status == "ok"]
        inputs = await self.dry_inputs() if passing and self.reviewing() else None
        if inputs is None or self.layer is None:
            return None
        source, registry, robots, shapes = inputs
        tcps = {t: {a: s.tcp.p for a, s in robot.arms.items()} for t, robot in robots.items()}
        layer, given = self.layer, last_fact(source, DEMO_FITS)
        released = released_poses(last_fact(source, DEMO_RELEASES), registry)

        def check(doc: MemoryDoc, names: Collection[str]) -> list[SpecGap]:
            return spec_check(self.package.keypoints, layer, shapes, registry, doc,
                              self.ctx.profile, self.config.review, tcps=tcps, robots=robots,
                              given=given, only=names, released=released)  # fmt: skip

        return dry_run_gate(check, passing)

    async def failed_values(self, n: int) -> dict[tuple[str, str | None], list[Any]]:
        """By Memory target, the values attempts up to `n` on this seed failed with and none
        passed with (`memory.Editor`'s `failed`)."""
        failed: dict[tuple[str, str | None], list[Any]] = {}
        passed: dict[tuple[str, str | None], list[Any]] = {}
        for k in range(1, n + 1):
            hash_ = self._read_by(k)
            if hash_ is None:
                continue
            read = await self.ctx.store.load(self.ctx.task, hash_)
            bucket = passed if attempt_passed(self.folder / f"attempt{k}") else failed
            for key, value in targets(read.doc).items():
                bucket.setdefault(key, []).append(value)
        out = {}
        for key, values in failed.items():
            good = [jsonable(v) for v in passed.get(key, [])]
            out[key] = [v for v in values if jsonable(v) not in good]
        return out

    def dry_run_gaps(self, n: int) -> list[SpecGap]:
        """The spec dry run attempt `n` ran under: the latest before it (none runs before an
        attempt whose Memory the previous one ran)."""
        for k in range(n, 0, -1):
            path = self.folder / f"{DRY_RUN}{k}" / "specs.json"
            if path.is_file():
                return [SpecGap.model_validate(g) for g in json.loads(path.read_bytes())["specs"]]
        return []

    def record_guards(
        self, n: int, expert: ExpertDiff | None, doc: MemoryDoc
    ) -> list[EditValidator]:
        """The Reviser's validators after attempt `n` that read the seed's record: `EditStreak`
        (`ReviewConfig.edit_streak`) and, with a review that found a first divergence,
        `AfterFirst` (`ReviewConfig.after_first`)."""
        review = self.config.review
        found: list[EditValidator] = []
        if review.edit_streak:
            streaks = edit_streaks(self.folder, n)
            found.append(EditStreak(streaks, review.edit_streak, _refit_points(doc)))
        improvised = improvised_only(expert, doc) if review.after_first and expert else {}
        if improvised and expert is not None and expert.first is not None:
            first = expert.stages[expert.first]
            found.append(AfterFirst(improvised, f"{first.arm} {first.keypoint}"))
        return found

    def stuck_lines(self, n: int, expert: ExpertDiff | None) -> list[str]:
        """Where the last `STUCK_TRIES` attempts up to `n` failed the same way, on the same
        stage, at the same point (`roles.stuck_failure`): the two revisions between them changed
        nothing that mattered there. Says why the Reviser's last edit of that spec was refused
        (the guard's reason, `r<n-1>/edits.jsonl`) and offers to refit it where the review found
        the first divergence. Information only; nothing is refused by it."""
        tries = [self.folder / f"attempt{k}" for k in range(n - STUCK_TRIES + 1, n + 1)]
        if n < STUCK_TRIES or not all((t / SUMMARY).is_file() for t in tries):
            return []
        found = stuck_failure(tries)
        if found is None:
            return []
        spec, reason, stage = found
        at = "" if stage is None else f" at stage {stage}"
        lines = [
            f"No progress: the last {STUCK_TRIES} attempts each failed first with {{{spec}}}"
            f"{at} ({reason}), sent to the same point (within 1 cm): the revisions between them "
            "did not change what ran there."
        ]
        refused = [
            record
            for record in _records(self.folder / "reviser" / f"r{n - 1}" / "edits.jsonl")
            if not record.accepted and spec in (getattr(record.op, a, None) for a in _OP_NAMES)
        ]
        if refused:
            reason = refused[-1].reason.splitlines()[0] if refused[-1].reason else "no reason"
            lines.append(f"Your last edit of {spec} ({refused[-1].op.op}) was refused: {reason}")
        first = None if expert is None or expert.first is None else expert.stages[expert.first]
        if first is not None and self._fits is not None:
            lines.append(
                f"The review's first divergence is at {first.arm} {first.keypoint}: "
                f'fit_spec(name="{spec}", at="{first.arm} {first.keypoint}") refits it there as '
                "the demonstration has it, or change another stage."
            )
        return lines

    def dry_run_lines(self, n: int) -> list[str]:
        """The spec dry run before attempt `n` as its Reviser reads it: the specs it found off
        the expert (none: nothing to add)."""
        path = self.folder / f"{DRY_RUN}{n}" / "specs.json"
        if not path.is_file():
            return []
        found = json.loads(path.read_bytes())
        if found.get("error"):
            return [dry_run_failed(found["error"])]
        gaps = [SpecGap.model_validate(g) for g in found["specs"]]
        # an unchecked spec is said too: a check that could not run is no pass
        shown = [gap for gap in gaps if gap.status != "ok" or gap.note.startswith("its target")]
        return render_specs(shown) if shown else []

    def _storable(self, editor: Editor) -> bool:
        """Whether the edited Memory may be stored: a version that fails the linter never is (the
        Memory stays as it was; the findings are in the event)."""
        left = editor.errors_left()
        if left:
            self.ctx.emit("lint_unfixed", errors=len(left), first=_listed(left[:3]))
        return not left

    async def unbind(self, memory: Version) -> Version:
        """`memory`, or where it binds the left or the right arm (the `fixed_arm` lint, as a
        resumed S1 can find it), its rewrite by role
        (`unbind_arms`) as the next draft: an Editor does not open over a binding."""
        rewrite = unbind_arms(memory.doc)
        if not rewrite.changes:
            return memory
        note = f"unbind_arms: {len(rewrite.changes)} changes"
        self.ctx.emit("unbind_arms", version=memory.meta.version, changes=len(rewrite.changes))
        return await self.ctx.store.commit(
            rewrite.doc,
            status="s1_draft",
            source=VersionSource(kind="s1", note=note),
            parent=memory.meta.version,
        )

    async def criteria(self, memory: Version, attempts: int) -> tuple[Version, dict[str, Any]]:
        """A Reviser writes the done criteria (`done_spec` parameters) aimed at the
        demonstration's final state, its last frame (`relations_demo.json` of the latest attempt
        that read it, in that attempt's roles; where none did, read on the latest attempt's
        frames and registry, `demo_target`); the relations the attempts ended with
        (`relations.json`), where there are any, only widen the tolerance. A change is the next
        draft. It edits the criteria alone (`EditRules.scope` done: the passes ran the rest as it
        is), and its finish is refused once while every criterion holds at some try's start (they
        would pass before the task is done); still so, they are not kept. A run finished before a
        stop is reused (its `AFTER` record), one cut short is set aside and run again. Returns
        the Memory and the record `exit.json` keeps: the criteria written, or why none were, and
        the attempts whose ends were not read."""
        ctx = self.ctx
        ends = []
        unread = []
        target = None
        for n in range(1, attempts + 1):
            path = self.folder / f"attempt{n}" / RELATIONS
            if path.is_file():
                ends.append((f"attempt {n}", json.loads(path.read_bytes())))
            else:
                unread.append(n)
            if (aimed := self.folder / f"attempt{n}" / DEMO_RELATIONS).is_file():
                target = json.loads(aimed.read_bytes())
        if target is None:
            target = await self.demo_target(attempts)
        record: dict[str, Any] = {"unread": unread}
        if target is None and not any(end["success"] for _, end in ends):
            why = "no demonstration to read, nor any success"
            ctx.emit("criteria", skipped=why, unread=unread)
            return memory, {**record, "written": 0, "skipped": why}
        directory = self.folder / "criteria"
        if (found := await self._after(directory, memory, None)) is not None:
            skipped = json.loads((directory / AFTER).read_bytes()).get("skipped")
            return found, {**record, "written": len(done_criteria(found.doc)), "skipped": skipped}
        set_aside(directory)
        memory = await self.unbind(memory)
        rules = ctx.memory.rules("s1").model_copy(update={"scope": "done"})
        editor = Editor(
            memory.doc, rules=rules, lint=ctx.memory.lint, repair=True, validators=validators()
        )
        scenes = [*ends, *([] if target is None else [("the demonstration", target)])]

        def prompt(opened: EpisodeOpened | None) -> tuple[str, list[ContentPart]]:
            text = role_prompts(ctx).criteria(
                instruction=self.package.instruction, editor=editor, profile=ctx.profile
            )
            brief = relations_brief(ends, target)
            return text, [*brief, *end_brief(self.package, self.end)]

        async def started(note: str) -> str | None:
            held = held_at_start(done_criteria(editor.doc), scenes)
            if not held:
                return None
            return (
                f"every criterion already holds at the start of {', '.join(held)}, so they "
                "would pass before the task is done: add or tighten one the task changes"
            )

        await self.run(
            Episode(
                task=ctx.task,
                stage="s1",
                role="reviser",
                directory=directory,
                prompt=prompt,
                tier=self.config.reviser_tier,
                editor=editor,
                memory=editor.doc,
                memory_label=f"v{memory.meta.version:03d}",
                finish_guard=started,
                step_limit=_step_limit(self.folder, attempts),
                extra={"protocol": "imitate", "seed": self.package.seed, "criteria": True},
            )
        )
        _lines(directory / "edits.jsonl", editor.records)
        changes = diff(memory.doc, editor.doc)
        held = held_at_start(done_criteria(editor.doc), scenes)
        ctx.emit("criteria", changes=len(changes.params) + len(changes.text), held=held)
        skipped = None
        if held:
            skipped = f"the criteria hold at the start of {', '.join(held)}"
        elif not changes.empty and self._storable(editor):
            memory = await ctx.store.commit(
                editor.doc,
                status="s1_draft",
                source=VersionSource(kind="s1", ref=self._ref(directory), note="done criteria"),
                parent=memory.meta.version,
            )
        after = {"version": memory.meta.version, "hash": memory.meta.hash, "skipped": skipped}
        _write(directory / AFTER, after)
        return memory, {**record, "written": len(done_criteria(memory.doc)), "skipped": skipped}

    async def demo_target(self, attempts: int) -> dict[str, Any] | None:
        """The relations at the demonstration's final state read on the latest attempt's frames
        and registry with the package's frames (`extension.demo_relations`; the demo seed's
        scene is every attempt's), kept in `relations_demo.json` beside the attempts (read once);
        None where there is nothing to read it with (no attempt has a log, or no perception
        toolset is mounted); a reading that fails raises (it may be transient: the criteria step
        is retried, not skipped)."""
        path = self.folder / DEMO_RELATIONS
        if path.is_file():
            return dict(json.loads(path.read_bytes()))
        for n in range(attempts, 0, -1):
            summary = self.folder / f"attempt{n}" / SUMMARY
            if not summary.is_file():
                continue
            outcome = EpisodeOutcome.model_validate_json(summary.read_bytes())
            if outcome.episode_id is None:
                continue
            seed, _ = tries_frames([(n, outcome)])
            keypoints = self.package.keypoints.model_dump(mode="json")
            seed = seed.model_copy(
                update={
                    "frames": [*seed.frames, *self.package.frames],
                    "facts": {**seed.facts, DEMO: keypoints},
                }
            )
            found = await demo_relations(self.ctx.driver, seed, phase="s1")
            if found is not None:
                _write(path, found)
            return found
        return None

    async def fall_back(self, attempts: int) -> dict[str, Any]:
        """The best Memory this S1 tried (`fallback_ranking` under `fallback_rule`), lint-clean,
        with its done criteria (`criteria`): what `exit.json` records of an S1 that did not pass
        (`fallback`, None when no version qualifies or none is lint-clean, with why in
        `fallback_reason`; `s1_exit` `best_memory` or `none`; `converged: false`). Under
        `success`, `fallback.hinted_only` says its official successes all came in hinted
        attempts (it qualifies all the same: the rule asks for a success)."""
        ctx = self.ctx
        rule = self.config.fallback_rule
        ranked = fallback_ranking(self.folder, await ctx.store.versions(ctx.task), rule)
        basis = {"converged": False, "fallback_rule": rule}
        if not ranked:  # under `success`: no attempt had an official success, S1 stops
            ctx.emit("fallback", version=None, reason=NO_SUCCESS)
            return {"fallback": None, **basis, "s1_exit": "none", "fallback_reason": NO_SUCCESS}
        tried_fix = False
        for score, meta, evidence in ranked:
            memory = await ctx.store.load(ctx.task, meta.version)
            fixed = False
            if unlintable(memory, ctx):
                if tried_fix:  # one lint-fix pass, on the best version the linter fails
                    continue
                tried_fix = True
                memory = await self.fix_on_load(FALLBACK_FIX, memory)
                if unlintable(memory, ctx):
                    continue
                fixed = True
            written = None
            if self.config.done_criteria:
                memory, written = await self.criteria(memory, attempts)
            after = memory.meta  # after the lint fix and the done criteria, if they changed it
            found = {"version": after.version, "hash": after.hash, "picked": meta.version}
            found["evidence"] = {**evidence, "score": list(score), "fixed": fixed}
            if rule == "success":  # its official successes all came with the hint
                found["hinted_only"] = score[0] == 0
            ctx.emit("fallback", **found)
            return {"fallback": found, **basis, "s1_exit": "best_memory", "criteria": written}
        ctx.emit("fallback", version=None, reason=NO_LINT_CLEAN)
        return {"fallback": None, **basis, "s1_exit": "none", "fallback_reason": NO_LINT_CLEAN}

    async def exit(
        self,
        memory: Version,
        attempts: int,
        hinted: int,
        passed: bool,
        reason: Literal[
            "attempts exhausted", "unfitted specs", "memory fails the linter"
        ] = "attempts exhausted",
        criteria: dict[str, Any] | None = None,
        fallback: dict[str, Any] | None = None,
    ) -> ImitateResult:
        ctx = self.ctx
        why: Literal[
            "passed", "attempts exhausted", "unfitted specs", "memory fails the linter"
        ] = "passed" if passed else reason
        found = (fallback or {}).get("fallback") is not None
        result = ImitateResult(
            task=ctx.task,
            seed=self.package.seed,
            passed=passed,
            reason=why,
            attempts=attempts,
            hinted=hinted,
            unhinted=attempts - hinted,
            version=memory.meta.version,
            hash=memory.meta.hash,
            s1_exit="passed" if passed else "best_memory" if found else "none",
        )
        # attempts on disk after the exit: a resumed replay exited earlier than the run did (kept)
        later = sorted(
            int(m[1])
            for path in self.folder.glob("attempt*")
            if (m := _ATTEMPT.fullmatch(path.name)) and int(m[1]) > attempts
        )
        record: dict[str, Any] = {"infra_reruns": self.reruns, "demo_truth": bool(self.truth)}
        if criteria is not None:  # the done criteria step's record (`criteria`)
            record["criteria"] = criteria
        record["after_exit"] = [f"attempt{n}" for n in later]
        record |= fallback or {}
        _write(self.folder / EXIT, {**result.model_dump(mode="json"), **record})
        await ctx.hooks.on_stop(ctx.task, why, None)
        ctx.emit("stop", reason=why, version=memory.meta.version, attempts=attempts)
        return result

    def _ref(self, directory: Path) -> str:
        return directory.relative_to(self.ctx.directory).as_posix()


def _first(
    effects: Sequence[Callable[[MemoryDoc, MemoryDoc, Any], str | None]],
) -> Callable[[MemoryDoc, MemoryDoc, Any], str | None]:
    """One `Editor` effect from several: the first refusal any gives."""

    def effect(before: MemoryDoc, after: MemoryDoc, op: Any) -> str | None:
        return next((why for check in effects if (why := check(before, after, op))), None)

    return effect


SPEC_KINDS: Final = ("grasp_spec", "place_spec", "hinge_spec", "path_spec")


def settled_guard(
    names: Collection[str], before: MemoryDoc, after: MemoryDoc, op: Any
) -> str | None:
    """The S1 Reviser's `Editor` effect after a review that found every close matching the
    expert and holding, on an attempt that ended on a finish the official check did not confirm
    (`expert_diff.settled`): an edit that changes one of those grasps' parameters is refused, as
    the attempt gives no evidence about them. Every other section stays open."""
    changed = [
        name
        for name in sorted(names)
        if name in before.params
        and (
            name not in after.params
            or jsonable(after.params[name].value) != jsonable(before.params[name].value)
        )
    ]
    if not changed:
        return None
    return (
        f"{', '.join(changed)} matched the expert and held in this attempt, which failed at its "
        "end (a finish the official check did not confirm): the attempt is no evidence about "
        "it. Leave it, and look at the end state, the done check and the last stage"
    )


LATCH_BACK_M: Final = 0.002
"""An edit moves a placement or a lift back from the latch when it does so by more than this
(m): a rounding of what was there is no move."""
_LIFTED: Final = re.compile(r"\blifted\(\s*([a-z0-9_]+)\s*,\s*([^,)]+)")
_TABLE_GAP: Final = re.compile(r"\bgap_z\(\s*([a-z0-9_]+)\s*,\s*table\b")


def _lift_heights(doc: MemoryDoc, done: str) -> dict[str, float]:
    """Each object a stage's completion text `done` holds `lifted(<object>, <height>)` to, with
    the height (a number or a `{param}` of `doc`; one it cannot read is left out)."""
    heights = {}
    for role, given in _LIFTED.findall(done):
        given = given.strip()
        param = doc.params.get(given.strip("{}")) if given.startswith("{") else None
        try:
            heights[role] = param.value if isinstance(param, NumberParam) else float(given)
        except ValueError:
            continue
    return heights


def latch_guard(
    motions: Mapping[Arm, Vec3], base: np.ndarray, before: MemoryDoc, after: MemoryDoc, op: Any
) -> str | None:
    """The S1 Reviser's `Editor` effect after an attempt that ended on a finish the official
    check did not confirm, of a demonstration still moving what it held when its success latched
    (`motions`: `goal.moving_at_latch`, world; `base`: the robot's base frame, 4x4
    world): the demonstration's end is the least that passes, so an edit that moves the end back
    from the latch is refused: a placement's offset lowered (when an arm rose into the latch) or,
    for a robot-frame or world-axes placement, moved back against the arm's motion; a lift's
    height lowered (a `lifted(...)` height, a lift done criterion's value); a `lifted(<object>,
    ...)` replaced by a two-sided `gap_z(<object>, table, ...)` band."""
    rising = any(d[2] >= RISING_M for d in motions.values())
    moved = []
    for name, old in before.params.items():
        new = after.params.get(name)
        if isinstance(old, PlaceSpecParam) and isinstance(new, PlaceSpecParam):
            delta = np.subtract(new.value.offset_m, old.value.offset_m)
            if rising and delta[2] < -LATCH_BACK_M:
                moved.append(f"{name} lowered {-delta[2]:.3f} m")
            spec = old.value
            arm = spec.arm if spec.arm in motions else None
            if arm is None and spec.arm is None and len(motions) == 1:
                arm = next(iter(motions))
            if arm is None or (spec.target.kind != "robot_frame" and spec.frame != "world_axes"):
                continue
            way = np.asarray(motions[arm], dtype=float)
            if spec.frame != "world_axes":
                way = base[:3, :3].T @ way
            size = float(np.linalg.norm(way[:2]))
            back = -float(delta[:2] @ way[:2]) / size if size >= MOVED_M else 0.0
            if back > LATCH_BACK_M:
                moved.append(f"{name} moved {back:.3f} m back against the {arm} arm's motion")
        elif rising and isinstance(old, DoneSpecParam) and isinstance(new, DoneSpecParam):
            if lift_criterion(old.value) and new.value.value < old.value.value - LATCH_BACK_M:
                moved.append(f"{name} lowered {old.value.value - new.value.value:.3f} m")
    stages = {stage.name: stage.done for stage in before.workflow}
    for stage in after.workflow if rising else []:
        was = stages.get(stage.name)
        if was is None:
            continue
        old_h, new_h = _lift_heights(before, was), _lift_heights(after, stage.done)
        for role, height in old_h.items():
            if role in new_h and new_h[role] < height - LATCH_BACK_M:
                moved.append(f"{stage.name}'s lift of {role} lowered {height - new_h[role]:.3f} m")
            elif role not in new_h and role in _TABLE_GAP.findall(stage.done):
                moved.append(f"{stage.name}'s lift of {role} made a two-sided gap_z band")
    if not moved:
        return None
    return (
        f"{'; '.join(moved)}: this attempt ended on a finish the official check did not "
        "confirm, and the demonstration was still moving what it held into its success latch, "
        "so its end is the least that passes: the miss was short of the latch, never past it. "
        "Do not lower, pull back or band a placement or lift; if the end needs to change, move it "
        "further the way the demonstration moved into the latch"
    )


STUCK_TRIES: Final = 3
_OP_NAMES: Final = ("param", "name")
"""Attempts in a row (two revisions between them) failing alike before the S1 Reviser is told
its revisions made no progress (`stuck_lines`)."""


def identity_guard(
    before: MemoryDoc, after: MemoryDoc, op: Any, targets: Collection[str] = ()
) -> str | None:
    """The S1 Reviser's `Editor` effect after a review that blames the identification
    (`expert_diff` `identification`): an edit to a spec it already had is refused, as the review's
    gaps were measured on the wrong objects. Defining a spec, and every other
    section, stays open; so does setting the `target` of a spec in `targets`, whose own call the
    attempt's tool refused for its target (`target_refusals`: independent of the review's
    gaps, such as a place spec targeting the role the arm holds)."""
    if isinstance(op, SetCategory) and op.field == "target" and op.param in targets:
        return None
    changed = [
        name
        for name, param in before.params.items()
        if param.kind in SPEC_KINDS
        and name in after.params
        and jsonable(after.params[name].value) != jsonable(param.value)
    ]
    if not changed:
        return None
    return (
        f"the expert comparison blames the identification, not {', '.join(changed)}: its gaps "
        "were measured on the wrong objects. Fix OBJECTS (how each role is described), and leave "
        "the specs as they are"
    )


# the place tool's refusals of a spec for its target (its `target`, toolsets.manipulation)
_TARGET_REFUSALS = (
    "the place spec's target is the object the arm holds",
    "a place spec names its target by role",
)


def target_refusals(directory: Path, specs: Collection[str]) -> frozenset[str]:
    """The Memory specs (of `specs`) an attempt's model call passed by name (`{spec}`) whose
    tool refused it for the spec's target (a tool error carrying one of the place tool's two
    target refusals, `_TARGET_REFUSALS`): an error of the spec's own, which the attempt's tool
    found whatever the review's gaps say."""
    path = directory / "tool_calls.jsonl"
    if not path.is_file():
        return frozenset()
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        call = json.loads(line)
        if call.get("origin") != "model" or call.get("code") != "tool_error":
            continue
        text = " ".join(
            str(part.get("text", ""))
            for part in call.get("content") or []
            if isinstance(part, dict)
        )
        if not any(phrase in text for phrase in _TARGET_REFUSALS):
            continue
        arguments = call.get("arguments") or {}
        found |= {
            str(value).strip("{}")
            for key in ("grasp", "place", "hinge", "path")
            if isinstance(value := arguments.get(key), str) and value.strip("{}") in specs
        }
    return frozenset(found)


def _pointing(keypoints: KeypointSet, arm: Arm) -> Vec3:
    """An arm's gripper axis (`KeypointSet.pointing`), the other arm's for an arm the
    demonstration did not use (regrouped keypoints moved onto it)."""
    return keypoints.pointing.get(arm) or next(iter(keypoints.pointing.values()))


def kept_verdict(
    attempt: Path,
    protocol: Mapping[str, Any],
    referee: RefereeConfig,
    segments: str | None,
) -> bool:
    """The verdict a resume keeps for a finished attempt (`protocol`: its `config.json`'s).
    Refused: an attempt judged with other referee values or other motion starts (`segments`):
    resume an S1 with the referee it started with. It reads files only."""
    stored = bool(json.loads((attempt / REFEREE / VERDICT).read_bytes())["verdict"]["passed"])
    recorded = protocol.get("referee")
    judged = None if recorded is None else RefereeConfig.model_validate(recorded)
    if judged == referee and _same_segments(protocol.get("referee_segments"), segments):
        return stored
    raise ValueError(
        f"{attempt} was judged with the referee {recorded} and motion starts "
        f"{protocol.get('referee_segments')}, not {referee.model_dump(mode='json')} and "
        f"{segments}: resume this S1 with the referee it started with (and its package's motion "
        "starts), or use a new run"
    )


def resume_plan(folder: Path, config: ImitateConfig) -> dict[str, Any]:
    """What a resume of the S1 in `folder` (`<task>/S1/<seed>/`) under `config` would do, read
    from its files alone (nothing is judged or written): its finished attempts as the schedule
    replays them (`hinted`, `passed`: as counted, failed where a revision followed it), the
    attempt the replay exits at (None: it goes on), and the next attempt it would run. A verdict
    `kept_verdict` refuses raises."""
    segments = segments_key(config.demo)
    rows: list[dict[str, Any]] = []
    for n in itertools.count(1):
        attempt = folder / f"attempt{n}"
        if not (attempt / "config.json").is_file():
            break
        protocol = json.loads((attempt / "config.json").read_bytes())["protocol"]
        hinted = bool(protocol["hinted"])
        if not (attempt / REFEREE / VERDICT).is_file():
            if not (folder / f"attempt{n + 1}" / "config.json").is_file():
                break  # cut short: it runs again
            rows.append({"attempt": n, "hinted": hinted, "passed": False})  # history
            continue
        kept = kept_verdict(attempt, protocol, config.referee, segments)
        revision = folder / "reviser" / f"r{n}"
        following = (folder / f"attempt{n + 1}" / "config.json").is_file()
        revised = (revision / AFTER).is_file() or (following and (revision / SUMMARY).is_file())
        passed = kept and not revised
        rows.append({"attempt": n, "hinted": hinted, "passed": passed})
    window: list[bool] = []  # the unhinted verdicts, as the schedule replays them
    exit_at = None
    for row in rows:
        if row["hinted"]:
            continue
        window.append(row["passed"])
        if row["passed"] and exits(window, config):
            exit_at = row["attempt"]
            break
    return {
        "attempts": rows,
        "exit_at": exit_at,
        "next": None if exit_at is not None else len(rows) + 1,
        "memory": _reads(folder, exit_at, len(rows)),
    }


def chosen_end_targets(folder: Path) -> tuple[list[EndTarget], int | None]:
    """The demonstration's end as targets among the attempts in `folder` that read it (hinted
    ones, whose frames seed the demonstration), with the attempt they come from (None: no
    attempt has read it): the reading that places the most roles, the latest of those. The
    demonstration's end is the same on every attempt of its seed; what differs is the registry
    a try ends with, which the reading replays the demonstration's grips on, so one that places
    fewer roles lost them to that try."""
    found = []
    for path in folder.glob(f"attempt*/{END_TARGETS}"):
        number = path.parent.name.removeprefix("attempt")
        if number.isdigit():
            data = json.loads(path.read_text(encoding="utf-8"))
            targets = [EndTarget.model_validate(t) for t in data.get("targets", [])]
            roles = {r for t in targets for r in (t.a, t.b, *t.peers) if r is not None}
            placed = sum(covers([], targets, r) for r in roles)
            found.append((placed, int(number), targets))
    if not found:
        return [], None
    _, n, targets = max(found, key=lambda f: (f[0], f[1]))
    return targets, n


def _reads(folder: Path, exit_at: int | None, finished: int) -> str | None:
    """The hash of the Memory a resume goes on with (before its lint-fix on load, if any): the
    one the replayed exit's attempt read, else what the last finished attempt left (its
    revision's, or the one it read); None with no attempt."""
    n = exit_at or finished
    if n == 0:
        return None
    if exit_at is None and (after := folder / "reviser" / f"r{n}" / AFTER).is_file():
        return str(json.loads(after.read_bytes())["hash"])
    return str(json.loads((folder / f"attempt{n}" / "config.json").read_bytes())["memory"]["hash"])


def _step_limit(folder: Path, attempts: int) -> int | None:
    """The task's step limit, as the attempts' outcomes recorded it (None: none did)."""
    for n in range(attempts, 0, -1):
        path = folder / f"attempt{n}" / SUMMARY
        if path.is_file():
            found = EpisodeOutcome.model_validate_json(path.read_bytes()).step_limit
            if found is not None:
                return found
    return None


def progress_text(keypoints: KeypointSet, progress: Mapping[Arm, ArmProgress]) -> str:
    """The keypoint progress a hinted Actor reads: per arm the keypoints passed, missed ones and
    the next with their closest approach (attempt minus keypoint), and the rest."""
    lines = ["Keypoint progress of the demonstration after this action:"]
    for arm, kps in keypoints.arms.items():
        state = progress[arm]
        passed = [
            kp.name for kp, p in zip(kps, state.keypoints, strict=True) if p.status == "passed"
        ]
        parts = [f"passed {', '.join(passed)}"] if passed else []
        later = []
        for kp, p in zip(kps, state.keypoints, strict=True):
            if p.status == "missed" or kp.index == state.target:
                parts.append(describe(arm, kp, p).removeprefix(f"{arm} "))
            elif p.status == "pending":
                later.append(f"{kp.name} ({kp.phase})")
        if later:
            parts.append(f"then {', '.join(later)}")
        lines.append(f"- {arm}: " + ("; ".join(parts) if parts else "no keypoints"))
    return "\n".join(lines)


def _actions(path: Path) -> dict[int, str]:
    """Each turn's admitted action with its arguments (rationale left out)."""
    found: dict[int, str] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        turn = json.loads(line)
        if turn["action"] is None:
            continue
        for exchange in turn["exchanges"]:
            for call in exchange["calls"]:
                if call["name"] == turn["action"] and call["status"] == "executed":
                    arguments = call["arguments"]
                    if isinstance(arguments, dict):
                        arguments = {k: v for k, v in arguments.items() if k != "rationale"}
                    found[turn["turn"]] = f"{call['name']} {json.dumps(arguments)}"[:200]
    return found


def _shapes(path: Path) -> dict[str, ObjectShape]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return {name: ObjectShape.model_validate(shape) for name, shape in loaded.items()}


def _hinted(attempt: Path) -> bool:
    """Whether the attempt in `attempt` was hinted (its `config.json`; not hinted without one)."""
    path = attempt / "config.json"
    if not path.is_file():
        return False
    return bool(json.loads(path.read_bytes()).get("protocol", {}).get("hinted"))


RAN: Final = frozenset(
    {
        ReceiptReason.ok,
        ReceiptReason.clamped,
        ReceiptReason.stopped_short,
        ReceiptReason.budget_exhausted,
    }
)
"""Receipts of a call that ran its spec (also any that used steps past its look and is no hard
failure): only such a call carries a refused Memory spec out as stored (`replaced_specs`); one
refused as invalid, as a repeat or otherwise before it ran does not."""


def ran(call: Call) -> bool:
    """Whether `call` ran what it passed without a hard failure (`RAN`)."""
    reason = call.reason
    if reason is None or reason in HARD_FAILURES:
        return False
    used = call.result.get("steps_used", 0) or 0
    return reason in RAN or used > (call.result.get("look_steps", 0) or 0)


def replaced_specs(doc: MemoryDoc, attempt: Path) -> dict[str, str]:
    """The Memory specs (of `doc`, the Memory an attempt read) the attempt in `attempt` passed (by
    name, or written out as stored: `expert_diff.memory_specs`) in a call refused before anything
    moved and never carried out as stored afterwards (no later call that passed it ran without a
    hard failure: `ran`), each with the first refusal's reason and note and, where a later call
    ran a spec of the same kind in its place, what that spec changed from the stored one (the
    direction a revision can move the stored spec). An unhinted pass with such a spec was made
    with whatever the Actor wrote in its place, so it fails."""
    refused: dict[str, str] = {}
    replaced: dict[str, str] = {}
    for call in attempt_calls(attempt):
        names = memory_specs(doc, call)
        reason = call.reason
        unmoved = call.result.get("steps_used", 0) == call.result.get("look_steps", 0)
        if names and reason in REFUSED and unmoved:  # `ActionReceipt.unmoved`
            note = str(call.result.get("note") or "")
            for name in names:
                refused.setdefault(name, f"{reason}: {note[:160]}" if note else str(reason))
            continue
        if not ran(call):
            continue
        for name in names:
            refused.pop(name, None)
            replaced.pop(name, None)
        for name in refused:
            if name in names or name in replaced:
                continue
            stored = doc.params[name].value
            for _, written in written_specs(call):
                if type(written) is type(stored):
                    changes = "; ".join(spec_changes(stored, written)) or "none"
                    replaced[name] = f"{call.name} in turn {call.turn} then ran: {changes}"
                    break
            else:
                other = [n for n in names if type(doc.params[n].value) is type(stored)]
                if other:
                    replaced[name] = (
                        f"{call.name} in turn {call.turn} then ran the Memory's {other[0]}"
                    )
    return {
        name: "; ".join(filter(None, [why, replaced.get(name)])) for name, why in refused.items()
    }


def attempt_passed(attempt: Path) -> bool | None:
    """An S1 attempt's verdict as it was judged; None before it was judged."""
    if (path := attempt / REFEREE / VERDICT).is_file():
        return bool(json.loads(path.read_bytes())["verdict"]["passed"])
    return None


def fallback_ranking(
    folder: Path, versions: Sequence[VersionMeta], rule: FallbackRule = "success"
) -> list[tuple[tuple[float, ...], VersionMeta, dict[str, Any]]]:
    """S1's versions (`versions`, in its store) best first, by the attempts in `folder` that read
    each (`config.json`'s Memory; set-aside attempts and those that ended on infrastructure left
    out). Under `rule` `success` only the versions that an attempt with an
    official success read (none: S1 has no fallback), by unhinted official successes, then
    official successes in all (hinted ones too), then the version number (the later wins). Under
    `progress`, every version, by unhinted referee passes, unhinted official successes, the best
    prefix of required keypoints (neither standoff nor retreat) one arm passed in order, as a
    fraction of its required ones, hinted passes, hinted successes, then the version number (so
    without any signal the latest draft leads). Each with its score and evidence. Files only."""
    reads: dict[str, list[dict[str, Any]]] = {}
    for path in folder.glob("attempt*"):
        if not _ATTEMPT.fullmatch(path.name) or not (path / SUMMARY).is_file():
            continue
        summary = json.loads((path / SUMMARY).read_bytes())
        if summary.get("end_class") == "infra":
            continue
        config = json.loads((path / "config.json").read_bytes())
        reads.setdefault(str(config["memory"]["hash"]), []).append(
            {
                "attempt": int(path.name.removeprefix("attempt")),
                "hinted": bool(config["protocol"]["hinted"]),
                "passed": bool(attempt_passed(path)),
                "success": bool(summary.get("success")),
                "prefix": _prefix(path / REFEREE / VERDICT),
            }
        )
    ranked = []
    for meta in versions:
        rows = reads.get(meta.hash, [])
        unhinted = [row for row in rows if not row["hinted"]]
        hinted = [row for row in rows if row["hinted"]]
        evidence: dict[str, Any] = {"attempts": sorted(row["attempt"] for row in rows)}
        score: tuple[float, ...]
        if rule == "success":
            successes = sum(row["success"] for row in rows)
            if not successes:
                continue
            score = (sum(row["success"] for row in unhinted), successes, meta.version)
            evidence["rule"] = rule
        else:
            prefix = max((row["prefix"] for row in rows), default=0.0)
            score = (
                sum(row["passed"] for row in unhinted),
                sum(row["success"] for row in unhinted),
                prefix,
                sum(row["passed"] for row in hinted),
                sum(row["success"] for row in hinted),
                meta.version,
            )
        ranked.append((score, meta, evidence))
    return sorted(ranked, key=lambda found: found[0], reverse=True)


def _prefix(path: Path) -> float:
    """The most of an arm's required keypoints (neither standoff nor retreat) passed in order, as
    a fraction of them, in a referee's `keypoints.json` (0 without one)."""
    if not path.is_file():
        return 0.0
    best = 0.0
    for arm in json.loads(path.read_bytes()).get("progress", {}).values():
        required = [kp for kp in arm["keypoints"] if kp["kind"] not in ("standoff", "retreat")]
        passed = next(
            (i for i, kp in enumerate(required) if kp["status"] != "passed"), len(required)
        )
        if required:
            best = max(best, passed / len(required))
    return round(best, 4)


def _first_named(diff: ExpertDiff | None) -> frozenset[str] | None:
    """The Memory parameters the review names at its first divergence (`memory.Editor`'s
    `first`); None without one, or after an identification error."""
    if diff is None or diff.first is None or diff.verdict == "identification":
        return None
    return frozenset(given.split(" ")[0] for given in diff.stages[diff.first].parameters)


def _refit_points(doc: MemoryDoc) -> dict[str, str]:
    """By spec the demonstration fitted (its source `demo <arm K>`), where `fit_spec` refits it."""
    prefix = f"{FITTED_SOURCE} "
    return {
        name: param.source.removeprefix(prefix)
        for name, param in doc.params.items()
        if param.source.startswith(prefix)
    }


def edit_streaks(folder: Path, n: int) -> dict[tuple[str, str | None], int]:
    """By target (`memory.review.target_of`), how many revisions in a row up to the one after
    attempt `n - 1` edited it (accepted), each followed by an attempt that passed no more of the
    demonstration's required keypoints (`_prefix`) than the best unhinted attempt before it (a
    hinted attempt shows no progress: its hints did the work). A revision that does not edit a
    target ends its run; an unhinted attempt passing more resets every count, and a refit
    (`fit_spec`) of a parameter resets its own."""
    streaks: dict[tuple[str, str | None], int] = {}
    first = folder / "attempt1"
    best = 0.0 if _hinted(first) else _prefix(first / REFEREE / VERDICT)
    for k in range(1, n):
        records = [r for r in _records(folder / "reviser" / f"r{k}" / "edits.jsonl") if r.accepted]
        refit = {r.op.name for r in records if isinstance(r.op, FitSpec)}
        edited = {t for r in records if (t := target_of(r.op)) is not None}
        streaks = {
            t: streaks.get(t, 0) + 1 for t in edited if t[0] not in refit
        }  # only what this revision edited stays in a run
        attempt = folder / f"attempt{k + 1}"
        if _hinted(attempt):
            continue
        after = _prefix(attempt / REFEREE / VERDICT)
        if after > best:
            streaks = {}
        best = max(best, after)
    return streaks


def improvised_only(diff: ExpertDiff, doc: MemoryDoc) -> dict[str, str]:
    """The Memory parameters `diff` names only at stages after its first divergence, every one
    of them a call the Actor improvised (its rationale named no workflow stage of `doc`, or it
    passed arguments its stage's call does not name), each with the first such stage; a stage
    no call reached names its parameters as the Memory has them, which keeps them editable."""
    if diff.first is None:
        return {}
    stages = {stage.name for stage in doc.workflow}
    before = {g.split(" ")[0] for s in diff.stages[: diff.first + 1] for g in s.parameters}
    found: dict[str, str] = {}
    kept: set[str] = set()
    for stage in diff.stages[diff.first + 1 :]:
        if stage.call is None:  # nothing ran there: the Memory's parameters are not in question
            kept.update(g.split(" ")[0] for g in stage.parameters)
            continue
        improvised = bool(stage.invented) or stage.stage not in stages
        for given in stage.parameters:
            name = given.split(" ")[0]
            if name in before:
                continue
            if improvised:
                found.setdefault(name, f"{stage.arm} {stage.keypoint}")
            else:
                kept.add(name)
    return {name: where for name, where in found.items() if name not in kept}


def _named(diff: ExpertDiff | None) -> frozenset[str]:
    """The Memory parameters the review names at the stages that diverged: new evidence about
    them (the Reviser may set back what the previous revision changed there). None after an
    identification error: the stages were measured on the wrong objects."""
    if diff is None or diff.verdict == "identification":
        return frozenset()
    return frozenset(
        given.split(" ")[0]
        for stage in diff.stages
        if stage.verdict is not None and stage.verdict not in NOT_MEMORY
        for given in stage.parameters
    )


def revision_ledger(folder: Path, n: int, *, last: int = LEDGER_TRIES) -> list[str]:
    """The S1 Reviser's record of the seed so far, oldest first: each of the `last` attempts
    before `n` with how it ended (hinted or not, the official check, the referee's verdict and
    first reason) and the revision that followed it (what it changed; not the cause it gave,
    which was a claim that later revisions would read as a fact). The
    Reviser sees one try in full; this tells it which values earlier tries ran with and what came
    of them, so that it does not undo a change a later try passed with or repeat one that failed
    the same way."""
    lines: list[str] = []
    for k in range(max(1, n - last), n):
        attempt = folder / f"attempt{k}"
        if not (attempt / SUMMARY).is_file():
            continue
        lines.append(_tried(attempt, k))
        lines += _revised(folder / "reviser" / f"r{k}", k)
    return lines


def _tried(attempt: Path, k: int) -> str:
    outcome = EpisodeOutcome.model_validate_json((attempt / SUMMARY).read_bytes())
    hinted = bool(json.loads((attempt / "config.json").read_bytes())["protocol"]["hinted"])
    official = "latched" if outcome.success else "never latched"
    head = f"- attempt {k} ({'hinted' if hinted else 'unhinted'}): official success {official}"
    judged = attempt / REFEREE / VERDICT
    if judged.is_file():
        verdict = json.loads(judged.read_bytes())["verdict"]
        reasons = [r for r in verdict.get("reasons", []) if "latched" not in r]
        head += "; referee " + ("passed" if verdict.get("passed") else "failed")
        if reasons and not verdict.get("passed"):
            head += f" ({_clip(reasons[0], 200)})"
    return f"{head}; ended {outcome.end.reason}"


def _revised(directory: Path, k: int) -> list[str]:
    """What revision `k` changed (its `CHANGES`; without that record, its accepted edits),
    as ledger lines, and whether it ended without its finish (its edits may be half done)."""
    found = directory / CHANGES
    if found.is_file():
        changes = MemoryDiff.model_validate_json(found.read_bytes())
        done = [line.removeprefix("- ") for line in render_diff(changes).splitlines()]
        done = [] if done == ["(no changes)"] else done
    else:  # each place's last write: a chain of steps on one value shows where it ended
        latest = {}
        for record in _records(directory / "edits.jsonl"):
            if record.accepted:
                where, value = _edited(record.op)
                latest.pop(where, None)
                latest[where] = value
        done = [where + value for where, value in latest.items()]
    summary = directory / SUMMARY
    if not done and not summary.is_file():
        return []
    shown = done[:_LEDGER_CHANGES]
    more = len(done) - len(shown)
    what = "; ".join(_clip(line, 160) for line in shown) + (f"; and {more} more" if more else "")
    lines = [f"  then revision r{k} changed: {what or 'nothing'}"]
    if summary.is_file():  # its stated cause stays out: a claim, not a record
        end = json.loads(summary.read_bytes()).get("end") or {}
        if end.get("reason") != "finish":
            lines.append(f"  it ended without its finish ({end.get('reason')})")
    return lines


def _edited(op: Any) -> tuple[str, str]:
    """An accepted edit as a ledger entry, for a revision with no `CHANGES` record: where it
    wrote and, for a number or category, the value."""
    data = op.model_dump(mode="json", exclude_none=True)
    where = data.get("param") or data.get("stage") or data.get("section") or data.get("lesson")
    where = where or data.get("name") or ""
    if data.get("field"):
        where = f"{where}.{data['field']}"
    if data["op"] in ("set_number", "set_category"):
        return f"params.{where}", f" set to {json.dumps(data.get('value'))}"
    return f"{data['op']} {where}".strip(), ""


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _write(path: Path, data: dict[str, Any]) -> None:
    write_atomic(path, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())


def _records(path: Path) -> list[EditRecord]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EditRecord.model_validate_json(line) for line in lines if line.strip()]


def _lines(path: Path, records: Sequence[EditRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")


def grasp_keypoints(
    keypoints: KeypointSet, layer: DemoTruth, shapes: Mapping[str, ObjectShape]
) -> list[str]:
    """The demonstration's grasps: per arm, each keypoint where it closes on an object (the
    referee's `acted_on`, else the object whose outline the TCP is at: `spec_check.grasp_objects`;
    a close on nothing before moving is no grasp), as `<arm> K<n>`."""
    poses = {frame.frame: frame.poses for frame in layer.frames}
    found = []
    for arm, kps in keypoints.arms.items():
        objects = grasp_objects(kps, poses, shapes)
        found += [
            f"{arm} {kp.name}"
            for kp, obj in zip(kps, objects, strict=True)
            if kp.cue == "close" and kp.phase != "preshape" and obj is not None
        ]
    return found


def demo_arms(keypoints: KeypointSet) -> dict[str, Any]:
    """The `DEMO_ARMS` fact: the demonstration's closes in frame order, each its arm, keypoint
    and TCP position."""
    closes = sorted(
        (kp.frame, arm, kp)
        for arm, kps in keypoints.arms.items()
        for kp in kps
        if kp.cue == "close"
    )
    return {
        "grasps": [
            {"arm": arm, "keypoint": f"{arm} {kp.name}", "tcp": list(kp.tcp.p)}
            for _, arm, kp in closes
        ]
    }
