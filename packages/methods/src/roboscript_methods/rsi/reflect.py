"""`p4_reflect`: one round of reflection on an evaluation's records (docs/concepts/learning.md).

The view (`learning.eval_view`) holds D parts and the official success; with `RsiConfig.diagnosis`
it must hold the truth diagnosis too (the upper-bound group), and only then: a view and a config
that disagree are refused, so a run cannot read privilege it did not declare. Before any role
runs, `learning.seeds.check_reflection` holds the records' seeds to the block the evaluation
reflects on. Every episode needs its key (block, task, official index, and the arm when the
round has several) and arm, given as data by the launcher (the benchmark's numbering; `rsi`
imports no benchmark); keys are unique in a round. The lint's Task Memory names
(`LintContext.memories`) of a task the run did not give are read from its episodes'
`memory_before.md` (`lint.memory_names`). Each task's demonstration package
(`Reflection.demos`) gives `rsi_frames` its final keyframe's RGB images, nothing else.

A round runs, through the Hub at `p4_reflect`, each role under the recipe's rules (`recipes/rsi`:
prompts, model-call limits, review passes):

1. The episodes' digests (`digest`) and their triage by task, by code (episodes flagged as
   infrastructure's left out): every episode a success
   (both arms) -> its reflector may only vote, and does not run while the document has nothing it
   could vote on; some succeeded and some failed -> a full reflector; none succeeded -> its
   reflector may only add traps and corrections of a Task Memory step, and the task goes to the
   gap ledger.
2. One reflector per task, in parallel (at most `Reflection.concurrency`), staging proposals for
   its task's part and candidate GLOBAL lessons (`staging`).
3. The curator, once: reads every proposal and lesson, stages GLOBAL's adds and promotions
   (evidence from several tasks), withdraws duplicates.
4. The reviewer: a verdict on every staged proposal, against its evidence and for genericity; the
   authors revise what it rejected once, and it reviews those again (`ReflectRules.passes`),
   with what the curator withdrew as covered by a proposal it rejected (`Staging.reinstate`).
5. The accepted proposals into the round's editor in staging's fixed order.
6. Consolidation, over the document so far (`_consolidate`): the curator merges entries of one
   part that prescribe the same behavior in overlapping situations (the oldest id stays, the
   evidence is united) and retires a task entry that only repeats a GLOBAL one; the reviewer
   accepts each only if the quoted texts prescribe the same behavior, and the curator revises
   what it rejected once for a second review (`ReflectRules.passes`); every entry failing the
   current lint (its vocabulary may have grown since the entry was written) is reworded or
   retired the same way, and one still failing after the second review retires by code
   (`LINT_RETIRED`), so the frozen document passes its own lint; none costs budget. It is
   skipped when no part holds two entries, no task entry stands beside a GLOBAL one and no entry
   fails the lint. The round's document is then committed and frozen in the store as `G<round>`.

`consolidate_only` runs step 6 alone over `start` (a round's document, `resume_round`).

The round's directory (append-only: a rerun goes to a new one) holds `inputs/` (the episodes: key,
seed, arm, directory, file hashes, digest), `reflect/<role>/<label>/pass<n>/` (each role's
conversation), `reflect/frames_viewed.jsonl`, `edits.jsonl` (every proposal with its verdicts and
editor record, and every proposal staging refused, by pass, author and number),
`consolidation.jsonl` (the same for the consolidation step), `G<r>.md` and `G<r>.json` (the frozen
document), `reflection.json` (the run: config, tiers, seeds, episodes, triage, each role's end,
turns, model calls and tokens, the proposals' counts, the document's name and hash) and `gaps.md`
(tasks whose episodes all failed, and the failure causes the reflectors named).
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import anyio
from pydantic import BaseModel, ConfigDict

from roboscript.agent.recording import Recorder, read_lines
from roboscript.agent.session import AgentSession, Brain, Limits, ModelClients
from roboscript.blobs import BlobStore, LocalBlobStore
from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.hub import OpenEpisode
from roboscript.contracts.ids import TaskName
from roboscript.contracts.tools import TextPart
from roboscript.hub import Hub
from roboscript.learning.eval_view import EvalRecordView, episode_memory
from roboscript.learning.seeds import SeedSplit, check_reflection
from roboscript.records import EventWriter, write_atomic
from roboscript_methods.rsi.digest import (
    digest_episode,
    headline,
    render_digest,
    successes_ended,
)
from roboscript_methods.rsi.doc import EpisodeKey, RsiDoc, heading, render
from roboscript_methods.rsi.edits import MergeEntries, RetireEntry, RsiEditor
from roboscript_methods.rsi.lint import (
    LintContext,
    LintItem,
    MemoryNames,
    lint,
    memory_names,
)
from roboscript_methods.rsi.method import (
    CURATOR,
    REFLECT,
    REFLECTOR,
    REVIEWER,
    RsiConfig,
)
from roboscript_methods.rsi.recipe import ReflectRules, RsiRecipe
from roboscript_methods.rsi.staging import Lesson, Seat, Staged, Staging, Triage, counts, spent
from roboscript_methods.rsi.store import FileRsiStore, round_name
from roboscript_methods.rsi.toolset import (
    LESSONS_NOTE,
    RoundData,
    RoundEpisode,
    RsiTools,
    describe,
)

RECORD: Final = "reflection.json"
EDITS: Final = "edits.jsonl"
CONSOLIDATION: Final = "consolidation"
CONSOLIDATION_EDITS: Final = "consolidation.jsonl"
LINT_RETIRED: Final = "fails lint after consolidation"
GAPS: Final = "gaps.md"
INPUTS: Final = "inputs"
ROLES_DIR: Final = "reflect"
RECORDED: Final = (
    "summary.json",
    "turns.jsonl",
    "events.jsonl",
    "config.json",
    "memory_before.md",
    "memory_used.md",  # library mode: the Memories in use at the end, in memory_before.md's place
)
"""The files of an episode's records whose hashes `inputs/episodes.jsonl` keeps."""

TRIAGE_NOTE: Final[dict[Triage, str]] = {
    "full": "Some episodes of this task succeeded and some failed: compare them.",
    "vote_only": "Every episode of this task succeeded: you may only vote on existing entries.",
    "trap_only": (
        "Every episode of this task failed: you may only add traps (a precursor to watch for and "
        "what to do about it) and corrections of a Task Memory step (`corrects`); the task also "
        "goes to the gap ledger."
    ),
}


@dataclass(frozen=True)
class Reflection:
    """What a round runs with: a Hub declaring `p4_reflect` with the `rsi` toolset mounted
    (`tools`, the same instance); the roles' brains (`brain(role, label)`, a new one each); the
    reflection's blob store (images it shows are copied there) and the records' one (`source`,
    read only; None: no images); the store the document goes to; the round's directory; the
    recipe; what the lint knows of the run; each task's demonstration package directory (only
    its final keyframe's RGB images are shown); the model clients; the log; how many reflectors
    run at once."""

    hub: Hub
    tools: RsiTools
    brain: Callable[[str, str], Brain]
    blobs: BlobStore
    store: FileRsiStore
    directory: Path
    recipe: RsiRecipe
    context: LintContext
    source: LocalBlobStore | None = None
    demos: Mapping[TaskName, Path] = field(default_factory=dict)
    models: ModelClients | None = None
    events: EventWriter | None = None
    concurrency: int = 4


class Reflected(BaseModel):
    """The document a round froze."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: ArtifactRef
    doc: RsiDoc


async def reflect(
    run: Reflection,
    config: RsiConfig,
    view: EvalRecordView,
    split: SeedSplit,
    *,
    round: int,
    keys: Sequence[EpisodeKey],
    arms: Sequence[str] | None = None,
    start: RsiDoc | None = None,
    spent: Mapping[TaskName | None, int] | None = None,
    consolidate_only: bool = False,
) -> Reflected:
    """Reflect on `view` into round `round`'s document (from `start`, the previous round's after
    its counts, else empty): commit and freeze it in `run.store` as `G<round>`. `keys` and `arms`
    give each episode of the view, in order, its key and the arm it ran in (None: one arm,
    `baseline`). A second reflection pass over the same round (`resume_round`) starts from the
    first pass's document with `spent`, the edits that pass applied by part, counted against the
    round's budgets. `consolidate_only`: only the consolidation step, over `start`."""
    name = round_name(round)
    if consolidate_only and start is None:
        raise ValueError("consolidate_only consolidates a document: give `start`")
    outputs = (
        INPUTS,
        ROLES_DIR,
        EDITS,
        CONSOLIDATION_EDITS,
        RECORD,
        GAPS,
        f"{name}.json",
        f"{name}.md",
    )
    if written := [o for o in outputs if (run.directory / o).exists()]:
        raise ValueError(
            f"{run.directory} holds {written}: a round's rerun goes to a new directory"
        )
    wanted = {"D", "P"} if config.diagnosis else {"D"}
    if set(view.tiers) != wanted:
        raise ValueError(
            f"the view holds tiers {sorted(view.tiers)} and the config reads {sorted(wanted)}: "
            "the truth diagnosis is read when `diagnosis` says so, and only then"
        )
    seeds = sorted({episode.seed for episode in view.episodes})
    check_reflection(split, config.evaluation, seeds)
    episodes = _round_episodes(run, view, keys, arms)
    rules = run.recipe.reflect
    triage = _triage(episodes)
    doc = start or RsiDoc()
    context = _context(run, episodes)
    staging = Staging(
        doc, run.recipe.rules, rules, context, round, [e.key for e in episodes.values()], spent
    )
    data = RoundData(
        episodes=episodes,
        staging=staging,
        memories=_memories(episodes),
        source=run.source,
        rules=rules,
        demos=run.demos,
    )
    await anyio.to_thread.run_sync(_write_inputs, run.directory, episodes, rules)
    ran: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}

    for task in sorted({e.key.task for e in episodes.values()} - set(triage)):
        skipped[task] = "every episode was flagged as the infrastructure's"
    seats = []
    for task, how in triage.items() if not consolidate_only else ():
        if how == "vote_only" and not (doc.part(None) or doc.part(task)):
            skipped[task] = "every episode succeeded and the document has no entry to vote on"
            continue
        seats.append(Seat(role=REFLECTOR, label=task, task=task, triage=how))
    ran += await _parallel(run, data, [(s, _reflector_brief(data, s, round)) for s in seats])

    if consolidate_only:
        skipped[CURATOR] = "consolidation only"
    elif staging.proposals:
        curator = Seat(role=CURATOR, label=CURATOR)
        ran.append(await _role(run, data, curator, _curator_brief(data, triage, round)))
    else:
        skipped[CURATOR] = "nothing was staged"

    for pass_ in range(1, rules.passes + 1) if not consolidate_only else ():
        if pass_ > 1:
            staging.reinstate(pass_)
            authors = sorted(
                {p.author for p in staging.proposals if p.status == "rejected" and p.pass_ < pass_}
            )
            revisers = [_seat_of(data, triage, author, pass_) for author in authors]
            briefs = [(s, _revision_brief(data, s)) for s in revisers]
            ran += await _parallel(run, data, briefs)
        if not staging.under_review(pass_):
            continue
        reviewer = Seat(role=REVIEWER, label=REVIEWER, pass_=pass_)
        ran.append(await _role(run, data, reviewer, _reviewer_brief(data, pass_)))
        staging.close_review(pass_)
    staging.finish(rules.passes)
    editor = RsiEditor(doc, run.recipe.rules, context, round, staging.episodes, staging.spent)
    staging.apply(editor)
    final, consolidation, roles, why = await _consolidate(run, data, editor.doc, round, context)
    ran += roles
    if why:
        skipped[CONSOLIDATION] = why

    await anyio.to_thread.run_sync(_write_edits, run.directory, staging, data)
    if consolidation is not None:
        await anyio.to_thread.run_sync(_write_consolidation, run.directory, consolidation)
    record = {
        "round": round,
        "config": config.model_dump(mode="json"),
        "tiers": sorted(view.tiers),
        "seeds": seeds,
        "episodes": [e.label for e in episodes.values()],
        "triage": triage,
        "skipped": skipped,
        "roles": ran,
        "usage": _usage(ran),
        "proposals": counts(staging.proposals),
        "refused_at_staging": len(staging.refusals),
        "consolidation": None
        if consolidation is None
        else {
            "proposals": counts(consolidation.proposals),
            "refused_at_staging": len(consolidation.refusals),
            "lint_retired": [r.entry for r in consolidation.retired],
            "entries": [
                sum(e.readable for _, e in editor.doc.entries()),
                sum(e.readable for _, e in final.entries()),
            ],
        },
        "frames_viewed": len(data.viewed),
        "rsi": {"name": name, "hash": final.content_hash()},
        "start": {
            "hash": doc.content_hash(),
            "entries": sum(1 for _ in doc.entries()),
            "spent": {heading(part): n for part, n in sorted(staging.spent.items(), key=str)},
        },
    }
    text = (json.dumps(record, indent=2) + "\n").encode()
    await anyio.to_thread.run_sync(write_atomic, run.directory / RECORD, text)
    ref = await run.store.commit_round(round, final, reason=REFLECT)
    document = (json.dumps(final.model_dump(mode="json"), indent=2) + "\n").encode()
    await anyio.to_thread.run_sync(write_atomic, run.directory / f"{name}.json", document)
    text = render(final).encode()
    await anyio.to_thread.run_sync(write_atomic, run.directory / f"{name}.md", text)
    gaps = _gaps(round, triage, episodes, ran).encode()
    await anyio.to_thread.run_sync(write_atomic, run.directory / GAPS, gaps)
    return Reflected(ref=ref, doc=final)


def _round_episodes(
    run: Reflection, view: EvalRecordView, keys: Sequence[EpisodeKey], arms: Sequence[str] | None
) -> dict[str, RoundEpisode]:
    if len(keys) != len(view.episodes) or (arms is not None and len(arms) != len(keys)):
        raise ValueError("give every episode of the view its key (and arm), in order")
    found: dict[str, RoundEpisode] = {}
    for index, (episode, key) in enumerate(zip(view.episodes, keys, strict=True)):
        if key.task != episode.task:
            raise ValueError(
                f"{episode.task} seed {episode.seed}: key {key.label()} of another task"
            )
        if key.label() in found:
            raise ValueError(f"episode key {key.label()} given twice: keys are unique in a round")
        arm = arms[index] if arms is not None else key.arm or "baseline"
        if key.arm is not None and key.arm != arm:
            raise ValueError(f"episode key {key.label()} names another arm than {arm}")
        digest = digest_episode(episode, key, arm, run.recipe.reflect)
        found[key.label()] = RoundEpisode(key=key, arm=arm, episode=episode, digest=digest)
    return found


def _triage(episodes: Mapping[str, RoundEpisode]) -> dict[str, Triage]:
    """By the episodes' outcomes; episodes flagged as infrastructure's are no outcome of the
    strategy and count for nothing (a task with only those is left out)."""
    tasks: dict[str, list[bool]] = {}
    for episode in episodes.values():
        if not episode.digest.infra:
            tasks.setdefault(episode.key.task, []).append(episode.digest.success)
    how: dict[str, Triage] = {}
    for task in sorted(tasks):
        outcomes = tasks[task]
        how[task] = "vote_only" if all(outcomes) else "trap_only" if not any(outcomes) else "full"
    return how


def _context(run: Reflection, episodes: Mapping[str, RoundEpisode]) -> LintContext:
    """`run.context`, with the Task Memory names of each task it lacks read from the records
    (every version the task's episodes read, their names joined)."""
    found: dict[TaskName, list[MemoryNames]] = {}
    for task, texts in _memory_texts(episodes).items():
        if task not in run.context.memories:
            found[task] = [memory_names(text) for text in texts]
    joined = {
        task: MemoryNames(
            params=tuple(dict.fromkeys(p for n in names for p in n.params)),
            stages=tuple(dict.fromkeys(s for n in names for s in n.stages)),
        )
        for task, names in found.items()
    }
    return replace(run.context, memories={**run.context.memories, **joined})


def _memory_texts(episodes: Mapping[str, RoundEpisode]) -> dict[TaskName, list[str]]:
    texts: dict[TaskName, list[str]] = {}
    for episode in episodes.values():
        directory = episode.episode.directory
        text = None if directory is None else episode_memory(directory)
        if text is not None:
            texts.setdefault(episode.key.task, [])
            if text not in texts[episode.key.task]:
                texts[episode.key.task].append(text)
    return texts


def _memories(episodes: Mapping[str, RoundEpisode]) -> dict[TaskName, str]:
    memories = {}
    for task, found in _memory_texts(episodes).items():
        many = f"\n\n[{len(found)} versions were read; this is the first]"
        note = "" if len(found) == 1 else many
        memories[task] = found[0] + note
    return memories


async def _parallel(
    run: Reflection, data: RoundData, seats: Sequence[tuple[Seat, str]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(seats)
    limiter = anyio.CapacityLimiter(run.concurrency)

    async def one(index: int, seat: Seat, brief: str) -> None:
        async with limiter:
            results[index] = await _role(run, data, seat, brief)

    async with anyio.create_task_group() as group:
        for index, (seat, brief) in enumerate(seats):
            group.start_soon(one, index, seat, brief)
    return [result for result in results if result is not None]


async def _role(
    run: Reflection, data: RoundData, seat: Seat, brief: str, prompt_name: str | None = None
) -> dict[str, Any]:
    """One role's episode as `seat`, briefed with `brief`, under the recipe's prompt
    `prompt_name` (None: the role's); what it did."""
    path = run.recipe.prompts / f"{prompt_name or seat.role}.md"
    prompt = path.read_text(encoding="utf-8")
    limit = getattr(run.recipe.reflect.limits, seat.role)
    session = await run.hub.open_episode(OpenEpisode(stage=REFLECT, role=seat.role))
    episode_id = session.episode_id
    run.tools.attach(episode_id, data, seat)
    directory = run.directory / ROLES_DIR / seat.role / seat.label / f"pass{seat.pass_}"
    recorder = await anyio.to_thread.run_sync(Recorder, directory, run.events)
    try:
        agent = AgentSession(
            episode=session,
            hub=run.hub,
            instructions=prompt,
            blobs=run.blobs,
            recorder=recorder,
            models=run.models,
            brief=[TextPart(text=brief)],
            limits=Limits(max_model_calls=limit),
        )
        result = await run.brain(seat.role, seat.label).run(agent)
    finally:
        run.tools.detach(episode_id)
        await run.hub.close_episode(episode_id)
    return {
        "role": seat.role,
        "label": seat.label,
        "pass": seat.pass_,
        "triage": seat.triage if seat.role == REFLECTOR else None,
        "end": result.end.reason,
        "note": result.end.note,
        "turns": result.turns,
        "model_calls": result.model_calls,
        "usage": result.usage.model_dump(),
        "elapsed_s": round(result.elapsed_s, 1),
    }


async def _consolidate(
    run: Reflection, data: RoundData, doc: RsiDoc, round: int, context: LintContext
) -> tuple[RsiDoc, Staging | None, list[dict[str, Any]], str]:
    """The round's last step over `doc`: the document after it, its staging (None: skipped),
    what ran, and why it was skipped."""
    if not run.recipe.reflect.consolidate:
        return doc, None, [], "the recipe's reflect rules turn consolidation off"
    failing = [i for i in lint(doc, context) if i.error]
    if not failing and not _overlaps_possible(doc):
        why = (
            "no part holds two entries, no task entry stands beside GLOBAL, nothing fails the lint"
        )
        return doc, None, [], why
    rules = run.recipe.rules
    episodes = data.staged.episodes
    staging = Staging(doc, rules, run.recipe.reflect, context, round, episodes, consolidating=True)
    own = replace(data, staging=staging)
    passes = run.recipe.reflect.passes
    ran = []
    for pass_ in range(1, passes + 1):
        curator = Seat(role=CURATOR, label=CONSOLIDATION, pass_=pass_)
        if pass_ == 1:
            brief = _consolidation_brief(doc, round, failing)
        elif staging.rejected(CONSOLIDATION, pass_ - 1):
            brief = _revision_brief(own, curator)
        else:
            break
        ran.append(await _role(run, own, curator, brief, f"{CURATOR}-{CONSOLIDATION}"))
        if staging.under_review(pass_):
            reviewer = Seat(role=REVIEWER, label=CONSOLIDATION, pass_=pass_)
            brief = _consolidation_review_brief(staging, pass_)
            ran.append(await _role(run, own, reviewer, brief, f"{REVIEWER}-{CONSOLIDATION}"))
            staging.close_review(pass_)
    staging.finish(passes)
    editor = RsiEditor(doc, rules, context, round, episodes, consolidating=True)
    staging.apply(editor)
    # the frozen document passes its own lint: what the roles left failing retires
    for entry_id in dict.fromkeys(i.entry for i in lint(editor.doc, context) if i.error):
        record = editor.edit(RetireEntry(id=entry_id, reason=LINT_RETIRED))
        assert record.accepted, record.reason
        staging.retired.append(record)
    return editor.doc, staging, ran, ""


def _overlaps_possible(doc: RsiDoc) -> bool:
    sizes = {part: sum(e.readable for e in doc.section(part)) for part in _parts(doc)}
    beside = sizes.get(None, 0) > 0 and any(n for part, n in sizes.items() if part is not None)
    return beside or any(n >= 2 for n in sizes.values())


def _parts(doc: RsiDoc) -> list[TaskName | None]:
    return [None, *sorted(doc.task_entries)]


def _quoted(doc: RsiDoc, part: TaskName | None) -> list[str]:
    return [f"- {e.id}: {e.text}" for e in doc.section(part) if e.readable]


def _consolidation_brief(doc: RsiDoc, round: int, failing: Sequence[LintItem]) -> str:
    lines = [
        f"# Round {round}: consolidate the document",
        "Merges, retirements and rewordings here cost no edit budget. Each part's entries as the "
        "Actor reads them:",
    ]
    for part in _parts(doc):
        if quoted := _quoted(doc, part):
            lines += ["", f"## {heading(part)}", *quoted]
    lines += ["", "## Entries failing the current lint (reword with `modify`, or retire)"]
    lines += [f"- {i.entry}: {i.rule} {i.match!r}: {i.message}" for i in failing] or ["(none)"]
    return "\n".join(lines)


def _consolidation_review_brief(staging: Staging, pass_: int) -> str:
    under = staging.under_review(pass_)
    lines = [
        f"# Consolidation review: {len(under)} proposals",
        "Give every one a verdict with rsi_verdict; one left without is rejected.",
    ]
    for proposal in under:
        edit = proposal.edit
        ids = edit.ids if isinstance(edit, MergeEntries) else [getattr(edit, "id", "")]
        lines += ["", describe(proposal)]
        if proposal.revises:
            lines.append(f"It revises: {describe(staging.get(proposal.revises))}")
        lines.append("The entries it touches:")
        lines += [f"- {i}: {staging.doc.find(i)[1].text}" for i in ids]
        failing = [f for f in lint(staging.doc, staging.context) if f.entry in ids and f.error]
        lines += [f"Fails the lint: {f.entry} {f.rule} {f.match!r}: {f.message}" for f in failing]
        task = staging.doc.find(ids[0])[0]
        if isinstance(edit, RetireEntry) and task is not None:
            lines += [f"GLOBAL, as the Actor of {task} reads it beside that part:"]
            lines += _quoted(staging.doc, None) or ["(none)"]
    return "\n".join(lines)


def _seat_of(data: RoundData, triage: Mapping[str, Triage], author: str, pass_: int) -> Seat:
    if author == CURATOR:
        return Seat(role=CURATOR, label=CURATOR, pass_=pass_)
    return Seat(role=REFLECTOR, label=author, task=author, triage=triage[author], pass_=pass_)


def _reflector_brief(data: RoundData, seat: Seat, round: int) -> str:
    assert seat.task is not None
    staging = data.staged
    own = [e for e in data.episodes.values() if e.key.task == seat.task]
    rows = [headline(e.digest) for e in own]
    instruction = own[0].digest.instruction
    scope = (
        f"Edits TASK {seat.task} takes this round: {staging.rules.budget(False, round)}; "
        f"candidate GLOBAL lessons you may stage: {staging.reflect.lessons}."
    )
    return "\n".join(
        [
            f"# Round {round}: reflect on TASK {seat.task}",
            TRIAGE_NOTE[seat.triage],
            scope,
            f"An instruction of this task: {instruction}",
            f"From the records, {_ended(data, seat.task)} (each arm's last reported gripper).",
            "",
            "## Episodes (rsi_episode shows each digest, rsi_frames its keyframes)",
            *rows,
        ]
    )


def _ended(data: RoundData, task: str) -> str:
    return successes_ended(e.digest for e in data.episodes.values() if e.key.task == task)


def _curator_brief(data: RoundData, triage: Mapping[str, Triage], round: int) -> str:
    staging = data.staged
    lines = [
        f"# Round {round}: curate GLOBAL",
        f"GLOBAL takes {staging.rules.budget(True, round)} edits this round; an entry needs "
        f"evidence from {staging.rules.global_min_tasks} distinct tasks.",
        "",
        "## Tasks (successes / episodes, triage)",
    ]
    for task, how in triage.items():
        outcomes = [
            e.digest.success
            for e in data.episodes.values()
            if e.key.task == task and not e.digest.infra
        ]
        lines.append(f"- {task}: {sum(outcomes)}/{len(outcomes)}, {how}")
    lines += ["", "## Staged proposals and lessons"]
    lines += [describe(p) for p in staging.proposals]
    return "\n".join(lines)


def _reviewer_brief(data: RoundData, pass_: int) -> str:
    under = data.staged.under_review(pass_)
    lines = [
        f"# Review pass {pass_}: {len(under)} proposals",
        "Give every one a verdict with rsi_verdict; one left without is rejected.",
        "",
    ]
    lines += [describe(p) for p in under]
    tasks = sorted({p.author for p in under if p.author != CURATOR})
    if tasks:
        lines += [
            "",
            "## How each task's successes ended (from the records: each arm's last "
            "reported gripper)",
        ]
        lines += [f"- {task}: {_ended(data, task)}" for task in tasks]
    lessons = [p for p in data.staged.proposals if isinstance(p.edit, Lesson)]
    if lessons:
        lines += ["", f"## Lessons ({LESSONS_NOTE}): not under review"]
        lines += [describe(p) for p in lessons]
    if pass_ > 1:
        lines += ["", "## What they revise"]
        lines += [describe(data.staged.get(p.revises)) for p in under if p.revises]
    return "\n".join(lines)


def _revision_brief(data: RoundData, seat: Seat) -> str:
    rejected = data.staged.rejected(seat.label, seat.pass_ - 1)
    lines = [
        f"# Revision: the reviewer rejected {len(rejected)} of your proposals",
        "Revise each at most once with rsi_edit(edit, revises=<its id>), fixing what the reason "
        "says, or leave it (it is dropped). One rejected as covered or as restating the Task "
        "Memory: unless the sentence the reason quotes prescribes the same concrete behavior, "
        "revise it to make the difference explicit. Then finish.",
        "",
    ]
    lines += [describe(p) for p in rejected]
    return "\n".join(lines)


def _write_inputs(
    directory: Path, episodes: Mapping[str, RoundEpisode], rules: ReflectRules
) -> None:
    rows = []
    by_task: dict[str, list[str]] = {}
    for label, episode in episodes.items():
        records = episode.episode.directory
        files = {}
        if records is not None:
            for name in RECORDED:
                if (path := records / name).is_file():
                    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        text = render_digest(episode.digest, rules)
        rows.append(
            {
                "key": episode.key.model_dump(mode="json"),
                "label": label,
                "seed": episode.episode.seed,
                "arm": episode.arm,
                "success": episode.digest.success,
                "directory": None if records is None else str(records),
                "files": files,
                "digest": episode.digest.model_dump(mode="json"),
                "text": text,
            }
        )
        by_task.setdefault(episode.key.task, []).append(text)
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_atomic(directory / INPUTS / "episodes.jsonl", lines.encode())
    for task, texts in by_task.items():
        write_atomic(directory / INPUTS / "digests" / f"{task}.md", "\n\n".join(texts).encode())


def resume_round(directory: Path, round: int) -> tuple[RsiDoc, dict[TaskName | None, int]]:
    """What a second reflection pass over round `round` starts from: the document the pass in
    `directory` wrote (`G<round>.json`) and the edits it applied by part (from its `edits.jsonl`),
    which count against the round's budgets. Its entries keep their ids and provenance."""
    name = round_name(round)
    doc = RsiDoc.model_validate_json((directory / f"{name}.json").read_text(encoding="utf-8"))
    record = json.loads((directory / RECORD).read_text(encoding="utf-8"))
    if record["round"] != round:
        raise ValueError(f"{directory} reflected round {record['round']}, not {round}")
    proposals = [
        Staged.model_validate({k: v for k, v in line.items() if k != "kind"})
        for line in read_lines(directory / EDITS)
        if line["kind"] == "proposal"
    ]
    return doc, spent(proposals)


def _write_consolidation(directory: Path, staging: Staging) -> None:
    proposals = sorted(staging.proposals, key=lambda p: int(p.id.rsplit("-", 1)[1]))
    lines = [
        json.dumps({"kind": "proposal", **p.model_dump(mode="json")}, ensure_ascii=False)
        for p in proposals
    ]
    lines += [
        json.dumps({"kind": "refused_at_staging", **r}, ensure_ascii=False)
        for r in staging.refusals
    ]
    lines += [
        json.dumps({"kind": "lint_retired", **r.model_dump(mode="json")}, ensure_ascii=False)
        for r in staging.retired
    ]
    write_atomic(directory / CONSOLIDATION_EDITS, "".join(f"{line}\n" for line in lines).encode())


def _write_edits(directory: Path, staging: Staging, data: RoundData) -> None:
    # by pass, author and number: the roles of a pass ran in parallel, the file does not show it
    proposals = sorted(
        staging.proposals, key=lambda p: (p.pass_, p.author, int(p.id.rsplit("-", 1)[1]))
    )
    refusals = sorted(staging.refusals, key=lambda r: (str(r["pass"]), str(r["author"])))
    lines = [
        json.dumps({"kind": "proposal", **p.model_dump(mode="json")}, ensure_ascii=False)
        for p in proposals
    ]
    lines += [json.dumps({"kind": "refused_at_staging", **r}, ensure_ascii=False) for r in refusals]
    write_atomic(directory / EDITS, "".join(f"{line}\n" for line in lines).encode())
    viewed = "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in data.viewed)
    write_atomic(directory / ROLES_DIR / "frames_viewed.jsonl", viewed.encode())


def _usage(ran: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {"model_calls": 0, "elapsed_s": 0.0}
    for role in ran:
        total["model_calls"] += role["model_calls"]
        total["elapsed_s"] = round(total["elapsed_s"] + role["elapsed_s"], 1)
        for name, value in role["usage"].items():
            total[name] = total.get(name, 0) + value
    return total


def _gaps(
    round: int,
    triage: Mapping[str, Triage],
    episodes: Mapping[str, RoundEpisode],
    ran: Sequence[Mapping[str, Any]],
) -> str:
    lines = [f"# Gap ledger, round {round}", "", "## Tasks whose episodes all failed", ""]
    failed = [task for task, how in triage.items() if how == "trap_only"]
    for task in failed:
        n = sum(e.key.task == task for e in episodes.values())
        lines.append(f"- {task}: 0/{n}")
    if not failed:
        lines.append("(none)")
    lines += ["", "## What the reflectors named (their finish notes)", ""]
    notes = [r for r in ran if r["role"] == REFLECTOR and r.get("note")]
    for role in notes:
        lines += [f"### {role['label']} (pass {role['pass']}, {role['triage']})", role["note"], ""]
    if not notes:
        lines.append("(none)")
    return "\n".join(lines).rstrip() + "\n"
