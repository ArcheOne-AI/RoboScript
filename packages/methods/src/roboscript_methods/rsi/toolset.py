"""The `rsi` toolset: the reflection's tools over one round (docs/concepts/learning.md), tier D
only.

A reflection attaches the round (`RoundData`: its valid episodes with their digests, the staged
proposals, the Task Memory each task's Actor read, the records' blob store) and the caller's
`Seat` to each of its episodes (`attach`), so the toolset runs in the reflection's process, as the
`memory` toolset runs beside the runner. Which role sees which tool is `p4_reflect`'s visibility
table (`method.P4_REFLECT`):

- `rsi_episodes(task, arm)`: the round's episodes of a task, one headline each;
- `rsi_episode(key, turns)`: an episode's digest (`digest`), and a page of its turns (and, in a
  round whose config reads the truth diagnosis, the upper-bound group, its diagnosis);
- `rsi_frames(key, frames, cameras)`: keyframe images (the Actor's own observations, read from the
  records' blob store and copied into the reflection's, at most `ReflectRules.frames_per_call`),
  and `demo_final`, the demonstration's final keyframe (its package's success frame, RGB only,
  never its truth layer: a demonstration is what a real robot has too), each view logged as a
  `frames_viewed` event and in the round's record;
- `rsi_doc()`: the document the round started from, with every entry's evidence, counts, votes
  and status (never the Actor's view), the budgets, and the proposals staged so far;
- `rsi_memory(task)`: the Task Memory as that task's Actor read it, so that entries neither repeat
  nor contradict it;
- `rsi_edit(edit, revises)`: stage a proposal (`staging`: checked at once, applied only after the
  review); `rsi_drop(proposal, reason)`: the curator withdraws one;
  `rsi_verdict(proposal, accept, reason)`: the reviewer's verdict.

Listings and keys only ever name the round's valid episodes; a key outside them is an error.
"""

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Final

import anyio
from pydantic import BaseModel, Field

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.errors import ToolError
from roboscript.contracts.ids import EpisodeId, TaskName
from roboscript.contracts.observation import Observation
from roboscript.contracts.tools import ContentPart, ImagePart, TextPart
from roboscript.demos import BLOBS_DIR, frame_alias
from roboscript.learning.demo import load_package
from roboscript.learning.eval_view import EvalEpisode, FrameImage, frame_images
from roboscript.tools import ToolContext, Toolset, tool
from roboscript_methods.rsi.digest import EpisodeDigest, headline, render_digest, render_turns
from roboscript_methods.rsi.doc import EpisodeKey, render
from roboscript_methods.rsi.edits import RsiEditor
from roboscript_methods.rsi.recipe import ReflectRules
from roboscript_methods.rsi.staging import (
    Lesson,
    Proposed,
    Refusal,
    Seat,
    Staged,
    StagedEdit,
    Staging,
)

LESSONS_NOTE: Final = (
    "the reflectors' candidate GLOBAL lessons: the curator's input only, neither entries nor "
    "proposals, so nothing duplicates them"
)
DEMO_FINAL: Final = "demo_final"
"""The frame alias `rsi_frames` takes for the demonstration's final keyframe."""


@dataclass(frozen=True)
class RoundEpisode:
    """One valid episode of the round: its key, arm, the view's episode and its digest."""

    key: EpisodeKey
    arm: str
    episode: EvalEpisode
    digest: EpisodeDigest

    @property
    def label(self) -> str:
        return self.key.label()


@dataclass
class RoundData:
    """What a round's tools read and write: the episodes by label (in order), the staging (None:
    a round that stages nothing), the Task Memory text by task, the records' blob store (None: no
    images), the rules, each task's demonstration package directory (its final keyframe's RGB
    images only: `rsi_frames`), and every frame view."""

    episodes: Mapping[str, RoundEpisode]
    staging: Staging | None
    memories: Mapping[TaskName, str]
    source: LocalBlobStore | None
    rules: ReflectRules
    demos: Mapping[TaskName, Path] = field(default_factory=dict)
    viewed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def staged(self) -> Staging:
        if self.staging is None:
            raise ToolError("this round stages no proposals")
        return self.staging


class TextResult(BaseModel):
    text: str


class FramesResult(BaseModel):
    text: str
    images: list[ImagePart]


def _render_text(result: TextResult) -> list[ContentPart]:
    return [TextPart(text=result.text)]


def _render_frames(result: FramesResult) -> list[ContentPart]:
    return [TextPart(text=result.text), *result.images]


Key = Annotated[str, Field(description="An episode's key as listed: block/task/epNNN.")]


class RsiTools(Toolset):
    name = "rsi"

    def __init__(self) -> None:
        self._seats: dict[EpisodeId, tuple[RoundData, Seat]] = {}

    def attach(self, episode_id: EpisodeId, round: RoundData, seat: Seat) -> None:
        """Have the episode's calls read `round` as `seat`, until `detach`."""
        if episode_id in self._seats:
            raise ValueError(f"episode {episode_id} has a round attached already")
        self._seats[episode_id] = (round, seat)

    def detach(self, episode_id: EpisodeId) -> None:
        del self._seats[episode_id]

    @tool(kind="pure", tier="D", render=_render_text)
    async def rsi_episodes(
        self,
        ctx: ToolContext,
        task: Annotated[TaskName, Field(description="The task whose episodes to list.")],
        arm: Annotated[str | None, Field(description="One arm only; null: every arm.")] = None,
    ) -> TextResult:
        """List this round's episodes of a task: key, seed, arm, official success and what went
        wrong, one line each."""
        round, _ = self._seat(ctx)
        rows = [
            headline(e.digest)
            for e in round.episodes.values()
            if e.key.task == task and (arm is None or e.arm == arm)
        ]
        if not rows:
            where = f" in {arm}" if arm else ""
            raise ToolError(f"this round read no episode of {task!r}{where}")
        successes = sum(e.digest.success for e in round.episodes.values() if e.key.task == task)
        top = f"{task}: {len(rows)} episodes listed; {successes} succeeded across arms"
        return TextResult(text="\n".join([top, *rows]))

    @tool(kind="pure", tier="D", render=_render_text)
    async def rsi_episode(
        self,
        ctx: ToolContext,
        key: Key,
        turns: Annotated[
            str | None,
            Field(
                description="A page of turns, `first-last` (e.g. `1-12`); null: the digest only."
            ),
        ] = None,
    ) -> TextResult:
        """An episode's digest (outcome, failed and refused calls by turn, false finish,
        infrastructure flags, keyframes to ask images of, entries the Actor cited); with `turns`,
        also those turns in full (observations, the model's text, calls and results; images by
        label)."""
        round, _ = self._seat(ctx)
        episode = self._episode(round, key)
        text = render_digest(episode.digest, round.rules)
        if episode.key.task in round.demos:
            text += (
                f"\n- {DEMO_FINAL}: the demonstration's last keyframe, to compare the objects in "
                "the last keyframe with (its gripper and arm pose are not goals)"
            )
        if turns is not None:
            first, last = _range(turns)
            page, stopped = render_turns(episode.episode, first, last, round.rules.page_chars)
            more = "" if stopped > last else f"\n\n[page full: go on from turn {stopped}]"
            text += f"\n\n# turns {first}-{last}\n{page or '(none)'}{more}"
        # only a view the config opened to the diagnosis holds it (`reflect`): the upper bound
        diagnosis = [p.content for p in episode.episode.parts if p.kind == "diagnosis"]
        for content in diagnosis:
            shown = json.dumps(content, ensure_ascii=False, sort_keys=True)
            text += f"\n\n# truth diagnosis (upper-bound group only)\n{shown}"
        return TextResult(text=text)

    @tool(kind="pure", tier="D", render=_render_frames)
    async def rsi_frames(
        self,
        ctx: ToolContext,
        key: Key,
        frames: Annotated[
            list[str], Field(description="Frame aliases from the digest's keyframes.", min_length=1)
        ],
        cameras: Annotated[
            list[str] | None, Field(description="Cameras to show; null: every camera.")
        ] = None,
    ) -> FramesResult:
        """Images of an episode's keyframes as the Actor saw them, a few per call; `demo_final`
        is the demonstration's last keyframe (its object states are the goal; it can come before
        the final release, so its gripper and arm pose are not), when the round has it."""
        round, seat = self._seat(ctx)
        episode = self._episode(round, key)
        found: list[tuple[FrameImage, LocalBlobStore, str]] = []
        missing = 0
        own = [f for f in frames if f != DEMO_FINAL]
        if DEMO_FINAL in frames:
            package = round.demos.get(episode.key.task)
            if package is None:
                raise ToolError(f"this round has no demonstration of {episode.key.task}")
            demo = await anyio.to_thread.run_sync(_demo_final, package, cameras or ())
            found += [(i, LocalBlobStore(package / BLOBS_DIR), "demonstration") for i in demo]
        if own:
            if round.source is None or episode.episode.directory is None:
                raise ToolError("this round has no images of its episodes")
            directory, source = episode.episode.directory, round.source
            try:
                taken = await anyio.to_thread.run_sync(
                    frame_images, directory, source, own, cameras or ()
                )
            except LookupError as error:
                raise ToolError(str(error)) from None
            found += [(i, source, f"step {i.step}") for i in taken.images]
            missing = len(taken.missing)
        cap = round.rules.frames_per_call
        images, lines = [], []
        for image, store, at in found[:cap]:
            data = await anyio.to_thread.run_sync(store.get_bytes, image.image)
            # the same bytes under the same id; the source's ref keeps the image's shape
            await ctx.blobs.aput_bytes(data, image.image.media_type, image.image.meta)
            label = f"{image.frame} {image.camera} ({at})"
            images.append(ImagePart(blob=image.image, label=label))
            lines.append(label)
        if len(found) > cap:
            lines.append(f"[{len(found) - cap} more images: ask for fewer frames or cameras]")
        if missing:
            lines.append(f"[{missing} images are not in the records' blob store]")
        view = {
            "role": seat.role,
            "label": seat.label,
            "pass": seat.pass_,
            "key": key,
            "shown": [i.label for i in images],
            "missing": missing,
        }
        round.viewed.append(view)
        ctx.emit("frames_viewed", view)
        return FramesResult(text="\n".join(lines) or "(no images)", images=images)

    @tool(kind="pure", tier="D", render=_render_text)
    async def rsi_doc(self, ctx: ToolContext) -> TextResult:
        """The experience document this round started from (every entry with its kind, status,
        counts, votes, provenance and evidence), the edits each part takes this round, and the
        proposals staged so far."""
        round, seat = self._seat(ctx)
        staging = round.staged
        budgets = (
            "Consolidation: merges within a part and retirements cost no budget."
            if staging.consolidating
            else RsiEditor(
                staging.doc,
                staging.rules,
                staging.context,
                staging.round,
                staging.episodes,
                staging.spent,
            ).budgets()
        )
        shown = [p for p in staging.proposals if seat.role != "reflector" or p.author == seat.label]
        lessons = [p for p in shown if isinstance(p.edit, Lesson)]
        lines = [budgets, "", render(staging.doc), "# PROPOSALS"]
        lines += [describe(p) for p in shown if p not in lessons] or ["(none)"]
        lines += ["", f"# LESSONS ({LESSONS_NOTE})"]
        lines += [describe(p) for p in lessons] or ["(none)"]
        return TextResult(text="\n".join(lines))

    @tool(kind="pure", tier="D", render=_render_text)
    async def rsi_memory(
        self, ctx: ToolContext, task: Annotated[TaskName, Field(description="A task of the round.")]
    ) -> TextResult:
        """The Task Memory as that task's Actor read it, with its workflow steps: entries must not
        repeat it, and depart from it only as the correction of a named step (`corrects`), which
        the Actor then follows instead of that step."""
        round, _ = self._seat(ctx)
        if task not in round.memories:
            raise ToolError(f"no Task Memory of {task!r} was read in this round")
        return TextResult(text=round.memories[task])

    @tool(kind="memory", tier="D", render=_render_text)
    async def rsi_edit(
        self,
        ctx: ToolContext,
        edit: Annotated[StagedEdit, Field(description="One proposal, tagged by `op`.")],
        revises: Annotated[
            str | None,
            Field(description="In the revision pass: the id of your rejected proposal it revises."),
        ] = None,
    ) -> TextResult:
        """Propose one change of the experience document: add an entry (a `rule`: when it
        applies, what to do, how to check it worked, and with `corrects` the Task Memory step it
        replaces; or a `trap`: a precursor to watch for and what to do about it) with evidence,
        modify an entry, merge entries, retire one, vote an entry helpful or harmful for one
        episode, or (a task's reflector) stage a candidate GLOBAL `lesson` for the curator.
        Evidence and votes name episodes of this round by key and
        the turn or step. A proposal is checked now (lint, caps, duplicates, budget, what your
        role may propose) and refused with the reason if it fails; it is applied only after the
        reviewer accepts it."""
        round, seat = self._seat(ctx)
        try:
            staged = round.staged.stage(seat, _keyed(edit, round), revises)
        except Refusal as error:
            ctx.emit("refused", {"author": seat.label, "reason": str(error)})
            return TextResult(text=f"Refused: {error}")
        ctx.emit("staged", staged.model_dump(mode="json"))
        return TextResult(text=f"Staged {describe(staged)}")

    @tool(kind="memory", tier="D", render=_render_text)
    async def rsi_drop(
        self,
        ctx: ToolContext,
        proposal: Annotated[str, Field(description="A staged proposal's id, e.g. P-x-01.")],
        reason: Annotated[str, Field(min_length=1)],
        covered_by: Annotated[
            str | None,
            Field(
                description="Your staged proposal that covers it (it comes back for review if "
                "that one is rejected); null: withdrawn for good (a duplicate, no evidence)."
            ),
        ] = None,
    ) -> TextResult:
        """Withdraw a staged proposal or lesson, with the reason: one a GLOBAL proposal of yours
        covers (name it), or one that duplicates another or that the evidence does not carry."""
        round, seat = self._seat(ctx)
        try:
            dropped = round.staged.drop(seat, proposal, reason, covered_by)
        except Refusal as error:
            return TextResult(text=f"Refused: {error}")
        ctx.emit("withdrawn", {"proposal": proposal, "reason": reason, "covered_by": covered_by})
        return TextResult(text=f"Withdrawn {dropped.id}.")

    @tool(kind="memory", tier="D", render=_render_text)
    async def rsi_verdict(
        self,
        ctx: ToolContext,
        proposal: Annotated[str, Field(description="A proposal under review, e.g. P-004.")],
        accept: bool,
        reason: Annotated[str, Field(min_length=1)],
    ) -> TextResult:
        """Accept or reject one proposal under review, with the reason: does its evidence (the
        episodes' digests and frames) show what it says, and is it generic (no task-specific
        trick in GLOBAL, no seed, coordinate or instance, nothing the Task Memory already sets)?"""
        round, seat = self._seat(ctx)
        try:
            judged = round.staged.verdict(seat, proposal, accept, reason)
        except Refusal as error:
            return TextResult(text=f"Refused: {error}")
        ctx.emit("verdict", {"proposal": proposal, "accept": accept, "reason": reason})
        left = len(round.staged.under_review(seat.pass_))
        return TextResult(text=f"{judged.id} {judged.status}; {left} left to review.")

    def _seat(self, ctx: ToolContext) -> tuple[RoundData, Seat]:
        found = self._seats.get(ctx.episode_id)
        if found is None:
            raise ToolError("No reflection round is attached to this episode.")
        return found

    def _episode(self, round: RoundData, key: str) -> RoundEpisode:
        if key not in round.episodes:
            raise ToolError(f"{key!r} is no episode of this round: use rsi_episodes for the keys")
        return round.episodes[key]


def _keyed(edit: Proposed, round: RoundData) -> Proposed:
    """`edit` with each episode it names keyed as the round keys it: an arm named where the
    round's key has none (a round of one arm) is dropped when it is that episode's arm, since
    the digests show every episode's arm and a model copies it."""

    def fix(value: Any) -> Any:
        if isinstance(value, list):
            return [fix(v) for v in value]
        if not isinstance(value, dict):
            return value
        value = {k: fix(v) for k, v in value.items()}
        if {"block", "task", "episode"} <= set(value) and value.get("arm") is not None:
            key = EpisodeKey(block=value["block"], task=value["task"], episode=value["episode"])
            found = round.episodes.get(key.label())
            if found is not None and found.arm == value["arm"]:
                value["arm"] = None
        return value

    data = fix(edit.model_dump(mode="json"))
    return type(edit).model_validate(data)


def _demo_final(package: Path, cameras: Collection[str]) -> list[FrameImage]:
    """The RGB images of a demonstration package's final keyframe (its success frame), by
    camera: `package.json` and its blobs only, never the package's truth layer."""
    demo = load_package(package)
    final = demo.keypoints.success_frame
    assert final is not None  # a package requires its success frame
    alias = frame_alias(final)
    entry = next((e for e in demo.frames if e.alias == alias), None)
    if entry is None:
        raise ToolError(f"{package}: the package lacks its final frame {alias}")
    store = LocalBlobStore(package / BLOBS_DIR)
    seen = Observation.model_validate_json(store.get_bytes(entry.observation))
    return [
        FrameImage(
            frame=DEMO_FINAL, frame_id=entry.frame_id, step=entry.step, camera=name, image=view.rgb
        )
        for name, view in sorted(seen.cameras.items())
        if not cameras or name in cameras
    ]


def describe(proposal: Staged) -> str:
    """A proposal in one line, as the roles read it."""
    edit = proposal.edit.model_dump(mode="json", exclude_defaults=True)
    revises = f" revising {proposal.revises}" if proposal.revises else ""
    revises += f"; covered by {proposal.covered_by}" if proposal.covered_by else ""
    verdicts = "".join(
        f"; review {v.pass_}: {'accept' if v.accept else 'reject'} ({v.reason})"
        for v in proposal.verdicts
    )
    note = f"; {proposal.note}" if proposal.note else ""
    return (
        f"{proposal.id} [{proposal.status}] by {proposal.author} (pass {proposal.pass_}{revises}): "
        f"{json.dumps(edit, ensure_ascii=False, sort_keys=True)}{verdicts}{note}"
    )


def _range(turns: str) -> tuple[int, int]:
    first, _, last = turns.partition("-")
    try:
        a, b = int(first), int(last or first)
    except ValueError:
        raise ToolError(f"turns {turns!r}: write first-last, e.g. 1-12") from None
    if a < 0 or b < a:
        raise ToolError(f"turns {turns!r}: first-last with first <= last")
    return a, b
