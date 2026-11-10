"""Staged proposals of one reflection round (docs/concepts/learning.md): what the reflectors and the
curator propose, the reviewer's verdicts, one revision of what it rejected, and the deterministic
application of what it accepted.

A proposal is an edit of the document (`edits`: add, modify, merge, retire, vote) or a candidate
GLOBAL lesson (`Lesson`: a reflector's, read by the curator, never applied). Staging checks a
proposal at once, by a dry run on a scratch editor holding the round's document with its author's
live proposals applied (the lint, the caps, dedup, evidence, the budget of the part), and refuses
it with the reason if the dry run does; it also holds each seat to what it may propose:

- a task's reflector (`Seat.task`) edits only its task's part (an add, a modify, a merge within
  it, a retire) and votes on the entries its episodes read (GLOBAL's and its task's), and stages
  at most `ReflectRules.lessons` lessons with evidence from its own task; by triage, a task whose
  episodes all succeeded only votes (`vote_only`), one whose episodes all failed only adds traps
  (`trap_only`: entries and lessons of kind `trap`);
- the curator edits only GLOBAL (adds with evidence from several tasks, promotions by
  `merge(into_global)`, modifies, retires, votes) and may withdraw any staged proposal (`drop`),
  naming the proposal of its own that covers it: if the reviewer rejects that one, what it
  withdrew comes back for the second review (`reinstate`);
- the reviewer gives each staged proposal of the pass a verdict (`verdict`); a proposal left
  without one is rejected.

A rejected proposal of the first pass may be revised once by its author (a new proposal naming it,
`revises`), reviewed in the second pass; one rejected again, or not revised, is dropped. Accepted
proposals go to the round's editor in a fixed order (`apply`): GLOBAL first (a task's entry near a
GLOBAL one is then refused as a duplicate), then each task's part by task name, then the votes,
each by author and the author's proposal number (ids are numbered per author, so neither depends
on which reflector ran first). Every proposal ends with a status and its editor record.

A consolidating staging (`consolidating`: the round's last step, over the document its review
left) seats only the curator, who stages merges within a part and retirements of any part, and the
reviewer, in one pass, and rewordings (`modify`) of entries that fail the current lint; they cost
no budget (`RsiEditor` `consolidating`).
"""

from collections.abc import Collection, Iterable, Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.ids import TaskName
from roboscript.contracts.tools import ToolModel
from roboscript_methods.rsi.doc import EntryKind, EpisodeKey, Evidence, RsiDoc, Situation, Text
from roboscript_methods.rsi.edits import (
    AddEntry,
    Edit,
    EditRecord,
    MergeEntries,
    ModifyEntry,
    RetireEntry,
    RsiEditor,
    VoteEntry,
)
from roboscript_methods.rsi.lint import LintContext
from roboscript_methods.rsi.recipe import ReflectRules, RsiRules

Role = Literal["reflector", "curator", "reviewer"]
Triage = Literal["vote_only", "full", "trap_only"]
Status = Literal[
    "staged", "lesson", "withdrawn", "accepted", "rejected", "dropped", "applied", "refused"
]


class Lesson(ToolModel):
    """A candidate GLOBAL lesson from one task's episodes, for the curator (never applied): when
    it applies, what to do, how to check it, its kind and its evidence (the task's episodes)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["lesson"] = "lesson"
    kind: EntryKind = "rule"
    when: Situation
    do: Text
    check: Text
    evidence: list[Evidence] = Field(min_length=1)


Proposed = AddEntry | ModifyEntry | MergeEntries | RetireEntry | VoteEntry | Lesson
StagedEdit = Annotated[Proposed, Field(discriminator="op")]


class Seat(BaseModel):
    """Who proposes: the role, its label (a task's name, `curator`, `reviewer`), the task of a
    reflector, its triage, and the review pass it works in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    label: str
    task: TaskName | None = None
    triage: Triage = "full"
    pass_: int = Field(1, ge=1)


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pass_: int
    accept: bool
    reason: str


class Staged(BaseModel):
    """One proposal: its id (`P-<author>-01`, numbered per author, so that ids and the order of
    application do not depend on which reflector ran first), author, pass, what it proposes, the
    proposal it revises, its status, the verdicts, a note (why withdrawn or dropped) and its editor
    record."""

    model_config = ConfigDict(extra="forbid")

    id: str
    author: str
    role: Role
    pass_: int
    edit: StagedEdit
    revises: str | None = None
    covered_by: str | None = None
    status: Status
    verdicts: list[Verdict] = []
    note: str = ""
    record: EditRecord | None = None


class Refusal(ValueError):
    """A proposal staging does not take; its message is the reason."""


class Staging:
    """The proposals of one round over `doc` (the document the round starts from); `spent`: the
    edits an earlier reflection pass over the same round applied, by part (`spent`), which count
    against this pass's budgets."""

    def __init__(
        self,
        doc: RsiDoc,
        rules: RsiRules,
        reflect: ReflectRules,
        context: LintContext,
        round: int,
        episodes: Collection[EpisodeKey],
        spent: Mapping[TaskName | None, int] | None = None,
        *,
        consolidating: bool = False,
    ) -> None:
        self.doc = doc
        self.consolidating = consolidating
        self.spent: dict[TaskName | None, int] = dict(spent or {})
        self.rules = rules
        self.reflect = reflect
        self.context = context
        self.round = round
        self.episodes = frozenset(episodes)
        self.proposals: list[Staged] = []
        self.refusals: list[dict[str, object]] = []
        self.retired: list[EditRecord] = []
        """Consolidation: entries retired by code for failing the lint after the review."""

    def get(self, proposal_id: str) -> Staged:
        for proposal in self.proposals:
            if proposal.id == proposal_id:
                return proposal
        raise Refusal(f"there is no proposal {proposal_id}")

    def stage(self, seat: Seat, edit: Proposed, revises: str | None = None) -> Staged:
        """Stage `edit` for `seat`, or `Refusal` (logged in `refusals`) with the reason."""
        try:
            return self._stage(seat, edit, revises)
        except Refusal as error:
            self.refusals.append(
                {
                    "author": seat.label,
                    "role": seat.role,
                    "pass": seat.pass_,
                    "edit": edit.model_dump(mode="json"),
                    "revises": revises,
                    "reason": str(error),
                }
            )
            raise

    def _stage(self, seat: Seat, edit: Proposed, revises: str | None) -> Staged:
        self._allowed(seat, edit)
        self._revision(seat, revises)
        if isinstance(edit, Lesson):
            self._lesson(seat, edit)
            status: Status = "lesson"
        else:
            record = self._dry_run(seat.label, edit)
            if not record.accepted:
                raise Refusal(f"the document would refuse it: {record.reason}" + _lint(record))
            status = "staged"
        number = sum(p.author == seat.label for p in self.proposals) + 1
        staged = Staged(
            id=f"P-{seat.label}-{number:02d}",
            author=seat.label,
            role=seat.role,
            pass_=seat.pass_,
            edit=edit,
            revises=revises,
            status=status,
        )
        self.proposals.append(staged)
        return staged

    def drop(
        self, seat: Seat, proposal_id: str, reason: str, covered_by: str | None = None
    ) -> Staged:
        """The curator withdraws a staged proposal or lesson, as covered by another proposal
        (`covered_by`: one of its own, staged; if that one is not applied in the end, the
        withdrawn one comes back, `reinstate`) or for good (a duplicate, no evidence)."""
        if seat.role != "curator":
            raise Refusal("only the curator withdraws proposals")
        proposal = self.get(proposal_id)
        if proposal.status not in ("staged", "lesson"):
            raise Refusal(f"{proposal_id} is {proposal.status}, not staged")
        if getattr(proposal.edit, "corrects", None) is not None:
            raise Refusal(
                f"{proposal_id} corrects a Task Memory step: it stays in its task's part for the "
                "reviewer, never withdrawn into a GLOBAL lesson"
            )
        if covered_by is not None:
            cover = self.get(covered_by)
            if cover.id == proposal.id or cover.status != "staged" or cover.author != seat.label:
                raise Refusal(f"{covered_by} is not one of your staged proposals")
        proposal.status = "withdrawn"
        proposal.note = reason
        proposal.covered_by = covered_by
        return proposal

    def reinstate(self, pass_: int) -> list[Staged]:
        """Proposals withdrawn as covered by one the reviewer rejected come back: a proposal
        under review in `pass_`, a lesson as a lesson. Those reinstated."""
        back = []
        for proposal in self.proposals:
            if proposal.status != "withdrawn" or proposal.covered_by is None:
                continue
            cover = self.get(proposal.covered_by)
            if cover.status not in ("rejected", "dropped", "refused", "withdrawn"):
                continue
            lesson = isinstance(proposal.edit, Lesson)
            proposal.status = "lesson" if lesson else "staged"
            if not lesson:
                proposal.pass_ = pass_
            proposal.note = f"back: {cover.id}, which covered it, was {cover.status}"
            back.append(proposal)
        return back

    def verdict(self, seat: Seat, proposal_id: str, accept: bool, reason: str) -> Staged:
        """The reviewer's verdict on a proposal staged for this pass."""
        if seat.role != "reviewer":
            raise Refusal("only the reviewer gives verdicts")
        proposal = self.get(proposal_id)
        open_ = proposal.status in ("staged", "accepted", "rejected")
        if not open_ or proposal.pass_ != seat.pass_:
            raise Refusal(f"{proposal_id} is not under review in pass {seat.pass_}")
        proposal.verdicts = [v for v in proposal.verdicts if v.pass_ != seat.pass_]
        proposal.verdicts.append(Verdict(pass_=seat.pass_, accept=accept, reason=reason))
        proposal.status = "accepted" if accept else "rejected"
        return proposal

    def under_review(self, pass_: int) -> list[Staged]:
        return [p for p in self.proposals if p.pass_ == pass_ and p.status == "staged"]

    def close_review(self, pass_: int) -> None:
        """Proposals of `pass_` the reviewer left without a verdict are rejected."""
        for proposal in self.under_review(pass_):
            proposal.verdicts.append(Verdict(pass_=pass_, accept=False, reason="no verdict given"))
            proposal.status = "rejected"

    def rejected(self, author: str, pass_: int) -> list[Staged]:
        return [
            p
            for p in self.proposals
            if p.author == author and p.pass_ == pass_ and p.status == "rejected"
        ]

    def finish(self, last_pass: int) -> None:
        """Every rejected proposal is dropped: a first-pass one was revised (or not) already, a
        last-pass one had its second review."""
        for proposal in self.proposals:
            if proposal.status == "rejected":
                proposal.status = "dropped"
                revised = any(p.revises == proposal.id for p in self.proposals)
                if proposal.pass_ < last_pass:
                    proposal.note = "rejected; revised" if revised else "rejected; not revised"
                else:
                    proposal.note = "rejected in the last review"

    def apply(self, editor: RsiEditor) -> None:
        """The accepted proposals into `editor`, in the module's order."""
        for proposal in sorted(
            (p for p in self.proposals if p.status == "accepted"), key=self._order
        ):
            assert not isinstance(proposal.edit, Lesson)
            record = editor.edit(proposal.edit)
            proposal.record = record
            proposal.status = "applied" if record.accepted else "refused"

    # checks

    def _allowed(self, seat: Seat, edit: Proposed) -> None:
        if seat.role == "reviewer":
            raise Refusal("the reviewer proposes nothing: give verdicts")
        if self.consolidating:
            merge = isinstance(edit, MergeEntries) and not edit.into_global
            other = isinstance(edit, RetireEntry | ModifyEntry)
            if seat.role != "curator" or not (merge or other):
                raise Refusal(
                    "consolidation: the curator merges entries within one part, retires one, or "
                    "rewords one that fails the lint"
                )
            return
        if seat.role == "curator":
            if isinstance(edit, Lesson):
                raise Refusal("lessons are the reflectors': stage a GLOBAL add instead")
            if self._part(edit) is not None:
                raise Refusal("the curator edits GLOBAL only; a task's part is its reflector's")
            return
        if seat.triage == "vote_only" and not isinstance(edit, VoteEntry):
            raise Refusal("every episode of this task succeeded: you may only vote")
        trap = isinstance(edit, (AddEntry, Lesson)) and edit.kind == "trap"
        correction = isinstance(edit, AddEntry) and edit.corrects is not None
        if seat.triage == "trap_only" and not (trap or correction):
            raise Refusal(
                "every episode of this task failed: you may only add traps and corrections of a "
                "Task Memory step"
            )
        if isinstance(edit, Lesson):
            return
        part = self._part(edit)
        if isinstance(edit, VoteEntry):
            if part not in (None, seat.task):
                raise Refusal(f"your episodes read GLOBAL and TASK {seat.task} only")
            if edit.episode.task != seat.task:
                raise Refusal(f"vote with an episode of {seat.task}")
            return
        if part != seat.task or (isinstance(edit, MergeEntries) and edit.into_global):
            raise Refusal(f"you edit TASK {seat.task} only; propose GLOBAL as a lesson")
        cited = edit.evidence if isinstance(edit, AddEntry | ModifyEntry) else []
        if others := sorted({e.task for e in cited} - {seat.task}):
            raise Refusal(f"cite episodes of {seat.task} only, not of {', '.join(others)}")

    def _revision(self, seat: Seat, revises: str | None) -> None:
        if seat.pass_ == 1:
            if revises is not None:
                raise Refusal("nothing is under revision in the first pass")
            return
        if revises is None:
            raise Refusal("in the revision pass every proposal revises one that was rejected")
        target = self.get(revises)
        if target.author != seat.label or target.status != "rejected" or target.pass_ >= seat.pass_:
            raise Refusal(f"{revises} is not one of your rejected proposals")
        if any(p.revises == revises for p in self.proposals):
            raise Refusal(f"{revises} is revised already")

    def _lesson(self, seat: Seat, lesson: Lesson) -> None:
        if not self.reflect.lessons:
            raise Refusal("this recipe has no GLOBAL part: stage no lessons")
        lessons = [p for p in self.proposals if p.author == seat.label and p.status == "lesson"]
        if len(lessons) >= self.reflect.lessons:
            raise Refusal(f"you have staged {len(lessons)} lessons, the most a task stages")
        keys = [e.key for e in lesson.evidence]
        if unknown := sorted({k.label() for k in keys if k not in self.episodes}):
            raise Refusal(f"no episode this round read is {', '.join(unknown)}")
        if others := sorted({k.task for k in keys} - {seat.task}):
            raise Refusal(f"cite episodes of {seat.task} only, not of {', '.join(others)}")

    def _dry_run(self, author: str, edit: Edit) -> EditRecord:
        """`edit` on a scratch editor with `author`'s live proposals applied first."""
        scratch = RsiEditor(
            self.doc,
            self.rules,
            self.context,
            self.round,
            self.episodes,
            self.spent,
            consolidating=self.consolidating,
        )
        for proposal in self.proposals:
            live = proposal.status in ("staged", "accepted")
            if proposal.author == author and live and not isinstance(proposal.edit, Lesson):
                scratch.edit(proposal.edit)
        return scratch.edit(edit)

    def _part(self, edit: Edit) -> TaskName | None:
        """The part an edit lands in (None: GLOBAL)."""
        if isinstance(edit, AddEntry):
            return edit.task
        if isinstance(edit, MergeEntries):
            if edit.into_global:
                return None
            return _part_of(edit.ids[0])
        return _part_of(edit.id)

    def _order(self, proposal: Staged) -> tuple[int, int, str, str, int]:
        edit = proposal.edit
        assert not isinstance(edit, Lesson)
        part = self._part(edit)
        number = int(proposal.id.rsplit("-", 1)[1])
        vote = int(isinstance(edit, VoteEntry))
        return (vote, int(part is not None), part or "", proposal.author, number)


def spent(proposals: Iterable[Staged]) -> dict[TaskName | None, int]:
    """The edits `proposals` (a reflection pass's) applied, by part (None: GLOBAL); votes cost
    none."""
    found: dict[TaskName | None, int] = {}
    for proposal in proposals:
        record = proposal.record
        if record is None or not record.accepted or record.op == "vote" or record.entry is None:
            continue
        part = _part_of(record.entry)
        found[part] = found.get(part, 0) + 1
    return found


def _part_of(entry_id: str) -> TaskName | None:
    if entry_id.startswith("G-"):
        return None
    return TaskName(entry_id.removeprefix("T-").rsplit("-", 1)[0])


def _lint(record: EditRecord) -> str:
    return "".join(f"; {i.rule} {i.match!r}: {i.message}" for i in record.findings if i.error)


def counts(proposals: Iterable[Staged]) -> dict[str, int]:
    """Proposals by status."""
    found: dict[str, int] = {}
    for proposal in proposals:
        found[proposal.status] = found.get(proposal.status, 0) + 1
    return dict(sorted(found.items()))
