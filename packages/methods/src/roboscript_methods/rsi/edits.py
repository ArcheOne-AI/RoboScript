"""Typed edits of an RSI document (docs/concepts/learning.md): add, modify, merge,
retire and vote, applied by code, never a rewrite of the whole document (whole rewrites let
content collapse, ACE 2510.04618).

An `RsiEditor` holds the document of one reflection round (`round`, from 1) and the recipe's rules
(`RsiRules`). Every edit is checked, and a refused one changes nothing, costs nothing and says why:

- a task's part names a task of the run (`LintContext.tasks`);
- new evidence points at an episode the round's reflection read (`episodes`: the valid keys,
  block, task and official index, given as data);
- the entry's fields within their caps (`caps`); an add carries at most `caps.evidence` refs, and
  a modify or merge keeps the newest `caps.evidence` of the union;
- no duplicate: the entry an add, modify or merge writes is refused when both its `when` and its
  `do` are near those of another readable entry of its part, or of GLOBAL for a task's entry
  (word-level difflib ratio at least `dedup_threshold`, each field on its own; text with no word
  is never a duplicate), with that entry's id: modify or merge it instead;
- an add to GLOBAL, or a merge into it (`into_global`: the promotion of task entries), needs
  evidence from `global_min_tasks` distinct tasks, counting only valid keys;
- the round's edit budget of the part the edit lands in (`RsiRules.budget`: GLOBAL as a whole,
  each task's part on its own, decaying by round); votes cost none;
- the part an Actor reads stays within its token budget (`global_tokens` with the notes,
  `task_tokens`), or at least does not grow while over it;
- the lint (`lint.lint`): an edit leaving a new error is refused; warnings come with it;
- a merge takes entries of one kind (`rule` or `trap`), and the merged entry keeps it; a modify
  never changes it;
- `corrects` (the Task Memory step a rule replaces) only on a rule of a task's part, naming a
  workflow step of that task's Task Memory (`LintContext.memories`).

A vote records a valid episode as helpful or harmful for an entry (`votes_helpful`,
`votes_harmful`), one per episode and entry, carried through merges; votes are information for
later reflectors, apart from the paired counts (`helpful`, `harmful`) that `counters` keeps and
that alone retire or promote. New and merged entries start on probation. Each attempt is an
`EditRecord` (the round's `edits.jsonl`).

A consolidating editor (`consolidating`, the round's last step) takes only merges within a part,
retirements, and modifies of entries that fail the current lint (a vocabulary can grow after an
entry was written), and charges no budget for them; every entry it writes must pass the lint. Its
merge keeps the oldest source's id (and its provenance, with the merge in its history) in that
entry's place, so no other entry's id moves.
"""

import difflib
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.ids import TaskName
from roboscript.contracts.tools import ToolModel
from roboscript_methods.rsi.doc import (
    GLOBAL,
    Change,
    EntryId,
    EntryKind,
    EpisodeKey,
    Evidence,
    RsiDoc,
    RsiEntry,
    Situation,
    Text,
    entry_number,
    heading,
    next_id,
)
from roboscript_methods.rsi.lint import LintContext, LintItem, lint, lint_entry, memory_stages
from roboscript_methods.rsi.recipe import RsiRules


class AddEntry(ToolModel):
    """A new entry in GLOBAL (`task` null) or in a task's part."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["add"] = "add"
    task: TaskName | None = None
    kind: EntryKind = "rule"
    when: Situation
    do: Text
    check: Text
    evidence: list[Evidence] = Field(min_length=1)
    corrects: Text | None = Field(
        default=None,
        description="A rule of a task's part only: the name of the Task Memory workflow step it "
        "corrects (the Actor then follows the entry instead of that step); null otherwise.",
    )


class ModifyEntry(ToolModel):
    """New wording of an entry that is not retired, and more evidence for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["modify"] = "modify"
    id: EntryId
    when: Situation | None = None
    do: Text | None = None
    check: Text | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    corrects: Text | None = Field(
        default=None,
        description="Set or change the Task Memory step a rule corrects; null leaves it as is.",
    )


class MergeEntries(ToolModel):
    """Entries into one new entry with all their evidence and counts; they retire. Of one part,
    or, `into_global`, of any parts into GLOBAL (promotion)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["merge"] = "merge"
    ids: list[EntryId] = Field(min_length=2)
    into_global: bool = False
    when: Situation
    do: Text
    check: Text
    corrects: Text | None = Field(
        default=None,
        description="A rule of a task's part only: the name of the Task Memory workflow step it "
        "corrects (the Actor then follows the entry instead of that step); null otherwise.",
    )


class RetireEntry(ToolModel):
    """An entry no Actor should read any longer (no gain measured, or wrong)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["retire"] = "retire"
    id: EntryId
    reason: Text


class VoteEntry(ToolModel):
    """An episode the round read where an entry helped or harmed (`vote`): one vote per episode
    and entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["vote"] = "vote"
    id: EntryId
    vote: Literal["helpful", "harmful"]
    episode: EpisodeKey


Edit = AddEntry | ModifyEntry | MergeEntries | RetireEntry | VoteEntry
RsiEdit = Annotated[Edit, Field(discriminator="op")]


class EditRecord(BaseModel):
    """One edit attempt: its round, the edit as asked, the part it landed in (heading), whether it
    was accepted, the entry it wrote or touched, why not, the lint's findings, and the edits left
    in that part this round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round: int
    op: str
    edit: dict[str, Any]
    part: str | None = None
    accepted: bool
    entry: str | None = None
    reason: str = ""
    findings: list[LintItem] = []
    left: int | None = None


class Refused(ValueError):
    """An edit that does not apply; its message is the reason."""


TASK_NAME: Final = re.compile(r"^[A-Za-z0-9_]+$")
WORD: Final = re.compile(r"\w+")


class RsiEditor:
    """The document one reflection round edits (`doc`, as edited so far), under `rules`, with what
    the lint knows of the run (`context`) and the episodes the round's reflection read
    (`episodes`, the only keys new evidence and votes may name). One editor serves every
    reflector of the round, so the round's budgets hold across them; `spent`, the edits an earlier
    reflection pass of the same round applied by part, counts against them."""

    def __init__(
        self,
        doc: RsiDoc,
        rules: RsiRules,
        context: LintContext,
        round: int,
        episodes: Collection[EpisodeKey],
        spent: Mapping[TaskName | None, int] | None = None,
        *,
        consolidating: bool = False,
    ) -> None:
        if round < 1:
            raise ValueError(f"rounds count from 1, not {round}")
        self.doc = doc
        self.rules = rules
        self.context = context
        self.round = round
        self.episodes = frozenset(episodes)
        # a second reflection pass over the same round starts with what the first one spent
        self.used: Counter[TaskName | None] = Counter(spent or {})
        self.consolidating = consolidating
        self.records: list[EditRecord] = []

    def left(self, task: TaskName | None) -> int:
        """The edits GLOBAL (`task` None) or `task`'s part still accepts this round."""
        return self.rules.budget(task is None, self.round) - self.used[task]

    def budgets(self) -> str:
        """The edits left this round, as a reflector reads them."""
        each = self.rules.budget(False, self.round)
        touched = "".join(
            f"; TASK {task}: {self.left(task)}"
            for task in sorted(t for t in self.used if t is not None)
        )
        return (
            f"Round {self.round} edits left: GLOBAL {self.left(None)}; each task's part {each}"
            f"{touched}. Votes cost none."
        )

    def edit(self, op: Edit) -> EditRecord:
        """Apply `op` if it passes every check (module docstring)."""
        try:
            if self.consolidating and not _consolidates(op):
                raise Refused(
                    "consolidation merges entries within one part, retires one, or rewords one "
                    "that fails the lint"
                )
            doc, entry, task = self._apply(op)
        except Refused as error:
            return self._record(op, str(error))
        charged = op.op != "vote" and not self.consolidating
        if op.op != "vote":
            if charged and self.left(task) <= 0:
                budget = self.rules.budget(task is None, self.round)
                why = f"{heading(task)}'s edit budget for round {self.round} ({budget}) is spent"
                return self._record(op, why, task, entry)
            limit = self.rules.global_tokens if task is None else self.rules.task_tokens
            tokens = self.rules.tokens(doc.budget_text(task))
            if tokens > limit and tokens > self.rules.tokens(self.doc.budget_text(task)):
                why = f"{heading(task)} would read about {tokens} tokens, over its {limit}"
                return self._record(op, why, task, entry)
        if self.consolidating and op.op in ("merge", "modify"):
            written = doc.find(entry)[1]
            if errors := [i for i in lint_entry(written, task, self.context) if i.error]:
                return self._record(op, "the entry would still fail the lint", task, entry, errors)
        before = {(i.entry, i.rule, i.match) for i in lint(self.doc, self.context)}
        new = [i for i in lint(doc, self.context) if (i.entry, i.rule, i.match) not in before]
        if errors := [i for i in new if i.error]:
            return self._record(op, "the lint refuses it", task, entry, errors)
        self.doc = doc
        if charged:
            self.used[task] += 1
        return self._record(op, "", task, entry, new, accepted=True)

    def _record(
        self,
        op: Edit,
        reason: str,
        task: TaskName | None = None,
        entry: str | None = None,
        findings: list[LintItem] | None = None,
        *,
        accepted: bool = False,
    ) -> EditRecord:
        """`entry` None: refused before the edit found its part."""
        landed = entry is not None
        record = EditRecord(
            round=self.round,
            op=op.op,
            edit=op.model_dump(mode="json"),
            part=heading(task) if landed else None,
            accepted=accepted,
            entry=entry,
            reason=reason,
            findings=findings or [],
            left=self.left(task) if landed else None,
        )
        self.records.append(record)
        return record

    def _apply(self, op: Edit) -> tuple[RsiDoc, str, TaskName | None]:
        """The document after `op`, the entry it wrote or touched and that entry's part; `Refused`
        when it does not apply."""
        doc, r = self.doc, self.round
        if isinstance(op, AddEntry):
            if op.task is not None:
                self._known_task(op.task)
            self._valid([e.key for e in op.evidence])
            if len(op.evidence) > self.rules.caps.evidence:
                cap = self.rules.caps.evidence
                raise Refused(f"an entry keeps at most {cap} evidence refs, not {len(op.evidence)}")
            if op.task is None:
                self._spans_tasks(op.evidence)
            self._corrects(op.task, op.kind, op.corrects)
            entry = self._entry(
                id=next_id(doc, op.task),
                kind=op.kind,
                when=op.when,
                do=op.do,
                check=op.check,
                evidence=tuple(dict.fromkeys(op.evidence)),
                corrects=op.corrects,
                added=r,
                history=(Change(round=r, op="add"),),
            )
            self._no_duplicate(op.task, entry, ())
            return _put(doc, op.task, [*doc.section(op.task), entry]), entry.id, op.task
        if isinstance(op, MergeEntries):
            return self._merge(op)
        task, entry = _readable(doc, op.id)
        if isinstance(op, ModifyEntry):
            failing = [i for i in lint_entry(entry, task, self.context) if i.error]
            if self.consolidating and not failing:
                raise Refused(f"{entry.id} passes the lint: consolidation rewords no other entry")
            names = ("when", "do", "check", "corrects")
            fields = {k: v for k in names if (v := getattr(op, k)) is not None}
            if op.corrects is not None:
                self._corrects(task, entry.kind, op.corrects)
            if not fields and not op.evidence:
                raise Refused("a modify changes a field or adds evidence")
            self._valid([e.key for e in op.evidence])
            note = ", ".join([*fields, *(["evidence"] if op.evidence else [])])
            changed = self._entry(
                **{
                    **entry.model_dump(),
                    **fields,
                    "evidence": self._keep((*entry.evidence, *op.evidence)),
                    "modified": tuple(dict.fromkeys((*entry.modified, r))),
                    "history": (*entry.history, Change(round=r, op="modify", note=note)),
                }
            )
            self._no_duplicate(task, changed, (entry.id,))
        elif isinstance(op, RetireEntry):
            history = (*entry.history, Change(round=r, op="retire", note=op.reason))
            changed = entry.model_copy(update={"status": "retired", "history": history})
        else:
            self._valid([op.episode])
            if op.episode in entry.voted:
                raise Refused(f"episode {op.episode.label()} has voted on {entry.id} already")
            field = f"votes_{op.vote}"
            history = (
                *entry.history,
                Change(round=r, op="vote", note=f"{op.vote} {op.episode.label()}"),
            )
            votes = (*getattr(entry, field), op.episode)
            changed = entry.model_copy(update={field: votes, "history": history})
        section = [changed if e.id == entry.id else e for e in doc.section(task)]
        return _put(doc, task, section), entry.id, task

    def _merge(self, op: MergeEntries) -> tuple[RsiDoc, str, TaskName | None]:
        doc, r = self.doc, self.round
        if len(set(op.ids)) != len(op.ids):
            raise Refused("a merge takes distinct entries")
        found = [_readable(doc, entry_id) for entry_id in op.ids]
        parts = {task for task, _ in found}
        if not op.into_global and len(parts) != 1:
            raise Refused("a merge takes entries of one part, or goes into GLOBAL (into_global)")
        target: TaskName | None = None if op.into_global else next(iter(parts))
        if target is None and (fixes := [e.id for _, e in found if e.corrects]):
            raise Refused(
                f"{', '.join(fixes)} correct a Task Memory step: a correction stays in its task's "
                "part, never merged into GLOBAL"
            )
        kinds = {entry.kind for _, entry in found}
        if len(kinds) != 1:
            raise Refused("a merge takes entries of one kind (rules or traps)")
        evidence = self._keep(tuple(e for _, entry in found for e in entry.evidence))
        if target is None:
            self._spans_tasks(evidence)
        self._corrects(target, next(iter(kinds)), op.corrects)
        helpful = _union(entry.votes_helpful for _, entry in found)
        harmful = _union(entry.votes_harmful for _, entry in found)
        # an episode voted both ways on the sources votes on the merged entry neither way
        both = set(helpful) & set(harmful)
        note = "from " + ", ".join(op.ids)
        if both:
            note += "; votes dropped both ways: " + ", ".join(sorted(k.label() for k in both))
        # consolidating, the oldest source carries the merge under its own id, in its place
        kept = min((e for _, e in found), key=lambda e: entry_number(e.id))
        provenance: dict[str, Any] = {"id": next_id(doc, target), "added": r}
        history = (Change(round=r, op="merge", note=note),)
        if self.consolidating:
            provenance = {"id": kept.id, "added": kept.added, "modified": (*kept.modified, r)}
            history = (*kept.history, *history)
        merged = self._entry(
            **provenance,
            kind=next(iter(kinds)),
            when=op.when,
            do=op.do,
            check=op.check,
            evidence=evidence,
            corrects=op.corrects,
            helpful=sum(entry.helpful for _, entry in found),
            harmful=sum(entry.harmful for _, entry in found),
            votes_helpful=tuple(k for k in helpful if k not in both),
            votes_harmful=tuple(k for k in harmful if k not in both),
            history=history,
        )
        self._no_duplicate(target, merged, tuple(op.ids))
        if self.consolidating:
            section = [
                merged
                if entry.id == kept.id
                else entry.model_copy(
                    update={
                        "status": "retired",
                        "history": (
                            *entry.history,
                            Change(round=r, op="merged_into", note=merged.id),
                        ),
                    }
                )
                if entry.id in op.ids
                else entry
                for entry in doc.section(target)
            ]
            return _put(doc, target, section), merged.id, target
        for part in sorted(parts, key=lambda t: "" if t is None else t):
            retired = [
                entry.model_copy(
                    update={
                        "status": "retired",
                        "history": (
                            *entry.history,
                            Change(round=r, op="merged_into", note=merged.id),
                        ),
                    }
                )
                if entry.id in op.ids
                else entry
                for entry in doc.section(part)
            ]
            doc = _put(doc, part, retired)
        return _put(doc, target, [*doc.section(target), merged]), merged.id, target

    def _entry(self, **fields: Any) -> RsiEntry:
        caps = self.rules.caps
        for name, cap in (("when", caps.when), ("do", caps.do), ("check", caps.check)):
            if len(fields[name]) > cap:
                raise Refused(f"`{name}` holds {len(fields[name])} characters, over its cap {cap}")
        return RsiEntry(**fields)

    def _corrects(self, task: TaskName | None, kind: str, step: str | None) -> None:
        """A correction is a rule of a task's part naming a step of that task's Task Memory."""
        if step is None:
            return
        if task is None:
            raise Refused(
                "GLOBAL corrects no Task Memory step: a correction goes in its task's part"
            )
        if kind != "rule":
            raise Refused("only a rule corrects a Task Memory step")
        stages = memory_stages(self.context, task)
        if step not in stages:
            known = ", ".join(stages) or "none known"
            raise Refused(f"{step!r} is no step of {task}'s Task Memory (its steps: {known})")

    def _known_task(self, task: str) -> None:
        if not TASK_NAME.match(task) or task not in self.context.tasks:
            raise Refused(f"{task!r} is no task of this run")

    def _valid(self, keys: Iterable[EpisodeKey]) -> None:
        if unknown := sorted({k.label() for k in keys if k not in self.episodes}):
            raise Refused(f"no episode this round read is {', '.join(unknown)}")

    def _keep(self, evidence: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
        """The newest `caps.evidence` distinct refs, oldest first."""
        return tuple(dict.fromkeys(evidence))[-self.rules.caps.evidence :]

    def _spans_tasks(self, evidence: Iterable[Evidence]) -> None:
        tasks = {e.task for e in evidence if e.key in self.episodes}
        if len(tasks) < self.rules.global_min_tasks:
            raise Refused(
                f"{GLOBAL} needs evidence from {self.rules.global_min_tasks} distinct tasks among "
                f"this round's episodes, this has {len(tasks)}"
            )

    def _no_duplicate(self, task: TaskName | None, entry: RsiEntry, skip: tuple[str, ...]) -> None:
        """`entry` (to land in `task`'s part) near no readable entry of that part, nor of GLOBAL
        for a task's entry, but those in `skip` (what it replaces)."""
        others = [*self.doc.section(task), *(self.doc.global_entries if task is not None else ())]
        for other in others:
            if not other.readable or other.id in skip:
                continue
            near = [_similar(a, b) for a, b in ((entry.when, other.when), (entry.do, other.do))]
            if all(ratio >= self.rules.dedup_threshold for ratio in near):
                raise Refused(
                    f"near entry {other.id} (when {near[0]:.2f}, do {near[1]:.2f}): modify or "
                    "merge it instead"
                )


def _similar(a: str, b: str) -> float:
    """Word-level difflib ratio of `a` and `b` (casefolded, Unicode words); 0 when either holds no
    word, so text the normalization empties is never a duplicate."""
    words_a, words_b = WORD.findall(a.casefold()), WORD.findall(b.casefold())
    if not words_a or not words_b:
        return 0.0
    return difflib.SequenceMatcher(None, words_a, words_b, autojunk=False).ratio()


def _consolidates(op: Edit) -> bool:
    merge = isinstance(op, MergeEntries) and not op.into_global
    return merge or isinstance(op, RetireEntry | ModifyEntry)


def _union(groups: Iterable[tuple[EpisodeKey, ...]]) -> tuple[EpisodeKey, ...]:
    return tuple(dict.fromkeys(key for group in groups for key in group))


def _readable(doc: RsiDoc, entry_id: str) -> tuple[TaskName | None, RsiEntry]:
    try:
        task, entry = doc.find(entry_id)
    except KeyError:
        raise Refused(f"there is no entry {entry_id}") from None
    if not entry.readable:
        raise Refused(f"entry {entry_id} is retired")
    return task, entry


def _put(doc: RsiDoc, task: TaskName | None, entries: list[RsiEntry]) -> RsiDoc:
    if task is None:
        return doc.model_copy(update={"global_entries": tuple(entries)})
    return doc.model_copy(update={"task_entries": {**doc.task_entries, task: tuple(entries)}})
