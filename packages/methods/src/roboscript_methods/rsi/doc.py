"""The RSI document (docs/concepts/learning.md): an itemized playbook of experience a reflection
wrote from evaluation records, apart from the Task Memory, as an artifact of kind `rsi`.

Two parts: `GLOBAL`, experience across tasks (no task names, object instances or numbers: `lint`),
and one `TASK <name>` part per task. An entry is a `rule` (when it applies, what to do) or a `trap`
(a precursor to watch for and what to avoid; the only kind a task whose episodes all failed may
add). It has a stable id the code assigns (`G-07` in GLOBAL, `T-<task>-03` in a task's part; never
reused), `when` it applies (observable), what to `do` (a behavior, no numeric parameters), how to
`check` it worked (observable), its `evidence` (block, task, episode, turn or step of the records
it came from), its paired `helpful` and `harmful` counts (`counters`: the only counts that move its
status), the reflectors' votes (episodes that voted it helpful or harmful, information for later
reflectors), its `status` (`probation` when new, `active`, `retired`) and its provenance: the round
it was added, the rounds it was changed, and every change. A retired entry stays for the record; no
Actor reads it. The document holds no workflow and no parameters: those are the Task Memory's,
which wins where the two differ.

`render` is the whole document as Markdown for people and the reflector (the store's `.md`); the
JSON form is the model itself. `context` is what an evaluated Actor reads for one overlay layer
(`agent.overlay`): GLOBAL's or one task's readable entries (active or on probation), ordered by id,
with no evidence, counts, provenance or status; "" when there are none, so a document with no
readable entries leaves the Actor's prompt byte for byte as it was.
"""

import hashlib
import re
from typing import Annotated, ClassVar, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
)

from roboscript.contracts.artifact import Artifact
from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.ids import TaskName
from roboscript.contracts.tools import ToolModel

KIND: Final = "rsi"
GLOBAL: Final = "GLOBAL"

EntryId = Annotated[str, StringConstraints(pattern=r"^(?:G|T-[A-Za-z0-9_]+)-[0-9]{2,}$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LEAD: Final = re.compile(r"^(?:when|whenever|if|watch for)\b[\s:,]*", re.IGNORECASE)


def _situation(text: str) -> str:
    if found := LEAD.match(text):
        raise ValueError(
            f"`when` starts with {found.group().strip()!r}: write the situation itself; the "
            "document puts 'When:' or 'Watch for:' before it"
        )
    return text


def _clause(text: str) -> str:
    """A situation as it reads after "When:" or "Watch for:": no closing period, and a capital
    first letter lowered unless the word is an acronym (its second letter a capital too)."""
    text = text.rstrip(" .")
    if len(text) > 1 and text[0].isupper() and not text[1].isupper():
        text = text[0].lower() + text[1:]
    return text


Situation = Annotated[Text, AfterValidator(_situation)]
"""An edit's `when`: the situation alone (`RsiEntry.text` labels it)."""
Status = Literal["probation", "active", "retired"]
EntryKind = Literal["rule", "trap"]
READABLE: Final[frozenset[str]] = frozenset({"probation", "active"})
ID_NUMBER: Final = re.compile(r"-([0-9]+)$")

CONTEXT_NOTES: Final = (
    "Patches for errors found in earlier evaluations: apply them. A patch that starts "
    '"Instead of Task Memory step <step>" replaces what the Task Memory says for that step; '
    "elsewhere the Task Memory holds.",
    "When you apply an entry, write its id in your scratchpad.",
)
"""Above the first readable part an Actor reads: the entries are patches for errors found in
earlier evaluations; a correction of a Task Memory step wins for that step, the Task Memory
otherwise; name what you apply (the citations `counters` credits)."""


class EpisodeKey(ToolModel):
    """An evaluated episode: its seed block, its task, its official index in that block (the
    benchmark's numbering; supplied to the reflection as data) and,
    in a round that reflects on several arms (runs of the same block, e.g. by two models), its
    arm (None: the round's only one), so the same official episode can appear once per arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block: str = Field(min_length=1)
    task: TaskName
    episode: NonNegativeInt
    arm: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Only when the episode's key names an arm (`<arm>:block/...`); else null.",
    )

    def label(self) -> str:
        at = "" if self.arm is None else f"{self.arm}:"
        return f"{at}{self.block}/{self.task}/ep{self.episode:03d}"


class Evidence(ToolModel):
    """Where an entry comes from: an evaluated episode (block, task, official index, arm as in
    `EpisodeKey`) and the turn or env step in it (None: the episode as a whole)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block: str = Field(min_length=1)
    task: TaskName
    episode: NonNegativeInt
    turn: NonNegativeInt | None = None
    step: NonNegativeInt | None = None
    arm: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Only when the episode's key names an arm (`<arm>:block/...`); else null.",
    )

    @property
    def key(self) -> EpisodeKey:
        """The episode it points at."""
        return EpisodeKey(block=self.block, task=self.task, episode=self.episode, arm=self.arm)

    def label(self) -> str:
        at = "" if self.turn is None else f" turn {self.turn}"
        at += "" if self.step is None else f" step {self.step}"
        return f"{self.key.label()}{at}"


ChangeOp = Literal[
    "add", "modify", "merge", "merged_into", "retire", "vote", "count", "promote", "auto_retire"
]


class Change(BaseModel):
    """One change of an entry, in the round it happened: `op` and what it did (`note`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round: PositiveInt
    op: ChangeOp
    note: str = ""


class RsiEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: EntryId
    kind: EntryKind = "rule"
    when: Text
    do: Text
    check: Text
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    corrects: Text | None = None
    """A rule of a task's part that corrects one step of that task's Task Memory: the step's
    name; the Actor reads the entry as replacing what the Task Memory says for that step."""
    helpful: NonNegativeInt = 0
    """Paired counts (`counters`): the only counts retirement and promotion read."""
    harmful: NonNegativeInt = 0
    votes_helpful: tuple[EpisodeKey, ...] = ()
    """The reflectors' votes, one per episode (helpful or harmful, never both): information for
    later reflectors, never read by retirement or promotion."""
    votes_harmful: tuple[EpisodeKey, ...] = ()
    status: Status = "probation"
    added: PositiveInt
    modified: tuple[PositiveInt, ...] = ()
    uncited: NonNegativeInt = 0
    """Consecutive evaluated rounds no Actor cited it (`counters`)."""
    history: tuple[Change, ...] = ()

    @property
    def text(self) -> str:
        """What the Actor reads of it: a rule as when, do and check (a correction first names
        the Task Memory step it replaces); a trap as the precursor to watch for, what to do
        about it, and the check."""
        when = _clause(self.when)
        if self.kind == "trap":
            return f"Watch for: {when} \u2192 {self.do} Check: {self.check}"
        instead = f"Instead of Task Memory step {self.corrects}: " if self.corrects else ""
        return f"{instead}When: {when} \u2192 {self.do} Check: {self.check}"

    @property
    def readable(self) -> bool:
        return self.status in READABLE

    @property
    def voted(self) -> frozenset[EpisodeKey]:
        return frozenset((*self.votes_helpful, *self.votes_harmful))


class RsiDoc(Artifact):
    """`global_entries`: GLOBAL; `task_entries`: each task's part, by task."""

    kind: ClassVar[str] = KIND

    global_entries: tuple[RsiEntry, ...] = ()
    task_entries: dict[TaskName, tuple[RsiEntry, ...]] = Field(default_factory=dict)

    def section(self, task: TaskName | None) -> tuple[RsiEntry, ...]:
        """GLOBAL (`task` None) or a task's part."""
        return self.global_entries if task is None else self.task_entries.get(task, ())

    def entries(self) -> list[tuple[TaskName | None, RsiEntry]]:
        """Every entry with its part (None: GLOBAL)."""
        found: list[tuple[TaskName | None, RsiEntry]] = [(None, e) for e in self.global_entries]
        for task in sorted(self.task_entries):
            found += [(task, entry) for entry in self.task_entries[task]]
        return found

    def find(self, entry_id: str) -> tuple[TaskName | None, RsiEntry]:
        for task, entry in self.entries():
            if entry.id == entry_id:
                return task, entry
        raise KeyError(entry_id)

    def content_hash(self) -> str:
        """sha256 of the canonical JSON without default-valued fields (schema-stable)."""
        data = self.model_dump(mode="json", exclude_defaults=True)
        return hashlib.sha256(canonical_json(data).encode()).hexdigest()

    def part(self, task: TaskName | None) -> str:
        """GLOBAL's or `task`'s readable entries as an Actor reads them, under the part's heading,
        without the notes (what the token budget counts); "" with none."""
        readable = sorted((e for e in self.section(task) if e.readable), key=_order)
        if not readable:
            return ""
        lines = [f"# EXPERIENCE: {heading(task)}"]
        lines += [f"- [{entry.id}] {entry.text}" for entry in readable]
        return "\n".join(lines)

    def budget_text(self, task: TaskName | None) -> str:
        """What a part's token budget counts: GLOBAL's readable entries with the notes, or one
        task's part without them."""
        text = self.part(task)
        if task is not None or not text:
            return text
        return "\n".join([*CONTEXT_NOTES, text])

    def context(self, task: TaskName | None) -> str:
        """The overlay layer's text (`agent.overlay.ContextArtifact`): GLOBAL (`task` None) or
        `task`'s part, the notes above the first readable part (GLOBAL, or the task's part when
        GLOBAL has none); "" with no readable entries."""
        text = self.part(task)
        if not text or (task is not None and self.part(None)):
            return text
        heading_line, _, rest = text.partition("\n")
        return "\n".join([heading_line, *CONTEXT_NOTES, rest])


def heading(task: TaskName | None) -> str:
    return GLOBAL if task is None else f"TASK {task}"


def entry_number(entry_id: str) -> int:
    match = ID_NUMBER.search(entry_id)
    assert match is not None  # EntryId's pattern
    return int(match.group(1))


def _order(entry: RsiEntry) -> int:
    return entry_number(entry.id)


def next_id(doc: RsiDoc, task: TaskName | None) -> str:
    """The next id of GLOBAL (`G-NN`) or `task`'s part (`T-<task>-NN`), retired ids included."""
    number = max((entry_number(e.id) for e in doc.section(task)), default=0) + 1
    return f"G-{number:02d}" if task is None else f"T-{task}-{number:02d}"


def render(doc: RsiDoc) -> str:
    """The whole document as Markdown: each part under its heading, every entry with its status,
    counts, provenance and evidence."""
    lines: list[str] = []
    for task in [None, *sorted(doc.task_entries)]:
        if lines:
            lines.append("")
        lines.append(f"# EXPERIENCE: {heading(task)}")
        for entry in sorted(doc.section(task), key=_order):
            changed = "".join(f", changed r{r}" for r in entry.modified)
            kind = "" if entry.kind == "rule" else f"{entry.kind}; "
            lines.append(
                f"- [{entry.id}] ({kind}{entry.status}; helpful {entry.helpful}, harmful "
                f"{entry.harmful}; votes +{len(entry.votes_helpful)} -{len(entry.votes_harmful)}; "
                f"added r{entry.added}{changed}) {entry.text}"
            )
            lines.append(f"  evidence: {'; '.join(e.label() for e in entry.evidence)}")
    return "\n".join(lines) + "\n"
