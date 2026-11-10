"""The Memory library (docs/concepts/learning.md): every task's
frozen Task Memory as one library that a P3 Actor browses and draws from itself, instead of
being handed the Memory of its task id (an id binding turns semantics into
an id; an open task recombines skills learnt in other contexts).

- An entry's id is opaque and stable: `mem-` and the first 12 hex digits of its version's content
  hash (`entry_id`), never the task name; its card (`make_card`) is a deterministic extract of the
  Memory (no model call): the task it came from, its TASK statement as the goal, its OBJECTS roles,
  the tools its workflow calls, its stages and its done criteria. Same Memory, same id and card.
- `LibraryIndex` (`library.json`, written by `python -m roboscript.memory.library`) lists the
  frozen versions and the store each lives in, so that per-task stores (one per pipeline unit)
  form one library; without one, the host's manifest and store are the library.
- `LibraryConfig.exclude` / `exclude_own` take entries out for generalization tests (the task's
  own Memory left out: leave-one-out); `EpisodeLibrary` is one episode's view, which the
  `library` toolset reads and the host binds the Memories in use from.
- Long text goes out in parts that fit the Hub's default Actor text cap (`PART_CHARS`), so that
  reading several Memories respects the context budget the Actor already has.
"""

import argparse
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from roboscript.contracts.ids import MemoryEntryId
from roboscript.contracts.tools import ContentPart
from roboscript.memory.config import StoreSpec
from roboscript.memory.doc import MemoryDoc, TaskLabel, view
from roboscript.memory.manifest import ManifestEntry, read_manifest
from roboscript.memory.objects import object_roles
from roboscript.memory.params import DoneSpecParam
from roboscript.memory.store import Hash, Version
from roboscript.records import write_atomic

PREFIX: Final = "mem-"
PART_CHARS: Final = 3600
"""Characters of one page of cards or one part of a Memory: under the Hub's default text cap of
an Actor's result (`hub.config.RenderConfig.max_text_chars`, 4000), so nothing is cut."""
_ROLE: Final = 50  # characters of a role's description on a card
_GOAL: Final = 120  # of the goal
_DONE: Final = 100  # of the final stage's done line
_CALL: Final = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_WORD: Final = re.compile(r"[a-z0-9]+")


def entry_id(content_hash: str) -> MemoryEntryId:
    """A library entry's id: `mem-` and 12 hex digits of its Memory version's content hash."""
    return MemoryEntryId(PREFIX + content_hash[:12])


class Card(BaseModel):
    """What a library entry is about, extracted from its Memory: the task it came from, its goal
    (the Memory's INSTRUCTION), its object roles (role: description, shortened), the tools its
    workflow calls, its stages by name and its done criteria (the final stage's done line,
    shortened, and each `done_spec` as metric and roles). `brief()`, one line for the index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: MemoryEntryId
    task: TaskLabel
    goal: str
    roles: list[str]
    tools: list[str]
    stages: list[str]
    done: list[str]

    def render(self) -> str:
        lines = [f"[{self.id}] from task {self.task}", f"  goal: {self.goal or '(none)'}"]
        lines.append(f"  objects: {'; '.join(self.roles) or '(none)'}")
        lines.append(f"  tools: {', '.join(self.tools) or '(none)'}")
        lines.append(f"  stages: {' | '.join(self.stages) or '(none)'}")
        if self.done:
            lines.append(f"  done: {'; '.join(self.done)}")
        return "\n".join(lines)

    def brief(self) -> str:
        stages = ", ".join(self.stages)
        return f"[{self.id}] {self.task}: {self.goal or '(no goal)'} | stages: {stages}"


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def workflow_tools(doc: MemoryDoc) -> list[str]:
    """The tools a workflow's calls name, in order, each once: the leading name of every
    `;`-separated call of every stage (a predicate nested in a call's arguments is not a tool)."""
    found: dict[str, None] = {}
    for stage in doc.workflow:
        for segment in stage.call.split(";"):
            if (match := _CALL.match(segment)) is not None:
                found[match[1]] = None
    return list(found)


def make_card(id_: MemoryEntryId, task: str, doc: MemoryDoc) -> Card:
    """`doc`'s card: deterministic, so the same Memory always has the same card."""
    try:
        roles = [f"{r}: {_short(d, _ROLE)}" for r, d in object_roles(doc).items()]
    except ValueError:  # a malformed OBJECTS line: the card shows the section's first words
        roles = [_short(doc.objects, _GOAL)] if doc.objects.strip() else []
    final = doc.workflow[-1].done if doc.workflow else ""
    done = [_short(final, _DONE)] if final.strip() else []
    for param in doc.params.values():
        if isinstance(param, DoneSpecParam):
            spec = param.value
            done.append(f"{spec.metric}({spec.a if spec.b is None else f'{spec.a}, {spec.b}'})")
    return Card(
        id=id_,
        task=task,
        goal=_short(doc.statement, _GOAL),
        roles=roles,
        tools=workflow_tools(doc),
        stages=[stage.name for stage in doc.workflow],
        done=done,
    )


@dataclass(frozen=True)
class LibraryEntry:
    """One frozen Task Memory of the library: its id and card, the task it came from and the
    version as an evaluated Actor reads it (the method's rewrite of a frozen Memory applied)."""

    id: MemoryEntryId
    task: TaskLabel
    version: Version
    card: Card

    def record(self) -> dict[str, Any]:
        meta = self.version.meta
        return {"id": self.id, "task": self.task, "version": meta.version, "hash": meta.hash}


def library_entries(versions: Iterable[tuple[str, Version, Version]]) -> list[LibraryEntry]:
    """The entries of (task, stored version, version as the Actor reads it), in order: ids from
    the stored version's hash. ValueError for two entries of one id or one task."""
    entries: list[LibraryEntry] = []
    for task, stored, read in versions:
        id_ = entry_id(stored.meta.hash)
        entries.append(LibraryEntry(id_, task, read, make_card(id_, task, read.doc)))
    ids: list[str] = [e.id for e in entries]
    if twice := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"library entries with one id twice: {twice}")
    tasks = [e.task for e in entries]
    if twice := sorted({t for t in tasks if tasks.count(t) > 1}):
        raise ValueError(f"library with two Memories of one task: {twice}")
    return entries


def memory_text(entry: LibraryEntry) -> str:
    """An entry's Memory as the Actor reads a Task Memory (`view(doc, detail=False)`), titled by
    its id and source task."""
    doc = entry.version.doc
    text = view(doc, detail=False)
    head = f"# TASK MEMORY: {doc.task}"
    assert text.startswith(head)
    return f"# TASK MEMORY {entry.id} (from task {entry.task})" + text[len(head) :]


def split_parts(text: str, size: int = PART_CHARS) -> list[str]:
    """`text` in parts of at most `size` characters, cut at section (`## `) boundaries, else at
    line ends, else hard."""
    parts: list[str] = []
    current = ""
    for block in re.split(r"(?m)(?=^## )", text):
        while len(block) > size:  # a section longer than a part: at line ends
            cut = block.rfind("\n", 0, size)
            cut = cut + 1 if cut > 0 else size
            if current:
                parts.append(current)
                current = ""
            parts.append(block[:cut])
            block = block[cut:]
        if len(current) + len(block) > size:
            parts.append(current)
            current = ""
        current += block
    if current.strip():
        parts.append(current)
    return [p.rstrip() + "\n" for p in parts if p.strip()]


def card_pages(
    entries: Sequence[LibraryEntry], size: int = PART_CHARS, *, brief: bool = False
) -> list[str]:
    """The entries' cards (`brief`: one line each) as pages of at most `size` characters, a card
    never split."""
    pages: list[str] = []
    current: list[str] = []
    for entry in entries:
        card = entry.card.brief() if brief else entry.card.render()
        if current and len("\n\n".join([*current, card])) > size:
            pages.append("\n\n".join(current))
            current = []
        current.append(card)
    if current:
        pages.append("\n\n".join(current))
    return pages


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower().replace("_", " ")))


def search(entries: Sequence[LibraryEntry], query: str, k: int) -> list[LibraryEntry]:
    """The `k` entries that match `query` best: by the number of the query's words their card has
    (the goal, roles and stages counting twice), then the number their whole Memory has, then
    library order; entries matching none of its words are left out. Deterministic."""
    wanted = _words(query)
    scored = []
    for order, entry in enumerate(entries):
        card = entry.card
        key = _words(" ".join([card.task, card.goal, *card.roles, *card.stages]))
        rest = _words(card.render()) | _words(view(entry.version.doc, detail=False))
        score = (2 * len(wanted & key) + len(wanted & rest), -order)
        if score[0] > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored[:k]]


_ROLE_LINE: Final = re.compile(r"^- ([A-Za-z0-9_]+):")


def combine(task: str, docs: Sequence[MemoryDoc]) -> MemoryDoc:
    """The Memories in use as one, for what binds them (their specs and numbers, roles' arm
    sides, done criteria): parameters and OBJECTS roles of all, a role or parameter in two the
    later one's; workflows one after another, a stage name already taken suffixed `_<n>`. Text
    is not merged: the Actor reads each Memory itself."""
    params: dict[str, Any] = {}
    roles: dict[str, list[str]] = {}
    workflow = []
    names: set[str] = set()
    for doc in docs:
        params.update(doc.params)
        current: list[str] | None = None
        for line in doc.objects.splitlines():
            if (match := _ROLE_LINE.match(line)) is not None:
                current = roles[match[1]] = [line]
            elif current is not None and line.strip():
                current.append(line)
        for stage in doc.workflow:
            name, n = stage.name, 1
            while name in names:
                n += 1
                name = f"{stage.name}_{n}"
            names.add(name)
            renamed = stage if name == stage.name else stage.model_copy(update={"name": name})
            workflow.append(renamed)
    objects = "\n".join(line for block in roles.values() for line in block)
    return MemoryDoc(task=task, objects=objects, workflow=workflow, params=params)


def role_clashes(entries: Sequence["LibraryEntry"]) -> list[str]:
    """The OBJECTS roles that two of `entries` name for other objects (one name, two
    descriptions), each as `role (id: description; id: description)`: binding both would bind
    one of them to the other's object. A role two name alike is one object, shared. ValueError
    for a malformed OBJECTS line."""
    seen: dict[str, tuple[str, str]] = {}
    found = []
    for entry in entries:
        for role, description in object_roles(entry.version.doc).items():
            if role not in seen:
                seen[role] = (entry.id, description)
            elif seen[role][1] != description:
                first, was = seen[role]
                found.append(f"{role} ({first}: {was}; {entry.id}: {description})")
    return found


class LibraryConfig(BaseModel):
    """P3's library: `index`, a `LibraryIndex` file (None: the host's manifest and store are the
    library); `exclude`, tasks whose Memories it leaves out; `exclude_own`, each episode's
    library leaves out the Memory of the episode's own task (leave-one-out). The default is the
    full library, the task's own Memory included."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: Path | None = None
    exclude: list[TaskLabel] = []
    exclude_own: bool = False


MemoryAccess = Literal["library", "bound"]
"""How a P3 Actor gets Task Memory: `library`, it browses every task's frozen Memory and uses
the ones it chooses (`memory.library`); `bound`, the Memory of its task id is in its prompt."""


class IndexEntry(BaseModel):
    """A frozen version of the library and the Memory store that holds it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskLabel
    version: int = Field(ge=0)
    hash: Hash
    store: StoreSpec

    @property
    def manifest_entry(self) -> ManifestEntry:
        return ManifestEntry(version=self.version, hash=self.hash)


class LibraryIndex(BaseModel):
    """`library.json`: the library's frozen versions, each with its store, in library order."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.memory.library/1"] = Field(default="rs.memory.library/1", alias="schema")
    entries: list[IndexEntry]


def index_from_manifests(manifests: Iterable[tuple[Path, Path]]) -> LibraryIndex:
    """An index of every task of each (manifest, store root) pair, in that order, each task's
    store the file store at its root."""
    entries = []
    for manifest_path, root in manifests:
        manifest = read_manifest(manifest_path)
        store = StoreSpec(name="file", options={"root": str(root)})
        for task, entry in manifest.tasks.items():
            entries.append(
                IndexEntry(task=task, version=entry.version, hash=entry.hash, store=store)
            )
    tasks = [e.task for e in entries]
    if twice := sorted({t for t in tasks if tasks.count(t) > 1}):
        raise ValueError(f"two manifests name {twice}: one Memory per task")
    return LibraryIndex(entries=entries)


def read_index(path: Path) -> LibraryIndex:
    return LibraryIndex.model_validate_json(path.read_bytes())


def write_index(path: Path, index: LibraryIndex) -> None:
    write_atomic(path, (index.model_dump_json(indent=2) + "\n").encode())


# One episode's library


@dataclass(frozen=True)
class Progress:
    """Where an episode stands when the Actor touches the library: env steps used, hard failures
    so far (`retry_guard`'s count) and seconds on its wall clock (the guard's: the runtime's
    identifications are credited back, so it may stand still or go back)."""

    steps: int
    failures: int
    elapsed_s: float


UseOutcome = list[ContentPart]
"""What adding Memories to the set in use did, for the Actor (the identification of their roles,
what now gates finish)."""


@dataclass
class EpisodeLibrary:
    """One P3 episode's library: the entries it sees (`entries`, in library order; `excluded`,
    the tasks left out of it), the Memories in use (`in_use`, in the order added), the host's
    `bind` (it binds the set in use: its specs, roles and done criteria; returns what to tell the
    Actor) and `progress`, and the record of every read and change (`timeline`)."""

    entries: list[LibraryEntry]
    excluded: list[str]
    own: MemoryEntryId | None
    bind: Callable[[list[LibraryEntry], list[LibraryEntry]], Awaitable[UseOutcome]]
    progress: Callable[[], Progress]
    in_use: list[str] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    opened: float = field(default_factory=time.monotonic)

    def get(self, id_: str) -> LibraryEntry | None:
        return next((e for e in self.entries if e.id == id_), None)

    def note(self, kind: str, **data: Any) -> dict[str, Any]:
        """Append a timeline record of `kind` with where the episode stands, and return it."""
        now = self.progress()
        record = {
            "kind": kind,
            **data,
            "step": now.steps,
            "failures_before": now.failures,
            "after_failure": now.failures > 0,
            "t_s": round(time.monotonic() - self.opened, 1),  # monotonic, since it opened
            "clock_s": round(now.elapsed_s, 1),
        }
        self.timeline.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        """What the episode's summary records: the library it saw, the entries read (in the order
        first read), the set in use at the end, every change in order, and the own Memory's
        part in them."""
        read: list[str] = []
        for record in self.timeline:
            if record["kind"] == "read" and record["id"] not in read:
                read.append(record["id"])
        used: list[str] = []
        for record in self.timeline:
            for id_ in record.get("added", []):
                if id_ not in used:
                    used.append(id_)
        tasks: dict[str, str] = {e.id: e.task for e in self.entries}
        return {
            "seen": [e.id for e in self.entries],
            "excluded": self.excluded,
            "own": self.own,
            "read": [{"id": i, "task": tasks[i]} for i in read],
            "used": [{"id": i, "task": tasks[i]} for i in used],
            "in_use_end": [{"id": i, "task": tasks[i]} for i in self.in_use],
            "own_read": self.own is not None and self.own in read,
            "own_used": self.own is not None and self.own in used,
            "own_in_use_end": self.own is not None and self.own in self.in_use,
            "timeline": self.timeline,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a Memory library index (library.json) from P3 manifests."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--unit-manifest",
        type=Path,
        action="append",
        default=[],
        help="a pipeline unit's P3 manifest (<unit>/P3/manifest.json): its store is the unit",
    )
    parser.add_argument(
        "--manifest",
        nargs=2,
        type=Path,
        action="append",
        default=[],
        metavar=("MANIFEST", "STORE_ROOT"),
        help="a manifest and the root of the file store that holds its versions",
    )
    args = parser.parse_args()
    pairs = [(m.resolve(), m.resolve().parent.parent) for m in args.unit_manifest]
    pairs += [(m.resolve(), r.resolve()) for m, r in args.manifest]
    if not pairs:
        parser.error("name at least one manifest")
    index = index_from_manifests(pairs)
    write_index(args.out, index)
    print(f"{args.out}: {len(index.entries)} entries")


if __name__ == "__main__":
    main()
