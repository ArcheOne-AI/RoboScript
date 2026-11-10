"""The `library` toolset (docs/concepts/learning.md): an evaluated Actor's
access to the Memory library (`memory.library`), for the whole episode, error recovery included.

- `library_index`: the cards of the episode's library, a page at a time;
- `library_search`: the cards that match a text best;
- `library_read`: one entry's full Memory, a part at a time (parts fit the Actor's text cap, so
  reading several Memories goes through the context budget the Actor already has);
- `library_use`: add entries to the set of Memories in use, or drop them. The host binds the set
  in use (`EpisodeLibrary.bind`): their specs and numbers for calls that name them, their roles
  identified in the scene, their done criteria gating `finish`. Reads alone bind nothing. A use
  that fails (roles that clash, an identification that failed, the tool's timeout) leaves the set
  in use as it was and is logged (`use_failed`, with its reason) before its error reaches the
  Actor; it never ends the episode.

The Agent host attaches each episode's `EpisodeLibrary` (`attach`), so the toolset runs
in-process, beside it. Every call is a `library.<kind>` event and a timeline record (where the
episode stood: steps, hard failures so far), which the episode's summary lists.
"""

from typing import Annotated, Final

from anyio import get_cancelled_exc_class
from pydantic import BaseModel, Field

from roboscript.contracts.errors import ToolError
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.memory.library import (
    EpisodeLibrary,
    card_pages,
    memory_text,
    search,
    split_parts,
)
from roboscript.tools import ToolContext, Toolset, tool
from roboscript.toolsets.perception import IDENTIFY_TIMEOUT_S

MAX_HITS: Final = 8
USE_TIMEOUT_S: Final = 2 * IDENTIFY_TIMEOUT_S
"""`library_use` identifies each added Memory's roles in the scene (the opening hook: an
identification and, for roles it left open, one more), which the default 60 s tool timeout would
cut off."""


class LibraryText(BaseModel):
    """A page of cards or a part of a Memory, and which of how many."""

    text: str
    page: int
    pages: int


class UseResult(BaseModel):
    """The set in use after the call, and what binding it did."""

    in_use: list[str]
    added: list[str]
    dropped: list[str]
    content: list[ContentPart] = []


def _render_text(result: LibraryText) -> list[ContentPart]:
    return [TextPart(text=result.text)]


def _render_use(result: UseResult) -> list[ContentPart]:
    head = f"In use now: {', '.join(result.in_use) or 'none'}."
    if result.added:
        head += f" Added: {', '.join(result.added)}."
    if result.dropped:
        head += f" Dropped: {', '.join(result.dropped)}."
    return [TextPart(text=head), *result.content]


class LibraryTools(Toolset):
    name = "library"

    def __init__(self) -> None:
        self._libraries: dict[EpisodeId, EpisodeLibrary] = {}

    def attach(self, episode_id: EpisodeId, library: EpisodeLibrary) -> None:
        if episode_id in self._libraries:
            raise ValueError(f"episode {episode_id} has a library attached already")
        self._libraries[episode_id] = library

    def detach(self, episode_id: EpisodeId) -> EpisodeLibrary | None:
        return self._libraries.pop(episode_id, None)

    @tool(kind="memory", tier="D", render=_render_text)
    async def library_index(
        self,
        ctx: ToolContext,
        page: Annotated[int, Field(ge=1, description="The page of cards; 1: the first.")] = 1,
    ) -> LibraryText:
        """List the Memory library, one line per Task Memory that practice on some task froze
        (its id, the task it came from, its goal and its stages), a page at a time; the
        full cards (object roles, tools, done criteria) come with library_search."""
        library = self._library(ctx)
        pages = card_pages(library.entries, brief=True) or ["The library is empty."]
        if page > len(pages):
            raise ToolError(f"there are {len(pages)} pages of cards")
        self._log(ctx, library, "index", page=page)
        text = f"Memory library, page {page} of {len(pages)} ({len(library.entries)} entries):\n\n"
        return LibraryText(text=text + pages[page - 1], page=page, pages=len(pages))

    @tool(kind="memory", tier="D", render=_render_text)
    async def library_search(
        self,
        ctx: ToolContext,
        query: Annotated[str, Field(min_length=1, description="What you need, in words.")],
        k: Annotated[int, Field(ge=1, le=MAX_HITS, description="How many cards at most.")] = 4,
    ) -> LibraryText:
        """Find the library's Memories whose cards and text match words best (objects, actions,
        parts, stages): their cards, best first."""
        library = self._library(ctx)
        hits = search(library.entries, query, k)
        self._log(ctx, library, "search", query=query, hits=[e.id for e in hits])
        if not hits:
            return LibraryText(text=f"No entry matches {query!r}.", page=1, pages=1)
        pages = card_pages(hits)
        return LibraryText(text=pages[0], page=1, pages=1)

    @tool(kind="memory", tier="D", render=_render_text)
    async def library_read(
        self,
        ctx: ToolContext,
        id: Annotated[str, Field(description="An entry's id, as its card gives it.")],
        part: Annotated[int, Field(ge=1, description="The part of a long Memory; 1: first.")] = 1,
    ) -> LibraryText:
        """Read one library entry's full Task Memory (instruction, objects, workflow, parameters,
        lessons, notes), a part at a time. Reading uses nothing: add it with library_use for its
        specs, roles and done criteria to apply."""
        library = self._library(ctx)
        entry = library.get(id)
        if entry is None:
            raise ToolError(f"no entry {id!r} in this library: see library_index")
        parts = split_parts(memory_text(entry))
        if part > len(parts):
            raise ToolError(f"{id} has {len(parts)} parts")
        self._log(ctx, library, "read", id=entry.id, task=entry.task, part=part)
        head = f"[{entry.id}, part {part} of {len(parts)}]\n"
        return LibraryText(text=head + parts[part - 1], page=part, pages=len(parts))

    @tool(kind="memory", tier="D", timeout_s=USE_TIMEOUT_S, render=_render_use)
    async def library_use(
        self,
        ctx: ToolContext,
        add: Annotated[
            list[str] | None, Field(description="Entry ids to add to the Memories in use.")
        ] = None,
        drop: Annotated[
            list[str] | None, Field(description="Entry ids to take out of the Memories in use.")
        ] = None,
    ) -> UseResult:
        """Change the set of Memories you use, at any time: one, several combined, or more added
        later (after a failure, say). The set in use applies: calls may name its specs and
        numbers by name, its object roles are identified in the scene, and its done criteria
        gate finish (drop a Memory whose criteria do not describe this task's end). Where two
        Memories in use name a spec alike, the one added last wins."""
        library = self._library(ctx)
        add, drop = add or [], drop or []
        unknown = [i for i in [*add, *drop] if library.get(i) is None]
        if unknown:
            raise ToolError(f"no entries {unknown} in this library: see library_index")
        if not add and not drop:
            raise ToolError("name entries to add or to drop")
        if both := sorted(set(add) & set(drop)):
            raise ToolError(f"{both} both added and dropped")
        new = [i for i in dict.fromkeys(add) if i not in library.in_use]
        gone = [i for i in dict.fromkeys(drop) if i in library.in_use]
        kept = [i for i in library.in_use if i not in gone]
        in_use = [*kept, *new]
        added = [e for i in new if (e := library.get(i)) is not None]
        using = [e for i in in_use if (e := library.get(i)) is not None]
        try:
            content = await library.bind(using, added)
        except BaseException as error:  # a refusal, a failed identification, the tool's timeout
            # the set in use stands as it was: the record says what was asked and why it failed
            cut = isinstance(error, get_cancelled_exc_class())
            reason = "cancelled: the tool's timeout or the episode's end" if cut else repr(error)
            self._log(
                ctx, library, "use_failed", asked_add=new, asked_drop=gone, reason=reason[:500]
            )
            raise
        library.in_use = in_use
        self._log(ctx, library, "use", added=new, dropped=gone, in_use=list(in_use))
        return UseResult(in_use=in_use, added=new, dropped=gone, content=content)

    def _library(self, ctx: ToolContext) -> EpisodeLibrary:
        library = self._libraries.get(ctx.episode_id)
        if library is None:
            raise ToolError("No Memory library is attached to this episode.")
        return library

    def _log(self, ctx: ToolContext, library: EpisodeLibrary, kind: str, **data: object) -> None:
        ctx.emit(kind, library.note(kind, **data))
