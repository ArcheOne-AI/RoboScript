"""The `memory` toolset (docs/concepts/learning.md, docs/concepts/agent-runtime.md):
`memory_read` and `memory_edit` over the Memory an episode edits: a demo-reading role's draft, a
reviewing role's Memory or its per-seed cache. The runner attaches that `Editor` to the episode
(`attach`), so the toolset runs in-process, beside the runner; the editor enforces the budget, the
revision's limits and the linter, so a model cannot spend beyond its budget. Every edit, accepted or
rejected, is logged as a `memory.edit` event, with the linter's findings.
"""

from collections.abc import Callable, Sequence
from typing import Annotated, Final, Literal

from pydantic import BaseModel, Field

from roboscript.contracts.errors import ToolError
from roboscript.contracts.ids import EpisodeId
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.memory import EditOp, Editor, Finding, SkillEditOp, diff, render_diff, view
from roboscript.memory.doc import content_hash
from roboscript.records import replay
from roboscript.tools import ToolContext, Toolset, tool

ReadSection = Literal[
    "statement", "objects", "workflow", "lessons", "notes", "scratch", "params", "diff"
]


class MemoryView(BaseModel):
    """The Memory as text, with the budget (None: unlimited) and the numeric targets this revision
    may still change."""

    text: str
    budget: int | None
    used: int
    left: int | None
    numbers_left: int | None


class EditResult(BaseModel):
    accepted: bool
    op: str
    cost: int
    left: int | None
    reason: str = ""
    findings: list[Finding] = []
    clamped: str = ""


def _render_view(result: MemoryView) -> list[ContentPart]:
    return [TextPart(text=result.text)]


def _render_edit(result: EditResult) -> list[ContentPart]:
    if result.accepted:
        units = result.left
        left = "" if units is None else f"; {units} budget unit{'s' * (units != 1)} left"
        lines = [f"Accepted {result.op} (cost {result.cost}{left})."]
        if result.clamped:
            lines.append(f"Clamped: {result.clamped}.")
    else:
        lines = [f"Rejected {result.op}: {result.reason}"]
    for f in result.findings:
        rule = f.rule if f.level == "error" else f"{f.rule}, a warning"
        lines.append(f"- {f.where}: {f.match!r} ({rule}): {f.message}")
    return [TextPart(text="\n".join(lines))]


EditCheck = Callable[[ToolContext, EditOp], None]
"""A method's check of a model's edit before the editor applies it, with the episode's facts in
`ctx` (a demonstration's fitted specs, say): it raises `ToolError` to refuse the edit."""


class MemoryTools(Toolset):
    """`checks`: the method's edit checks, run in order on every `memory_edit`
    (`learning.runtime.MethodRuntime.edit_checks`)."""

    name = "memory"

    def __init__(self, checks: Sequence[EditCheck] = ()) -> None:
        self._editors: dict[EpisodeId, Editor] = {}
        self.checks = tuple(checks)

    def attach(self, episode_id: EpisodeId, editor: Editor) -> None:
        """Have the episode's memory calls read and edit `editor`'s document, until `detach`."""
        if episode_id in self._editors:
            raise ValueError(f"episode {episode_id} has a Memory attached already")
        self._editors[episode_id] = editor

    def detach(self, episode_id: EpisodeId) -> Editor:
        return self._editors.pop(episode_id)

    @tool(kind="memory", tier="D", render=_render_view)
    async def memory_read(
        self,
        ctx: ToolContext,
        section: Annotated[
            ReadSection | None,
            Field(description="One section, the parameters, or the changes so far; null: all."),
        ] = None,
    ) -> MemoryView:
        """Read the edit budget left, what each edit costs, and the Memory you are editing: all of
        it (the parameters first), or one section (for a long Memory)."""
        editor = self._editor(ctx)
        doc = editor.doc
        if section == "diff":
            text = "## CHANGES\n" + render_diff(diff(editor.base, doc)) + "\n"
        elif section == "scratch" and doc.scratch is None:
            text = "There is no SCRATCH: only a cache has one.\n"
        else:
            text = view(doc, section)
        return MemoryView(
            text=f"{_status(editor)}\n\n{text}",  # the status first: the Hub may cut long text
            budget=editor.budget,
            used=editor.used,
            left=editor.left,
            numbers_left=editor.numbers_left,
        )

    @tool(kind="memory", tier="D", render=_render_edit)
    async def memory_edit(
        self,
        ctx: ToolContext,
        op: Annotated[EditOp, Field(description="The edit: one operation, tagged by `op`.")],
    ) -> EditResult:
        """Change the Memory by one operation: set a number or a category (a parameter or a spec
        field), add, edit or remove a lesson, edit a workflow stage, add, remove or move a stage
        or set its completion predicates, define or remove a parameter, replace a text section,
        or note something in SCRATCH. Edits cost budget units (see memory_read); an edit that
        breaks the budget, bounds or the linter is rejected with the reason and
        costs nothing; a numeric edit beyond its trust radius moves the radius toward its value
        (the result says so)."""
        for check in self.checks:
            check(ctx, op)
        editor = self._editor(ctx)
        record = editor.edit(op)
        ctx.emit("edit", record.model_dump(mode="json"))
        if (tape := replay.active()) is not None:
            doc = editor.doc
            scratch = doc.model_dump(mode="json", include={"scratch"})["scratch"]
            after = {"doc": content_hash(doc.replace(created=None)), "scratch": scratch}
            tape.observe("memory.edit", {**record.model_dump(mode="json"), **after})
        return EditResult(
            accepted=record.accepted,
            op=record.op.op,
            cost=record.cost,
            left=record.left,
            reason=record.reason,
            findings=record.findings,
            clamped=record.clamped,
        )

    def _editor(self, ctx: ToolContext) -> Editor:
        editor = self._editors.get(ctx.episode_id)
        if editor is None:
            raise ToolError("No Memory is attached to this episode.")
        return editor


class SkillMemoryTools(MemoryTools):
    """`MemoryTools` whose `memory_edit` also defines a written skill's fitted parameters (a
    `skill_spec`, `SkillEditOp`): the memory toolset of a stack whose skillsets fit written
    skills (`tools.skillsets.fits_skills`), which the runner mounts in its place. A stack without
    them is shown the plain `MemoryTools` schema."""

    @tool(kind="memory", tier="D", render=_render_edit)
    async def memory_edit(
        self,
        ctx: ToolContext,
        op: Annotated[SkillEditOp, Field(description="The edit: one operation, tagged by `op`.")],
    ) -> EditResult:
        return await super().memory_edit(ctx, op)

    memory_edit.__doc__ = MemoryTools.memory_edit.__doc__


NO_TRUST: Final = (
    "No trust radius applies to this revision (the `trust` the parameters list does not bind "
    "it): make a correction in one edit, not in steps."
)


CLAMPED: Final = (
    "An edit asking for more than its trust radius is moved the radius toward it (the result says "
    "so) and the next revision may go further: ask for the correction the evidence shows."
)


def _status(editor: Editor) -> str:
    if editor.budget is None:
        lines = ["Budget: none applies."]
    else:
        costs = ", ".join(f"{op} {cost}" for op, cost in editor.rules.costs)
        lines = [f"Budget: {editor.left} of {editor.budget} units left. Costs: {costs}."]
    trust = editor.rules.trust_radius
    if editor.numbers_left is not None:
        within = ", each within its trust radius." if trust else "."
        lines.append(f"This revision may change {editor.numbers_left} more numeric targets{within}")
    elif trust:
        lines.append("Each numeric change stays within its trust radius.")
    else:
        lines.append(NO_TRUST)
    if trust and editor.rules.clamp_to_trust:
        lines.append(CLAMPED)
    return "\n".join(lines)
