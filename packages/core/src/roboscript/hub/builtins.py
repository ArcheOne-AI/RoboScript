"""The Hub's own tools: `finish`, handled by the pipeline's finish step (never forwarded), and the
episode's `scratchpad`, notes a model keeps for itself within one episode
(docs/concepts/learning.md), such as P3's adjustments while recovering: at most `SCRATCHPAD_CHARS`
characters, logged as `hub.scratchpad` events, gone with the episode and never written to the
Memory."""

from typing import Annotated, Final

from pydantic import BaseModel, Field

from roboscript.contracts.errors import ToolError
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.tools import ToolContext, Toolset, tool

FINISH: Final = "hub.finish"
SCRATCHPAD_CHARS: Final = 2000


class Finished(BaseModel):
    note: str


class Scratchpad(BaseModel):
    text: str
    chars: int
    limit: int


def _render_finished(result: Finished) -> list[ContentPart]:
    return [TextPart(text="Finished: the episode is over.")]


def _render_scratchpad(result: Scratchpad) -> list[ContentPart]:
    if not result.text:
        return [TextPart(text="Scratchpad: empty.")]
    head = f"Scratchpad ({result.chars} of {result.limit} characters):"
    return [TextPart(text=f"{head}\n{result.text}")]


class HubTools(Toolset):
    name = "hub"

    @tool(kind="control", tier="D", render=_render_finished)
    async def finish(
        self,
        ctx: ToolContext,
        note: Annotated[str, Field(description="A sentence or two on how the task ended.")],
    ) -> Finished:
        """End the episode, once the task is done or cannot be done; no tool calls are accepted
        afterwards."""
        return Finished(note=note)

    @tool(kind="pure", tier="D", render=_render_scratchpad)
    async def scratchpad(
        self,
        ctx: ToolContext,
        note: Annotated[str | None, Field(description="A note to add; null: only show it.")] = None,
        replace: Annotated[
            bool, Field(description="Replace the whole scratchpad with the note.")
        ] = False,
    ) -> Scratchpad:
        """Your notes for this episode only, at most 2000 characters: a note is added as a new
        line (or replaces them all). Once written, the scratchpad is shown at the start of every
        turn; it ends with the episode and never reaches the Task Memory."""
        text: str = ctx.state.get("text", "")
        if note is not None:
            new = note if replace else "\n".join(part for part in (text, note) if part)
            if len(new) > SCRATCHPAD_CHARS:
                raise ToolError(
                    f"the scratchpad holds {SCRATCHPAD_CHARS} characters and this makes "
                    f"{len(new)}: replace it with a shorter version"
                )
            ctx.state["text"] = text = new
            ctx.emit("scratchpad", {"replace": replace, "text": text})
        return Scratchpad(text=text, chars=len(text), limit=SCRATCHPAD_CHARS)
