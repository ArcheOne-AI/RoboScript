"""The Mock benchmark's toolset, for conformance checks without a simulator or a model
(docs/concepts/agent-runtime.md).

`look` returns a new frame and a known image through the blob store, `act` is a slow counted
action whose receipt reason can be forced, `fail` always fails, and `hidden` exists but no
phase of `testing.phases.TestMethod` shows it.
"""

from typing import Annotated

import anyio
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import ToolError
from roboscript.contracts.ids import new_id
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.tools import FrameRef, Rationale, ToolContext, Toolset, tool


class MockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act_seconds: NonNegativeFloat = 2.0


class Observed(BaseModel):
    frame: FrameRef
    image: BlobRef


class Done(BaseModel):
    ok: bool = True


def mock_image() -> np.ndarray:
    """What `look` shows: 48x64 RGB, red growing to the right and green downwards."""
    y, x = np.mgrid[0:48, 0:64]
    return np.stack([x * 4, y * 5, np.full_like(x, 128)], axis=-1).astype(np.uint8)


class Mock(Toolset):
    name = "mock"
    Config = MockConfig
    config: MockConfig

    def __init__(self) -> None:
        self.acts = 0  # actions run, over all episodes

    @tool(kind="perception", tier="D")
    async def look(self, ctx: ToolContext) -> Observed:
        """Look at the scene: a new frame with its camera image."""
        image = await ctx.blobs.aput_image(mock_image(), meta={"camera": "overhead"})
        return Observed(frame=new_id("frame"), image=image)

    @tool(kind="execution", tier="D")
    async def act(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        outcome: Annotated[
            ReceiptReason, Field(description="The receipt reason to report (for testing).")
        ] = ReceiptReason.ok,
    ) -> ActionReceipt:
        """Perform one physical action; it takes a while."""
        await anyio.sleep(self.config.act_seconds)
        self.acts += 1
        ctx.state["acts"] = ctx.state.get("acts", 0) + 1
        note = f"action {ctx.state['acts']} of this episode"
        return ActionReceipt(reason=outcome, steps_used=1, note=note)

    @tool(kind="pure", tier="D")
    async def fail(self, ctx: ToolContext) -> Done:
        """Always fails."""
        raise ToolError("the mock tool failed on purpose")

    @tool(kind="pure", tier="D")
    async def hidden(self, ctx: ToolContext) -> Done:
        """Exists, but no stage sees it."""
        return Done()
