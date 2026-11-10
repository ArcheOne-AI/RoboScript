"""A toolset for tests of env-bound execution without the manipulation library: `hold` keeps the
robot still for some counted actions through the episode's env (`env.agent`), e.g. to drive an
episode behind the P3 bridge with a scripted brain. Only test stacks mount it."""

from typing import Annotated

from pydantic import Field

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.ids import BatchId, new_id
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.tools import Rationale, ToolContext, Toolset, tool


class Hold(Toolset):
    name = "hold"

    @tool(kind="execution", tier="D")
    async def hold(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        steps: Annotated[int, Field(ge=1, le=100, description="How many actions to hold for.")],
    ) -> ActionReceipt:
        """Keep both arms and grippers where they are for `steps` counted actions."""
        before = await ctx.env.robot(ctx.env_episode)
        step = ActionStep(qpos=before.qpos_cmd)
        batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=[step] * steps)
        result = await ctx.env.execute(ctx.env_episode, batch)
        after = result.states[-1].robot if result.states else before
        left = None if after.step_limit is None else after.step_limit - after.step
        return ActionReceipt(
            reason=ReceiptReason.ok,
            steps_used=result.executed,
            steps_left=left,
            note=f"held still for {result.executed} actions ({result.stopped})",
        )
