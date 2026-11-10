"""A skillset of your own: `pickup`, one operation API for the Mock robot.

A skillset is data (a manifest) plus tools added to a core toolset (`ToolsetExtension`). Its tools
run while the Agent executes, so they are tier D and read only the robot's own state. This one
extends `hold`, the test toolset that executes through the env, because the Mock benchmark mounts
no planner; a skillset for a real benchmark extends `manipulation` and takes its planner and
executor from the bindings (see `docs/how-to/add-a-skillset.md`).
"""

from pathlib import Path
from typing import Annotated, Final

from pydantic import Field

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.ids import BatchId, new_id
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.robot import Arm
from roboscript.contracts.skills import SkillSet
from roboscript.embodiments import load_embodiment
from roboscript.tools import Rationale, ToolContext, tool
from roboscript.tools.skillsets import SkillSetPlugin, skill
from roboscript.tools.spec import ToolsetExtension


class PickUp(ToolsetExtension):
    extends = "hold"  # the core toolset whose provider mounts this tool

    @tool(kind="execution", tier="D")
    async def pick_up(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        arm: Arm,
        tcp: Annotated[Vec3, Field(description="Where the TCP closes (world, meters).")],
        lift: Annotated[float, Field(gt=0, le=0.3, description="How far to lift (meters).")],
    ) -> ActionReceipt:
        """Close one arm's gripper on a point and lift what it holds: open, close, lift, one
        counted action."""
        if ctx.context.embodiment is None:
            raise LookupError(f"episode {ctx.episode_id} names no embodiment")
        spec = load_embodiment(ctx.context.embodiment).arms[arm]
        robot = await ctx.env.robot(ctx.env_episode)
        arm_state = robot.arms[arm]
        start = spec.action[0]
        # Move the wrist so that the TCP lands on the target; keep the orientation as it is.
        offset = [t - e for t, e in zip(arm_state.tcp.p, arm_state.ee.p, strict=True)]
        qpos = list(robot.qpos_cmd)
        qpos[start : start + 3] = [t - o for t, o in zip(tcp, offset, strict=True)]
        qpos[spec.gripper_index] = spec.gripper.open
        close = list(qpos)
        close[spec.gripper_index] = spec.gripper.closed
        up = list(close)
        up[start + 2] += lift
        steps = [ActionStep(qpos=q) for q in (qpos, close, up)]
        batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=steps)
        result = await ctx.env.execute(ctx.env_episode, batch)
        return ActionReceipt(
            reason=ReceiptReason.ok,
            steps_used=result.executed,
            note=f"{arm} closed at {tuple(round(v, 3) for v in tcp)} and lifted {lift:g} m",
        )


PICKUP_SKILLS: Final = SkillSetPlugin(
    manifest=SkillSet(
        name="pickup",
        version="1",
        layer="built",  # core | benchmark | robot | built
        envs=["mock"],
        embodiments=["mock_bimanual"],
        skills=[
            skill(
                PickUp.specs()[0],
                # which of the implementation steps this skill goes through, in this order
                ["parameters", "env_execute", "receipt"],
                arms=["left", "right"],  # the arm names its parameters take
            )
        ],
        guide=Path(__file__).with_name("guide.md").read_text(encoding="utf-8"),
    ),
    extensions=(PickUp,),
)
