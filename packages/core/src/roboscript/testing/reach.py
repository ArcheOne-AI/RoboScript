"""A skillset for testing skillset validation on MockEnv (docs/how-to/add-a-skillset.md): `reach`
moves an arm's TCP to a point, keeping its orientation, and opens or closes its gripper, in one
counted action; it extends the test `hold` toolset (`testing.hold`). Not registered (it would add a
tool to every stack that mounts `hold`): named by import path,
`roboscript.testing.reach:REACH_SKILLS`. It reads only the robot's own state, so it passes the
boundary check; a scripted planner reading the ground truth drives it to MockEnv's `lift_red` in the
ceiling step (`testing.validation`)."""

from typing import Annotated, Final, Literal

from pydantic import Field

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.ids import BatchId, new_id
from roboscript.contracts.receipt import ActionReceipt, ReceiptReason
from roboscript.contracts.robot import Arm
from roboscript.contracts.skills import SkillSet
from roboscript.embodiments import load_embodiment
from roboscript.spatial.rotation import quat_from_axis_angle, quat_multiply
from roboscript.tools import Rationale, ToolContext, tool
from roboscript.tools.skillsets import SkillSetPlugin, skill
from roboscript.tools.spec import ToolsetExtension

AXES: Final = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class Reach(ToolsetExtension):
    extends = "hold"

    @tool(kind="execution", tier="D")
    async def reach(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        arm: Arm,
        tcp: Annotated[Vec3, Field(description="Where the TCP goes (world, meters).")],
        gripper: Annotated[Literal["open", "closed"], Field(description="The gripper after.")],
    ) -> ActionReceipt:
        """Move one arm's TCP straight to a point, keeping its orientation, then set its
        gripper; one counted action."""
        if ctx.context.embodiment is None:
            raise LookupError(f"episode {ctx.episode_id} names no embodiment")
        profile = load_embodiment(ctx.context.embodiment)
        spec = profile.arms[arm]
        before = await ctx.env.robot(ctx.env_episode)
        qpos = list(before.qpos_cmd)
        start, stop = spec.action
        x, y, z, *angles = qpos[start:stop]
        turn = [quat_from_axis_angle(axis, angle) for axis, angle in zip(AXES, angles, strict=True)]
        q = quat_multiply(turn[2], quat_multiply(turn[1], turn[0])).tolist()
        now = profile.tcp_pose(arm, Pose(p=(x, y, z), q=tuple(q)))
        qpos[start : start + 3] = [e + t - n for e, t, n in zip((x, y, z), tcp, now.p, strict=True)]
        qpos[spec.gripper_index] = spec.gripper.open if gripper == "open" else spec.gripper.closed
        batch = ActionBatch(batch_id=BatchId(new_id("batch")), steps=[ActionStep(qpos=qpos)])
        result = await ctx.env.execute(ctx.env_episode, batch)
        return ActionReceipt(
            reason=ReceiptReason.ok,
            steps_used=result.executed,
            note=f"{arm} TCP at {tuple(round(v, 3) for v in tcp)}, gripper {gripper}",
        )


REACH_SKILLS: Final = SkillSetPlugin(
    manifest=SkillSet(
        name="reach",
        version="1",
        layer="built",
        envs=["mock"],
        embodiments=["mock_bimanual"],
        skills=[
            skill(
                Reach.specs()[0],
                ["parameters", "env_execute", "receipt"],
                arms=["left", "right"],
            )
        ],
    ),
    extensions=(Reach,),
)
