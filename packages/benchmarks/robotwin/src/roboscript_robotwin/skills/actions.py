"""RoboTwin's action skill on the core `manipulation` toolset: `rotate_about`
(docs/how-to/add-a-skillset.md: validated on RoboTwin only, so it mounts only with this skillset).
It builds on the toolset's handle `act` (one action through the episode's action API, its
coarse-then-fine refinement and its after-action hook); the turn itself is the core's action API
(`ActionApi.rotate_about`), which `together` and `check_reach` reach too.
"""

from roboscript.contracts.receipt import ActionReceipt
from roboscript.tools import Rationale, ToolContext, tool
from roboscript.tools.spec import ToolsetExtension
from roboscript.toolsets.manipulation import (
    ACTION_TIMEOUT_S,
    EXECUTOR,
    Manipulation,
    RotateAbout,
    render_receipt,
)
from roboscript.toolsets.manipulation.actions import (
    Angle,
    ArmName,
    HingeArg,
    Steps,
    TheObject,
)


class RoboTwinActions(ToolsetExtension):
    """`rotate_about`, run by the `manipulation` toolset."""

    extends = "manipulation"
    host: Manipulation

    @tool(
        kind="execution",
        tier="D",
        timeout_s=ACTION_TIMEOUT_S,
        render=render_receipt,
        needs=EXECUTOR,
    )
    async def rotate_about(
        self,
        ctx: ToolContext,
        rationale: Rationale,
        arm: ArmName,
        obj: TheObject,
        hinge: HingeArg,
        angle_deg: Angle,
        steps: Steps = None,
    ) -> ActionReceipt:
        """Turn the gripper, and what it holds, by angle_deg about a hinge of a registered
        object: a line on the box of one of its registered parts (a lid, a door) as the part
        was measured, in straight moves along the arc from where the gripper is now, positive
        angle_deg the way the hinge spec says. The part is measured again first; the hinge's
        side, height and offset are the given ones (the receipt notes where vision estimated
        it)."""
        action = RotateAbout(
            action="rotate_about", obj=obj, hinge=hinge, angle_deg=angle_deg, steps=steps
        )
        return await self.host.act(ctx, {arm: action})
