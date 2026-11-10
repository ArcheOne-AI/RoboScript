"""RoboTwin's measuring skills on the core `perception` toolset: `estimate_hinge` and
`measure_turn` (docs/how-to/add-a-skillset.md: validated on RoboTwin only, so they mount only with
this skillset). They build on the toolset's scene access (`SceneAccess.hinge`, `turn_offset`).
The core's grasps and turns ask `estimate_hinge` through the Hub when it is mounted
(`toolsets.manipulation.refine`), and go on unrefined without it.
"""

from typing import Annotated

from pydantic import Field

from roboscript.contracts.geometry import Vec3
from roboscript.contracts.robot import Arm
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.perception.measure import HingeEstimate, Side, TurnEstimate
from roboscript.tools import ObjectRef, ToolContext, tool
from roboscript.tools.spec import ToolsetExtension
from roboscript.toolsets.perception import Perception


def render_hinge(result: HingeEstimate) -> list[ContentPart]:
    p, d = result.point, result.direction
    text = (
        f"part {result.part} turns on its {result.side} side ({result.at}): line through "
        f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) along ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}), "
        f"radius {result.radius_m:.3f} m, confidence {result.confidence:.2f}."
    )
    return [TextPart(text=text)]


def render_turn(result: TurnEstimate) -> list[ContentPart]:
    text = (
        f"part {result.part} lies turned {result.offset_deg:+.0f} deg from where it is "
        f"registered, confidence {result.confidence:.2f} over {result.judged} points."
    )
    return [TextPart(text=text)]


class RoboTwinPerception(ToolsetExtension):
    """`estimate_hinge` and `measure_turn`, run by the `perception` toolset."""

    extends = "perception"
    host: Perception

    @tool(kind="perception", tier="D", render=render_hinge)
    async def estimate_hinge(
        self,
        ctx: ToolContext,
        obj: Annotated[ObjectRef, Field(description="The registered object.")],
        part: Annotated[str, Field(description="Its registered part that turns (a lid, a door).")],
        side: Annotated[
            Side | None, Field(description="The part's side the hinge is on; null: estimate it.")
        ] = None,
    ) -> HingeEstimate:
        """Where a registered part turns on its object: a line on one side of the part (upright
        for a thin standing part such as a door, along its bottom for a lying one, a lid), the
        side the body touches along it, opposite what sticks out of the part; its radius and
        confidence. Its part, side and at make the hinge rotate_about takes."""
        return await self.host.scene.hinge(ctx, obj, part, side)

    @tool(kind="perception", tier="D", timeout_s=120.0, render=render_turn)
    async def measure_turn(
        self,
        ctx: ToolContext,
        obj: Annotated[ObjectRef, Field(description="The registered object.")],
        part: Annotated[str, Field(description="Its registered part that turns (a lid, a door).")],
        point: Annotated[Vec3, Field(description="A point on the hinge line (world, m).")],
        direction: Annotated[Vec3, Field(description="The hinge line's direction (world).")],
        arm: Annotated[Arm, Field(description="The arm whose wrist camera looks too.")],
        span_deg: Annotated[
            float, Field(gt=0.0, le=180.0, description="The largest offset tried either way.")
        ],
    ) -> TurnEstimate:
        """How far a registered part lies turned about a hinge line from where it is registered,
        seen now by the head camera and the arm's wrist camera: the turn (deg, right-handed
        about the direction) at which the depth confirms most of its points."""
        return await self.host.scene.turn_offset(ctx, obj, part, (point, direction), span_deg, arm)
