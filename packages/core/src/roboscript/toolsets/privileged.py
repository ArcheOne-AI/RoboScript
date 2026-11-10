"""The `privileged` toolset (docs/concepts/agent-runtime.md,
docs/architecture.md): the generic tier-P toolset, a view of the simulator's ground
truth.

A method may give a learning phase's roles privileged data. `true_objects` reads the env's
ground truth through its control plane (the one route an agent-side toolset otherwise never
has), so the toolset is constructed with that plane, and only an env that declares the
truth-snapshot capability can have it mounted (`mountable`; a real robot never declares it). The
method's phases decide which roles see it; a tier-D episode, and every episode of an eval phase,
never sees or calls it (the Hub's tier checks hold for service calls too).
"""

from typing import Annotated, Final

import numpy as np
from pydantic import BaseModel, Field

from roboscript.contracts.capabilities import EnvCapabilities, missing
from roboscript.contracts.env import EnvControl
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.tools import ToolContext, Toolset, tool


class TrueObject(BaseModel):
    """A scene object as the simulator has it: its name, its pose (the asset's own frame) and
    its geometric center (its convex hulls' vertices' mean; the pose's position without hulls)."""

    name: str
    pose: Pose
    center: Vec3


class TrueObjects(BaseModel):
    """The scene's objects as the simulator has them now."""

    objects: list[TrueObject]


def render_true_objects(result: TrueObjects) -> list[ContentPart]:
    lines = ["The simulator's objects now (privileged: never write them into the Memory):"]
    for obj in result.objects:
        p, c = obj.pose.p, obj.center
        lines.append(
            f"- {obj.name}: center ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}), origin "
            f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), q ({', '.join(f'{v:.3f}' for v in obj.pose.q)})"
        )
    return [TextPart(text="\n".join(lines))]


REQUIRES: Final = EnvCapabilities(truth=True)
"""What an env must declare for the toolset to be mounted on it."""


def mountable(offered: EnvCapabilities) -> list[str]:
    """Why the toolset cannot be mounted on an env offering `offered`, one line each (empty: it
    can): the env must declare ground-truth snapshots."""
    return [f"the env does not declare {face}" for face in missing(REQUIRES, offered)]


class Privileged(Toolset):
    """The simulator's ground truth, tier P."""

    name = "privileged"

    def __init__(self, control: EnvControl) -> None:
        """`control`: the env's control plane (the one of the task this stack runs)."""
        self.control = control

    @tool(kind="pure", tier="P", render=render_true_objects)
    async def true_objects(
        self,
        ctx: ToolContext,
        centers: Annotated[
            bool, Field(description="Measure each object's center from its shape (slower).")
        ] = True,
    ) -> TrueObjects:
        """The simulator's scene objects now: name, pose and center. Privileged: never write them
        into the Memory."""
        snapshot = await self.control.gt_snapshot(ctx.env_episode, geometry=centers)
        objects = []
        for name, pose in snapshot.objects.items():
            hulls = snapshot.hulls.get(name)
            points = [v for hull in hulls or [] for v in hull]
            center = np.mean(points, axis=0) if points else np.asarray(pose.p)
            objects.append(
                TrueObject(name=name, pose=pose, center=tuple(float(v) for v in center))  # type: ignore[arg-type]
            )
        return TrueObjects(objects=objects)
