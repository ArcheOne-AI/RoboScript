"""The `geometry` toolset (docs/concepts/perception.md,
docs/concepts/agent-runtime.md): measuring with depth, no model.

`lift` and `fit` measure the points a region shows (a proposal, a box or a mask; see
`roboscript.toolsets.scene`); `table` gives the episode's table, fixed on the first frame
analyzed (the `TABLE` fact, which `identify` sets on the initial scene); `project` puts world
points into a camera's image and `measure_pixel` reads the surface under one pixel. Pixels are
continuous coordinates, as boxes are everywhere (`roboscript.contracts.vision`): pixel (c, r)
covers [c, c + 1) x [r, r + 1). All tools work on frames of the episode's store, so a reviewing
role measures the Actor's logged frames the same way.
"""

from functools import partial
from typing import Annotated, Final, Literal

import anyio
import numpy as np
from pydantic import BaseModel, Field, FiniteFloat

from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.perception.fit import fit, summarize
from roboscript.perception.lift import PixelMeasure, lift, measure_pixel
from roboscript.perception.table import estimated
from roboscript.perception.views import FrameGeometry
from roboscript.spatial.camera import project
from roboscript.tools import FrameRef, RegionRef, ToolContext, Toolset, tool
from roboscript.tools.context import ToolsetRuntime
from roboscript.toolsets.scene import VISIBLE, SceneAccess, SceneConfig

Shape = Literal["auto", "box", "cylinder", "plane", "line"]
Pixel = tuple[FiniteFloat, FiniteFloat]
_Camera = Annotated[
    str | None,
    Field(description="A camera `observe` lists; null: the embodiment's head camera."),
]


class LiftReport(BaseModel):
    """The points a region shows: how many, `solid` (standing above the table) or `flat` (level
    with it), the cameras they come from, and their summary: footprint center at half height,
    top center, size (long side, short side, height), the long side's yaw, and whether every view
    cut them at the image border (the center may be about 1 cm off)."""

    region: RegionRef
    frame: FrameRef
    points: int
    kind: Literal["solid", "flat"]
    cameras: list[str]
    center: Vec3
    top_center: Vec3
    size: Vec3
    yaw_deg: float
    partially_visible: bool


class FitReport(BaseModel):
    """A shape fitted to the points a region shows: whether it stands, its centers, its upright
    box (size along its axes, yaw of its x), a symmetry guess from the shape alone, the RMS
    distance of the points to the surface, and a cylinder's radius and axis."""

    region: RegionRef
    shape: str
    upright: bool
    center: Vec3
    top_center: Vec3
    bottom_center: Vec3
    size: Vec3
    yaw_deg: float
    symmetry: str
    residual: float
    radius: float | None
    axis: Vec3


_TABLE_SINCE: Final = "table_since"  # the frame the table was first reported on


class TableReport(BaseModel):
    """The table: its plane's height at the world origin and normal, the extent the anchor camera
    sees, its edges where a view shows them (null: unknown, never guessed), its landmarks and
    which of them an edge no view shows leaves estimated; `since`: the frame it was found on,
    when an earlier call reported it already."""

    z: float
    normal: Vec3
    extent: tuple[float, float, float, float]
    edges: dict[str, float | None]
    landmarks: dict[str, Vec3]
    estimated: list[str] = []
    since: FrameRef | None = None


def render_table(result: TableReport) -> list[ContentPart]:
    marks = ", ".join(
        f"{name} {_xyz(p)}{' (estimated)' if name in result.estimated else ''}"
        for name, p in result.landmarks.items()
    )
    if result.since is not None:
        text = f"The table is unchanged since {result.since} (found once per episode): {marks}."
        return [TextPart(text=text)]
    edges = ", ".join(
        f"{name}={'unseen' if v is None else f'{v:.3f}'}" for name, v in result.edges.items()
    )
    extent = ", ".join(f"{v:.3f}" for v in result.extent)
    text = f"Table: z={result.z:.3f}, extent (x0, x1, y0, y1) {extent}; edges {edges}. "
    text += f"Landmarks: {marks or 'none'}."
    if result.estimated:
        text += (
            " Estimated ones take a side no view shows from the robot's midline or the view's "
            "extent; to place at a demonstrated spot, a robot_frame target is exact."
        )
    return [TextPart(text=text)]


def _xyz(p: Vec3) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in p) + ")"


class ProjectedPoint(BaseModel):
    """Where a world point shows: its pixel (null behind the camera), its depth along the view,
    whether the pixel lies in the image, and whether the depth image shows the point there
    (null outside the image: false when something in front hides it)."""

    point: Vec3
    pixel: Pixel | None
    depth: float | None
    in_image: bool
    visible: bool | None


class Projected(BaseModel):
    frame: FrameRef
    camera: str
    points: list[ProjectedPoint]


class MeasuredPixel(PixelMeasure):
    frame: FrameRef


def render_lift(result: LiftReport) -> list[ContentPart]:
    cut = ", cut by the image border" if result.partially_visible else ""
    text = (
        f"{result.region}: {result.points} {result.kind} points from {', '.join(result.cameras)}"
        f"{cut}; center {_xyz(result.center)}, top center {_xyz(result.top_center)}, size "
        f"{_sizes(result.size)} m, yaw {result.yaw_deg:.1f} deg."
    )
    return [TextPart(text=text)]


def render_fit(result: FitReport) -> list[ContentPart]:
    radius = "" if result.radius is None else f", radius {result.radius:.3f} m"
    stands = "standing" if result.upright else "lying"
    text = (
        f"{result.region}: {stands} {result.shape}, center {_xyz(result.center)}, top center "
        f"{_xyz(result.top_center)}, box {_sizes(result.size)} m at yaw {result.yaw_deg:.1f} deg"
        f"{radius}, symmetry {result.symmetry}, residual {result.residual * 1000:.1f} mm."
    )
    return [TextPart(text=text)]


class Geometry(Toolset):
    """Lift, fit, the table, projections and pixel measurements."""

    name = "geometry"
    Config = SceneConfig
    config: SceneConfig

    async def start(self, runtime: ToolsetRuntime) -> None:
        self.scene = SceneAccess(self.config, runtime)

    async def stop(self) -> None:
        await self.scene.close()

    @tool(kind="pure", tier="D", render=render_lift)
    async def lift(
        self,
        ctx: ToolContext,
        region: Annotated[RegionRef, Field(description="A region from ground, segment, identify.")],
    ) -> LiftReport:
        """The world points a region shows, from every view it has: robot pixels, the table,
        floor and background surfaces dropped, fragments of one object merged, the largest
        object kept; summarized as center, top center, size and yaw."""
        picked = await self.scene.region(ctx, region)
        geometry = picked.geometry
        lifted = await anyio.to_thread.run_sync(lift, geometry, picked.masks, self.config.pipeline)
        if lifted is None:
            raise ToolError("that region shows nothing with depth above the table")
        s = await anyio.to_thread.run_sync(summarize, lifted.points, geometry.table.z)
        return LiftReport(
            region=region,
            frame=geometry.frame,
            points=len(lifted.points),
            kind=lifted.kind,
            cameras=list(lifted.masks),
            center=s.center,
            top_center=s.top_center,
            size=s.size,
            yaw_deg=round(s.yaw_deg, 1),
            partially_visible=lifted.partially_visible,
        )

    @tool(kind="pure", tier="D", render=render_fit)
    async def fit(
        self,
        ctx: ToolContext,
        region: Annotated[RegionRef, Field(description="A region from ground, segment, identify.")],
        shape: Annotated[
            Shape, Field(description="auto: the shape that explains the points best.")
        ] = "auto",
    ) -> FitReport:
        """Fit a shape to the points a region shows: a box (upright), a cylinder (standing or
        lying), a plane or a line, with its centers, box, symmetry and residual."""
        picked = await self.scene.region(ctx, region)
        z = picked.geometry.table.z
        lifted = await anyio.to_thread.run_sync(
            lift, picked.geometry, picked.masks, self.config.pipeline
        )
        if lifted is None or len(lifted.points) < 3:
            raise ToolError("that region shows nothing with depth above the table")
        f = await anyio.to_thread.run_sync(partial(fit, lifted.points, shape, z))
        return FitReport(
            region=region,
            shape=f.shape,
            upright=f.upright,
            center=f.center,
            top_center=f.top_center,
            bottom_center=f.bottom_center,
            size=f.obb.size,
            yaw_deg=round(f.yaw_deg, 1),
            symmetry=f.symmetry,
            residual=f.residual,
            radius=f.radius,
            axis=f.axis,
        )

    @tool(kind="pure", tier="D", render=render_table)
    async def table(
        self,
        ctx: ToolContext,
        frame: Annotated[
            FrameRef | None,
            Field(description="Where to find it if not known yet; null: the newest frame."),
        ] = None,
    ) -> TableReport:
        """The table: plane height, extent, the edges some view shows and its landmarks (table
        center, front-edge midpoint, robot midline; estimated where no view shows an edge). It is
        found once, on the episode's first analyzed frame, and never changes afterwards."""
        since = ctx.state.get(_TABLE_SINCE)
        if since is None:
            seen = await self.scene.frame(ctx, frame)
            t = await self.scene.table(ctx, seen.id)
            ctx.state[_TABLE_SINCE] = seen.id
        else:
            t = await self.scene.table(ctx, since)
        return TableReport(
            z=t.z,
            normal=t.normal,
            extent=t.extent,
            edges=t.edges,
            landmarks=t.landmarks,
            estimated=estimated(t),
            since=since,
        )

    @tool(kind="pure", tier="D")
    async def project(
        self,
        ctx: ToolContext,
        points: Annotated[
            list[Vec3], Field(min_length=1, max_length=20, description="World x, y, z in meters.")
        ],
        camera: _Camera = None,
        frame: Annotated[
            FrameRef | None, Field(description="The frame (its calibration); null: the newest.")
        ] = None,
    ) -> Projected:
        """Where world points show in a camera's image in a frame (wrist cameras move with
        their arm): pixel, depth along the view, and whether the depth image shows the point
        there or something in front hides it."""
        geometry = await self.scene.geometry(ctx, await self.scene.frame(ctx, frame))
        camera = self.scene.camera(ctx, geometry, camera)
        return Projected(
            frame=geometry.frame, camera=camera, points=_project(geometry, camera, points)
        )

    @tool(kind="pure", tier="D")
    async def measure_pixel(
        self,
        ctx: ToolContext,
        pixel: Annotated[Pixel, Field(description="u (column), v (row) in pixels.")],
        camera: _Camera = None,
        frame: Annotated[FrameRef | None, Field(description="The frame; null: the newest.")] = None,
    ) -> MeasuredPixel:
        """The surface under one pixel: its world point (median of a small window), the normal
        toward the camera, the camera depth, the height above the table and whether the robot
        covers it."""
        geometry = await self.scene.geometry(ctx, await self.scene.frame(ctx, frame))
        camera = self.scene.camera(ctx, geometry, camera)
        try:
            found = measure_pixel(geometry, camera, pixel)
        except ValueError as error:
            raise ToolError(str(error)) from None
        return MeasuredPixel(frame=geometry.frame, **dict(found))


def _project(geometry: FrameGeometry, camera: str, points: list[Vec3]) -> list[ProjectedPoint]:
    view = geometry.view(camera)
    image = view.image
    h, w = image.shape
    uv, z, front = project(np.asarray(points, dtype=float), image.K, image.cam2world)
    out = []
    for point, (u, v), depth, ahead in zip(points, uv, z, front, strict=True):
        if not ahead:
            out.append(
                ProjectedPoint(point=point, pixel=None, depth=None, in_image=False, visible=None)
            )
            continue
        c, r = int(np.rint(u)), int(np.rint(v))
        inside = 0 <= c < w and 0 <= r < h
        seen = None
        if inside:
            measured = float(view.depth[r, c])
            seen = measured > 0 and abs(measured - float(depth)) < VISIBLE
        out.append(
            ProjectedPoint(
                point=point,
                pixel=(float(u) + 0.5, float(v) + 0.5),  # continuous: pixel centers at + 0.5
                depth=float(depth),
                in_image=inside,
                visible=seen,
            )
        )
    return out


def _xyz(values: Vec3) -> str:
    return "(" + ", ".join(f"{round(v, 3) + 0.0:.3f}" for v in values) + ")"


def _sizes(values: Vec3) -> str:
    return " x ".join(f"{v:.3f}" for v in values)
