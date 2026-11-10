"""The `sensing` toolset (docs/concepts/agent-runtime.md): what the robot sees and feels,
through the env's agent plane only.

`observe` takes a new frame from the cameras with the robot state and puts it into the episode's
frame store (`ToolContext.frames`), its images overlaid from calibration and proprioception, and
lists the objects the episode's scene registry holds (where they are now: a held one where its
gripper holds it, the others where they were measured);
`get_image` and `get_camera` read any frame of the store, also one an earlier episode left (a
reviewing role's seed); `get_robot_state` reads proprioception now. Frames are immutable and
addressed by reference (`frame_2` for models), so a call never races an action. The per-turn
observation is `observe` on the Hub's system channel. The table grid appears once the store holds
the table estimate (the `TABLE` scene fact perception sets).
"""

from collections.abc import Sequence
from functools import partial
from typing import Annotated, Any

import anyio
import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self

from roboscript.blobs import IMAGE_TYPES, NPY, BlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.env import ObserveRequest
from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import Arm, ArmState, RobotState
from roboscript.contracts.tools import ContentPart, ImagePart, TextPart
from roboscript.embodiments import EmbodimentProfile, Gripper, load_embodiment
from roboscript.perception.registry import WHOLE, object_geometry
from roboscript.spatial.pose import pose_to_matrix
from roboscript.spatial.rotation import quat_to_matrix
from roboscript.tools import TABLE, FrameRef, ToolContext, ToolModel, Toolset, tool
from roboscript.toolsets.scene import ObjectRef, ObjectReport, describe, object_report
from roboscript.toolsets.sensing.overlay import Axes, Box, Mark, TableGrid, draw

__all__ = [
    "ArmReport",
    "CameraReport",
    "ImageShown",
    "Observed",
    "Overlay",
    "PixelBox",
    "Point",
    "RobotReport",
    "Sensing",
    "SensingConfig",
    "TableGrid",
    "View",
]


class SensingConfig(BaseModel):
    """`axes_m`: length of the TCP axes drawn in fixed views; `grid_m`: spacing of the table grid
    (meters). A gripper commanded closed holds something when its measured opening is >= the
    embodiment profile's `Gripper.holding_threshold`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    axes_m: PositiveFloat = 0.05
    grid_m: PositiveFloat = 0.1


class ArmReport(BaseModel):
    """One arm: measured ee and TCP world poses, the gripper's measured opening and its command
    (1 open, 0 closed), whether it holds something (None when the opening is not measured) and,
    from `get_robot_state`, the joints (measured, commanded; radians)."""

    ee: Pose
    tcp: Pose
    gripper: float | None
    gripper_cmd: float
    holding: bool | None
    joints: list[float] | None = None
    joints_cmd: list[float] | None = None


class View(BaseModel):
    camera: str
    image: BlobRef


class Observed(BaseModel):
    frame: FrameRef
    step: int
    step_limit: int | None
    arms: dict[Arm, ArmReport]
    views: list[View]
    overlays: list[str]
    objects: list[ObjectReport] = []


class RobotReport(BaseModel):
    step: int
    step_limit: int | None
    arms: dict[Arm, ArmReport]


class CameraReport(BaseModel):
    """Image size, intrinsics in pixels and the camera's pose in the world (OpenCV axes: x right,
    y down, z along the view)."""

    camera: str
    frame: FrameRef
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pose: Pose


class ImageShown(BaseModel):
    frame: FrameRef
    camera: str
    image: BlobRef
    overlays: list[str]


class Point(ToolModel):
    """A point to mark in the image: exactly one of `world` and `pixel`."""

    label: Annotated[str, Field(max_length=24, description="A short label drawn beside it.")] = ""
    world: Annotated[
        Vec3 | None,
        Field(description="World x, y, z in meters, projected with the frame's calibration."),
    ] = None
    pixel: Annotated[
        tuple[FiniteFloat, FiniteFloat] | None,
        Field(description="Pixel u (column), v (row)."),
    ] = None

    @model_validator(mode="after")
    def _one_position(self) -> Self:
        if (self.world is None) == (self.pixel is None):
            raise ValueError("give exactly one of world and pixel")
        return self


class PixelBox(ToolModel):
    """A box in pixels, corners (u0, v0) and (u1, v1)."""

    label: Annotated[str, Field(max_length=24, description="A short label drawn above it.")] = ""
    u0: FiniteFloat
    v0: FiniteFloat
    u1: FiniteFloat
    v1: FiniteFloat


class Overlay(ToolModel):
    """What to draw on an image."""

    robot: Annotated[
        bool, Field(description="TCP dots; in fixed views also the ee-TCP lines and TCP axes.")
    ] = True
    grid: Annotated[bool, Field(description="The table grid, once the table is known.")] = True
    points: list[Point] = Field(default_factory=list, max_length=20, description="Points to mark.")
    boxes: list[PixelBox] = Field(default_factory=list, max_length=20, description="Boxes to draw.")
    frames: Annotated[
        bool,
        Field(
            description="The registered objects' and parts' frames (x red, y green, z blue; "
            "where the shape leaves x open, the candidate x arrows labeled A, B, ...) and the "
            "world's x and y at the robot base."
        ),
    ] = False


def render_observed(result: Observed) -> list[ContentPart]:
    lines = [f"Observation {result.frame} at {_steps(result.step, result.step_limit)}."]
    lines += [_arm_line(arm, report) for arm, report in result.arms.items()]
    if result.objects:
        lines.append("Registered objects:")
        lines += [f"- {describe(report)}" for report in result.objects]
    if result.overlays:
        lines.append("Overlays: " + "; ".join(result.overlays) + ".")
    lines.append("Images: " + ", ".join(view.camera for view in result.views) + ".")
    images = [ImagePart(blob=view.image, label=view.camera) for view in result.views]
    return [TextPart(text="\n".join(lines)), *images]


def render_image(result: ImageShown) -> list[ContentPart]:
    drawn = "; ".join(result.overlays) or "no overlays"
    text = f"{result.camera} in {result.frame} ({drawn})."
    return [TextPart(text=text), ImagePart(blob=result.image, label=result.camera)]


def render_robot(result: RobotReport) -> list[ContentPart]:
    lines = [f"Robot at {_steps(result.step, result.step_limit)}."]
    for arm, report in result.arms.items():
        lines.append(_arm_line(arm, report))
        if report.joints is not None:
            lines.append(f"  {arm} joints measured {_numbers(report.joints)}")
        if report.joints_cmd is not None:
            lines.append(f"  {arm} joints commanded {_numbers(report.joints_cmd)}")
    return [TextPart(text="\n".join(lines))]


class Sensing(Toolset):
    """Cameras and proprioception of the episode's env."""

    name = "sensing"
    Config = SensingConfig
    config: SensingConfig

    def __init__(self) -> None:
        self._profiles: dict[str, EmbodimentProfile] = {}

    @tool(kind="perception", tier="D", render=render_observed)
    async def observe(
        self,
        ctx: ToolContext,
        views: Annotated[
            list[str] | None, Field(description="Cameras to look through; all when null.")
        ] = None,
        overlay: Annotated[
            bool, Field(description="Draw the TCPs, their axes and the table grid.")
        ] = True,
        frames: Annotated[
            bool, Field(description="Also draw the registered objects' frames (see show_frames).")
        ] = False,
    ) -> Observed:
        """Look now: a new frame from the cameras with the robot state. Overlays: TCP dots (left
        cyan, right orange); in fixed views a line from the ee to the TCP and the TCP axes (x red,
        y green, z blue); a table grid labeled in meters once the table is known."""
        profile = self._profile(ctx)
        cameras = _cameras(profile, views)
        request = ObserveRequest(cameras=cameras, depth=True)
        observation = await ctx.env.observe(ctx.env_episode, request)
        await ctx.frames.put(observation)
        drawn = Overlay(frames=frames) if overlay else None
        grid = _grid(ctx, drawn)
        shown = [
            View(camera=name, image=await self._image(ctx, observation, name, drawn, grid))
            for name in cameras
        ]
        registry = await ctx.frames.registry()
        return Observed(
            frame=observation.frame_id,
            step=observation.step,
            step_limit=observation.robot.step_limit,
            arms=self._arms(observation.robot, profile),
            views=shown,
            overlays=self._legend(drawn, grid),
            objects=[object_report(r, observation.robot) for r in registry.objects.values()],
        )

    @tool(kind="pure", tier="D", render=render_image)
    async def get_image(
        self,
        ctx: ToolContext,
        camera: Annotated[str, Field(description="One of the cameras `observe` lists.")],
        frame: Annotated[FrameRef, Field(description="A frame taken earlier, e.g. frame_2.")],
        overlay: Annotated[
            Overlay | None, Field(description="What to draw; the robot and grid when null.")
        ] = None,
    ) -> ImageShown:
        """One camera's image from an earlier frame, with overlays: the robot and table grid as in
        `observe`, and points (world coordinates projected with that frame's calibration, or
        pixels) and pixel boxes you give, in magenta."""
        observation = await _frame(ctx, frame)
        _check_camera(observation, camera)
        drawn = Overlay() if overlay is None else overlay
        grid = _grid(ctx, drawn)
        image = await self._image(ctx, observation, camera, drawn, grid)
        overlays = self._legend(drawn, grid)
        if drawn.points or drawn.boxes:
            overlays.append(f"in magenta {len(drawn.points)} point(s), {len(drawn.boxes)} box(es)")
        return ImageShown(frame=frame, camera=camera, image=image, overlays=overlays)

    @tool(kind="pure", tier="D", render=render_image)
    async def show_frames(
        self,
        ctx: ToolContext,
        objs: Annotated[
            list[ObjectRef] | None,
            Field(max_length=12, description="Registered objects; null: all of them."),
        ] = None,
        camera: Annotated[
            str | None, Field(description="A camera `observe` lists; null: the head camera.")
        ] = None,
        frame: Annotated[
            FrameRef | None, Field(description="A frame taken earlier; null: the newest.")
        ] = None,
    ) -> ImageShown:
        """The objects' frames drawn on a camera image, to read which way each one's axes point:
        x red, y green, z blue from the object's origin (and its registered parts'), labeled with
        its id; where the shape leaves x open, the candidate x arrows labeled A, B (C, D), A the
        current one (`orient` picks another); the world's x and y at the robot base."""
        profile = self._profile(ctx)
        chosen = frame if frame is not None else _newest(ctx)
        name = camera or next((c for c, m in profile.cameras.items() if m.kind == "head"), "")
        observation = await _frame(ctx, chosen)
        _check_camera(observation, name)
        drawn = Overlay(robot=False, grid=False, frames=True)
        image = await self._image(ctx, observation, name, drawn, None, objs)
        return ImageShown(
            frame=chosen, camera=name, image=image, overlays=self._legend(drawn, None)
        )

    @tool(kind="pure", tier="D")
    async def get_camera(
        self,
        ctx: ToolContext,
        camera: Annotated[str, Field(description="One of the cameras `observe` lists.")],
        frame: Annotated[FrameRef, Field(description="A frame taken earlier, e.g. frame_2.")],
    ) -> CameraReport:
        """A camera's calibration in a frame: image size, intrinsics (pixels) and its pose in the
        world with OpenCV axes (x right, y down, z along the view). Wrist cameras move with their
        arm, so every frame has its own."""
        observation = await _frame(ctx, frame)
        _check_camera(observation, camera)
        model = observation.cameras[camera].camera
        (fx, _, cx), (_, fy, cy), _ = model.K
        return CameraReport(
            camera=camera,
            frame=frame,
            width=model.width,
            height=model.height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            pose=Pose.from_matrix(model.cam2world),
        )

    @tool(kind="perception", tier="D", render=render_robot)
    async def get_robot_state(self, ctx: ToolContext) -> RobotReport:
        """The robot now: per arm the measured ee and TCP poses, the joints (measured and
        commanded), the gripper's opening and command and whether it holds something; the
        step count and limit."""
        robot = await ctx.env.robot(ctx.env_episode)
        arms = self._arms(robot, self._profile(ctx), joints=True)
        return RobotReport(step=robot.step, step_limit=robot.step_limit, arms=arms)

    def _profile(self, ctx: ToolContext) -> EmbodimentProfile:
        name = ctx.context.embodiment
        if name is None:
            raise LookupError(f"episode {ctx.episode_id} has no OpenEpisode.embodiment")
        if name not in self._profiles:
            self._profiles[name] = load_embodiment(name)
        return self._profiles[name]

    async def _image(
        self,
        ctx: ToolContext,
        observation: Observation,
        camera: str,
        overlay: Overlay | None,
        grid: TableGrid | None,
        objs: Sequence[str] | None = None,
    ) -> BlobRef:
        """The camera's RGB as PNG, with `overlay` drawn (its frames: of `objs`, else of every
        registered object); without one, the frame's own PNG."""
        captured = observation.cameras[camera]
        if overlay is None and _plain_png(captured.rgb):
            return captured.rgb  # the image as captured: nothing to encode again
        rgb = await _rgb(ctx.blobs, captured.rgb)
        if overlay is not None:
            fixed = self._profile(ctx).cameras[camera].kind == "head"
            marks = [Mark(p.label, p.world, p.pixel) for p in overlay.points]
            boxes = [Box(b.label, (b.u0, b.v0, b.u1, b.v1)) for b in overlay.boxes]
            axes: list[Axes] = []
            base = None
            if overlay.frames:
                axes = await frame_axes(ctx, observation.robot, objs)
                base = _base_point(self._profile(ctx), ctx)
            paint = partial(
                draw,
                rgb,
                captured.camera,
                robot=observation.robot if overlay.robot else None,
                axes_m=self.config.axes_m if fixed else None,
                grid=grid,
                grid_m=self.config.grid_m,
                marks=marks,
                boxes=boxes,
                axes=axes,
                base=base,
            )
            rgb = await anyio.to_thread.run_sync(paint)
        return await ctx.blobs.aput_image(rgb, meta={"camera": camera})

    def _arms(
        self, robot: RobotState, profile: EmbodimentProfile, joints: bool = False
    ) -> dict[Arm, ArmReport]:
        return {
            arm: _arm(state, profile.arms[arm].gripper, joints) for arm, state in robot.arms.items()
        }

    def _legend(self, overlay: Overlay | None, grid: TableGrid | None) -> list[str]:
        if overlay is None:
            return []
        legend = []
        if overlay.robot:
            legend.append("TCP dots (left cyan, right orange)")
            legend.append(
                f"in fixed views a line from the ee to the TCP and the TCP axes, "
                f"{self.config.axes_m:g} m (x red, y green, z blue)"
            )
        if grid is not None:
            legend.append(
                f"table grid every {self.config.grid_m:g} m on the table plane "
                f"(z {grid.z:.3f} m), labeled in meters"
            )
        if overlay.frames:
            legend.append(
                "object and part frames labeled by id (x red, y green, z blue; where x is open, "
                "candidate x arrows A, B, ..., A the current one), the world's x and y at the base"
            )
        return legend


def _arm(state: ArmState, gripper: Gripper, joints: bool) -> ArmReport:
    closed = state.gripper_cmd < (gripper.open + gripper.closed) / 2
    holding = (
        None if state.gripper is None else closed and state.gripper >= gripper.holding_threshold
    )
    return ArmReport(
        ee=state.ee,
        tcp=state.tcp,
        gripper=state.gripper,
        gripper_cmd=state.gripper_cmd,
        holding=holding,
        joints=state.joints if joints else None,
        joints_cmd=state.joints_cmd if joints else None,
    )


def _arm_line(arm: str, report: ArmReport) -> str:
    pointing = quat_to_matrix(report.tcp.q)[:, 0]
    opening = "not measured" if report.gripper is None else f"{report.gripper:.3f}"
    holding = {True: "holding something", False: "holding nothing", None: "holding unknown"}
    return (
        f"{arm} arm: TCP {_numbers(report.tcp.p)}, q {_numbers(report.tcp.q)}, pointing "
        f"{_numbers(pointing)}; ee {_numbers(report.ee.p)}; gripper command "
        f"{report.gripper_cmd:.2f}, opening {opening}, {holding[report.holding]}"
    )


def _steps(step: int, limit: int | None) -> str:
    if limit is None:
        return f"step {step} (no step limit)"
    return f"step {step} of {limit} ({limit - step} left)"


def _numbers(values: Sequence[float] | NDArray[Any]) -> str:
    return "(" + ", ".join(f"{round(float(v), 3) + 0.0:.3f}" for v in values) + ")"


def _cameras(profile: EmbodimentProfile, views: list[str] | None) -> list[str]:
    known = list(profile.cameras)
    if views is None:
        return known
    if unknown := [name for name in views if name not in profile.cameras]:
        raise ToolError(f"no camera {', '.join(unknown)}; the cameras are {', '.join(known)}")
    return list(dict.fromkeys(views))


async def _frame(ctx: ToolContext, frame: str) -> Observation:
    try:
        return await ctx.frames.get(frame)
    except LookupError:
        raise ToolError("that frame is not in this episode; use one from observe") from None


async def frame_axes(
    ctx: ToolContext, robot: RobotState, objs: Sequence[str] | None = None
) -> list[Axes]:
    """The frames of `objs` (every registered object when None) and their registered parts as
    they are with `robot`, for `overlay.draw`: an open x as its candidate turns."""
    registry = await ctx.frames.registry()
    wanted = list(registry.objects) if objs is None else list(objs)
    missing = [o for o in wanted if o not in registry.objects]
    if missing:
        raise ToolError(f"not registered in this episode: {', '.join(missing)}")
    found = []
    for obj in wanted:
        record = registry.objects[obj]
        now = object_geometry(record, robot)
        length = float(np.clip(0.6 * max(now.size[:2]), 0.04, 0.15))
        open_x = record.direction is not None and record.direction.weak
        found.append(
            Axes(f"{obj}", pose_to_matrix(now.pose), length, record.turns if open_x else ())
        )
        for name, part in now.parts.items():
            if name == WHOLE:
                continue
            short = float(np.clip(0.8 * max(part.size[:2]), 0.03, 0.08))
            turns = (0.0, 180.0) if len(part.candidates) > 1 else ()
            found.append(Axes(f"{obj}.{name}", pose_to_matrix(part.pose), short, turns))
    return found


def _base_point(profile: EmbodimentProfile, ctx: ToolContext) -> tuple[float, float, float] | None:
    """The robot base (the arms' planner bases' mean) at the table's height, if known."""
    bases = [arm.planner.base.p for arm in profile.arms.values() if arm.planner is not None]
    if not bases:
        return None
    x, y, z = np.mean(np.asarray(bases, dtype=float), axis=0)
    table = ctx.frames.fact(TABLE)
    return float(x), float(y), float(z if table is None else table["z"])


def _newest(ctx: ToolContext) -> FrameRef:
    """The newest frame the episode took, else the last one it was given."""
    entries = ctx.frames.entries()
    newest = ctx.frames.live() or (entries[-1] if entries else None)
    if newest is None:
        raise ToolError("no frame yet: observe first")
    return newest.alias


def _grid(ctx: ToolContext, overlay: Overlay | None) -> TableGrid | None:
    """The table to draw the grid on: the store's table estimate, if there is one."""
    table = ctx.frames.fact(TABLE)
    if overlay is None or not overlay.grid or table is None:
        return None
    x0, x1, y0, y1 = table["extent"]
    return TableGrid(z=float(table["z"]), extent=(x0, x1, y0, y1))


def _check_camera(observation: Observation, camera: str) -> None:
    if camera not in observation.cameras:
        cameras = ", ".join(observation.cameras)
        raise ToolError(f"that frame has no image from {camera!r}; it has {cameras}")


def _plain_png(ref: BlobRef) -> bool:
    """An RGB PNG (as `put_frame` stores frames), which models can be shown as it is."""
    return ref.media_type == IMAGE_TYPES["png"] and ref.shape is not None and ref.shape[2:] == (3,)


async def _rgb(blobs: BlobStore, ref: BlobRef) -> NDArray[np.uint8]:
    array = await (blobs.aget_array(ref) if ref.media_type == NPY else blobs.aget_image(ref))
    return np.ascontiguousarray(array[..., :3])
