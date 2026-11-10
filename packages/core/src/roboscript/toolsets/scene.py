"""What the perception, geometry and worldstate toolsets share (docs/concepts/perception.md,
docs/concepts/agent-runtime.md): one config, how a frame is analyzed, how pixel regions
are named, the object reports, the rules on what an Actor may change, and the grasp relations every
action tool applies.

- Frames: a tool's `frame` is an alias the Hub resolves (`frame_3`), or null for the newest.
- Cameras: a tool's `camera` is one the frame has, or null for the embodiment's head camera (the
  frame's anchor camera without one).
- Analysis (`SceneAccess.geometry`): the frame's views with the robot's pixels removed (the
  self-filter, from the motion planner's forward kinematics at the frame's measured joints, when
  `planner` is set; a robot the cameras never see, like MockEnv's, needs none) and the episode's
  table. The first frame analyzed fixes the table as the episode's `TABLE` fact (the one-time
  identification analyzes the initial scene first); every later analysis reuses it, so the table
  and its landmarks never move. The toolsets of an episode share its last few analyses and their
  proposals (`FrameStore.derived`, per pipeline config).
- Regions (`region_12` for models): pixels of a frame, named by a small JSON document: one of the
  frame's proposals (every view; a frame and the fixed table always give the same proposals), a
  box in one camera, or a mask in the blob store (`segment`'s).
- The registry is the episode's, in its frame store: tools read a copy (`frames.registry()`) and
  change it under the episode's lock (`frames.edit_registry()`). Inner steps go to the log as
  `perception.step` events.
- An Actor's registry changes only through the open-time identification, the grasp relations
  (`after_action`), an identity correction on the initial frame (`register` with `replaces` and
  a candidate region the identification showed) and parts of registered objects measured on the
  initial frame (`locate_part`, `register_part`); its `ground` and `segment` look at the initial
  frame only (`SceneAccess.initial`). The open-time identification leaves the `IDENTIFIED` fact.
"""

import functools
import hashlib
from collections import OrderedDict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import anyio
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError, model_validator
from typing_extensions import Self

from roboscript import rpc
from roboscript.contracts import motion
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.env import ObserveRequest
from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Pose, Vec3
from roboscript.contracts.ids import FrameId
from roboscript.contracts.motion import Planner
from roboscript.contracts.observation import Observation
from roboscript.contracts.receipt import ActionReceipt, GripEvent
from roboscript.contracts.robot import Arm, RobotState
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.manipulation.specs import HingeSpec
from roboscript.perception.config import PerceptionConfig
from roboscript.perception.criteria import DoneCheckConfig
from roboscript.perception.fit import vec3
from roboscript.perception.frames import CameraImage, Frame, Frames
from roboscript.perception.grounding import ProposalSource
from roboscript.perception.lift import box_mask
from roboscript.perception.locate import StepLog
from roboscript.perception.measure import (
    SIDES,
    HingeEstimate,
    Measured,
    Reason,
    Side,
    TurnEstimate,
    hinge,
    look_pose,
    measure,
    relocate,
    turn_offset,
)
from roboscript.perception.overlay import som_overlay
from roboscript.perception.parts import Part, PartKind
from roboscript.perception.prior import PartPrior
from roboscript.perception.proposals import Proposal, propose
from roboscript.perception.registry import (
    ObjectRecord,
    PartRecord,
    SceneRegistry,
    Step,
    object_geometry,
    silent,
)
from roboscript.perception.robot import RobotBody, SelfFilter, robot_body
from roboscript.perception.table import Table
from roboscript.perception.views import FrameGeometry, View, analyze
from roboscript.perception.vlm import VlmConfig
from roboscript.rpc import read_ready
from roboscript.settings import repo_root
from roboscript.spatial.camera import project
from roboscript.tools import (
    IDENTIFIED,
    TABLE,
    ObjectRef,
    RegionRef,
    SharedConfig,
    ToolContext,
    ToolsetRuntime,
)

KEEP: Final = 4
"""Analyses (with their proposals) an episode keeps."""
VISIBLE: Final = 0.01
"""A point shows in a camera when the depth at its pixel is its own within this (meters)."""


class SceneConfig(SharedConfig):
    """The config of the perception, geometry and worldstate toolsets
    (`configs/scene.example.yaml`): the pipeline's knobs, the services and the VLM backends. The
    three share it (a Hub refuses them with different ones), so that a region id names the same
    pixels in each. What holding means comes from the embodiment profile
    (`Gripper.holding_threshold`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pipeline: PerceptionConfig = PerceptionConfig()
    planner: str | None = None
    """URL of the motion planner service, whose forward kinematics place the robot for the
    self-filter; null: no self-filter (for a robot the cameras do not see)."""
    planner_ready_file: Path | None = None
    """In place of `planner`: the ready file of the motion service a stack starts for the run
    (`roboscript.stack.ready_path`), read when the toolset starts, so the service starts first;
    relative to the repository root."""
    vision: str | None = None
    """URL of the vision service (GroundingDINO, SAM 3), which must share the Hub's blob store;
    null: no cross-check and no `grounding_dino` or `sam3` backend."""
    vlm: VlmConfig = VlmConfig()
    report_ends: bool = True
    """`locate_part` reports the object's two ends along its long side with their heights where
    they differ (`PartReport.ends`), so a front can be named by height (`x_toward: low_end`).
    False: no such line."""
    done: DoneCheckConfig = DoneCheckConfig()
    """How `check_done` holds the done criteria to what they can tell apart."""

    @model_validator(mode="after")
    def _one_planner(self) -> Self:
        if self.planner is not None and self.planner_ready_file is not None:
            raise ValueError("set planner or planner_ready_file, not both")
        return self

    def planner_url(self) -> str | None:
        """The motion service's URL: `planner`, or the one its ready file names."""
        path = self.planner_ready_file
        if path is None:
            return self.planner
        ready = read_ready(path if path.is_absolute() else repo_root() / path)
        if ready is None:
            raise ValueError(f"no motion service has written {path}: start it first")
        return ready.url_for(Planner.PLANE)


class RegionKey(BaseModel):
    """What a region id names: a proposal of a frame, a box (continuous pixel coordinates) in one
    of its cameras, or a mask blob of one camera."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame: FrameId
    proposal: PositiveInt | None = None
    camera: str | None = None
    box: tuple[float, float, float, float] | None = None
    mask: BlobRef | None = None

    @property
    def id(self) -> str:
        return self.model_dump_json(exclude_none=True)


@dataclass(frozen=True, eq=False)
class Region:
    """A region's pixels (camera -> mask) and the analysis of its frame."""

    key: RegionKey
    geometry: FrameGeometry
    masks: dict[str, NDArray[np.bool_]]


class IdentifiedRole(BaseModel):
    """A role of the open-time identification: its object (None: none), why, the regions of
    the objects its backends showed (candidates for an identity correction) and, for a plural
    role (`order: rest`), all its objects in its order (`object` the first; empty otherwise)."""

    object: str | None
    reason: str
    candidates: list[str] = []
    instances: list[str] = []


class IdentifiedFact(BaseModel):
    """The `IDENTIFIED` fact: the initial frame the objects were identified on, and the roles."""

    frame: FrameId
    roles: dict[str, IdentifiedRole]


class ShownReport(BaseModel):
    """An object an identification backend showed: as a region (None: nothing to show), the
    backend, and the center of its points."""

    backend: str
    region: RegionRef | None
    center: Vec3 | None


class ObjectReport(BaseModel):
    """A registered object as it is now: center (footprint center, halfway up), top center, size
    along its box's axes, the yaw of its long side, shape and symmetry guesses, the confidence of
    its identity (`ambiguous`: its role's backends showed other objects, `candidates`), the arms
    holding it, how many steps old its measurement is, and how many ways its frame's x may point
    (`x_open`: more than one where the shape leaves it open; `show_frames`, `orient`)."""

    obj: ObjectRef
    name: str
    center: Vec3
    top_center: Vec3
    size: Vec3
    yaw_deg: float
    shape: str
    symmetry: str
    confidence: float
    ambiguous: bool = False
    candidates: list[ShownReport] = []
    held_by: list[Arm] = []
    stale_age: int = 0
    partially_visible: bool = False
    x_open: int = 1


PartRoute = Literal["only", "mark", "prior", "box"]
"""How `locate_part` found a part: the one candidate of its kind, the VLM's choice among the
candidates, where its prior says (the one candidate there, the nearest of several the VLM named
none of, or the object's points in the prior's box), or the VLM's box (a moving part none of the
candidates is)."""


class EndReport(BaseModel):
    """An end of an object's long side: its point (world) and how tall the object stands
    there."""

    center: Vec3
    height_m: float


class PartReport(BaseModel):
    """A registered part: its frame's origin in the world (the center of its box), its size along
    the frame's x, y and z, what x points toward (both ways when nothing decides it), its point
    count and, from `locate_part`, the kinds of part candidate it is, the route that found it
    and its prior (where it lies on the object, for the Memory to keep)."""

    obj: ObjectRef
    part: str
    center: Vec3
    size: Vec3
    x_toward: str | None
    both_ways: bool
    points: int
    kinds: list[PartKind] = []
    route: PartRoute | None = None
    prior: PartPrior | None = None
    ends: list[EndReport] = []
    """From `locate_part` (`SceneConfig.report_ends`): the object's two ends along its long
    side, lower first, where their heights tell them apart (`ObjectProxy.end_heights`)."""


def object_report(record: ObjectRecord, robot: RobotState | None) -> ObjectReport:
    now = object_geometry(record, robot)
    return ObjectReport(
        obj=record.id,
        name=record.name,
        center=vec3(now.center),
        top_center=vec3(now.top_center),
        size=vec3(now.size),
        yaw_deg=round(now.yaw_deg, 1),
        shape=now.shape,
        symmetry=now.symmetry,
        confidence=record.confidence,
        ambiguous=record.ambiguous,
        candidates=[
            ShownReport(backend=c.backend, region=c.region, center=c.center)
            for c in record.candidates
        ],
        held_by=list(now.held_by),
        stale_age=now.stale_age,
        partially_visible=now.partially_visible,
        x_open=len(record.turns) if record.direction and record.direction.weak else 1,
    )


ROUTE_WORDS: Final[dict[PartRoute, str]] = {
    "only": "the only candidate of its kind",
    "mark": "the VLM's choice among the candidates",
    "prior": "where its prior says",
    "box": "the VLM's box, no candidate matching",
}


def render_part(result: PartReport) -> list[ContentPart]:
    if result.both_ways:
        toward = "both ways"
    elif result.x_toward is not None:
        toward = f"toward {result.x_toward}"
    else:
        toward = "away from the object's body"
    size = " x ".join(f"{s:.3f}" for s in result.size)
    kind = f", a {' and '.join(result.kinds)} candidate" if result.kinds else ""
    route = f" (found by {ROUTE_WORDS[result.route]})" if result.route else ""
    prior = ""
    if result.prior is not None:
        at, extent = (_xyz(v) for v in (result.prior.at, result.prior.size))
        prior = f" Prior: at {at}, size {extent} of the object's box."
    text = (
        f"{result.obj} part {result.part}: center {_xyz(result.center)}, {size} m, x {toward}, "
        f"{result.points} points{kind}{route}.{prior}"
    )
    if len(result.ends) == 2:
        low, high = result.ends
        text += (
            f" The object's long side ends lower at {_xyz(low.center)} ({100 * low.height_m:.1f} "
            f"cm tall) and higher at {_xyz(high.center)} ({100 * high.height_m:.1f} cm): a spec's "
            "x_toward low_end or high_end names one of them by height alone, the same in every "
            "measurement."
        )
    return [TextPart(text=text)]


class MeasuredReport(BaseModel):
    """An object (or a part of it) measured again in `frame`: its frame (world, the registry's
    convention), size, how far it moved from where the registry had it, the confidence, why it
    is (not) usable (`ok`, `not_visible`, `low_confidence`, `inconsistent`: its footprint does
    not continue the old one, `held`: an arm holds it, and it follows the grasp, `relocated`:
    gone from where it was, found nearby, the confidence its size agreement), whether the
    registry now has it, the cameras that saw it and their points."""

    obj: ObjectRef
    part: str | None = None
    frame: FrameId
    pose: Pose
    size: Vec3
    shift_m: float
    confidence: float
    reason: Reason
    stored: bool
    cameras: list[str]
    points: int


class SceneMeasure(Protocol):
    """Measuring registered objects again, for the manipulation toolset's coarse-then-fine
    actions (`SceneAccess` implements it)."""

    async def remeasure(
        self, ctx: ToolContext, obj: str, *, frame: str | None = None, store: bool = True,
        cameras: list[str] | None = None,
    ) -> MeasuredReport: ...  # fmt: skip

    async def refine(
        self, ctx: ToolContext, target: str, arm: Arm, *, part: str | None = None,
        store: bool = True,
    ) -> MeasuredReport: ...  # fmt: skip

    async def look_pose(self, ctx: ToolContext, obj: str, arm: Arm) -> Pose: ...

    async def hinge(
        self, ctx: ToolContext, obj: str, part: str, side: Side | None = None
    ) -> HingeEstimate: ...

    async def turn_offset(
        self, ctx: ToolContext, obj: str, part: str, hinge: tuple[Vec3, Vec3], span_deg: float,
        arm: Arm,
    ) -> TurnEstimate: ...  # fmt: skip


def measured_report(
    obj: str, found: Measured, robot: RobotState | None, stored: bool
) -> MeasuredReport:
    """A measurement's report (`remeasure` with its `frame` and `store` stores the same one
    later: the measurement of a frame is deterministic)."""
    now = object_geometry(found.record, robot)
    return MeasuredReport(
        obj=obj,
        frame=found.record.frame,
        pose=Pose.from_list(now.pose),
        size=vec3(now.size),
        shift_m=round(found.shift_m, 4),
        confidence=found.confidence,
        reason=found.reason,
        stored=stored,
        cameras=list(found.cameras),
        points=found.points,
    )


def hinge_spec(found: HingeEstimate) -> HingeSpec:
    """The hinge `rotate_about` takes, from an estimate."""
    return HingeSpec(part=found.part, side=found.side, at=found.at)


def render_measured(result: MeasuredReport) -> list[ContentPart]:
    what = result.obj if result.part is None else f"{result.obj} part {result.part}"
    if result.reason == "not_visible":
        return [TextPart(text=f"{what}: not seen where the registry has it; nothing changed.")]
    if result.reason == "relocated":
        size = " x ".join(f"{s:.3f}" for s in result.size)
        kept = "registered there" if result.stored else "not registered (not sure)"
        text = (
            f"{what}: not where the registry had it; found {1000 * result.shift_m:.0f} mm away "
            f"(knocked or fallen), center {_xyz(result.pose.p)}, {size} m, confidence "
            f"{result.confidence:.2f}, {kept}."
        )
        return [TextPart(text=text)]
    if result.reason == "held":
        return [TextPart(text=f"{what}: held, so it moves with the gripper; not measured.")]
    size = " x ".join(f"{s:.3f}" for s in result.size)
    kept = "registered" if result.stored else f"not registered ({result.reason})"
    text = (
        f"{what}: center {_xyz(result.pose.p)}, {size} m, moved {1000 * result.shift_m:.0f} mm, "
        f"confidence {result.confidence:.2f}, {kept}; seen by {', '.join(result.cameras)}."
    )
    return [TextPart(text=text)]


BACKENDS: Final = {
    "som": "Set-of-Mark",
    "vlm": "the VLM's box",
    "grounding_dino": "GroundingDINO",
    "compare": "the look-alike comparison",
    "layout": "the look-alikes' order on the table, nearest first",
    "measured": "the size or position the instruction names, measured",
}


def describe(report: ObjectReport) -> str:
    """One line on an object for a model (ids already aliased)."""
    size = " x ".join(f"{s:.3f}" for s in report.size)
    text = f"{report.obj} {report.name}: center {_xyz(report.center)}, {size} m"
    text += f", yaw {report.yaw_deg:.1f} deg"
    if report.held_by:
        text += f", held by {' and '.join(report.held_by)}"
    if report.stale_age:
        text += f", measured {report.stale_age} steps ago"
    if report.partially_visible:
        text += ", partly outside the image (center may be ~1 cm off)"
    if report.x_open > 1:
        text += f", frame x open ({report.x_open} ways)"
    if report.ambiguous:
        others = "; ".join(shown_text(c) for c in report.candidates[1:]) or "another object"
        text += f", identity ambiguous (confidence {report.confidence:.1f}; {others})"
    elif report.confidence < 1:
        text += f", identity unconfirmed (confidence {report.confidence:.1f})"
    return text


def shown_text(shown: ShownReport) -> str:
    """What a backend showed, for a model: `GroundingDINO shows region_7 at (x, y, z)`."""
    where = "" if shown.center is None else f" at {_xyz(shown.center)}"
    return f"{BACKENDS.get(shown.backend, shown.backend)} shows {shown.region or 'nothing'}{where}"


class SceneAccess:
    """A toolset's access to frames, analyses, regions and the registry (see the module)."""

    def __init__(self, config: SceneConfig, runtime: ToolsetRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self.planner: Planner | None = None
        url = config.planner_url()
        if url is not None:
            self.planner = rpc.client(
                Planner,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
                url,
                token=None,
                timeout=10,
                timeouts=motion.TIMEOUTS,
            )
        analysis = config.pipeline.model_dump_json() + (url or "")
        self.key = hashlib.sha256(analysis.encode()).hexdigest()[:16]
        self._bodies: dict[str, RobotBody] = {}

    async def close(self) -> None:
        if self.planner is not None:
            await rpc.aclose(self.planner)

    def log(self, ctx: ToolContext) -> Step:
        events = self.runtime.events
        return silent if events is None else StepLog(events, ctx.episode_id)

    async def frame(self, ctx: ToolContext, ref: str | None) -> Frame:
        """A frame of the episode by id, or the newest."""
        try:
            return await Frames(ctx.frames, ctx.blobs).frame(ref or "live")
        except LookupError as error:
            raise ToolError(str(error)) from None

    def camera(self, ctx: ToolContext, geometry: FrameGeometry, camera: str | None) -> str:
        """The camera a tool means: the one named (it must be the frame's), or the embodiment's
        head camera, or the frame's anchor camera."""
        if camera is None:
            profile = self.profile(ctx)
            cameras = {} if profile is None else profile.cameras
            heads = [name for name, spec in cameras.items() if spec.kind == "head"]
            camera = next((name for name in heads if name in geometry.views), geometry.anchor)
        check_camera(geometry, camera)
        return camera

    async def geometry(self, ctx: ToolContext, frame: Frame) -> FrameGeometry:
        """The frame's analysis; the first one fixes the episode's table."""
        cache = self._cache(ctx, "geometry")
        if frame.id in cache:
            cache.move_to_end(frame.id)
            return cache[frame.id]  # type: ignore[no-any-return]
        fixed = stored_table(ctx)
        geometry = await self._analyze(ctx, frame, fixed)
        if stored_table(ctx) is None:
            ctx.frames.set_fact(TABLE, geometry.table)
            table = geometry.table
            self.log(ctx)(
                "table", frame=frame.id, z=table.z, edges=table.edges, method=table.method
            )
        cache[frame.id] = geometry
        while len(cache) > KEEP:
            cache.popitem(last=False)
        return geometry

    async def _analyze(self, ctx: ToolContext, frame: Frame, table: Table | None) -> FrameGeometry:
        """The frame's analysis with `table` (None: the frame's own), the robot filtered out."""
        profile = self.profile(ctx)
        shape = None
        if self.planner is not None:
            shape = await SelfFilter(
                self.planner, self._body(profile), self.config.pipeline.self_filter
            ).shape(frame.robot)
        return await analyze(frame, self.config.pipeline, shape, _base(profile), table=table)

    async def reference_table(self, ctx: ToolContext, observation: BlobRef) -> Table:
        """The table of another scene's frame, such as the demonstration's (an `Observation` in
        the blob store), found on that frame itself as an episode finds its own, with this
        config and the robot filtered out (`reference_table`)."""
        found = stored_reference(ctx, observation)
        if found is not None:
            return found
        seen = Observation.model_validate_json(await ctx.blobs.aget_bytes(observation))
        table = (await self._analyze(ctx, Frame(seen, ctx.blobs), None)).table
        self.log(ctx)("reference_table", frame=seen.frame_id, z=table.z, method=table.method)
        return keep_reference(ctx, observation, table)

    async def proposals(self, ctx: ToolContext, geometry: FrameGeometry) -> list[Proposal]:
        """The frame's proposals with this config, computed once per analysis."""
        cache = self._cache(ctx, "proposals")
        if geometry.frame not in cache:
            cache[geometry.frame] = await anyio.to_thread.run_sync(
                propose, geometry, self.config.pipeline
            )
            while len(cache) > KEEP:
                cache.popitem(last=False)
        return cache[geometry.frame]  # type: ignore[no-any-return]

    def source(self, ctx: ToolContext) -> ProposalSource:
        """The proposals as a backend asks for them: this toolset's, never another config's."""
        return functools.partial(self.proposals, ctx)

    async def region(self, ctx: ToolContext, region_id: str) -> Region:
        try:
            key = RegionKey.model_validate_json(region_id)
        except ValidationError:
            raise ToolError("that is not a region of this episode") from None
        geometry = await self.geometry(ctx, await self.frame(ctx, key.frame))
        if key.proposal is not None:
            proposals = await self.proposals(ctx, geometry)
            chosen = [p for p in proposals if p.id == key.proposal]
            if not chosen:
                raise ToolError("that is not a region of this episode")
            return Region(key, geometry, chosen[0].masks)
        assert key.camera is not None  # a box or a mask belongs to one camera
        if key.box is not None:
            shape = geometry.view(key.camera).image.shape
            return Region(key, geometry, {key.camera: box_mask(key.box, shape)})
        assert key.mask is not None
        mask = np.asarray(await ctx.blobs.aget_array(key.mask), dtype=bool)
        return Region(key, geometry, {key.camera: mask})

    async def table(self, ctx: ToolContext, ref: str | None) -> Table:
        """The episode's table, fixed now on the frame `ref` names (the newest) if it is not yet."""
        fixed = stored_table(ctx)
        if fixed is not None:
            return fixed
        return (await self.geometry(ctx, await self.frame(ctx, ref))).table

    async def robot(self, ctx: ToolContext, ref: str | None) -> RobotState:
        """The robot in the frame `ref` names; else now (from the env) or, without an env, in
        the newest frame."""
        if ref is None and ctx.context.env is not None:
            return await ctx.env.robot(ctx.env_episode)
        return (await self.frame(ctx, ref)).robot

    def identified(self, ctx: ToolContext) -> IdentifiedFact | None:
        """The open-time identification of the episode, if there was one."""
        fact = ctx.frames.fact(IDENTIFIED)
        return None if fact is None else IdentifiedFact.model_validate(fact)

    def initial(self, ctx: ToolContext, frame: FrameId, what: str, newest: bool = False) -> None:
        """Refuse an Actor `what` on a frame other than the initial one (the identification's);
        with `newest`, the episode's newest own frame passes too (`is_newest`: to ground again
        an object no arm holds, dropped or never taken, where it lies now)."""
        if ctx.context.role != "actor":
            return
        if newest and is_newest(ctx, frame):
            return
        done = self.identified(ctx)
        if done is None or done.frame != frame:
            where = "no frame" if done is None else "the initial frame, the one identified"
            raise ToolError(
                f"an Actor {what} only on the initial scene: use {where}; the objects are "
                "identified once and follow the grippers afterwards"
            )

    async def register_part(
        self,
        ctx: ToolContext,
        obj: str,
        part: str,
        region: Region | Part,
        frame: FrameId,
        x_toward: str | None,
        articulated: bool = False,
    ) -> PartReport:
        """Register as `part` of `obj` (seen in `frame`) the object's points that the region's
        cameras show inside its masks, or a part candidate's points, framed along the
        candidate's direction (`articulated`: it turns or slides on the body); not once an arm
        has taken the object (`unmoved`)."""
        self.initial(ctx, frame, "registers parts")
        robot = (await ctx.frames.get(frame)).robot
        async with ctx.frames.edit_registry() as registry:
            record = _object(registry, obj)
            unmoved(ctx, record)
            box, decided, kinds = None, False, []
            if isinstance(region, Part):
                inside, kinds = region.points, list(region.kinds)
                box, decided = (region.frame, region.extent), region.decided
            else:
                points = object_geometry(record, robot).points
                views, masks = region.geometry.views, region.masks
                inside = await anyio.to_thread.run_sync(part_points, points, views, masks)
            if len(inside) < 3:
                raise ToolError(
                    f"only {len(inside)} of the object's points lie in that region: box the part "
                    "tighter around where the object is, or look from another camera"
                )
            try:
                added = registry.register_part(
                    obj,
                    part,
                    inside,
                    x_toward=x_toward,
                    box=box,
                    decided=decided,
                    articulated=articulated,
                    robot=robot,
                    self_x_toward_unset=self.config.pipeline.frames.self_x_toward_unset,
                )
            except (LookupError, ValueError) as error:  # its message names the object by its id
                raise ToolError(str(error).replace(obj, "the object")) from None
            if articulated:  # x from the hinge toward the free edge, as `hinge` reads it
                found = hinge(registry, obj, part, robot=robot)
                if found.confidence >= self.config.pipeline.measure.hinge_frame:
                    sx, sy = SIDES[found.side]
                    added = registry.point_part(obj, part, (-sx, -sy, 0.0), robot)
                self.log(ctx)("part_hinge", object=obj, part=part, side=found.side,
                              confidence=found.confidence)  # fmt: skip
            pose = object_geometry(registry.get(obj), robot).parts[part].pose
        self.log(ctx)("register_part", object=obj, part=part, points=len(inside), kinds=kinds)
        return part_report(obj, added, pose, kinds)

    async def remeasure(
        self,
        ctx: ToolContext,
        obj: str,
        *,
        frame: str | None = None,
        store: bool = True,
        cameras: list[str] | None = None,
    ) -> "MeasuredReport":
        """`obj` measured again (`perception.measure`) in `frame`, else in a new observation of
        the scene as it is now; not seen where the registry has it, searched for nearby
        (`perception.measure.relocate`); stored when it is `ok`, or `relocated` and confident, and
        `store`. Its identity stays."""
        seen = await self._look(ctx, frame, cameras)
        geometry = await self.geometry(ctx, seen)
        async with ctx.frames.edit_registry() as registry:
            _object(registry, obj)
            run = functools.partial(
                measure, registry, geometry, obj, robot=seen.robot, cameras=cameras,
                pipeline=self.config.pipeline,
            )  # fmt: skip
            found = await anyio.to_thread.run_sync(run)
            if found.reason == "not_visible":  # gone from there, or hidden: search near it
                proposals = await self.proposals(ctx, geometry)
                search = functools.partial(
                    relocate, registry, geometry, obj, proposals, robot=seen.robot,
                    pipeline=self.config.pipeline,
                )  # fmt: skip
                found = await anyio.to_thread.run_sync(search)
            trust = self.config.pipeline.measure.trust
            stored = store and (
                found.reason == "ok" or (found.reason == "relocated" and found.confidence >= trust)
            )
            if stored:
                registry.objects[obj] = found.record
        report = measured_report(obj, found, seen.robot, stored)
        self.log(ctx)("remeasure", seen=seen.id, **report.model_dump(mode="json"))
        return report

    async def refine(
        self, ctx: ToolContext, target: str, arm: Arm, *, part: str | None = None,
        store: bool = True,
    ) -> "MeasuredReport":  # fmt: skip
        """`target` measured again from a new observation by the head camera and `arm`'s wrist
        camera (the arm near it), its parts fitted again; `part` is the one reported."""
        cameras = [c for c in self._cameras(ctx, arm) if c is not None]
        report = await self.remeasure(ctx, target, store=store, cameras=cameras)
        if part is None:
            return report
        registry = await ctx.frames.registry()
        record = _object(registry, target)
        if part not in record.parts:
            raise ToolError(f"the object has no part {part!r}: locate it first")
        robot = await self.robot(ctx, None)
        shape = object_geometry(record, robot).parts[part]
        return report.model_copy(
            update={"part": part, "pose": Pose.from_list(shape.pose), "size": vec3(shape.size)}
        )

    async def look_pose(self, ctx: ToolContext, obj: str, arm: Arm) -> Pose:
        """An ee pose of `arm` from which its wrist camera sees `obj` (`perception.measure`)."""
        wrist = self._cameras(ctx, arm)[1]
        if wrist is None:
            raise ToolError(f"the {arm} arm carries no camera")
        seen = await self.frame(ctx, None)
        geometry = await self.geometry(ctx, seen)
        check_camera(geometry, wrist)
        registry = await ctx.frames.registry()
        expected = object_geometry(_object(registry, obj), seen.robot)
        return look_pose(geometry, expected, arm, wrist, seen.robot, self.config.pipeline.measure)

    async def hinge(
        self, ctx: ToolContext, obj: str, part: str, side: Side | None = None
    ) -> HingeEstimate:
        """A hinge line of `part` on the rest of `obj` (`perception.measure.hinge`)."""
        registry = await ctx.frames.registry()
        robot = await self.robot(ctx, None)
        try:
            return hinge(registry, obj, part, side=side, robot=robot)
        except LookupError as error:
            raise ToolError(str(error).replace(obj, "the object")) from None

    async def observe(self, ctx: ToolContext) -> Frame:
        """A new observation of the scene as it is now (the episode needs its env)."""
        if ctx.context.env is None:
            raise ToolError("this episode has no env to observe")
        return await self._look(ctx, None, None)

    async def turn_offset(
        self, ctx: ToolContext, obj: str, part: str, hinge: tuple[Vec3, Vec3], span_deg: float,
        arm: Arm,
    ) -> TurnEstimate:  # fmt: skip
        """How far `part` lies turned about `hinge` from where the registry has it, in a new
        observation by the head camera and `arm`'s wrist camera
        (`perception.measure.turn_offset`)."""
        cameras = [c for c in self._cameras(ctx, arm) if c is not None]
        seen = await self._look(ctx, None, cameras)
        geometry = await self.geometry(ctx, seen)
        registry = await ctx.frames.registry()
        _object(registry, obj)
        run = functools.partial(
            turn_offset, registry, geometry, obj, part, hinge, span_deg, robot=seen.robot,
            config=self.config.pipeline.measure,
        )  # fmt: skip
        try:
            found = await anyio.to_thread.run_sync(run)
        except LookupError as error:
            raise ToolError(str(error).replace(obj, "the object")) from None
        self.log(ctx)("turn_offset", seen=seen.id, object=obj, **found.model_dump(mode="json"))
        return found

    async def _look(self, ctx: ToolContext, frame: str | None, cameras: list[str] | None) -> Frame:
        """The frame named, else a new observation (with the env), else the newest."""
        if frame is None and ctx.context.env is not None:
            observed = await ctx.env.observe(ctx.env_episode, ObserveRequest(cameras=cameras))
            frame = (await ctx.frames.put(observed)).frame_id
        return await self.frame(ctx, frame)

    def _cameras(self, ctx: ToolContext, arm: Arm) -> tuple[str | None, str | None]:
        """The embodiment's head camera and `arm`'s wrist camera (None: none)."""
        profile = self.profile(ctx)
        cameras = {} if profile is None else profile.cameras
        head = next((n for n, c in cameras.items() if c.kind == "head"), None)
        wrist = next((n for n, c in cameras.items() if c.kind == "wrist" and c.arm == arm), None)
        return head, wrist

    def profile(self, ctx: ToolContext) -> EmbodimentProfile | None:
        name = ctx.context.embodiment
        if name is None:
            if self.planner is not None:
                raise ToolError("the self-filter needs the episode's embodiment; none is set")
            return None
        return embodiment(name)

    def _cache(self, ctx: ToolContext, kind: str) -> "OrderedDict[FrameId, Any]":
        cache: OrderedDict[FrameId, Any] = ctx.frames.derived.setdefault(
            (kind, self.key), OrderedDict()
        )
        return cache

    def _body(self, profile: EmbodimentProfile | None) -> RobotBody:
        assert profile is not None  # `profile` refuses an episode without one
        if profile.name not in self._bodies:
            self._bodies[profile.name] = robot_body(profile)
        return self._bodies[profile.name]


async def after_action(
    ctx: ToolContext,
    receipt: ActionReceipt,
    *,
    targets: Mapping[Arm, str] | None = None,
    parts: Mapping[Arm, str] | None = None,
    presses: Collection[Arm] = (),
    pin: bool = True,
    square: bool = False,
) -> None:
    """The grasp relations after a physical action: the one post-action hook every action tool
    calls with its receipt. Each close that settled at or above the embodiment's holding
    threshold (`Gripper.holding_threshold`) attaches the object the grasp aimed at
    (`targets[arm]`, else the registered object nearest the TCP), or only its articulated part
    (`parts[arm]`, else the one the TCP is at), unless the close is a press (`presses`: closed
    fingers push and hold nothing), pinned in the frame the grasp was made in where a choice
    left its front open, or with the receipt's `GraspChoice.hold_frame` wherever its shape looks
    the same turned (`SceneRegistry.pin`, the receipt's `GraspChoice.frame_turn_deg`; not when
    `pin` is false), and with `square` a box turned about the TCP as flat pads square it against
    them (`SceneRegistry.attach`); each open and each lost hold detaches what the arm held and
    settles an object; in step order, with the robot where it was at each change
    (`receipt.tcp_path`), through `edit_registry`. Then each press the receipt reports with the
    gripper closed on nothing, or by an arm whose grasp was a press (`presses`: its approach runs
    with the fingers as they were), is kept on the object it met (`SceneRegistry.on_press`). Each
    decision is logged as a `<toolset>.step` event of the acting toolset."""
    if not (receipt.grips or receipt.presses) or ctx.context.embodiment is None:
        return

    def log(step: str, **data: Any) -> None:
        ctx.emit("step", {"step": step, **data})

    profile = embodiment(ctx.context.embodiment)
    final = await ctx.env.robot(ctx.env_episode)
    table = stored_table(ctx)

    def relate(registry: SceneRegistry) -> None:
        for event in receipt.grips:
            robot = _robot_at(final, receipt, event)
            gripper = profile.arms[event.arm].gripper
            threshold = gripper.holding_threshold
            if event.closed and event.opening is not None and event.opening >= threshold:
                target = (targets or {}).get(event.arm)
                aimed = target if target in registry.objects else None
                part = (parts or {}).get(event.arm) if aimed is not None else None
                chosen = receipt.grasp if receipt.arm == event.arm else None
                turn = None if chosen is None else chosen.frame_turn_deg
                registry.on_close(
                    event.arm,
                    robot,
                    threshold=threshold,
                    target=aimed,
                    part=part,
                    press=event.arm in presses,
                    turn_deg=turn or 0.0,
                    pin=pin,
                    hold_frame=chosen is not None and chosen.hold_frame,
                    square_axis=profile.poses.tcp_closing_axis if square else None,
                    closing_axis=profile.poses.tcp_closing_axis,
                    span_m=gripper.span_m,
                    log=log,
                )
            else:
                registry.on_open(event.arm, robot, table, log=log)
        for press in receipt.presses:
            if press.closed or press.arm in presses:  # an open grasp's approach is no press
                registry.on_press(press.arm, press.tcp, log=log)

    async with ctx.frames.edit_registry() as registry:
        if registry.objects:
            await anyio.to_thread.run_sync(relate, registry)


def is_newest(ctx: ToolContext, frame: str) -> bool:
    """Whether `frame` (id or alias) is the episode's newest own observation: the scene as it is
    now, camera data a real robot has."""
    live = ctx.frames.live()
    return live is not None and frame in (live.frame_id, live.alias)


def unmoved(ctx: ToolContext, record: ObjectRecord) -> None:
    """Refuse to measure a part of an object an arm has taken since it was measured
    (`ObjectRecord.moved`): the frames show it elsewhere than the registry has it."""
    if record.moved:
        fix = "an Actor measures parts before an arm takes the object"
        if ctx.context.role != "actor":
            fix = "refresh it first"
        raise ToolError(
            "an arm has taken that object since it was measured, so the frames no longer show "
            f"it where the registry has it: {fix}"
        )


def check_camera(geometry: FrameGeometry, camera: str) -> None:
    if camera not in geometry.views:
        raise ToolError(f"no camera {camera!r} in that frame; it has {', '.join(geometry.views)}")


REFERENCE_TABLE: Final = "reference_table"
"""The demonstration's table, found on its final frame (the `DONE` fact's `reference`):
`{"observation": <its blob id>, "table": <Table JSON>}`, what a lift and a release onto the table
are measured against (`worldstate.check_done`, `manipulation`'s robot-frame releases)."""


def stored_reference(ctx: ToolContext, observation: BlobRef) -> Table | None:
    fact = ctx.frames.fact(REFERENCE_TABLE)
    if fact is None or fact.get("observation") != observation.id:
        return None
    return Table.model_validate(fact["table"])


def keep_reference(ctx: ToolContext, observation: BlobRef, table: Table) -> Table:
    ctx.frames.set_fact(
        REFERENCE_TABLE, {"observation": observation.id, "table": table.model_dump(mode="json")}
    )
    return table


async def reference_table(
    ctx: ToolContext, observation: BlobRef, config: PerceptionConfig
) -> Table:
    """The demonstration's table as the episode knows it (`REFERENCE_TABLE`), else found on its
    frame `observation` with `config`, without a self-filter (a toolset with no planner of the
    scene's: the robot seldom stands within the plane's 3 mm band), and kept."""
    found = stored_reference(ctx, observation)
    if found is not None:
        return found
    seen = Observation.model_validate_json(await ctx.blobs.aget_bytes(observation))
    table = (await analyze(Frame(seen, ctx.blobs), config)).table
    return keep_reference(ctx, observation, table)


def stored_table(ctx: ToolContext) -> Table | None:
    fact = ctx.frames.fact(TABLE)
    return None if fact is None else Table.model_validate(fact)


def part_points(
    points: NDArray[np.float64],
    views: Mapping[str, View],
    masks: Mapping[str, NDArray[np.bool_]],
) -> NDArray[np.float64]:
    """The points (N, 3) some camera shows inside its mask: each one projected into the views
    whose mask is given, kept where it lands in the mask and the depth there is its own (not
    hidden behind something)."""
    keep = np.zeros(len(points), dtype=bool)
    for camera, mask in masks.items():
        view = views[camera]
        uv, z, front = project(points, view.image.K, view.image.cam2world)
        ok = front & np.isfinite(uv).all(axis=1)
        pixel = np.rint(uv[ok]).astype(np.int64)
        h, w = mask.shape
        inside = (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
        index = np.flatnonzero(ok)[inside]
        u, v = pixel[inside, 0], pixel[inside, 1]
        seen = mask[v, u] & (np.abs(view.depth[v, u] - z[index]) < VISIBLE)
        keep[index[seen]] = True
    return points[keep]


def part_report(
    obj: str, part: PartRecord, pose: NDArray[np.float64], kinds: list[PartKind] | None = None
) -> PartReport:
    """A part's report; `pose` is its frame in the world (7,); `kinds`: the part candidate's."""
    return PartReport(
        obj=obj,
        part=part.name,
        center=vec3(pose[:3]),
        size=part.size,
        x_toward=part.x_toward,
        both_ways=part.both_ways,
        points=len(part.points),
        kinds=kinds or [],
    )


def outline(image: CameraImage, objects: Mapping[str, NDArray[np.float64]]) -> NDArray[np.uint8]:
    """The image with each object's points (label -> world points) outlined and labeled, where
    they show in it."""
    h, w = image.shape
    masks, labels = {}, {}
    for number, (label, points) in enumerate(objects.items(), 1):
        uv, _, front = project(points, image.K, image.cam2world)
        pixel = np.rint(uv[front & np.isfinite(uv).all(axis=1)]).astype(np.int64)
        pixel = pixel[(pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0)]
        pixel = pixel[pixel[:, 1] < h]
        if not len(pixel):
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[pixel[:, 1], pixel[:, 0]] = True
        masks[number], labels[number] = mask, label
    drawn, _ = som_overlay(image.rgb, masks, labels=labels)
    return drawn


def _robot_at(final: RobotState, receipt: ActionReceipt, event: GripEvent) -> RobotState:
    """The robot at a grip event: each arm's TCP then (arms the action did not command stayed
    where they are now) and the event's arm's measured opening."""
    arms = dict(final.arms)
    for arm, path in receipt.tcp_path.items():
        if event.step < len(path):
            arms[arm] = arms[arm].model_copy(update={"tcp": path[event.step]})
    arms[event.arm] = arms[event.arm].model_copy(update={"gripper": event.opening})
    return final.model_copy(update={"arms": arms})


def _object(registry: SceneRegistry, obj: str) -> ObjectRecord:
    try:
        return registry.get(obj)
    except LookupError:
        raise ToolError("that object is not registered in this episode") from None


@functools.lru_cache(maxsize=8)
def embodiment(name: str) -> EmbodimentProfile:
    """An episode's embodiment profile by the name `OpenEpisode.embodiment` gives, loaded once."""
    return load_embodiment(name)


def _base(profile: EmbodimentProfile | None) -> NDArray[np.float64] | None:
    """Where the robot's base is (the robot-midline landmark), from the planner assets."""
    if profile is None:
        return None
    bases = [arm.planner.base.p for arm in profile.arms.values() if arm.planner is not None]
    return np.mean(bases, axis=0) if bases else None


def _xyz(values: Vec3) -> str:
    return "(" + ", ".join(f"{round(v, 3) + 0.0:.3f}" for v in values) + ")"
