"""The `perception` toolset (docs/concepts/perception.md,
docs/concepts/agent-runtime.md): find objects and parts.

- `identify` is system-only: the runner calls it as an Actor's or a demo-reading role's episode
  opens (`identify_objects`) and no model sees it. It is the one-time identification
  (`roboscript.perception.identify`) on a new observation of the initial scene, for the roles of
  the Memory's OBJECTS (abstract: what each does in the task, what kind of object it is) and the
  episode's live instruction, whose words decide which object plays each role; with no roles yet
  (a demo-reading role's first Memory), for the objects the instruction is about. Geometric
  proposals, one Set-of-Mark call for every role, GroundingDINO as a cross-check, one VLM box
  call for the roles left open or contested, then the three-way vote; the objects go into the
  episode's registry, and the `IDENTIFIED` fact keeps the initial frame and each role's
  candidate regions (what an Actor may correct an identity to).
- `ground`: candidates for a query in one camera, as regions, from a `roboscript.perception`
  backend: `vlm` (the VLM's box), `som` (its choice among the numbered proposals), `grounding_dino`
  or `geometric` (every proposal with its attribute card, no model).
- `segment`: a SAM 3 mask (for a text, a box, a point, or what a region shows), as a region.
- `locate` (a demo-reading role's): one more object by its description, through the same chain.
- `locate_part`: a part of a registered object among its geometric part candidates (by kind,
  else named by the VLM), registered in the candidate's frame; a moving part none of them is,
  boxed by the VLM instead.

An Actor grounds, segments and locates parts on the initial frame (its frame defaults to
it); the objects follow the grippers afterwards (`roboscript.toolsets.scene.after_action`).
It grounds and segments on the newest frame too, to find again an object no arm holds (one it
dropped or never took) and register it again (`worldstate.register` with `replaces`).
The VLM backends reach the model through the Hub's model factory (`ToolsetRuntime.models`) in a
gateway session of their own, `<episode id>.perception` (the episode's own session is its
brain's, which alone holds its permits key); GroundingDINO and SAM 3 need the vision service
(`SceneConfig.vision`), which shares the Hub's blob store. Each model call's prompt, answer,
tokens and time are in a `perception.step` event.
"""

from collections.abc import Collection, Mapping, Sequence
from functools import partial
from typing import Annotated, Any, Final, Literal

import anyio
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field, FiniteFloat

from roboscript import rpc
from roboscript.contracts import vision as vision_plane
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraModel
from roboscript.contracts.env import ObserveRequest
from roboscript.contracts.errors import InfraError, ToolError
from roboscript.contracts.geometry import OrientedBox, Pose, Vec3
from roboscript.contracts.hub import CallOutcome, ToolHub
from roboscript.contracts.ids import EpisodeId, FrameId, ToolName
from roboscript.contracts.robot import Arm
from roboscript.contracts.tools import ContentPart, ImagePart, TextPart
from roboscript.contracts.vision import Vision
from roboscript.kernel import entrypoints
from roboscript.manipulation.api import robot_base
from roboscript.manipulation.proxy import ALIKE, ObjectProxy
from roboscript.manipulation.specs import END_HEIGHT_MIN_M
from roboscript.memory import (
    MAX_DESCRIPTION,
    MAX_ROLES,
    MAX_TOWARD,
    Order,
    by_order,
    order_text,
    parse_order,
    rest_order,
)
from roboscript.perception.fit import vec3
from roboscript.perception.grounding import (
    GeometricGrounder,
    Grounder,
    Grounding,
    Mode,
    Resources,
    SegmentPrompt,
    mask_box,
)
from roboscript.perception.identify import (
    Answer,
    Identified,
    ask,
    detector_boxes,
    rest_instances,
    settle,
)
from roboscript.perception.lift import box_mask
from roboscript.perception.overlay import som_overlay
from roboscript.perception.parts import Part, PartKind, candidate_mask, candidates
from roboscript.perception.prior import PartPrior, part_prior, prior_distance, prior_points
from roboscript.perception.registry import ObjectGeometry, SceneRegistry, object_geometry
from roboscript.perception.views import FrameGeometry
from roboscript.perception.vision import VisionDetector, VisionSegmenter
from roboscript.perception.vlm import (
    BoxGrounder,
    SomGrounder,
    camera_words,
    choose_axis,
    instruction_roles,
    part_mark,
)
from roboscript.rpc import service_failures_as_infra
from roboscript.spatial.cloud import oriented_box
from roboscript.spatial.pose import pose_to_matrix
from roboscript.tools import (
    IDENTIFIED,
    FrameRef,
    ObjectRef,
    RegionRef,
    ToolContext,
    ToolModel,
    Toolset,
    tool,
)
from roboscript.tools.context import ToolsetRuntime
from roboscript.toolsets.scene import (
    BACKENDS,
    EndReport,
    IdentifiedFact,
    IdentifiedRole,
    MeasuredReport,
    PartReport,
    PartRoute,
    RegionKey,
    SceneAccess,
    SceneConfig,
    embodiment,
    outline,
    part_points,
    render_measured,
    render_part,
    unmoved,
)
from roboscript.toolsets.sensing.overlay import CANDIDATE_LABELS, Axes, close_up, draw

Backend = Literal["vlm", "som", "grounding_dino", "geometric"]
Reason = Literal["ok", "unconfirmed", "ambiguous", "not_found", "not_named", "registered"]
GROUP: Final = "roboscript.perception"
IDENTIFY_TIMEOUT_S: Final = 1200.0
"""`identify` asks up to four VLM rounds one after another, each of which may wait out its
retries (`VlmConfig.retry_waits_s`, 85 s) and a gateway `queue_timeout` (300 s); the role's wall
clock restarts after the open-time identification, so this is not the role's time."""
BUILT_IN: Final[dict[str, type]] = {
    "geometric": GeometricGrounder,
    "grounding_dino": VisionDetector,
    "som": SomGrounder,
    "vlm": BoxGrounder,
}
"""The core's backends, which register in `GROUP` too (used where the package is not installed
as a distribution, e.g. a source checkout on the simulator's interpreter)."""

PixelBox = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
Pixel = tuple[FiniteFloat, FiniteFloat]
_Camera = Annotated[
    str | None,
    Field(description="A camera `observe` lists; null: the embodiment's head camera."),
]


class LookPose(BaseModel):
    obj: ObjectRef
    arm: Arm
    pose: Pose


def render_look(result: LookPose) -> list[ContentPart]:
    p, q = result.pose.p, result.pose.q
    text = (
        f"{result.arm} ee pose to look at {result.obj}: position "
        f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), quaternion ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, "
        f"{q[3]:.4f})."
    )
    return [TextPart(text=text)]


class RoleQuery(ToolModel):
    role: Annotated[
        str,
        Field(
            min_length=1,
            max_length=40,
            description="The role as the Memory's OBJECTS names it (snake_case).",
        ),
    ]
    description: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_DESCRIPTION,
            description="What the role does in the task and what kind of object it is, as the "
            "Memory's OBJECTS says.",
        ),
    ]
    x_toward: Annotated[
        str | None,
        Field(
            max_length=MAX_TOWARD,
            description="Where the role's frame x points, as its OBJECTS x toward: line says "
            "(null: none).",
        ),
    ] = None
    front: Annotated[
        str | None,
        Field(
            max_length=MAX_DESCRIPTION,
            description="What common sense says is the front of the role's kind of object (a "
            "category prior: the end or side, and a part there), context for settling x toward "
            "(null: none).",
        ),
    ] = None
    order: Annotated[
        str | None,
        Field(
            max_length=MAX_DESCRIPTION,
            description="The role's place among look-alike objects, as its OBJECTS order: line "
            "says (`<rank> among <group> by <key>`; null: none).",
        ),
    ] = None
    carried: Annotated[
        bool,
        Field(
            description="The workflow grasps or places the role's object: it is something an "
            "arm carries, never a flat patch on the table.",
        ),
    ] = False


class Seen(BaseModel):
    """An object an answer shows: as a region (None: nothing to show), the backend that gave it,
    and the center of its points."""

    region: RegionRef | None
    backend: str
    center: Vec3 | None


class RoleFound(BaseModel):
    """A role: its object (None: none registered), why (`ok`: two backends agree, `unconfirmed`: no
    cross-check answer, `ambiguous`: the backends show different objects, or its object is
    registered for another role (`same_as`), `not_found`, `not_named`: the instruction asks for
    no object in it, `registered`: it was registered before), the confidence, center and size,
    the instruction's words for it, the arm the instruction has act on it (`arm`), the objects
    the backends showed (the chosen one first) and the backends outvoted."""

    role: str
    obj: ObjectRef | None
    reason: Reason
    confidence: float
    center: Vec3 | None = None
    size: Vec3 | None = None
    phrase: str | None = None
    arm: Arm | None = None
    candidates: list[Seen] = []
    dissent: list[str] = []
    same_as: ObjectRef | None = None
    note: str = ""
    instances: list[ObjectRef] = []


class Identification(BaseModel):
    frame: FrameRef
    camera: str
    roles: list[RoleFound]
    image: BlobRef | None


class GroundCandidate(BaseModel):
    region: RegionRef | None
    box: PixelBox | None
    point: Pixel | None
    score: float | None
    label: str | None


class Grounded(BaseModel):
    frame: FrameRef
    camera: str
    backend: str
    candidates: list[GroundCandidate]
    image: BlobRef
    latency_s: float


class Segmented(BaseModel):
    """The mask as a region (None when nothing matched), its pixel count, box and score."""

    frame: FrameRef
    camera: str
    region: RegionRef | None
    pixels: int
    box: PixelBox | None
    score: float | None
    image: BlobRef


async def identify_objects(
    hub: ToolHub,
    episode_id: EpisodeId,
    instruction: str,
    roles: Mapping[str, str] | None = None,
    towards: Mapping[str, str] | None = None,
    fronts: Mapping[str, str] | None = None,
    orders: Mapping[str, str] | None = None,
    carried: Collection[str] = (),
    frame: str | None = None,
) -> CallOutcome:
    """Identify once, on the system channel: the runner's call as an episode opens, with the
    episode's live instruction and the Memory's roles (role -> description; None: the objects the
    instruction is about), their `x toward:` words (role -> words), their kinds' fronts as
    common sense has them (role -> a category prior's words) and their `order:` lines (role ->
    the line's text) and the roles the workflow carries (`RoleQuery.carried`)."""
    arguments: dict[str, object] = {"instruction": instruction}
    if frame is not None:  # a frame of the store (a recording's), not a new observation
        arguments["frame"] = frame
    towards, fronts, orders = towards or {}, fronts or {}, orders or {}
    if roles:
        arguments["roles"] = [
            {"role": r, "description": d}
            | ({"x_toward": towards[r]} if r in towards else {})
            | ({"front": fronts[r]} if r in fronts else {})
            | ({"order": orders[r]} if r in orders else {})
            | ({"carried": True} if r in carried else {})
            for r, d in roles.items()
        ]
    return await hub.system_call(episode_id, ToolName("identify"), arguments)


def render_identification(result: Identification) -> list[ContentPart]:
    lines = [f"Objects in {result.frame}:"] + [f"- {_role_line(r)}" for r in result.roles]
    parts: list[ContentPart] = [TextPart(text="\n".join(lines))]
    if result.image is not None:
        parts.append(ImagePart(blob=result.image, label=result.camera))
    return parts


def render_grounded(result: Grounded) -> list[ContentPart]:
    lines = [
        f"{BACKENDS.get(result.backend, result.backend)} in {result.camera} of {result.frame}:"
    ]
    for number, c in enumerate(result.candidates, 1):
        where = f"box {_numbers(c.box)}" if c.box is not None else f"point {_numbers(c.point)}"
        score = "" if c.score is None else f", score {c.score:.2f}"
        label = "" if c.label is None else f": {c.label}"
        lines.append(f"{number}. {c.region or 'no region'}, {where}{score}{label}")
    if not result.candidates:
        lines.append("nothing found")
    return [TextPart(text="\n".join(lines)), ImagePart(blob=result.image, label=result.camera)]


def render_segmented(result: Segmented) -> list[ContentPart]:
    if result.region is None:
        text = f"SAM 3 found nothing in {result.camera} of {result.frame}."
    else:
        score = "" if result.score is None else f", score {result.score:.2f}"
        text = (
            f"SAM 3 mask {result.region} in {result.camera} of {result.frame}: "
            f"{result.pixels} pixels, box {_numbers(result.box)}{score}."
        )
    return [TextPart(text=text), ImagePart(blob=result.image, label=result.camera)]


def backend(name: str) -> Any:
    """A grounding backend's class by its entry-point name (`GROUP`), else the core's."""
    try:
        return entrypoints.load(GROUP, name)
    except LookupError:
        return BUILT_IN[name]


def _ends(whole: ObjectGeometry) -> list[EndReport]:
    """The object's two ends along its long side, lower first (world), where their heights
    differ by `END_HEIGHT_MIN_M` at least; else none."""
    proxy = ObjectProxy.from_geometry(whole)
    found = proxy.end_heights()
    if found is None or abs(found[0][1] - found[1][1]) < END_HEIGHT_MIN_M:
        return []
    return [
        EndReport(center=vec3((proxy.pose @ np.append(at, 1.0))[:3]), height_m=round(height, 4))
        for at, height in sorted(found, key=lambda end: end[1])
    ]


class Perception(Toolset):
    """Identify, ground, segment and locate objects and parts."""

    name = "perception"
    Config = SceneConfig
    config: SceneConfig

    def __init__(self, vision: Vision | None = None) -> None:
        """`vision`: the vision service in this process; by default a client of the configured
        URL."""
        self.vision = vision

    async def start(self, runtime: ToolsetRuntime) -> None:
        self.scene = SceneAccess(self.config, runtime)
        if self.vision is None and self.config.vision is not None:
            self.vision = rpc.client(
                Vision,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
                self.config.vision,
                token=None,
                timeout=10,
                timeouts=vision_plane.TIMEOUTS,
            )

    async def stop(self) -> None:
        await self.scene.close()
        if self.vision is not None:
            await rpc.aclose(self.vision)  # a no-op for an in-process service

    @tool(
        kind="perception",
        tier="D",
        timeout_s=IDENTIFY_TIMEOUT_S,
        render=render_identification,
        system_only=True,
    )
    async def identify(
        self,
        ctx: ToolContext,
        instruction: Annotated[
            str, Field(min_length=1, max_length=1000, description="The episode's instruction.")
        ],
        roles: Annotated[
            list[RoleQuery] | None,
            Field(
                max_length=MAX_ROLES,
                description="The Memory's roles; null: the objects the instruction is about.",
            ),
        ] = None,
        frame: Annotated[
            FrameRef | None,
            Field(description="The frame to look at; null: a new observation (or the newest)."),
        ] = None,
    ) -> Identification:
        """Find and register the objects of the task's roles, once, on the initial scene: the
        instruction's words decide which object plays each role. Set-of-Mark with the VLM over
        the geometric proposals, GroundingDINO as a cross-check, a VLM box where they leave a
        role open or disagree. Roles already registered keep their objects. A role with
        x_toward whose object's shape leaves its frame's x open gets it settled: one VLM call
        picks the candidate x arrow that points there."""
        with service_failures_as_infra("identify"):  # a service failing: void, run again
            return await self._identify_roles(ctx, instruction, roles, frame)

    async def _identify_roles(
        self,
        ctx: ToolContext,
        instruction: str,
        roles: list[RoleQuery] | None,
        frame: str | None,
    ) -> Identification:
        named = {r.role: r.description for r in roles or []}
        towards = {r.role: r.x_toward for r in roles or [] if r.x_toward}
        fronts = {r.role: r.front for r in roles or [] if r.front}
        carried = {r.role for r in roles or [] if r.carried}
        try:
            orders = {r.role: parse_order(r.order) for r in roles or [] if r.order}
        except ValueError as error:
            raise ToolError(f"order: {error}") from None
        if len(named) < len(roles or []):
            raise ToolError("name each role once")
        abstract = bool(named)
        if not named:
            named = await self._roles(ctx, instruction)
        registry = await ctx.frames.registry()
        known = {role: registry.named(role) for role in named}
        todo = {role: text for role, text in named.items() if known[role] is None}
        if todo and frame is None and ctx.context.env is not None:  # the scene as it is now
            observed = await ctx.env.observe(ctx.env_episode, ObserveRequest())
            frame = (await ctx.frames.put(observed)).frame_id
        seen = await self.scene.frame(ctx, frame)
        geometry = await self.scene.geometry(ctx, seen)
        found = {}
        instances: dict[str, list[str]] = {}
        if todo:
            plural = {r for r, o in orders.items() if o.rank is None}
            found, boxes = await self._identify(
                ctx, geometry, todo, instruction if abstract else None, carried, plural
            )
            rest = {r: o for r, o in orders.items() if o.rank is None and r in found}
            if rest:
                instances = await self._rest(ctx, geometry, found, rest, boxes)
            if orders:
                found = await self._ordered(ctx, found, orders, instances)
            await self._directions(
                ctx, geometry, seen.robot, {r: towards[r] for r in todo if r in towards}, fronts
            )
        registry = await ctx.frames.registry()
        results = []
        for role in named:
            if role in found:
                one = _role_found(found[role], registry, seen.id)
                if role in instances:
                    one = one.model_copy(update={"instances": instances[role]})
                results.append(one)
                continue
            record = known[role]
            assert record is not None
            now = object_geometry(record, seen.robot)
            results.append(
                RoleFound(
                    role=role,
                    obj=record.id,
                    reason="registered",
                    confidence=record.confidence,
                    center=vec3(now.center),
                    size=vec3(now.size),
                )
            )
        fact = IdentifiedFact(
            frame=seen.id,
            roles={
                r.role: IdentifiedRole(
                    object=r.obj,
                    reason=r.reason,
                    candidates=[c.region for c in r.candidates if c.region is not None],
                    instances=list(r.instances),
                )
                for r in results
            },
        )
        ctx.frames.set_fact(IDENTIFIED, fact)
        image = await self._outline(ctx, geometry, registry, list(named))
        return Identification(frame=seen.id, camera=geometry.anchor, roles=results, image=image)

    @tool(kind="perception", tier="D", timeout_s=IDENTIFY_TIMEOUT_S, render=render_identification)
    async def locate(
        self,
        ctx: ToolContext,
        name: Annotated[
            str,
            Field(
                min_length=1,
                max_length=40,
                description="The role to register it under, as the Memory's OBJECTS names it "
                "(snake_case).",
            ),
        ],
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                description="What the object looks like: its kind, colour, shape or material, "
                "in the instruction's words.",
            ),
        ],
        frame: Annotated[
            FrameRef | None, Field(description="The frame to look at; null: the newest.")
        ] = None,
    ) -> Identification:
        """Find one more object by its description and register it under a role, with the
        chain of the identification (Set-of-Mark, GroundingDINO as a cross-check, a VLM box where
        they leave it open or disagree). An object registered already is not registered again."""
        registry = await ctx.frames.registry()
        if registry.named(name) is not None:
            raise ToolError(
                f"{name!r} is registered already: register the right region with "
                "register(replaces=...)"
            )
        seen = await self.scene.frame(ctx, frame)
        geometry = await self.scene.geometry(ctx, seen)
        found, _ = await self._identify(ctx, geometry, {name: query}, None)
        registry = await ctx.frames.registry()
        image = await self._outline(ctx, geometry, registry, [name])
        role = _role_found(found[name], registry, seen.id)
        return Identification(frame=seen.id, camera=geometry.anchor, roles=[role], image=image)

    @tool(kind="perception", tier="D", timeout_s=300.0, render=render_grounded)
    async def ground(
        self,
        ctx: ToolContext,
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                description="What to find: its kind, colour, shape or material, in the "
                "instruction's words.",
            ),
        ],
        backend: Annotated[
            Backend,
            Field(
                description="vlm: the VLM's box; grounding_dino: detector boxes; som: the VLM's "
                "choice among numbered proposals; geometric: every proposal, the query unread."
            ),
        ] = "vlm",
        camera: _Camera = None,
        mode: Annotated[
            Mode, Field(description="box: boxes; point: points, each on a proposal if any.")
        ] = "box",
        frame: Annotated[
            FrameRef | None,
            Field(description="The frame to look at; null: the newest (an Actor's: the initial)."),
        ] = None,
    ) -> Grounded:
        """Find something in one camera's image. Candidates come as regions to lift, fit or
        register, with the overlay; detector scores rank and are not confidences. An Actor
        grounds on the initial frame, or on the newest to find again an object no arm holds (one
        dropped or never taken; register it with replaces)."""
        seen = await self.scene.frame(ctx, self._initial_frame(ctx, frame))
        self.scene.initial(ctx, seen.id, "grounds", newest=True)
        geometry = await self.scene.geometry(ctx, seen)
        camera = self.scene.camera(ctx, geometry, camera)
        async with self._models(ctx) as client:
            grounder: Grounder = self._grounder(ctx, backend, client)
            grounding = await grounder.ground(geometry, camera, query, mode)
        self.scene.log(ctx)("ground", **_ground_log(grounding))
        candidates = [
            GroundCandidate(
                region=_region(
                    seen.id, Answer(backend=backend, camera=camera, proposal=c.region, box=c.box)
                ),
                box=c.box,
                point=c.point,
                score=c.score,
                label=c.label,
            )
            for c in grounding.candidates
        ]
        image = await ctx.blobs.aput_image(grounding.overlay)
        return Grounded(
            frame=seen.id,
            camera=camera,
            backend=backend,
            candidates=candidates,
            image=image,
            latency_s=round(grounding.latency_s, 3),
        )

    @tool(kind="perception", tier="D", timeout_s=120.0, render=render_segmented)
    async def segment(
        self,
        ctx: ToolContext,
        camera: _Camera = None,
        text: Annotated[
            str | None, Field(max_length=120, description="A phrase: every instance, best first.")
        ] = None,
        box: Annotated[PixelBox | None, Field(description="u0, v0, u1, v1 in pixels.")] = None,
        point: Annotated[Pixel | None, Field(description="u, v in pixels.")] = None,
        region: Annotated[RegionRef | None, Field(description="A region to refine.")] = None,
        frame: Annotated[
            FrameRef | None,
            Field(
                description="The frame; null: the newest (an Actor's: the initial), or the "
                "region's."
            ),
        ] = None,
    ) -> Segmented:
        """A mask from SAM 3 in one camera: the best match for a text, the object in a box or
        under a point, or what a region shows there. Give exactly one of them. The mask comes
        back as a region to lift, fit or register. An Actor segments on the initial frame, or on
        the newest to find again an object no arm holds."""
        given = [x for x in (text, box, point, region) if x is not None]
        if len(given) != 1:
            raise ToolError("give exactly one of text, box, point and region")
        segmenter = VisionSegmenter(self._vision(), ctx.blobs)
        if region is not None:
            picked = await self.scene.region(ctx, region)
            geometry = picked.geometry
            self.scene.initial(ctx, geometry.frame, "segments", newest=True)
            camera = self.scene.camera(ctx, geometry, camera)
            if camera not in picked.masks or not picked.masks[camera].any():
                raise ToolError(f"that region shows nothing in {camera}")
            prompt = SegmentPrompt(region=picked.masks[camera])
        else:
            seen = await self.scene.frame(ctx, self._initial_frame(ctx, frame))
            self.scene.initial(ctx, seen.id, "segments", newest=True)
            geometry = await self.scene.geometry(ctx, seen)
            camera = self.scene.camera(ctx, geometry, camera)
            prompt = SegmentPrompt(text=text, box=box, point=point)
        found = await segmenter.segment(geometry, camera, prompt)
        mask = found.mask
        image = geometry.view(camera).image
        drawn, _ = await anyio.to_thread.run_sync(
            som_overlay, image.rgb, {1: mask} if mask.any() else {}
        )
        region_id, bounds = None, None
        if mask.any():
            blob = await ctx.blobs.aput_array(mask)
            region_id = RegionKey(frame=geometry.frame, camera=camera, mask=blob).id
            bounds = mask_box(mask)
        latency = round(found.latency_s, 3)
        log = {"camera": camera, "pixels": int(mask.sum()), "score": found.score}
        self.scene.log(ctx)("segment", frame=geometry.frame, latency_s=latency, **log)
        return Segmented(
            frame=geometry.frame,
            camera=camera,
            region=region_id,
            pixels=int(mask.sum()),
            box=bounds,
            score=found.score,
            image=await ctx.blobs.aput_image(drawn),
        )

    @tool(kind="perception", tier="D", timeout_s=300.0, render=render_part)
    async def locate_part(
        self,
        ctx: ToolContext,
        obj: Annotated[ObjectRef, Field(description="The object the part belongs to.")],
        part: Annotated[
            str,
            Field(
                min_length=1, max_length=40, description="The part's name, as the Memory names it."
            ),
        ],
        query: Annotated[
            str | None,
            Field(max_length=200, description="How to find it; null: the <part> of the <object>."),
        ] = None,
        kind: Annotated[
            PartKind | None,
            Field(
                description="The kind of part candidate it is, as the Memory keeps it: handle "
                "(sticks out of the body's side), end or neck (of a long or a narrowing object), "
                "rim or opening (of an open vessel), lid, button or rocker (on top, or on a thin "
                "upright object's face); null: any."
            ),
        ] = None,
        x_toward: Annotated[
            str | None,
            Field(
                description="A registered part of the object that the part's x points toward "
                "(the part's own name counts as null)."
            ),
        ] = None,
        articulated: Annotated[
            bool,
            Field(
                description="The part turns or slides on the object (a lid, a door, a drawer): "
                "a grasp of it moves the part, not the object."
            ),
        ] = False,
        prior: Annotated[
            PartPrior | None,
            Field(
                description="Where the part lies on the object, as a report's prior gave it (the "
                "Memory keeps it); null: anywhere."
            ),
        ] = None,
        camera: _Camera = None,
        frame: Annotated[
            FrameRef | None,
            Field(description="The frame to look at; null: the newest (an Actor's: the initial)."),
        ] = None,
    ) -> PartReport:
        """Find a part of a registered object and register it. Its candidates come from the
        object's geometry (see kind); with `kind`, only those of that kind (if none, those of a
        kind like it: a rocker the geometry reads as a button). The only one of a
        kind is taken as it is; else the VLM names the one the query describes among them,
        outlined and numbered in one camera. The part's frame: origin at the center of its box,
        z up, x from the object's body toward the part where that decides it (else along its
        long side, toward the part x_toward names, else both ways). A moving part (articulated)
        the geometry offers no candidate for, or none the VLM names, is boxed by the VLM from
        the query instead, and a moving part's x points from its hinge toward its free edge where
        the hinge estimate is confident. With a prior, only the candidates lying where it says
        are offered, nearest first (the nearest is taken when the VLM names none of several), and
        where none does, the object's points inside the prior's box make the part. The report
        gives the candidate's kinds and the part's prior, for the Memory to keep, and the route
        that found the part (only: the one candidate of its kind; mark: the VLM's choice among the
        candidates; prior: where the prior says; box: the VLM's box). To give the part's pixels
        yourself, use register_part with a region. An Actor locates parts on the initial frame
        only."""
        registry = await ctx.frames.registry()
        if obj not in registry.objects:
            raise ToolError("that object is not registered in this episode")
        record = registry.objects[obj]
        unmoved(ctx, record)  # before the model call; register_part checks again
        seen = await self.scene.frame(ctx, self._initial_frame(ctx, frame))
        self.scene.initial(ctx, seen.id, "registers parts")
        geometry = await self.scene.geometry(ctx, seen)
        camera = self.scene.camera(ctx, geometry, camera)
        whole = object_geometry(record, seen.robot)
        points, box, both_ways = whole.points, whole.obb, not record.decided
        found = await anyio.to_thread.run_sync(candidates, points, None, geometry.table.z)
        offered = of_kind(found, kind)
        listed = [list(c.kinds) for c in found]
        text = query or f"the {part.replace('_', ' ')} of the {record.name.replace('_', ' ')}"
        chosen: Part | None = None
        route: PartRoute = "only"
        missed = ""
        if prior is not None:
            offered, chosen = _by_prior(prior, box, points, both_ways, offered)
            route = "prior"
        if not offered and chosen is None and not articulated:
            what = "" if kind is None else f"{kind} "
            where = "" if prior is None else " where the prior says"
            raise ToolError(
                f"the object has no {what}part candidate{where} (its candidates: "
                f"{listed or 'none'}): give the part's pixels to register_part as a region"
            )
        if chosen is None and len(offered) == 1 and (kind is not None or prior is not None):
            chosen = offered[0]
        elif chosen is None and offered:
            route = "mark"
            chosen, missed = await self._part_mark(ctx, geometry, camera, obj, part, offered, text)
            if chosen is None and prior is not None:  # the VLM named none: the nearest
                chosen, route = offered[0], "prior"
        if chosen is None and not articulated:
            raise ToolError(
                f"{missed}: try another query or camera, or give the part's pixels to register_part"
            )
        if chosen is None:  # a moving link the shape offers no candidate for, or none the VLM saw
            route = "box"
            chosen = await self._part_box(ctx, geometry, camera, obj, part, points, text)
        self.scene.log(ctx)("part_candidates", object=obj, part=part, kind=kind, route=route,
                            candidates=listed, chosen=list(chosen.kinds))  # fmt: skip
        report = await self.scene.register_part(
            ctx, obj, part, chosen, seen.id, x_toward, articulated
        )
        ends = _ends(whole) if self.config.report_ends else []
        update = {"route": route, "prior": part_prior(box, chosen.points), "ends": ends}
        return report.model_copy(update=update)

    async def _part_box(
        self,
        ctx: ToolContext,
        geometry: FrameGeometry,
        camera: str,
        obj: str,
        part: str,
        points: Any,
        text: str,
    ) -> Part:
        """A moving part the VLM boxes in `camera`: the object's points the box shows, boxed
        upright (x along its long side, either way until its hinge decides it). A box that shows
        fewer than 3 of them missed the object (a part's description can send it elsewhere on
        the whole image), so the VLM is asked once more, shown the object close
        up (`close_crop`)."""
        view = geometry.view(camera)
        shown = await anyio.to_thread.run_sync(candidate_mask, points, view)
        inside = np.zeros((0, 3))
        for close in (False, True):
            rgb, (left, top), scale = (
                close_crop(view.image.rgb, shown) if close else (view.image.rgb, (0, 0), 1.0)
            )
            async with self._models(ctx) as client:
                grounder = BoxGrounder(
                    client, self.config.vlm, ctx.blobs, pipeline=self.config.pipeline
                )
                answer = await grounder.locate(rgb, text, camera, geometry.anchor)
            box = None
            if answer.box is not None:
                u0, v0, u1, v1 = answer.box
                box = (left + u0 / scale, top + v0 / scale, left + u1 / scale, top + v1 / scale)
            asked = answer.asked.model_dump(mode="json")
            self.scene.log(ctx)("part_box", object=obj, part=part, query=text, box=box,
                                close_up=close, asked=asked)  # fmt: skip
            if box is None:
                if close:
                    break
                continue
            mask = box_mask(box, view.image.shape)
            inside = await anyio.to_thread.run_sync(
                part_points, points, geometry.views, {camera: mask}
            )
            if len(inside) >= 3 or not shown.any():
                break
        if len(inside) < 3:
            raise ToolError(
                f"the VLM's box for {text!r} in {camera} holds {len(inside)} of the object's "
                "points, on the whole image and close up: try another query or camera"
            )
        return _boxed(inside)

    async def _part_mark(
        self,
        ctx: ToolContext,
        geometry: FrameGeometry,
        camera: str,
        obj: str,
        part: str,
        offered: Sequence[Part],
        text: str,
    ) -> tuple[Part | None, str]:
        """The candidate the VLM names for the part `text` describes, among those that show in
        `camera`; else None and why."""
        view = geometry.view(camera)
        masks = [await anyio.to_thread.run_sync(candidate_mask, c.points, view) for c in offered]
        shown = {i: m for i, m in enumerate(masks, 1) if m.any()}
        if not shown:
            return None, f"none of the object's part candidates shows in {camera}"
        overlay, _ = await anyio.to_thread.run_sync(som_overlay, view.image.rgb, shown)
        words = camera_words(camera, self.config.pipeline, geometry.anchor)
        async with self._models(ctx) as client:
            mark, asked = await part_mark(
                client, view.image.rgb, overlay, text, list(shown), words, self.config.vlm,
                ctx.blobs,
            )  # fmt: skip
        log = {"object": obj, "part": part, "query": text, "marks": list(shown), "mark": mark}
        self.scene.log(ctx)("part_mark", **log, asked=asked.model_dump(mode="json"))
        if mark is None:
            return None, f"the VLM saw {text!r} in none of the object's part candidates in {camera}"
        return offered[mark - 1], ""

    @tool(kind="perception", tier="D", timeout_s=120.0, render=render_measured)
    async def remeasure(
        self,
        ctx: ToolContext,
        obj: Annotated[ObjectRef, Field(description="The registered object to measure again.")],
        frame: Annotated[
            FrameRef | None, Field(description="A frame to measure in; null: a new look.")
        ] = None,
        cameras: Annotated[
            list[str] | None, Field(description="The cameras to use; null: all.")
        ] = None,
        store: Annotated[
            bool, Field(description="Register it when confident (false: only report).")
        ] = True,
    ) -> MeasuredReport:
        """Measure a registered object's pose and shape again from a new look at the scene, where
        the registry expects it, or nearby when it is no longer there (knocked over or pushed);
        its identity stays. Use it after it was set down or pushed, or when a check says it is
        off. It is registered only when the measurement is confident."""
        return await self.scene.remeasure(ctx, obj, frame=frame, store=store, cameras=cameras)

    @tool(kind="perception", tier="D", timeout_s=120.0, render=render_measured)
    async def refine(
        self,
        ctx: ToolContext,
        obj: Annotated[ObjectRef, Field(description="The registered object to measure again.")],
        arm: Annotated[Arm, Field(description="The arm whose wrist camera is near the object.")],
        part: Annotated[
            str | None, Field(description="A registered part to report; null: the object.")
        ] = None,
        store: Annotated[
            bool, Field(description="Register it when confident (false: only report).")
        ] = True,
    ) -> MeasuredReport:
        """Measure an object (and its parts) again with the head camera and a wrist camera close
        to it: move that arm to a pose near the object first (see look_pose), then refine before
        a fine grasp or placement. Its identity stays."""
        return await self.scene.refine(ctx, obj, arm, part=part, store=store)

    @tool(kind="perception", tier="D", render=render_look)
    async def look_pose(
        self,
        ctx: ToolContext,
        obj: Annotated[ObjectRef, Field(description="The registered object to look at.")],
        arm: Annotated[Arm, Field(description="The arm whose wrist camera looks.")],
    ) -> LookPose:
        """An ee pose for move_to_pose that brings the arm's wrist camera about 30 cm from the
        object, looking down at it, aimed past what the head camera's image cuts off; then call
        remeasure or refine."""
        return LookPose(obj=obj, arm=arm, pose=await self.scene.look_pose(ctx, obj, arm))

    async def _roles(self, ctx: ToolContext, instruction: str) -> dict[str, str]:
        """The objects an instruction is about, as roles to identify. An answer that names none
        (empty, or not the JSON asked for) is the model failing, not the task: an
        infrastructure error, so that the caller tries again."""
        profile = self.scene.profile(ctx)
        robot = "a robot" if profile is None else profile.description
        async with self._models(ctx) as client:
            named, asked = await instruction_roles(
                client, instruction, self.config.vlm, robot=robot
            )
        found = {"instruction": instruction, "roles": named, "asked": asked.model_dump(mode="json")}
        self.scene.log(ctx)("roles", **found)
        if not named:
            why = asked.error or "an empty answer"
            raise InfraError("not_ready", f"the grounding model named no object ({why})")
        return dict(list(named.items())[:MAX_ROLES])

    async def _directions(
        self,
        ctx: ToolContext,
        geometry: FrameGeometry,
        robot: Any,
        towards: Mapping[str, str],
        fronts: Mapping[str, str],
    ) -> None:
        """Each role's frame whose x is open turned to the candidate arrow the VLM sees point
        toward its `x toward:` words (its kind's front, from `fronts`, said as context), in the
        head camera, asked twice with the labels the other way round (roles at once): answers
        that name different arrows are a guess, so the frame
        stays open (grasps and placements try every candidate) rather than turned by a coin
        flip. A frame left open by an earlier choice is not asked again. A role whose call fails
        keeps its frame, its direction still weak, with the error logged."""
        registry = await ctx.frames.registry()
        records = {
            role: record
            for role, words in towards.items()
            if (record := registry.named(role)) is not None
            and record.direction is not None
            and record.direction.weak
            and record.direction.cue != "chosen"
        }
        if not records:
            return
        camera = self.scene.camera(ctx, geometry, None)
        image = geometry.view(camera).image
        model = CameraModel(
            name=camera,
            width=image.shape[1],
            height=image.shape[0],
            K=tuple(tuple(float(v) for v in row) for row in image.K),  # type: ignore[arg-type]
            cam2world=tuple(tuple(float(v) for v in row) for row in image.cam2world),  # type: ignore[arg-type]
        )
        words = camera_words(camera, self.config.pipeline, geometry.anchor)
        chosen: dict[str, float] = {}
        log = self.scene.log(ctx)

        async def settle_one(role: str, record: Any) -> None:
            now = object_geometry(record, robot)
            length = float(np.clip(0.6 * max(now.size[:2]), 0.04, 0.15))
            pose = pose_to_matrix(now.pose)
            labels = list(CANDIDATE_LABELS[: len(record.turns)])
            orders = (tuple(record.turns), tuple(reversed(record.turns)))
            answers: list[float | None] = []
            for turns in orders:  # the same question with the labels the other way round
                drawn = partial(draw, axes=[Axes("", pose, length, turns)])
                overlay = await anyio.to_thread.run_sync(drawn, image.rgb, model)
                near = await anyio.to_thread.run_sync(
                    close_up, overlay, model, now.obb, now.pose, length
                )
                try:
                    async with self._models(ctx) as client:
                        label, asked = await choose_axis(
                            client, image.rgb, overlay, near, role.replace("_", " "),
                            towards[role], labels, words, self.config.vlm, ctx.blobs,
                            fronts.get(role),
                        )  # fmt: skip
                except (InfraError, ToolError) as error:  # one arrow unasked: no reason to lose all
                    log("axis", object=record.id, role=role, toward=towards[role],
                        labels=labels, front=fronts.get(role), turns=turns, chosen=None,
                        error=str(error))  # fmt: skip
                    return
                log("axis", object=record.id, role=role, toward=towards[role], labels=labels,
                    front=fronts.get(role), turns=turns, chosen=label,
                    asked=asked.model_dump(mode="json"))  # fmt: skip
                answers.append(None if label is None else turns[labels.index(label)])
            if answers[0] is not None and answers[0] == answers[1]:
                chosen[record.id] = answers[0]

        async with anyio.create_task_group() as group:
            for role, record in records.items():
                group.start_soon(settle_one, role, record)
        if chosen:
            async with ctx.frames.edit_registry() as edit:
                for obj, turn in chosen.items():
                    edit.orient(obj, turn)

    async def _identify(
        self,
        ctx: ToolContext,
        geometry: FrameGeometry,
        roles: Mapping[str, str],
        instruction: str | None,
        carried: Collection[str] = (),
        plural: Collection[str] = (),
    ) -> tuple[dict[str, Identified], dict[str, list[PixelBox] | None]]:
        """The chain: every model call outside the registry's lock, then the claims under it;
        with GroundingDINO's boxes for each role (`detector_boxes`; more for a `plural` one)."""
        vlm, blobs, log = self.config.vlm, ctx.blobs, self.scene.log(ctx)
        proposals = await self.scene.proposals(ctx, geometry)
        source = self.scene.source(ctx)
        async with self._models(ctx) as client:
            evidence = await ask(
                geometry,
                roles,
                proposals,
                instruction=instruction,
                som=SomGrounder(client, vlm, blobs, source, self.config.pipeline),
                box=BoxGrounder(client, vlm, blobs, source, self.config.pipeline),
                detector=None if self.vision is None else VisionDetector(self.vision, blobs),
                segmenter=None if self.vision is None else VisionSegmenter(self.vision, blobs),
                config=self.config.pipeline,
                log=log,
                plural=plural,
            )
        robot = (await ctx.frames.get(geometry.frame)).robot
        frame = geometry.frame

        def region(answer: Answer) -> str | None:
            return _region(frame, answer)

        boxes = {role: detector_boxes(evidence, role) for role in roles}
        async with ctx.frames.edit_registry() as registry:
            found = await anyio.to_thread.run_sync(
                lambda: settle(
                    evidence,
                    registry,
                    robot=robot,
                    region=region,
                    config=self.config.pipeline,
                    log=log,
                    carried=carried,
                )
            )
        return found, boxes

    async def _ordered(
        self,
        ctx: ToolContext,
        found: dict[str, Identified],
        orders: Mapping[str, Order],
        instances: dict[str, list[str]],
    ) -> dict[str, Identified]:
        """The look-alike roles' objects sorted by their `order:` lines (`memory.by_order`): the
        registry's names and the identifications follow. A group's ranked roles are sorted
        among its rest role's objects too (`instances`, from `_rest`), which keeps those left,
        sorted by its key (`memory.rest_order`); `instances` is changed."""
        if ctx.context.embodiment is None:
            raise ToolError("ordering look-alike objects needs the episode's embodiment")
        base = robot_base(embodiment(ctx.context.embodiment)).matrix()
        bound = {role: f.object for role, f in found.items() if f.object is not None}
        extra: dict[str, list[str]] = {}
        for role, objs in instances.items():
            extra.setdefault(orders[role].group, []).extend(objs)
        async with ctx.frames.edit_registry() as registry:
            pool = {*bound.values(), *(o for objs in extra.values() for o in objs)}
            starts = {obj: registry.get(obj).fit.center for obj in pool}
            ordered = by_order(orders, bound, starts, base[:3, 1], base[:3, 3], extra)
            for role, objs in instances.items():
                order = orders[role]
                ranked = [r for r, o in orders.items() if o.rank is not None and r in bound]
                group = [r for r in ranked if orders[r].group == order.group]
                taken = {ordered[r] for r in group}
                left = {bound[r] for r in group} | set(objs)
                instances[role] = rest_order(
                    order, sorted(left - taken), starts, base[:3, 1], base[:3, 3]
                )
                if instances[role]:
                    ordered[role] = instances[role][0]
            names = {obj: role for role, obj in ordered.items() if bound.get(role) != obj}
            names |= {obj: role for role, objs in instances.items() for obj in objs}
            for obj, role in names.items():
                if registry.objects[obj].name != role:
                    registry.objects[obj] = registry.objects[obj].model_copy(update={"name": role})
        out = dict(found)
        for role, obj in ordered.items():
            if bound.get(role) == obj:
                continue
            order = orders[role]
            note = f"rank {order.rank} among {order.group} by {order.key}"
            if order.rank is None:
                note = f"{order_text(order)}: the first of those its group's ranks leave"
            out[role] = found[role].model_copy(update={"object": obj, "note": note})
        return out

    async def _rest(
        self,
        ctx: ToolContext,
        geometry: FrameGeometry,
        found: dict[str, Identified],
        rest: Mapping[str, Order],
        boxes: Mapping[str, list[PixelBox] | None],
    ) -> dict[str, list[str]]:
        """Each plural role's objects (`order: rest`): its identified one and every look-alike
        no role claims that GroundingDINO's boxes for its words show (`boxes`; None: every
        look-alike; `perception.identify.rest_instances`), registered under its name and
        sorted by its order's key (`memory.rest_order`); its identification names the first.
        `found` is changed."""
        if ctx.context.embodiment is None:
            raise ToolError("ordering look-alike objects needs the episode's embodiment")
        base = robot_base(embodiment(ctx.context.embodiment)).matrix()
        proposals = await self.scene.proposals(ctx, geometry)
        robot = (await ctx.frames.get(geometry.frame)).robot
        log = self.scene.log(ctx)
        out: dict[str, list[str]] = {}
        async with ctx.frames.edit_registry() as registry:
            for role, order in rest.items():
                first = found[role].object
                if first is None:
                    continue
                run = partial(
                    rest_instances, registry, geometry, first, proposals, robot=robot,
                    boxes=boxes.get(role), config=self.config.pipeline, log=log,
                )  # fmt: skip
                added = await anyio.to_thread.run_sync(run)
                starts = {o: registry.get(o).fit.center for o in [first, *added]}
                out[role] = rest_order(order, list(starts), starts, base[:3, 1], base[:3, 3])
                found[role] = found[role].model_copy(update={"object": out[role][0]})
        return out

    async def _outline(
        self,
        ctx: ToolContext,
        geometry: FrameGeometry,
        registry: SceneRegistry,
        roles: list[str],
    ) -> BlobRef | None:
        """The anchor image with the roles' registered objects outlined and named."""
        robot = (await ctx.frames.get(geometry.frame)).robot
        objects = {
            role: object_geometry(record, robot).points
            for role in roles
            if (record := registry.named(role)) is not None
        }
        if not objects:
            return None
        image = geometry.view(geometry.anchor).image
        drawn = await anyio.to_thread.run_sync(outline, image, objects)
        return await ctx.blobs.aput_image(drawn)

    def _initial_frame(self, ctx: ToolContext, frame: str | None) -> str | None:
        """An Actor's default frame is the initial one; the others' the newest."""
        if frame is not None or ctx.context.role != "actor":
            return frame
        done = self.scene.identified(ctx)
        return None if done is None else done.frame

    def _grounder(self, ctx: ToolContext, name: str, client: Any) -> Grounder:
        resources = Resources(
            config=self.config.pipeline,
            proposals=self.scene.source(ctx),
            client=client,
            vlm=self.config.vlm,
            blobs=ctx.blobs,
            vision=self.vision,
        )
        try:
            built: Grounder = backend(name).build(resources)
        except LookupError as error:
            raise ToolError(str(error)) from None
        return built

    def _models(self, ctx: ToolContext) -> Any:
        """A model client for this episode's grounding calls, in a gateway session of their own
        (the episode's session belongs to its brain, which alone holds a permits key for it)."""
        return self.runtime.models(session_id=f"{ctx.episode_id}.perception")

    def _vision(self) -> Vision:
        if self.vision is None:
            raise ToolError("GroundingDINO and SAM 3 need the vision service; none is configured")
        return self.vision


def _by_prior(
    prior: PartPrior, box: OrientedBox, points: Any, both_ways: bool, offered: Sequence[Part]
) -> tuple[list[Part], Part | None]:
    """The candidates lying where `prior` says, nearest first; with none, the part the object's
    points inside the prior's box make (None when too few are)."""
    near = [(d, i) for i, c in enumerate(offered)
            if (d := prior_distance(prior, box, c.points, both_ways)) is not None]  # fmt: skip
    if near:
        return [offered[i] for _, i in sorted(near)], None
    inside = prior_points(prior, box, points, both_ways)
    return [], _boxed(inside) if len(inside) else None


def _boxed(inside: Any) -> Part:
    """A part of the object's points `inside` (world), boxed upright (x along its long side,
    either way until its hinge decides it)."""
    box = oriented_box(inside, "upright")
    frame = np.eye(4)
    frame[:3, :3] = np.asarray(box.axes).T
    frame[:3, 3] = box.center
    return Part(kinds=(), frame=frame, extent=np.asarray(box.size), decided=False, points=inside)


def _role_found(identified: Identified, registry: SceneRegistry, frame: FrameId) -> RoleFound:
    candidates = [
        Seen(region=_region(frame, a), backend=a.backend, center=a.center)
        for a in identified.candidates
    ]
    found = RoleFound(
        role=identified.role,
        obj=identified.object,
        reason=identified.reason,
        confidence=identified.confidence,
        phrase=identified.phrase,
        arm=identified.arm,
        candidates=candidates,
        dissent=identified.dissent,
        same_as=identified.same_as,
        note=identified.note,
    )
    if identified.object is None:
        return found
    record = registry.get(identified.object)
    return found.model_copy(update={"center": record.fit.center, "size": vec3(record.fit.obb.size)})


def _region(frame: FrameId, answer: Answer) -> str | None:
    """The region an answer names: its proposal, else its mask (a split piece), else its box,
    else none."""
    if answer.proposal is not None:
        return RegionKey(frame=frame, proposal=answer.proposal).id
    if answer.mask is not None:
        return RegionKey(frame=frame, camera=answer.camera, mask=answer.mask).id
    if answer.box is not None:
        return RegionKey(frame=frame, camera=answer.camera, box=answer.box).id
    return None


def _role_line(r: RoleFound) -> str:
    where = ""
    if r.center is not None and r.size is not None:
        where = f" at {_numbers(r.center)}, {' x '.join(f'{s:.3f}' for s in r.size)} m"
    words = "" if not r.phrase else f' ("{r.phrase}")'
    if r.arm is not None:
        words += f" [the instruction's {r.arm} arm]"
    others = [c for c in r.candidates[1:] if c.region is not None]
    shown = "; ".join(_seen(c) for c in (others if r.obj is not None else r.candidates))
    if len(r.instances) > 1:  # a plural role: name each object to act on by its alias
        listed = ", ".join(r.instances)
        where += f"; with the others, {len(r.instances)} objects in its order: {listed}"
    if r.reason == "registered":
        return f"{r.role}: {r.obj}{where} (registered before)"
    if r.reason == "ok" and r.dissent:
        outvoted = ", ".join(BACKENDS.get(b, b) for b in r.dissent)
        return f"{r.role}{words}: {r.obj}{where} (two backends agree; {outvoted} showed {shown})"
    if r.reason == "ok":
        return f"{r.role}{words}: {r.obj}{where} (confirmed by a second backend)"
    if r.reason == "unconfirmed":
        return f"{r.role}{words}: {r.obj}{where} (unconfirmed: the cross-check found nothing)"
    if r.reason == "ambiguous" and r.same_as is not None:
        return (
            f"{r.role}{words}: ambiguous, not registered: its answer shows {r.same_as}, "
            "registered for another role"
        )
    if r.reason == "ambiguous":
        return f"{r.role}{words}: {r.obj}{where}, ambiguous: {shown}; check before acting on it"
    if r.reason == "not_named":
        return f"{r.role}: not in this instruction (it names no object for it): skip its stages"
    note = f"; {r.note}" if r.note else ""
    return f"{r.role}{words}: not found" + (f" ({shown})" if shown else "") + note


def _seen(c: Seen) -> str:
    where = "" if c.center is None else f" at {_numbers(c.center)}"
    return f"{BACKENDS.get(c.backend, c.backend)} shows {c.region}{where}"


def _ground_log(grounding: Grounding) -> dict[str, object]:
    return {
        "frame": grounding.frame,
        "camera": grounding.camera,
        "backend": grounding.backend,
        "query": grounding.query,
        "candidates": [c.model_dump(mode="json") for c in grounding.candidates],
        "latency_s": round(grounding.latency_s, 3),
        "asked": None if grounding.asked is None else grounding.asked.model_dump(mode="json"),
    }


def _numbers(values: Sequence[float] | None) -> str:
    return "(" + ", ".join(f"{round(float(v), 3) + 0.0:g}" for v in values or ()) + ")"


CROP_SIDE: Final = 448
"""A close-up for the VLM is enlarged to about this many pixels on its long side."""
CROP_MARGIN: Final = 0.25
"""A close-up shows this share of the object's pixel extent around it, at least 12 pixels."""


def close_crop(
    rgb: NDArray[np.uint8], shown: NDArray[np.bool_]
) -> tuple[NDArray[np.uint8], tuple[int, int], float]:
    """`rgb` cut around the pixels `shown` (an object's mask) with a margin and enlarged to about
    `CROP_SIDE`: the crop, its top-left pixel in `rgb` and its scale (crop pixels per image
    pixel). The whole image at scale 1 when nothing is shown."""
    if not shown.any():
        return rgb, (0, 0), 1.0
    rows, cols = np.flatnonzero(shown.any(axis=1)), np.flatnonzero(shown.any(axis=0))
    h, w = shown.shape
    pad = max(12, round(CROP_MARGIN * max(rows[-1] - rows[0], cols[-1] - cols[0])))
    top, bottom = max(0, rows[0] - pad), min(h, rows[-1] + 1 + pad)
    left, right = max(0, cols[0] - pad), min(w, cols[-1] + 1 + pad)
    scale = max(1.0, CROP_SIDE / max(bottom - top, right - left))
    cut = Image.fromarray(np.ascontiguousarray(rgb[top:bottom, left:right, :3]))
    size = (round(cut.width * scale), round(cut.height * scale))
    return np.asarray(cut.resize(size, Image.Resampling.LANCZOS)), (int(left), int(top)), scale


def of_kind(found: Sequence[Part], kind: PartKind | None) -> list[Part]:
    """The candidates of `kind` (all for None), else those of a kind that looks like it
    (`ALIKE`: a switch's rocker the geometry reads as a button, flat)."""
    if kind is None:
        return list(found)
    for kinds in ((kind,), ALIKE.get(kind, ())):
        offered = [c for c in found if set(kinds) & set(c.kinds)]
        if offered:
            return offered
    return []
