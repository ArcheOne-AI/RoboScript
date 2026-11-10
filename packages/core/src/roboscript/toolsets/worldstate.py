"""The `worldstate` toolset (docs/concepts/perception.md,
docs/concepts/agent-runtime.md): the episode's registered objects and what holds between
them.

`register` puts the object a region shows into the episode's scene registry (lifted from every
view the region has, fitted). An Actor's `register` is an identity correction only, on the
initial frame (the `IDENTIFIED` fact): under a role of the identification, from a region it
showed for that role, replacing the role's object (nothing for a role without one), before an
arm has taken that object (from then on the grasp relations own its pose), and never onto an
object registered for another role (`settle`'s claim check). `register_part` frames a part of a
registered object from a region (an Actor's on the initial frame; not once an arm has taken the
object), and `refresh` (a demo-reading role's) re-measures an object in a frame: nothing re-detects
objects on its own, they keep the pose they were identified at or follow the gripper holding
them. `orient` turns a registered object's frame to one of the candidate x arrows that
`show_frames` draws (an Actor's and a demo-reading role's; identity and geometry kept).
`relation` and `check` read the registry as it is now (or at a frame's robot state; on a frame of an
earlier try, as that try left it): spatial relations and the ten predicates, whose thresholds are
required arguments (values from the Memory's parameters); what holding and an open gripper mean
comes from the embodiment profile. Predicate reasons name objects by the fields that hold them
(`{a}`, `{obj}`), filled in with the objects' aliases when rendered.

Done tools (`check_done` and the like) belong to a method's own toolset (wicl's:
`roboscript_methods.wicl.toolsets.done`), which shares this toolset's scene registry.
"""

import re
from functools import partial
from typing import Annotated, Final, Literal

import anyio
from pydantic import BaseModel, Field

from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.robot import Arm, RobotState
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.embodiments import Gripper
from roboscript.perception.criteria import TOL_FLOOR_DEG, EndTarget
from roboscript.perception.identify import claimed
from roboscript.perception.locate import locate, refresh, scene
from roboscript.perception.predicates import Direction, PredicateResult, check
from roboscript.perception.registry import (
    SceneRegistry,
    object_geometry,
)
from roboscript.perception.relations import Relation, relation
from roboscript.tools import (
    DONE,
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
    ObjectReport,
    PartReport,
    Region,
    SceneAccess,
    SceneConfig,
    describe,
    is_newest,
    object_report,
    render_part,
)

Shape = Literal["auto", "box", "cylinder", "plane", "line"]
_Obj = Annotated[ObjectRef, Field(description="A registered object.")]
_ArmField = Annotated[Arm, Field(description="The arm.")]
_MEMORY = "a value from the Memory's parameters"


class Holding(ToolModel):
    name: Literal["holding"]
    arm: _ArmField


class GripperOpen(ToolModel):
    name: Literal["gripper_open"]
    arm: _ArmField
    fraction: Annotated[
        float,
        Field(ge=0.0, le=1.0, description=f"Share of the full opening, {_MEMORY}."),
    ]


class Lifted(ToolModel):
    name: Literal["lifted"]
    obj: _Obj
    height: Annotated[float, Field(description=f"Meters of its bottom above the table, {_MEMORY}.")]


class On(ToolModel):
    name: Literal["on"]
    a: _Obj
    b: _Obj
    tol: Annotated[
        float, Field(description=f"Meters of slack between a's bottom and b's top, {_MEMORY}.")
    ]
    min_overlap: Annotated[
        float, Field(ge=0.0, le=1.0, description=f"Share of a's footprint over b's, {_MEMORY}.")
    ]


class Inside(ToolModel):
    name: Literal["inside"]
    a: _Obj
    b: _Obj


class Beside(ToolModel):
    name: Literal["beside"]
    a: _Obj
    b: _Obj
    direction: Annotated[
        Direction,
        Field(
            description="The robot's left, right, front (nearer it) or back; arm_side / "
            "other_side: the side of the arm that took a, or of the other arm (a side the "
            "acting arm gives, not the task)."
        ),
    ]
    dmin: Annotated[float, Field(description=f"Meters, {_MEMORY}.")]
    dmax: Annotated[float, Field(description=f"Meters, {_MEMORY}.")]


class Near(ToolModel):
    name: Literal["near"]
    arm: _ArmField
    obj: _Obj
    distance: Annotated[float, Field(description=f"Meters from the TCP to its box, {_MEMORY}.")]


class Pressed(ToolModel):
    name: Literal["pressed"]
    obj: _Obj
    distance: Annotated[
        float,
        Field(description=f"Meters from its center to the last press, seen from above, {_MEMORY}."),
    ]


class Moved(ToolModel):
    name: Literal["moved"]
    obj: _Obj
    distance: Annotated[float, Field(description=f"Meters, {_MEMORY}.")]
    since: Annotated[
        int | None, Field(description="A step; null: since it was first measured.")
    ] = None


class YawAligned(ToolModel):
    name: Literal["yaw_aligned"]
    obj: _Obj
    ref: Annotated[ObjectRef | None, Field(description="Another object; or give ref_deg.")] = None
    ref_deg: Annotated[float | None, Field(description="A yaw in degrees; or give ref.")] = None
    tol: Annotated[float, Field(description=f"Degrees, {_MEMORY}.")]


Predicate = Annotated[
    Holding | GripperOpen | Lifted | On | Inside | Beside | Near | Pressed | Moved | YawAligned,
    Field(discriminator="name"),
]


class CheckReport(BaseModel):
    """A predicate's value (None: undecidable, e.g. no measured opening), what was measured, and
    the reason, which names objects by field (`{a}`)."""

    predicate: str
    value: bool | None
    measured: dict[str, float | None]
    reason: str
    objects: dict[str, ObjectRef]


class RelationReport(Relation):
    a: ObjectRef
    b: ObjectRef | None
    point: Vec3 | None


def render_object(result: ObjectReport) -> list[ContentPart]:
    return [TextPart(text=describe(result) + ".")]


def render_check(result: CheckReport) -> list[ContentPart]:
    reason = result.reason
    for field, alias in result.objects.items():
        reason = reason.replace("{" + field + "}", alias)
    value = {True: "true", False: "false", None: "undecided"}[result.value]
    return [TextPart(text=f"{result.predicate}: {value} ({reason})")]


def render_relation(result: RelationReport) -> list[ContentPart]:
    other = result.b if result.b is not None else f"point {result.point}"
    flags = [f for f in ("on", "inside", "above", "left_of", "in_front_of") if getattr(result, f)]
    dx, dy, dz = result.displacement
    text = (
        f"{result.a} relative to {other}: displacement ({dx:.3f}, {dy:.3f}, {dz:.3f}) m, "
        f"{result.distance_xy:.3f} m apart in xy, bottom-to-top gap {result.gap_z:.3f} m, "
        f"{result.overlap:.0%} of its footprint over the other; {', '.join(flags) or 'no flag'}."
    )
    return [TextPart(text=text)]


class WorldState(Toolset):
    """Register objects and parts, re-measure them, relate them and check predicates."""

    name = "worldstate"
    Config = SceneConfig
    config: SceneConfig

    async def start(self, runtime: ToolsetRuntime) -> None:
        self.scene = SceneAccess(self.config, runtime)

    async def stop(self) -> None:
        await self.scene.close()

    @tool(kind="perception", tier="D", render=render_object)
    async def register(
        self,
        ctx: ToolContext,
        name: Annotated[
            str,
            Field(
                min_length=1,
                max_length=40,
                description="The object's role, as the Memory's OBJECTS names it (snake_case).",
            ),
        ],
        region: Annotated[RegionRef, Field(description="A region from ground, segment, identify.")],
        shape: Annotated[
            Shape, Field(description="auto: the shape that explains the points best.")
        ] = "auto",
        replaces: Annotated[
            ObjectRef | None,
            Field(description="An object this one replaces (registered under a wrong identity)."),
        ] = None,
    ) -> ObjectReport:
        """Register the object a region shows, under a role: its points from every view the
        region has, fitted. With replaces, that object leaves the registry. Actions and checks
        take registered objects only. An Actor only corrects an identity: under a role, from a
        region the initial identification showed for that role, replacing the role's object
        (none if it has none), before an arm has taken it; or adds, without replaces, another
        object to a role the identification found several objects for (order: rest)."""
        picked = await self.scene.region(ctx, region)
        robot = (await ctx.frames.get(picked.geometry.frame)).robot
        add = partial(
            locate,
            name=name,
            geometry=picked.geometry,
            chosen=picked.masks,
            config=self.config.pipeline,
            shape=shape,
            log=self.scene.log(ctx),
        )
        done = self.scene.identified(ctx)
        role = None if done is None else done.roles.get(name)
        adding = role is not None and bool(role.instances) and replaces is None
        async with ctx.frames.edit_registry() as registry:
            if ctx.context.role == "actor":
                self._correction(ctx, registry, name, region, replaces, picked, robot, adding)
            taken = registry.named(name)
            if taken is not None and taken.id != replaces and not adding:
                raise ToolError(
                    f"{name!r} is registered already; to register this region instead, give "
                    "that object as replaces"
                )
            if replaces is not None:
                if replaces not in registry.objects:
                    raise ToolError("the object to replace is not registered in this episode")
                if registry.objects[replaces].grasps:
                    raise ToolError("the object to replace is held: open the gripper first")
            try:
                record = await anyio.to_thread.run_sync(lambda: add(registry))
            except LookupError:
                raise ToolError("that region shows nothing with depth above the table") from None
            if replaces is not None:
                if registry.objects[replaces].taken:  # found again: still the object an arm took
                    record = record.model_copy(update={"taken": True})
                    registry.objects[record.id] = record
                registry.remove(replaces)
        return object_report(record, robot)

    @tool(kind="perception", tier="D", render=render_part)
    async def register_part(
        self,
        ctx: ToolContext,
        obj: _Obj,
        part: Annotated[
            str,
            Field(
                min_length=1, max_length=40, description="The part's name, as the Memory names it."
            ),
        ],
        region: Annotated[RegionRef, Field(description="Pixels of the part.")],
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
    ) -> PartReport:
        """Register a part of an object: the object's points that the region's cameras show
        inside it, framed by geometry (origin at the center of their box, z up, x along its
        long side, toward the part x_toward names, else both ways). An Actor registers parts on
        the initial frame only."""
        picked = await self.scene.region(ctx, region)
        return await self.scene.register_part(
            ctx, obj, part, picked, picked.key.frame, x_toward, articulated
        )

    @tool(kind="perception", tier="D", render=render_object)
    async def refresh(
        self,
        ctx: ToolContext,
        obj: _Obj,
        frame: Annotated[
            FrameRef | None, Field(description="The frame to measure in; null: the newest.")
        ] = None,
    ) -> ObjectReport:
        """Re-measure a registered object in a frame: the proposal where the registry expects
        it, lifted and fitted again. Nothing re-measures objects on its own; use it when you
        know the registry is off (e.g. with the arm out of the way)."""
        seen = await self.scene.frame(ctx, frame)
        geometry = await self.scene.geometry(ctx, seen)
        proposals = await self.scene.proposals(ctx, geometry)
        async with ctx.frames.edit_registry() as registry:
            if obj not in registry.objects:
                raise ToolError("that object is not registered in this episode")
            measure = partial(
                refresh,
                registry,
                geometry,
                obj,
                proposals,
                robot=seen.robot,
                config=self.config.pipeline,
                log=self.scene.log(ctx),
            )
            try:
                record = await anyio.to_thread.run_sync(measure)
            except LookupError:
                raise ToolError(
                    "the object is not where the registry expects it in that frame; register it "
                    "again from a region (register with replaces)"
                ) from None
        return object_report(record, seen.robot)

    @tool(kind="perception", tier="D", render=render_object)
    async def orient(
        self,
        ctx: ToolContext,
        obj: _Obj,
        x: Annotated[
            Literal["A", "B", "C", "D"],
            Field(
                description="The candidate x arrow of show_frames to make the object's x: A as "
                "it is, B the half turn (for a near-square object the next quarter turn), ..."
            ),
        ],
    ) -> ObjectReport:
        """Turn a registered object's frame about its z so that its x is the candidate arrow `x`
        that show_frames draws, then keep it (its x counts as decided, unless the shape had
        pointed x the other way: then the other arrows stay candidates). The object stays what
        and where it is (identity, points, parts in the world, grasps); specs and checks read
        the new frame. Use it where the shape left x open and the image shows which end the
        Memory's `x toward:` names."""
        async with ctx.frames.edit_registry() as registry:
            if obj not in registry.objects:
                raise ToolError("that object is not registered in this episode")
            turns = registry.objects[obj].turns
            index = "ABCD".index(x)
            if index >= len(turns):
                raise ToolError(f"that object's frame has {len(turns)} candidate x arrows")
            record = registry.orient(obj, turns[index])
        self.scene.log(ctx)("orient", object=obj, x=x, turn_deg=turns[index])
        live = ctx.frames.live()
        robot = None
        if record.grasps and live is not None:  # held: where the newest frame's gripper has it
            robot = (await ctx.frames.get(live.alias)).robot
        return object_report(record, robot)

    @tool(kind="pure", tier="D", render=render_relation)
    async def relation(
        self,
        ctx: ToolContext,
        a: _Obj,
        tol: Annotated[float, Field(description=f"Slack of the flags in meters, {_MEMORY}.")],
        b: Annotated[
            ObjectRef | None, Field(description="The other object; or give point.")
        ] = None,
        point: Annotated[Vec3 | None, Field(description="A world point; or give b.")] = None,
        frame: Annotated[
            FrameRef | None, Field(description="At that frame's robot state; null: now.")
        ] = None,
    ) -> RelationReport:
        """How object a sits relative to object b (or a point) as the registry has them:
        displacement, xy distance, bottom-to-top gap, footprint overlap, and on, inside, above,
        left_of, in_front_of."""
        if (b is None) == (point is None):
            raise ToolError("give exactly one of b and point")
        robot = await self.scene.robot(ctx, frame)
        registry = await ctx.frames.registry(frame)
        unknown = [o for o in (a, b) if o is not None and o not in registry.objects]
        if unknown:
            raise ToolError("that object is not registered in this episode")
        first = object_geometry(registry.objects[a], robot)
        second = object_geometry(registry.objects[b], robot) if b is not None else point
        assert second is not None  # one of b and point is given
        found = await anyio.to_thread.run_sync(partial(relation, first, second, tol))
        return RelationReport(a=a, b=b, point=point, **dict(found))

    @tool(kind="pure", tier="D", render=render_check)
    async def check(
        self,
        ctx: ToolContext,
        predicate: Annotated[
            Predicate, Field(description="The predicate and its arguments, tagged by name.")
        ],
        frame: Annotated[
            FrameRef | None, Field(description="At that frame's robot state; null: now.")
        ] = None,
    ) -> CheckReport:
        """Check a predicate on the registered objects and the robot: holding, gripper_open,
        lifted, on, inside, beside, near, pressed (where an empty gripper last pushed on it and
        was stopped), moved, yaw_aligned. A failing predicate ends nothing; it tells you
        whether a stage is done."""
        robot = await self.scene.robot(ctx, frame)
        table = await self.scene.table(ctx, frame)
        registry = await ctx.frames.registry(frame)
        arguments = predicate.model_dump(exclude={"name"})
        objects = {k: v for k, v in arguments.items() if k in ("obj", "a", "b", "ref") and v}
        if any(o not in registry.objects for o in objects.values()):
            raise ToolError("that object is not registered in this episode")
        if isinstance(predicate, Holding | GripperOpen):
            gripper = self._gripper(ctx, predicate.arm)
            if isinstance(predicate, Holding):
                arguments["threshold"] = gripper.holding_threshold
            else:
                arguments["full"] = gripper.open
        floor = ""
        if isinstance(predicate, YawAligned):
            if (predicate.ref is None) == (predicate.ref_deg is None):
                raise ToolError("give exactly one of ref and ref_deg")
            arguments["ref"] = predicate.ref if predicate.ref is not None else predicate.ref_deg
            del arguments["ref_deg"]
            record = registry.objects[predicate.obj]
            if predicate.ref_deg is not None and record.grasps and record.name in _held_ends(ctx):
                return CheckReport(
                    predicate="yaw_aligned",
                    value=None,
                    measured={},
                    reason=HELD_YAW.format(role=record.name),
                    objects=objects,
                )
            if predicate.tol < TOL_FLOOR_DEG:  # read at the floor, as check_done reads criteria
                arguments["tol"] = TOL_FLOOR_DEG
                floor = (
                    f" (tol {predicate.tol:g} deg read as {TOL_FLOOR_DEG:g}: perception's error "
                    "on a yaw)"
                )
        now = await anyio.to_thread.run_sync(scene, registry, robot, table)
        result: PredicateResult = check(now, predicate.name, **arguments)
        reason = result.reason + floor
        for field, obj in objects.items():
            reason = re.sub(rf"\b{re.escape(obj)}\b", "{" + field + "}", reason)
        return CheckReport(
            predicate=result.predicate,
            value=result.value,
            measured=result.measured,
            reason=reason,
            objects=objects,
        )

    def _correction(
        self,
        ctx: ToolContext,
        registry: SceneRegistry,
        name: str,
        region: str,
        replaces: str | None,
        picked: Region,
        robot: RobotState,
        adding: bool = False,
    ) -> None:
        """An Actor's `register`, under the registry's lock: an identity correction of role
        `name` (see the module), or a ToolError that says what is wrong; or a role's object found
        again on the newest frame (`relocated`: a region of a later frame than the initial one,
        the scene as it is now, replacing the role's object, which no arm holds: one dropped,
        knocked over or never taken); or (`adding`) another object of a plural role, from a
        region no registered object shows."""
        if adding:
            owner = claimed(registry, picked.geometry, picked.masks, robot)
            if owner is not None:
                other = registry.objects[owner].name
                raise ToolError(
                    f"that region shows the object registered for {other!r}: add to {name!r} an "
                    "object no role has"
                )
            return
        done = self.scene.identified(ctx)
        role = None if done is None else done.roles.get(name)
        current = registry.named(name)
        relocated = (
            role is not None
            and done is not None
            and picked.geometry.frame != done.frame
            and is_newest(ctx, picked.geometry.frame)
            and current is not None
            and current.id == replaces
            and not current.grasps
        )
        if not relocated and (role is None or region not in role.candidates):
            raise ToolError(
                "an Actor registers only to correct an identity (a role of the initial "
                "identification and a region it showed for that role), to find again on the "
                "newest frame a role's object no arm holds (with that object as replaces), or "
                "to add an object to a role the identification found several objects for"
            )
        if (None if current is None else current.id) != replaces:
            what = "nothing: it has no object" if current is None else "the object it has"
            raise ToolError(f"to correct {name!r}, replace {what}")
        if current is not None and current.moved and not relocated:
            raise ToolError(
                f"an arm has taken the object of {name!r} since the initial frame, and the "
                "grasps own its pose now: an identity is corrected only before that"
            )
        owner = claimed(registry, picked.geometry, picked.masks, robot, exclude=replaces)
        if owner is not None:
            # Grounding again with the same words finds the same region: say what else works.
            # (Registry ids are not the model's aliases, so the roles are named instead.)
            other = registry.objects[owner].name
            again = (
                ""
                if current is None
                else f", or refresh {name!r}'s object to re-measure it where it was registered"
            )
            raise ToolError(
                f"that region shows the object registered for {other!r}: an object is registered "
                f"once. Find {name!r} in a region that holds only it: ground it in another camera "
                f"or frame, or with words that tell it apart from {other!r}{again}; if that object "
                f"is {name!r}'s instead, correct {other!r} first"
            )

    def _gripper(self, ctx: ToolContext, arm: Arm) -> Gripper:
        profile = self.scene.profile(ctx)
        if profile is None:
            raise ToolError("the episode names no embodiment, so its grippers are unknown")
        return profile.arms[arm].gripper


HELD_YAW: Final = (
    "undecided: the demonstration ends holding {role}, and its end gives no yaw for an object it "
    "ends holding (only one set down ends at a yaw), so a world yaw is no goal for it: do not "
    "turn it for this; check_done measures where it ends (its lift, its place relative to the "
    "robot's base)"
)


def _held_ends(ctx: ToolContext) -> set[str]:
    """The roles the demonstration's end holds (its `holds` end targets in the `DONE` fact; none
    before an end is known)."""
    fact = ctx.frames.fact(DONE)
    if fact is None:
        return set()
    ends = [EndTarget.model_validate(end) for end in fact.get("end", [])]
    return {t.a for t in ends if t.kind == "holds"}
