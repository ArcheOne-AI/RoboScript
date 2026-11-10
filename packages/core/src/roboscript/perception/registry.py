"""The scene registry: the objects and parts the agent registered in one episode, as plain data.

Runtime policy: objects are identified once, on the initial scene at
the start of the episode (`register`); afterwards nothing re-detects or recognizes them. Moved
objects get their identity and pose from grasp relations only. After a close whose settled
measured opening shows something between the fingers (at or above the embodiment's holding
threshold, `Gripper.holding_threshold`, the rule receipts use too), `on_close` attaches the object
the grasp aimed at (without one: the registered object nearest the TCP within `REACH`, near ties
going to the nearest center); its pose is then the TCP x the transform recorded at close, with
the section between the fingers centered on the TCP. Several arms may hold one object (a
handover, a two-handed lift); it follows the arm that took it first. A close on an articulated
part (registered `articulated`: a lid, a door, a drawer, which turns or slides on the object's
body) attaches that part alone, when the grasp aimed at it or, aimed at nothing, when the TCP lies
within `GRIP_RADIUS` of its box: the part then follows the TCP (whose arc or line keeps it on
its hinge or slide) and the body stays; an open leaves the part where its grasp puts it. A rigid
part (a vessel's handle) is the object's: grasping it attaches the object. After an open (or a lost
hold), `on_open` ends that arm's hold, and once no arm holds the object it is left at the pose
its grasp implies and settled: a model-free snap of its height onto what lies under its
footprint, the top surface of a registered object's points there (the highest 5 mm height band
of them that covers `SUPPORT_SHARE` of the footprint, so a support's side faces never pull it
down: a block's top face, a basket's inside bottom, a bowl's inner surface; the bottom of a
container whose inside no camera saw) or the table plane when its center is over the table,
whichever is higher. With `center_on_close` (`PerceptionConfig.center_on_close`, set by
`register`; off by default), a close also models the fingers pushing what they close on to
their middle: along the closing axis the caller gives, the fitted shape's section through the
TCP (a cylinder's chord, else its box's) is centered on the TCP when the points between the
fingers show it from one side only (their span `ONE_SIDED_M` short of the chord: a camera sees
the near face of what the fingers close on, never the far one); seen from both sides, the
points' span is centered instead; a chord wider than the open fingers (`Gripper.span_m`) is not
what they closed on, and the pose stays. A model of the close only: nothing is measured after
it. Objects never grasped keep their initial pose. A press (`on_press`: the executor saw an
empty gripper stopped pushing, `ActionReceipt.presses`) is kept on the registered object no arm
holds that it reaches (within `PRESS_REACH` of the footprint horizontally, above the perceived
top by at most that object's `top_uncertainty` plus `PRESS_REACH`), the nearest, in that
object's frame. Each of these decisions is logged as a step (`Step`). `refresh` (re-measuring
from new points) is an explicit call, for the Demo Reader; nothing calls it automatically.
Changes are serialized by a lock per registry, since tool calls run in
worker threads. An object identified as `ambiguous` keeps the candidates the backends showed.

An object's frame is its fit's (`Fit.frame`): z up, x along the footprint's long side (the sign
is not decided), origin at the bottom center of a standing object and at the axis center of a
lying one; a refresh keeps x the way closest to the previous frame, so parts stay put. A part's
frame: a part candidate's (`perception.parts`: z up, x from the object's body toward the part
when that decides it), else z up and x along the part's horizontal long side; the origin at the
center of the part's box; x turned toward the `x_toward` part if one is named, and with neither a
decided x nor `x_toward` both ways (two candidates); y = z x x. Parts are stored in the object
frame and move with it; an articulated part moved by a grasp keeps its new pose there
(`ObjectRecord.part_poses`), its registered pose (`PartRecord.pose`) stays the measured one.
`stale_age` counts env steps since the object was last measured. `ObjectGeometry` is the typed
view manipulation's perceived-object proxy consumes; it depends on no toolset.
"""

import functools
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer, PrivateAttr

from roboscript.contracts.geometry import OrientedBox, Pose, Vec3
from roboscript.contracts.ids import FrameId
from roboscript.contracts.robot import Arm, RobotState
from roboscript.perception.config import DEFAULT_FRAMES, FrameConfig
from roboscript.perception.direction import Direction, orient, part_x, reframe
from roboscript.perception.fit import Fit, Shape, Symmetry, fit, vec3
from roboscript.perception.lift import Lifted, touches_border
from roboscript.perception.parts import far_end
from roboscript.perception.proposals import Proposal
from roboscript.perception.table import Table
from roboscript.spatial.cloud import oriented_box
from roboscript.spatial.pose import pose_compose, pose_inverse, pose_to_matrix
from roboscript.spatial.rotation import (
    quat_angle_deg,
    quat_from_axis_angle,
    quat_from_matrix,
    quat_to_matrix,
)

WHOLE: Final = "whole"
"""The default part: the object's own box."""
REACH: Final = 0.05
"""Without a target, `on_close` attaches a registered object within this distance of the TCP."""
PRESS_REACH: Final = 0.02
"""Contact slack of a press: a press stopped with the TCP horizontally within this distance of an
object's footprint, or vertically above the perceived top by this much beyond the top's own
uncertainty (`top_uncertainty`), pressed on that object."""
TOP_BASE: Final = 0.01
"""The top of a well-seen object is known to the centimeter of its box."""
TOP_LOW: Final = 0.03
"""What a top seen only at a grazing angle may be perceived lower than it is: a top face the
camera does not see is replaced by the highest side points, which lie this much below the top
at most (scaled by the share of the top face that was not seen)."""
TOP_PARTIAL: Final = 0.01
"""An object cut at the image border adds this to the uncertainty of its box."""
TOP_ERROR: Final = 0.04
"""The bound of `top_uncertainty`."""
TOP_CELL: Final = 0.005
"""Cells of the footprint in which the top face is counted as seen."""
TIE: Final = 0.01
"""Objects whose boxes lie within this of the nearest one's are a tie: the nearest center wins."""
GRIP_RADIUS: Final = 0.02
"""The section between the fingers: the object's points this close to the closing axis."""
SQUARE_MIN_DEG: Final = 5.0
SQUARE_MAX_DEG: Final = 40.0
"""Flat pads closing on a box at an angle turn it about the TCP until the box axis nearest the
closing axis lies along it; `attach` with
`square_axis` follows that turn when it is at least `SQUARE_MIN_DEG` (within perception's error) and
at most `SQUARE_MAX_DEG` (nearer a diagonal the box may turn either way)."""
ONE_SIDED_M: Final = 0.005
"""The points between the fingers show a section from one side when their span along the closing
axis is this much shorter than the fitted shape's chord."""
SETTLE_SLACK: Final = 0.02
"""A registered object's points support a released one up to this above its bottom (placing may
press it down a little)."""
SUPPORT_SHARE: Final = 0.25
"""An object's points support a released one when they cover this share of its footprint (in
5 mm cells): the sliver of a neighbor its box overlaps does not."""
_CELL: Final = 0.005
_BAND: Final = 0.005
"""A support's top surface is a height band this thick."""

Step = Callable[..., None]
"""Logs one step: `step(name, **data)` (e.g. `perception.locate.StepLog`)."""


def silent(step: str, **data: Any) -> None:
    """A `Step` that logs nothing."""


def _as_points(value: Any) -> NDArray[np.float64]:
    return np.asarray(value, dtype=float).reshape(-1, 3)


def _dump_points(value: NDArray[np.float64]) -> list[list[float]]:
    rounded: list[list[float]] = np.round(value, 5).tolist()  # 0.01 mm
    return rounded


Points = Annotated[
    NDArray[np.float64],
    BeforeValidator(_as_points),
    PlainSerializer(_dump_points, when_used="json"),
]
"""(N, 3) points in meters: an array in memory, nested lists in JSON."""


class PartRecord(BaseModel):
    """A part: `pose` and `points` in the object frame, as measured; `size` along the part
    frame's x, y, z; `both_ways`: neither the part's shape nor `x_toward` decided x, so the frame
    turned 180 deg about z is an equal candidate; `articulated`: it turns or slides on the
    object's body, so a grasp moves it alone."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    pose: Pose
    size: Vec3
    x_toward: str | None
    both_ways: bool
    points: Points
    articulated: bool = False


class PartGrasp(BaseModel):
    """An arm holding an articulated part: the part and its frame in the TCP frame."""

    model_config = ConfigDict(frozen=True)

    part: str
    pose: Pose


class Measurement(BaseModel):
    """Where an object's frame was (world) when measured in `frame` at `step`."""

    model_config = ConfigDict(frozen=True)

    frame: FrameId
    step: int
    pose: Pose


class Pressed(BaseModel):
    """Where an arm's empty gripper was stopped pushing on an object (`on_press`): the
    TCP then, in the object's frame."""

    model_config = ConfigDict(frozen=True)

    arm: Arm
    point: Vec3


class Shown(BaseModel):
    """What one identification backend showed for an object's role: a region of the frame (an
    id the perception toolsets name; None: nothing to show) and the center of its points."""

    model_config = ConfigDict(frozen=True)

    backend: str
    region: str | None
    center: Vec3 | None


class ObjectRecord(BaseModel):
    """One registered object. `fit` and `points` are world geometry as measured in `frame` at
    `step` (`partially_visible`: every view that measured it cut it at the image border, so its
    center may be off by a centimeter); `pose` is where its frame is (the fit's frame, or where
    holding left it); `grasps` holds the object frame in the TCP frame of each arm holding it, in
    the order they took it, the latest last (it follows the first); `carried`: an arm has held it
    since it was last measured (see `moved`); `taken`: an arm has held it since it was registered.
    `part_grasps`: the arms holding one of its articulated parts; `part_poses`: where grasps
    left its articulated parts (object frame); `presses`: where empty fingers were
    stopped pushing on it, in order (`on_press`). `ambiguous`: the
    identification's backends showed different objects for its role, `candidates` (this one
    first). `direction`: how the frame's x was decided (`perception.direction`; None: as the fit
    left it). `arm`: the arm the instruction has act on it, as the identification read the
    instruction for its role (None: it says none). `far_end`: its footprint's far end
    (`perception.parts.far_end`, world, where `points` are) as its first measurement found it,
    carried through later ones: a frame turned toward it (`x_toward: far_end`) must not turn with
    the next view's points (None: it had none)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str
    name: str
    fit: Fit
    points: Points
    confidence: float
    frame: FrameId
    step: int
    pose: Pose
    partially_visible: bool = False
    ambiguous: bool = False
    candidates: list[Shown] = []
    parts: dict[str, PartRecord] = {}
    grasps: dict[Arm, Pose] = {}
    carried: bool = False
    taken: bool = False
    part_grasps: dict[Arm, PartGrasp] = {}
    part_poses: dict[str, Pose] = {}
    presses: list[Pressed] = []
    history: list[Measurement] = []
    direction: Direction | None = None
    arm: Arm | None = None
    far_end: Vec3 | None = None
    handled: Arm | None = None
    """The arm that took it last (kept once it lets go): the arm that acted on it, which an
    arm-relative side follows (`perception.sides`)."""

    @property
    def decided(self) -> bool:
        """Whether its frame's x is decided (else `direction.turns` are equal candidates)."""
        if self.direction is None:
            return self.fit.symmetry == "none" and self.fit.shape == "cylinder"
        return not self.direction.weak and self.direction.cue != "round"

    @property
    def x_known(self) -> bool:
        """Whether its frame's x points one known way: decided by its shape, or chosen toward a
        named end (`toolsets.perception`'s axis choice, asked twice and answered alike; its other
        turn is kept for grasps to fall back on, but x as it stands is the chosen one)."""
        return self.decided or (self.direction is not None and self.direction.cue == "chosen")

    @property
    def turns(self) -> tuple[float, ...]:
        """The turns (deg) its frame may take, labeled A, B, ... in frame overlays: those of a
        weak direction, else the frame and its half turn."""
        if self.direction is not None and self.direction.weak:
            return self.direction.turns
        return (0.0, 180.0)

    @property
    def symmetry(self) -> Symmetry:
        """The yaw turns it is taken to look the same under: none once x is decided by a cue
        or a choice (a weak shape cue only points it), else the fit's."""
        direction = self.direction
        if self.decided and direction is not None and direction.cue in ("shape", "chosen"):
            return "none"
        return self.fit.symmetry

    @property
    def held_by(self) -> Arm | None:
        """The arm the object follows (the first of those holding it)."""
        return next(iter(self.grasps), None)

    @property
    def moved(self) -> bool:
        """Whether the grasp relations own its pose: an arm holds it, or has held it since it was
        last measured, so the frames before show it elsewhere than the registry has it."""
        return self.carried or bool(self.grasps)


@dataclass(frozen=True, eq=False)
class PartGeometry:
    """A part in the world: `pose` (7,) and its `candidates` (one, or two when both ways)."""

    name: str
    pose: NDArray[np.float64]
    candidates: tuple[NDArray[np.float64], ...]
    size: NDArray[np.float64]
    points: NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class ObjectGeometry:
    """An object as it is now, in the world: frame `pose` (7,), box `obb` (center, size along its
    own axes, orientation), centers, points, parts (`whole` included), the arms holding it (the
    one it follows first), and how many steps old its measurement is."""

    id: str
    name: str
    shape: Shape
    upright: bool
    symmetry: Symmetry
    pose: NDArray[np.float64]
    obb: OrientedBox
    center: NDArray[np.float64]
    top_center: NDArray[np.float64]
    bottom_center: NDArray[np.float64]
    points: NDArray[np.float64]
    parts: Mapping[str, PartGeometry]
    held_by: tuple[Arm, ...]
    stale_age: int
    partially_visible: bool
    direction: Direction | None = None
    shape_symmetry: Symmetry | None = None
    """The yaw turns its measured shape looks the same under (the fit's), whatever decided x
    (`symmetry` is "none" once a cue or a choice did); None: as `symmetry`."""
    far_end: NDArray[np.float64] | None = None
    """Its footprint's far end as its first measurement found it (`ObjectRecord.far_end`), where
    it is now (world)."""
    handled_by: Arm | None = None
    """The arm that took it last, kept once it lets go (`ObjectRecord.handled`)."""

    @property
    def size(self) -> NDArray[np.float64]:
        return np.asarray(self.obb.size)

    @property
    def yaw_deg(self) -> float:
        x = quat_to_matrix(self.pose[3:])[:, 0]
        return float(np.degrees(np.arctan2(x[1], x[0])))


Source = ArrayLike | Lifted | Proposal
_F = TypeVar("_F", bound=Callable[..., Any])


def _serialized(method: _F) -> _F:
    @functools.wraps(method)
    def locked(self: "SceneRegistry", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast(_F, locked)


class SceneRegistry(BaseModel):
    """One episode's registered objects (`o1`, `o2`, ...); a plain model, dumped as JSON."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    objects: dict[str, ObjectRecord] = {}
    counter: int = 0
    base: tuple[float, float] | None = None
    """The robot base's world x, y: the anchor frames point away from (set by `register`)."""
    center_on_close: bool = False
    """A close centers what it holds on the TCP along the closing axis (see the module; set by
    `register` from `PerceptionConfig.center_on_close`), so every copy of the registry, a
    demonstration's replay among them, models the close alike."""
    _lock: Any = PrivateAttr(default_factory=threading.RLock)

    def get(self, obj_id: str) -> ObjectRecord:
        if obj_id not in self.objects:
            raise LookupError(
                f"no registered object {obj_id!r}; registered: {sorted(self.objects)}"
            )
        return self.objects[obj_id]

    def named(self, name: str) -> ObjectRecord | None:
        """The object registered under `name`, if any (the first, should there be several)."""
        return next((o for o in self.objects.values() if o.name == name), None)

    @_serialized
    def to_json(self) -> str:
        """The registry as JSON, taken under its lock (`model_validate_json` reads it back)."""
        return self.model_dump_json()

    @_serialized
    def clone(self) -> "SceneRegistry":
        """An independent registry with the same objects (records are immutable: a change
        replaces them), taken under the lock."""
        return SceneRegistry(
            objects=dict(self.objects), counter=self.counter, center_on_close=self.center_on_close
        )

    @_serialized
    def remove(self, obj_id: str) -> ObjectRecord:
        """Take an object out (e.g. one registered under the wrong identity); not a held one."""
        obj = self.get(obj_id)
        if obj.grasps:
            raise ValueError(f"{obj_id} is held by {_arms(obj)}: open it first")
        del self.objects[obj_id]
        return obj

    @_serialized
    def register(
        self,
        name: str,
        source: Source,
        *,
        frame: FrameId,
        step: int,
        table_z: float | None = None,
        shape: Shape | Literal["auto"] = "auto",
        confidence: float = 1.0,
        ambiguous: bool = False,
        candidates: Sequence[Shown] = (),
        base: ArrayLike | None = None,
        frames: FrameConfig = DEFAULT_FRAMES,
        center_on_close: bool | None = None,
    ) -> ObjectRecord:
        """A new object from world points (or a lifted mask or a proposal), fitted as `shape`,
        its frame pointed by the convention (`perception.direction`; `base`: the robot base's
        world x, y, the anchor); `center_on_close`, when given, sets the registry's."""
        points = _points_of(source)
        fitted = fit(points, shape, table_z)
        if base is not None:
            self.base = (float(np.asarray(base)[0]), float(np.asarray(base)[1]))
        if center_on_close is not None:
            self.center_on_close = center_on_close
        pose, direction = orient(fitted, points, table_z, self.base, frames)
        fitted = reframe(fitted, pose)
        self.counter += 1
        record = ObjectRecord(
            id=f"o{self.counter}",
            name=name,
            fit=fitted,
            points=points,
            confidence=confidence,
            frame=frame,
            step=step,
            pose=fitted.frame,
            partially_visible=_cut(source),
            ambiguous=ambiguous,
            candidates=list(candidates),
            history=[Measurement(frame=frame, step=step, pose=fitted.frame)],
            direction=direction,
        )
        record = record.model_copy(update={"far_end": _far_end(record)})
        self.objects[record.id] = record
        return record

    def remeasure(
        self,
        obj_id: str,
        source: Source,
        *,
        frame: FrameId,
        step: int,
        table_z: float | None = None,
        robot: RobotState | None = None,
        confidence: float | None = None,
        frames: FrameConfig = DEFAULT_FRAMES,
    ) -> ObjectRecord:
        """The record re-measured from new points (fitted the same way), not stored. The new frame
        keeps x the way closest to the old one (a near-square footprint's axes may swap, so it
        may take a quarter turn); a held object needs the `robot` state of the measurement, and
        its grasp transforms are updated."""
        old = self.get(obj_id)
        points = _points_of(source)
        fitted = _shape_kept(fit(points, old.fit.shape, table_z), old.fit)
        previous = Pose.from_list(object_geometry(old, robot).pose)
        pose = _continue_frame(fitted.frame, previous, _turns(fitted, old, frames))
        kept = None  # the first measurement's far end, where it is now
        if old.far_end is not None:
            back = np.asarray(np.linalg.inv(old.fit.frame.matrix()), dtype=np.float64)
            moved = pose_to_matrix(previous.as_list()) @ back
            kept = vec3(_apply(moved, old.far_end))
        grasps = {}
        if old.grasps and robot is not None:
            for arm in old.grasps:
                tcp = robot.arms[arm].tcp.as_list()
                grasps[arm] = Pose.from_list(pose_compose(pose_inverse(tcp), pose.as_list()))
        return old.model_copy(
            update={
                "fit": reframe(fitted, pose),  # measured where it now is
                "points": points,
                "confidence": old.confidence if confidence is None else confidence,
                "frame": frame,
                "step": step,
                "pose": pose,
                "grasps": grasps,
                "carried": False,  # measured where it is: until an arm takes it again
                "partially_visible": _cut(source),
                "history": [*old.history, Measurement(frame=frame, step=step, pose=pose)],
                "far_end": kept,
            }
        )

    @_serialized
    def refresh(
        self,
        obj_id: str,
        source: Source,
        *,
        frame: FrameId,
        step: int,
        table_z: float | None = None,
        robot: RobotState | None = None,
        confidence: float | None = None,
        frames: FrameConfig = DEFAULT_FRAMES,
    ) -> ObjectRecord:
        """`remeasure` and store (an explicit call, e.g. a Reviser measuring logged frames)."""
        record = self.remeasure(
            obj_id,
            source,
            frame=frame,
            step=step,
            table_z=table_z,
            robot=robot,
            confidence=confidence,
            frames=frames,
        )
        self.objects[obj_id] = record
        return record

    @_serialized
    def point_part(
        self, obj_id: str, part: str, x: ArrayLike, robot: RobotState | None = None
    ) -> PartRecord:
        """The part's frame turned about its z by the quarter turn that brings its x nearest the
        world direction `x` (horizontal; where the object is with `robot`), its x then decided."""
        obj = self.get(obj_id)
        record = obj.parts[part]
        pose = pose_to_matrix(object_geometry(obj, robot).pose)
        want = pose[:3, :3].T @ np.asarray(x, dtype=float)  # in the object frame
        frame = record.pose.matrix()
        current = frame[:3, 0]
        delta = float(np.arctan2(current[0] * want[1] - current[1] * want[0], current @ want))
        quarter = round(delta / (np.pi / 2))
        turn = np.eye(4)
        turn[:3, :3] = quat_to_matrix(quat_from_axis_angle([0.0, 0.0, 1.0], quarter * np.pi / 2))
        size = record.size if quarter % 2 == 0 else (record.size[1], record.size[0], record.size[2])
        turned = record.model_copy(
            update={"pose": Pose.from_matrix(frame @ turn), "size": size, "both_ways": False}
        )
        self.objects[obj_id] = obj.model_copy(update={"parts": {**obj.parts, part: turned}})
        return turned

    @_serialized
    def orient(self, obj_id: str, turn_deg: float) -> ObjectRecord:
        """The object's frame turned about its z by `turn_deg`, its x then decided (`chosen`),
        unless the turn goes against a weak shape cue's pointing: two pieces of evidence that
        disagree decide nothing, so the old turns stay candidates, the new frame first. Its
        identity, geometry and parts stay where they are in the world."""
        old = self.get(obj_id)
        turn = np.eye(4)
        turn[:3, :3] = quat_to_matrix(quat_from_axis_angle([0.0, 0.0, 1.0], np.radians(turn_deg)))
        back = np.linalg.inv(turn)

        def turned(pose: Pose) -> Pose:
            return Pose.from_matrix(pose.matrix() @ turn)

        def local(pose: Pose) -> Pose:
            return Pose.from_matrix(back @ pose.matrix())

        parts = {
            name: part.model_copy(
                update={"pose": local(part.pose), "points": _apply(back, part.points)}
            )
            for name, part in old.parts.items()
        }
        strength = 0.0 if old.direction is None else old.direction.strength
        turns: tuple[float, ...] = (0.0,)
        against = old.direction is not None and old.direction.cue == "shape" and old.direction.weak
        if against and not math.isclose(turn_deg % 360.0, 0.0, abs_tol=1e-6):
            assert old.direction is not None
            others = {(t - turn_deg) % 360.0 for t in old.direction.turns}
            turns = tuple(sorted(others | {0.0}, key=lambda t: (t != 0.0, t)))
        record = old.model_copy(
            update={
                "pose": turned(old.pose),
                "fit": reframe(old.fit, turned(old.fit.frame)),
                "parts": parts,
                "part_poses": {name: local(p) for name, p in old.part_poses.items()},
                "grasps": {arm: turned(g) for arm, g in old.grasps.items()},
                "history": [m.model_copy(update={"pose": turned(m.pose)}) for m in old.history],
                "direction": Direction(cue="chosen", strength=strength, turns=turns),
            }
        )
        self.objects[obj_id] = record
        return record

    @_serialized
    def register_part(
        self,
        obj_id: str,
        part: str,
        points: ArrayLike,
        *,
        x_toward: str | None = None,
        box: tuple[ArrayLike, ArrayLike] | None = None,
        decided: bool = False,
        articulated: bool = False,
        robot: RobotState | None = None,
        self_x_toward_unset: bool = True,
    ) -> PartRecord:
        """A part of `obj_id` from world points taken where the object is now (a held one where
        `robot`'s TCP holds it); `box`: its frame (4x4 world, z up, origin at the box's center)
        and extent, as a part candidate gives them (`perception.parts`), with x `decided`;
        `articulated`: it turns or slides on the body (a grasp moves it alone). `x_toward`
        naming the part itself (an object's "x toward the handle" copied onto the handle) says
        nothing about the part's own x: it is taken as unset (as given, toward the part's own
        registered center or refused when it has none, unless `self_x_toward_unset`)."""
        if part == WHOLE:
            raise ValueError(f"{WHOLE!r} is the object's own box; name the part")
        if x_toward == part and self_x_toward_unset:
            x_toward = None
        obj = self.get(obj_id)
        now = object_geometry(obj, robot)
        pose = now.pose
        pts = _as_points(points)
        if box is None:
            fitted = oriented_box(pts, "upright")
            center, x, size = fitted.center, fitted.axes[0], fitted.size
        else:
            frame = np.asarray(box[0], dtype=float)
            center, x, size = frame[:3, 3], frame[:3, 0], np.asarray(box[1], dtype=float)
        if x_toward is not None:
            if x_toward not in obj.parts:
                raise LookupError(f"{obj_id} has no part {x_toward!r} yet: register it first")
            target = _apply(pose_to_matrix(pose), obj.parts[x_toward].pose.p)
            if x @ (target - center) < 0:
                x = -x
        elif not decided:
            x = part_x(center, x, now.center, self.base)
        world = _yaw_pose(center, float(np.arctan2(x[1], x[0])))
        inverse = pose_inverse(pose)
        record = PartRecord(
            name=part,
            pose=Pose.from_list(pose_compose(inverse, world)),
            size=(float(size[0]), float(size[1]), float(size[2])),
            x_toward=x_toward,
            both_ways=x_toward is None and not decided,
            points=_apply(pose_to_matrix(inverse), pts),
            articulated=articulated,
        )
        self.objects[obj_id] = obj.model_copy(update={"parts": {**obj.parts, part: record}})
        return record

    @_serialized
    def attach(
        self,
        obj_id: str,
        arm: Arm,
        robot: RobotState,
        closing_axis: Vec3 | None = None,
        square_axis: Vec3 | None = None,
        span_m: float | None = None,
    ) -> ObjectRecord:
        """`arm` holds `obj_id` from now on: record its current frame in the TCP frame. With
        `closing_axis` (a unit vector in the TCP frame) and the registry's `center_on_close`, a
        free object is centered on the TCP along it as closing fingers push it to their middle
        (`_section_offset`; `span_m`: the gap between the open fingers). With `square_axis` (the
        fingers' closing axis in the TCP frame), a free box is first turned about the TCP so
        that its axis nearest that one lies along it (`squared`)."""
        obj = self.get(obj_id)
        now = object_geometry(obj, robot)
        tcp = robot.arms[arm].tcp.as_list()
        to_tcp = pose_inverse(tcp)
        pose = now.pose
        if square_axis is not None and not obj.grasps and obj.fit.shape == "box":
            pose = squared(pose, now.obb.q, tcp, square_axis)
            if pose is not now.pose:
                now = object_geometry(obj.model_copy(update={"pose": Pose.from_list(pose)}))
        grasp = pose_compose(to_tcp, pose)
        if closing_axis is not None and self.center_on_close and not obj.grasps:
            axis = np.asarray(closing_axis, dtype=float)
            axis /= np.linalg.norm(axis)
            offset = _section_offset(obj, now, pose_to_matrix(to_tcp), axis, span_m)
            if offset is not None:
                grasp[:3] -= offset * axis
        update: dict[str, Any] = {
            "grasps": {
                **{a: g for a, g in obj.grasps.items() if a != arm},
                arm: Pose.from_list(grasp),
            },
            "carried": True,
            "taken": True,
            "handled": arm,
        }
        if not obj.grasps:
            update["pose"] = Pose.from_list(pose_compose(tcp, grasp))
        record = obj.model_copy(update=update)
        self.objects[obj_id] = record
        return record

    @_serialized
    def detach(self, obj_id: str, arm: Arm, robot: RobotState) -> ObjectRecord:
        """`arm` lets go of `obj_id`; once no arm holds it, it stays where its grasp puts it
        (the TCP x the recorded transform), unsettled."""
        obj = self.get(obj_id)
        if arm not in obj.grasps:
            raise ValueError(f"{obj_id} is not held by the {arm} arm")
        pose = Pose.from_list(object_geometry(obj, robot).pose)
        grasps = {a: g for a, g in obj.grasps.items() if a != arm}
        record = obj.model_copy(update={"grasps": grasps, "pose": pose})
        self.objects[obj_id] = record
        return record

    @_serialized
    def settle(
        self, obj_id: str, table: Table, robot: RobotState, log: Step = silent
    ) -> ObjectRecord:
        """Snap a free object's height onto what lies under its footprint: another registered
        object's points there no higher than `SETTLE_SLACK` above its bottom, when they cover
        `SUPPORT_SHARE` of the footprint (their median height); the bottom of another object
        whose footprint holds its center but none of whose points lie under it (a container
        whose inside no camera saw); the table plane when its center is over the table. The
        highest wins; only z changes (no model, no physics). With nothing under it, it stays."""
        obj = self.get(obj_id)
        if obj.grasps:
            raise ValueError(f"{obj_id} is held by {_arms(obj)}: open it first")
        here = object_geometry(obj, robot)
        bottom = float(_corners(here.obb)[:, 2].min())
        center = np.asarray(here.obb.center[:2])
        area = max(float(np.prod(here.obb.size[:2])), _CELL**2)
        heights: dict[str, float] = {}
        if table.over(center):
            heights["table"] = float(table.plane_z(center))
        for other in self.objects.values():
            if other.id == obj_id:
                continue
            below = object_geometry(other, robot)
            points = below.points[_footprint(below.points[:, :2], here.obb)]
            low = points[points[:, 2] <= bottom + SETTLE_SLACK]
            surface = _surface(low, area)
            if surface is not None:
                heights[other.id] = surface
            elif not len(points) and _footprint(center[None], below.obb)[0]:
                heights[other.id] = float(_corners(below.obb)[:, 2].min())
        if not heights:
            log("settle", object=obj_id, support=None, bottom_z=bottom)
            return obj
        support = max(heights, key=heights.__getitem__)
        drop = heights[support] - bottom
        record = obj.model_copy(
            update={"pose": Pose(p=vec3(np.add(obj.pose.p, (0.0, 0.0, drop))), q=obj.pose.q)}
        )
        self.objects[obj_id] = record
        log("settle", object=obj_id, support=support, z=heights[support], dz=drop, under=heights)
        return record

    @_serialized
    def on_close(
        self,
        arm: Arm,
        robot: RobotState,
        *,
        threshold: float,
        target: str | None = None,
        part: str | None = None,
        press: bool = False,
        reach: float = REACH,
        closing_axis: Vec3 | None = None,
        square_axis: Vec3 | None = None,
        span_m: float | None = None,
        turn_deg: float = 0.0,
        pin: bool = True,
        hold_frame: bool = False,
        log: Step = silent,
    ) -> ObjectRecord | None:
        """After `arm` closed (on `robot`, the settled state): when its measured opening is at or
        above `threshold` (the embodiment's `Gripper.holding_threshold`), attach `target` (the
        object the grasp aimed at), or without one the registered object nearest the TCP within
        `reach` of its box (a tie within `TIE` goes to the nearest center); its articulated
        `part` (the one the grasp aimed at; without one, the articulated part whose box the TCP
        lies within `GRIP_RADIUS` of) alone, if there is one. A `press` (closed fingers pushing)
        attaches nothing. What the arm already holds stays; None when it holds nothing known.
        An attached object whose front a choice named while its turns stay open, or with
        `hold_frame` any whose shape looks the same turned, is pinned (`pin`, unless `pin` is
        false), turned by `turn_deg`: the turn of its frame the grasp was made in.
        `closing_axis`, `square_axis` and `span_m` as `attach` takes them."""
        opening = robot.arms[arm].gripper
        if opening is None:
            raise ValueError(f"grasp relations need the {arm} gripper's measured opening")
        held = self._held(arm) or self._part_holder(arm)
        if held is not None:
            return held
        tcp = np.asarray(robot.arms[arm].tcp.p)
        decision: dict[str, Any] = {"arm": arm, "opening": opening, "threshold": threshold}
        if opening < threshold:
            log("attach", **decision, object=None, reason="the fingers closed on nothing")
            return None
        if press:
            log("attach", **decision, object=None, reason="a press holds nothing")
            return None
        chosen = target if target is not None else self._nearest(tcp, robot, reach)
        if chosen is None:
            log("attach", **decision, object=None, reason=f"nothing registered within {reach} m")
            return None
        now = object_geometry(self.get(chosen), robot)
        gap = _box_gap(tcp, now.obb)
        moving = _articulated(self.get(chosen), part, now, tcp)
        if moving is not None:
            record = self.attach_part(chosen, moving, arm, robot)
            log("attach", **decision, object=chosen, part=moving, target=target, gap=gap)
            return record
        before = self.get(chosen).pose
        record = self.attach(chosen, arm, robot, closing_axis, square_axis, span_m)
        squared_deg = float(quat_angle_deg(record.pose.q, before.q))
        if set(record.grasps) == {arm} and squared_deg >= SQUARE_MIN_DEG:
            log("square", object=chosen, arm=arm, turn_deg=round(squared_deg, 1))
        # Off, or a second arm joining a hold: the frame stays as it is.
        first = pin and set(record.grasps) == {arm}
        if first and (pinned := self.pin(chosen, turn_deg, hold=hold_frame)) is not record:
            log("pin", object=chosen, turn_deg=turn_deg)
            record = pinned
        log(
            "attach", **decision, object=chosen, target=target, gap=gap, held_by=list(record.grasps)
        )
        return record

    @_serialized
    def pin(self, obj_id: str, turn_deg: float = 0.0, hold: bool = False) -> ObjectRecord:
        """An object whose front a choice named (`orient`) but whose turns stay open (the choice
        went against a weak shape cue), or with `hold` any object whose shape looks the same
        turned a half or quarter turn (its fit's symmetry, whatever decided its x), turned by
        `turn_deg` and decided: held, the frame a grasp took it in is the one its placements and
        paths were written in, so a placement takes the orientation written (an open frame takes
        the equivalent turn nearest how it is held: a front-to-back flip once the wrist turns
        past a quarter turn on the way, as a demonstration's may). Any other object as it is."""
        old = self.get(obj_id)
        direction = old.direction
        if hold and old.fit.symmetry in ("pi", "half_pi"):
            decided = direction is not None and direction.cue == "chosen" and not direction.weak
            if decided and math.isclose(turn_deg % 360.0, 0.0, abs_tol=1e-6):
                return old
            record = self.orient(obj_id, turn_deg)
            strength = 0.0 if record.direction is None else record.direction.strength
            decided = Direction(cue="chosen", strength=strength)
            record = record.model_copy(update={"direction": decided})
            self.objects[obj_id] = record
            return record
        if direction is None or direction.cue != "chosen" or not direction.weak:
            return old
        return self.orient(obj_id, turn_deg)  # a chosen frame: no turn left open

    @_serialized
    def on_press(
        self, arm: Arm, tcp: ArrayLike, *, reach: float = PRESS_REACH, log: Step = silent
    ) -> ObjectRecord | None:
        """`arm`'s empty gripper was stopped pushing with its TCP at `tcp` (world): a
        press on the registered object no arm holds whose box lies nearest the TCP, within
        `reach` (`nearest_free`), kept in that object's frame (`ObjectRecord.presses`). None:
        nothing registered there."""
        point = np.asarray(tcp, dtype=float)
        found = self.nearest_free(point, reach)
        if found is None:
            log("press", arm=arm, object=None, reason=f"nothing registered within {reach} m")
            return None
        obj_id, gap = found
        obj = self.get(obj_id)
        local = _apply(np.linalg.inv(pose_to_matrix(obj.pose.as_list())), point)
        pressed = Pressed(arm=arm, point=vec3(local))
        record = obj.model_copy(update={"presses": [*obj.presses, pressed]})
        self.objects[obj_id] = record
        log("press", arm=arm, object=obj_id, gap=gap)
        return record

    def nearest_free(self, point: NDArray[np.float64], reach: float) -> tuple[str, float] | None:
        """The registered object no arm holds that a press stopped at `point` (world) reaches
        (`_press_gap`: within `reach` of the footprint horizontally, vertically above the
        perceived top by at most its `top_uncertainty` beyond the reach), the nearest by that
        gap, with it (0 over the footprint above the top); None if there is none."""
        near = {
            obj_id: _press_gap(point, object_geometry(obj).obb, top_uncertainty(obj))
            for obj_id, obj in self.objects.items()
            if not obj.grasps and not obj.part_grasps
        }
        near = {obj_id: gap for obj_id, gap in near.items() if gap <= reach}
        if not near:
            return None
        chosen = min(near, key=near.__getitem__)
        return chosen, near[chosen]

    @_serialized
    def attach_part(self, obj_id: str, part: str, arm: Arm, robot: RobotState) -> ObjectRecord:
        """`arm` holds the articulated `part` of `obj_id` from now on: record the part's frame
        in the TCP frame; the object stays where it is."""
        obj = self.get(obj_id)
        world = object_geometry(obj, robot).parts[part].pose
        grasp = pose_compose(pose_inverse(robot.arms[arm].tcp.as_list()), world)
        held = PartGrasp(part=part, pose=Pose.from_list(grasp))
        record = obj.model_copy(update={"part_grasps": {**obj.part_grasps, arm: held}})
        self.objects[obj_id] = record
        return record

    @_serialized
    def detach_part(self, obj_id: str, arm: Arm, robot: RobotState) -> ObjectRecord:
        """`arm` lets go of the part it holds: the part stays where its grasp puts it."""
        obj = self.get(obj_id)
        held = obj.part_grasps[arm]
        world = object_geometry(obj, robot).parts[held.part].pose
        local = pose_compose(pose_inverse(object_geometry(obj, robot).pose), world)
        record = obj.model_copy(
            update={
                "part_grasps": {a: g for a, g in obj.part_grasps.items() if a != arm},
                "part_poses": {**obj.part_poses, held.part: Pose.from_list(local)},
            }
        )
        self.objects[obj_id] = record
        return record

    @_serialized
    def on_open(
        self, arm: Arm, robot: RobotState, table: Table | None, log: Step = silent
    ) -> ObjectRecord | None:
        """After `arm` opened (or lost its hold): end its hold, and settle the object once no arm
        holds it (without a table estimate it stays where the grasp left it); a part it held stays
        where its grasp puts it. None when the arm held nothing."""
        holder = self._part_holder(arm)
        if holder is not None:
            record = self.detach_part(holder.id, arm, robot)
            part = holder.part_grasps[arm].part
            log(
                "detach",
                arm=arm,
                object=record.id,
                part=part,
                pose=record.part_poses[part].as_list(),
            )
            return record
        held = self._held(arm)
        if held is None:
            return None
        record = self.detach(held.id, arm, robot)
        log(
            "detach",
            arm=arm,
            object=record.id,
            pose=record.pose.as_list(),
            held_by=list(record.grasps),
        )
        if record.grasps or table is None:
            return record
        return self.settle(record.id, table, robot, log)

    def geometry(self, obj_id: str, robot: RobotState | None = None) -> ObjectGeometry:
        """The object now: a held one where `robot`'s TCP holds it; `stale_age` against
        `robot.step` (0 without a robot state)."""
        return object_geometry(self.get(obj_id), robot)

    def _held(self, arm: Arm) -> ObjectRecord | None:
        return next((o for o in self.objects.values() if arm in o.grasps), None)

    def _part_holder(self, arm: Arm) -> ObjectRecord | None:
        return next((o for o in self.objects.values() if arm in o.part_grasps), None)

    def _nearest(self, tcp: NDArray[np.float64], robot: RobotState, reach: float) -> str | None:
        near: dict[str, tuple[float, float]] = {}
        for obj_id, obj in self.objects.items():
            now = object_geometry(obj, robot)
            gap = _box_gap(tcp, now.obb)
            if gap <= reach:
                near[obj_id] = (gap, float(np.linalg.norm(tcp - now.center)))
        if not near:
            return None
        closest = min(gap for gap, _ in near.values())
        tied = [obj_id for obj_id, (gap, _) in near.items() if gap <= closest + TIE]
        return min(tied, key=lambda obj_id: near[obj_id][1])


def object_geometry(obj: ObjectRecord, robot: RobotState | None = None) -> ObjectGeometry:
    pose = np.asarray(obj.pose.as_list())
    if obj.grasps:
        if robot is None:
            raise ValueError(f"{obj.id} is held by {_arms(obj)}: give the robot state")
        arm, grasp = next(iter(obj.grasps.items()))
        pose = pose_compose(robot.arms[arm].tcp.as_list(), grasp.as_list())
    moved = pose_to_matrix(pose) @ np.linalg.inv(obj.fit.frame.matrix())
    obb = obj.fit.obb
    obb_rotation = moved[:3, :3] @ quat_to_matrix(obb.q)
    center = _apply(moved, obb.center)
    whole = np.concatenate([center, quat_from_matrix(obb_rotation)])
    parts = {
        WHOLE: _part(WHOLE, whole, obb.size, _apply(moved, obj.points), both_ways=not obj.decided)
    }
    holders = {g.part: arm for arm, g in obj.part_grasps.items()}
    for name, part in obj.parts.items():
        measured = pose_compose(pose, part.pose.as_list())
        world = pose_compose(pose, obj.part_poses.get(name, part.pose).as_list())
        if name in holders:
            if robot is None:
                raise ValueError(
                    f"{obj.id}'s {name} is held by the {holders[name]} arm: give the robot state"
                )
            arm = holders[name]
            world = pose_compose(robot.arms[arm].tcp.as_list(), obj.part_grasps[arm].pose.as_list())
        carried = pose_to_matrix(world) @ np.linalg.inv(pose_to_matrix(measured))
        points = _apply(carried @ pose_to_matrix(pose), part.points)
        parts[name] = _part(name, world, part.size, points, both_ways=part.both_ways)
    return ObjectGeometry(
        id=obj.id,
        name=obj.name,
        shape=obj.fit.shape,
        upright=obj.fit.upright,
        symmetry=obj.symmetry,
        pose=pose,
        obb=OrientedBox(center=vec3(center), size=obb.size, q=_quat(obb_rotation)),
        center=_apply(moved, obj.fit.center),
        top_center=_apply(moved, obj.fit.top_center),
        bottom_center=_apply(moved, obj.fit.bottom_center),
        points=_apply(moved, obj.points),
        parts=parts,
        held_by=tuple(obj.grasps),
        stale_age=0 if robot is None else robot.step - obj.step,
        partially_visible=obj.partially_visible,
        direction=obj.direction,
        shape_symmetry=obj.fit.symmetry,
        far_end=None if obj.far_end is None else _apply(moved, obj.far_end),
        handled_by=obj.handled,
    )


def start_geometry(obj: ObjectRecord) -> ObjectGeometry:
    """An object's geometry where it was first measured, as nothing held or moved it then: on
    the scene it started on, its start. Holds and moved parts recorded since are dropped (a held
    object would otherwise need the robot state, which says where it is now, not where it began)."""
    first = obj.history[0].pose if obj.history else obj.pose
    unmoved = {"pose": first, "grasps": {}, "part_grasps": {}, "part_poses": {}}
    return object_geometry(obj.model_copy(update=unmoved))


def _shape_kept(new: Fit, old: Fit) -> Fit:
    """A re-measured box keeps the turns an earlier, closer fit found its shape the same under
    when the new points fit a box too loosely to say ("none" from the residual alone): a rigid
    object does not turn irregular because a camera added noisier points, and a pose grasp or
    a placement tries only the turns its shape is taken to have (with x decided, none at all)."""
    loose = (
        new.shape == old.shape == "box" and new.symmetry == "none" and new.residual > old.residual
    )
    if loose and old.symmetry in ("pi", "half_pi"):
        return new.model_copy(update={"symmetry": old.symmetry})
    return new


def _turns(new: Fit, old: ObjectRecord, frames: FrameConfig) -> Symmetry:
    """The turns a re-measured frame may take to continue the old one: any the new fit or the
    old frame's weak direction allows, and the quarter turns when either footprint is near
    square (its long side may have swapped); x then keeps the old way, decided or not."""
    if new.symmetry == "continuous":
        return "continuous"
    if old.direction is not None and old.direction.weak and len(old.direction.turns) == 4:
        return "half_pi"
    if new.symmetry == "half_pi" or any(
        fitted.upright and fitted.obb.size[0] < frames.square * fitted.obb.size[1]
        for fitted in (new, old.fit)
    ):
        return "half_pi"
    return "pi"


def _surface(points: NDArray[np.float64], area: float) -> float | None:
    """The height of the highest level surface among points under a footprint of `area`: the
    top of the highest `_BAND` height band whose points cover `SUPPORT_SHARE` of it (in `_CELL`
    cells), their median; None if no band does."""
    z = points[:, 2]
    for level in np.unique(np.round(z / _BAND))[::-1]:
        band = points[np.abs(z / _BAND - level) <= 0.5]
        cells = len(np.unique(np.floor(band[:, :2] / _CELL).astype(np.int64), axis=0))
        if cells * _CELL**2 >= SUPPORT_SHARE * area:
            return float(np.median(band[:, 2]))
    return None


def _articulated(
    obj: ObjectRecord, part: str | None, now: ObjectGeometry, tcp: NDArray[np.float64]
) -> str | None:
    """The articulated part a close on `obj` takes: `part` if it is one, else (none aimed at)
    the articulated part whose box the TCP lies nearest, within `GRIP_RADIUS`."""
    if part is not None:
        record = obj.parts.get(part)
        return part if record is not None and record.articulated else None
    near = {
        name: _local_gap(tcp, now.parts[name])
        for name, record in obj.parts.items()
        if record.articulated
    }
    near = {name: gap for name, gap in near.items() if gap <= GRIP_RADIUS}
    return min(near, key=near.__getitem__) if near else None


def _local_gap(point: NDArray[np.float64], part: PartGeometry) -> float:
    """Distance from a point to a part's box (0 inside)."""
    local = (point - part.pose[:3]) @ quat_to_matrix(part.pose[3:])
    return float(np.linalg.norm(np.maximum(np.abs(local) - part.size / 2, 0.0)))


def _arms(obj: ObjectRecord) -> str:
    arms = list(obj.grasps)
    return f"the {arms[0]} arm" if len(arms) == 1 else "both arms"


def _part(
    name: str, pose: NDArray[np.float64], size: Vec3, points: NDArray[np.float64], both_ways: bool
) -> PartGeometry:
    turned = pose_compose(pose, [0, 0, 0, *quat_from_axis_angle([0.0, 0.0, 1.0], np.pi)])
    return PartGeometry(
        name=name,
        pose=pose,
        candidates=(pose, turned) if both_ways else (pose,),
        size=np.asarray(size),
        points=points,
    )


def _continue_frame(new: Pose, old: Pose, symmetry: Symmetry) -> Pose:
    """`new` turned about z by the multiple of 180 deg (90 deg for square footprints; none for
    round ones, which keep the old yaw) that brings its x closest to `old`'s."""
    old_x = quat_to_matrix(old.q)[:, 0]
    old_yaw = float(np.arctan2(old_x[1], old_x[0]))
    if symmetry == "continuous":
        return _pose(np.asarray(new.p), old_yaw)
    new_x = quat_to_matrix(new.q)[:, 0]
    yaw = float(np.arctan2(new_x[1], new_x[0]))
    step = np.pi / 2 if symmetry == "half_pi" else np.pi
    options = yaw + step * np.arange(round(2 * np.pi / step))
    best = options[np.argmin(np.abs(np.angle(np.exp(1j * (options - old_yaw)))))]
    return _pose(np.asarray(new.p), float(best))


def _corners(obb: OrientedBox) -> NDArray[np.float64]:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    rotated = (signs * np.asarray(obb.size) / 2) @ quat_to_matrix(obb.q).T
    corners: NDArray[np.float64] = rotated + np.asarray(obb.center, dtype=float)
    return corners


def _section_offset(
    obj: ObjectRecord,
    now: ObjectGeometry,
    to_tcp: NDArray[np.float64],
    axis: NDArray[np.float64],
    span_m: float | None,
) -> float | None:
    """Where the section the fingers close on is centered along the closing `axis` (unit, TCP
    frame; `to_tcp`: world to TCP, 4x4), when the fitted shape's chord on the axis through the
    TCP fits between the open fingers (`span_m`): the chord's middle when the points between
    the fingers (within `GRIP_RADIUS` of it) span `ONE_SIDED_M` less than it, else those
    points' span's. None (nothing to center) without points there, without a chord, or with
    one wider than the fingers open (they closed on something else: a tool's handle, not its
    head)."""
    local = _apply(to_tcp, now.points)
    along = local @ axis
    section = along[np.linalg.norm(local - np.outer(along, axis), axis=1) <= GRIP_RADIUS]
    chord = _chord(obj, now, to_tcp, axis)
    if not len(section) or chord is None:
        return None
    low, high = chord
    if span_m is not None and high - low > span_m:
        return None
    if float(section.max() - section.min()) <= high - low - ONE_SIDED_M:
        return (low + high) / 2
    return float(section.min() + section.max()) / 2


def _chord(
    obj: ObjectRecord, now: ObjectGeometry, to_tcp: NDArray[np.float64], axis: NDArray[np.float64]
) -> tuple[float, float] | None:
    """Where the line through the TCP along `axis` (TCP frame) enters and leaves the fitted
    shape (TCP-frame coordinates along it): a cylinder's surface (its radius about its axis,
    between its end faces), else its box; None when the line misses it."""
    rotation = to_tcp[:3, :3]
    box = now.obb
    box_r = rotation @ quat_to_matrix(box.q)
    center = _apply(to_tcp, np.asarray(box.center)[None])[0]
    half = np.asarray(box.size) / 2
    # the box's slabs, in its own frame: the line is origin -center, direction axis
    o, d = box_r.T @ -center, box_r.T @ axis
    if obj.fit.shape == "cylinder" and obj.fit.radius is not None:
        k = 2 if obj.fit.upright else 0  # the cylinder's axis among the box's
        rest = [i for i in range(3) if i != k]
        c = _apply(to_tcp, np.asarray(now.center)[None])[0]
        oc = box_r.T @ -c
        a = float(d[rest] @ d[rest])
        b = float(oc[rest] @ d[rest])
        disc = b * b - a * (float(oc[rest] @ oc[rest]) - obj.fit.radius**2)
        if a < 1e-12 or disc < 0:
            return None
        low, high = (-b - math.sqrt(disc)) / a, (-b + math.sqrt(disc)) / a
        slabs = [k]  # and its end faces
    else:
        low, high = -math.inf, math.inf
        slabs = [0, 1, 2]
    for i in slabs:
        if abs(d[i]) < 1e-12:
            if abs(o[i]) > half[i]:
                return None
            continue
        t1, t2 = sorted(((-half[i] - o[i]) / d[i], (half[i] - o[i]) / d[i]))
        low, high = max(low, t1), min(high, t2)
    return (float(low), float(high)) if low < high else None


def _box_gap(point: NDArray[np.float64], obb: OrientedBox) -> float:
    """Distance from a point to a box (0 inside)."""
    local = (point - np.asarray(obb.center)) @ quat_to_matrix(obb.q)
    return float(np.linalg.norm(np.maximum(np.abs(local) - np.asarray(obb.size) / 2, 0.0)))


def top_uncertainty(obj: ObjectRecord) -> float:
    """How far below its true top an object's perceived top may lie (m), from how it was
    measured: the centimeter of a well-seen box; more as less of the top face was seen (the share
    of the footprint's `TOP_CELL` cells holding points in the top centimeter: a top seen at a
    grazing angle leaves only a rim of side points there) and when the object was cut at the image
    border; at most `TOP_ERROR`."""
    geometry = object_geometry(obj)
    obb, points = geometry.obb, geometry.points
    rot = quat_to_matrix(obb.q)
    x = rot[:2, 0] / max(float(np.linalg.norm(rot[:2, 0])), 1e-9)
    size = np.maximum(np.asarray(obb.size[:2], dtype=float), 1e-6)
    cells = np.ceil(size / TOP_CELL).astype(int)
    seen = 0.0
    if len(points):
        top = points[points[:, 2] >= points[:, 2].max() - 0.01]
        offset = top[:, :2] - np.asarray(obb.center[:2])
        local = np.stack([offset @ x, offset @ (-x[1], x[0])], axis=1)
        inside = np.all(np.abs(local) <= size / 2, axis=1)
        index = np.floor((local[inside] + size / 2) / TOP_CELL).astype(int)
        index = np.minimum(index, cells - 1)
        seen = len(np.unique(index, axis=0)) / float(np.prod(cells))
    spread = TOP_BASE + TOP_LOW * (1.0 - min(seen, 1.0))
    spread += TOP_PARTIAL if obj.partially_visible else 0.0
    return float(min(spread, TOP_ERROR))


def _press_gap(point: NDArray[np.float64], obb: OrientedBox, top_error: float) -> float:
    """How far a press stopped at `point` is from pressing on a box: measured horizontally
    outside the footprint, and vertically only as above the top by more than `top_error` (how
    far below its true top the top may be perceived) or below the bottom, never as the 3-D
    distance to the box, which a top perceived low would make too large. Above the footprint and
    not far above the top is 0."""
    rot = quat_to_matrix(obb.q)
    x = rot[:2, 0] / max(float(np.linalg.norm(rot[:2, 0])), 1e-9)
    offset = point[:2] - np.asarray(obb.center[:2])
    local = np.array([offset @ x, offset @ (-x[1], x[0])])
    horizontal = np.maximum(np.abs(local) - np.asarray(obb.size[:2]) / 2, 0.0)
    half = float(np.abs(rot[2]) @ (np.asarray(obb.size) / 2))
    height = float(point[2] - obb.center[2])
    vertical = max(height - half - top_error, -height - half, 0.0)
    return float(np.hypot(np.linalg.norm(horizontal), vertical))


def _footprint(xy: NDArray[np.float64], obb: OrientedBox) -> NDArray[np.bool_]:
    """Which points' xy (N, 2) lie within a box's footprint (its x axis taken horizontally)."""
    x = quat_to_matrix(obb.q)[:2, 0]
    x = x / max(float(np.linalg.norm(x)), 1e-9)
    offset = xy - np.asarray(obb.center[:2])
    local = np.stack([offset @ x, offset @ (-x[1], x[0])], axis=1)
    inside: NDArray[np.bool_] = np.all(np.abs(local) <= np.asarray(obb.size[:2]) / 2, axis=1)
    return inside


def _cut(source: Source) -> bool:
    """Whether every view the source comes from cuts it at the image border (unknown for bare
    points: False)."""
    if isinstance(source, Lifted):
        return source.partially_visible
    if isinstance(source, Proposal):
        return all(touches_border(m) for m in source.masks.values() if m.any())
    return False


def _points_of(source: Source) -> NDArray[np.float64]:
    if isinstance(source, (Lifted, Proposal)):
        return source.points
    return _as_points(source)


def squared(
    pose: NDArray[np.float64], box_q: ArrayLike, tcp: ArrayLike, axis: ArrayLike
) -> NDArray[np.float64]:
    """`pose` (7,) of a box whose axes `box_q` gives, turned about the TCP (`tcp`, 7) so that its
    axis nearest the fingers' closing `axis` (TCP frame) lies along it, when that is
    `SQUARE_MIN_DEG` to `SQUARE_MAX_DEG` away; else `pose` itself."""
    tcp = np.asarray(tcp, dtype=float)
    closing = quat_to_matrix(tcp[3:]) @ np.asarray(axis, dtype=float)
    closing /= np.linalg.norm(closing)
    axes = quat_to_matrix(np.asarray(box_q, dtype=float))
    along = axes.T @ closing
    k = int(np.argmax(np.abs(along)))
    nearest = axes[:, k] * np.sign(along[k])
    angle = math.degrees(math.acos(min(1.0, abs(float(along[k])))))
    if not SQUARE_MIN_DEG <= angle <= SQUARE_MAX_DEG:
        return pose
    turn = np.eye(4)
    axis_angle = quat_from_axis_angle(np.cross(nearest, closing), math.radians(angle))
    turn[:3, :3] = quat_to_matrix(axis_angle)
    turn[:3, 3] = tcp[:3] - turn[:3, :3] @ tcp[:3]  # about the TCP
    return np.asarray(pose_compose(_pose7(turn), pose), dtype=np.float64)


def _pose7(m: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([m[:3, 3], quat_from_matrix(m[:3, :3])])


def _pose(origin: NDArray[np.float64], yaw: float) -> Pose:
    return Pose.from_list(_yaw_pose(origin, yaw))


def _yaw_pose(origin: ArrayLike, yaw: float) -> NDArray[np.float64]:
    return np.concatenate([np.asarray(origin, dtype=float), quat_from_axis_angle([0, 0, 1], yaw)])


def _far_end(obj: ObjectRecord) -> Vec3 | None:
    """`perception.parts.far_end` of a just measured object (its frame where it was measured),
    in the world."""
    pose = obj.fit.frame.matrix()
    inverse = np.asarray(np.linalg.inv(pose), dtype=np.float64)
    local = _apply(inverse, obj.points)
    end = far_end(local, _apply(inverse, obj.fit.obb.center))
    return None if end is None else vec3(_apply(pose, end))


def _apply(m: NDArray[np.float64], points: ArrayLike) -> NDArray[np.float64]:
    p = np.asarray(points, dtype=float)
    moved: NDArray[np.float64] = p @ m[:3, :3].T + m[:3, 3]
    return moved


def _quat(rotation: NDArray[np.float64]) -> tuple[float, float, float, float]:
    w, x, y, z = quat_from_matrix(rotation).tolist()
    return w, x, y, z
