"""The perceived-object proxy (docs/concepts/embodied-core.md): the object surface the pose
functions read (`posemath.SceneObject`), over measured geometry instead of an asset.

Its frame is the object's: z up, x along the footprint's long side, origin where the geometry puts
it (the scene registry's object frame, or a measured frame carried by the true pose). x points to
the object's own front where perception decided it (`x_known`: a single handle, a decisive shape
cue, a choice from the Memory's words), else the part's `turns` are as good (`synthesis` tries
them all): the half turn, and the quarter turns of a near-square footprint. Parts are frames and
boxes in the object frame, `whole` the object's own box; a part also has kinds (`perception.parts`:
handle, rim, neck, ...; a moving link's hinge or slide). `parts_of` gives the parts a spec names:
the part of that name, else every part of that kind, else of a kind the part classifier gives such
parts too (`ALIKE`: a neck may come out a button, a rocker a button); `part` gives
the first of them, or says which parts there are. `aim` gives where a frame's x turned toward a
part kind points (`GraspSpec.x_toward`): that part, else such a part the object's points show
(`perception.parts`), else the object's one asymmetric protrusion, so that a spec written toward a
handle holds on a model whose handle the registry did not name, or for `far_end` the end of
its footprint farthest from the footprint's center (an asymmetric object without a decided part: a
pan whose handle came out merged with its rim); `decided_kind` is the kind of the object's single
part whose direction the geometry decides, `front` that, else `low_end` where the ends of its long
side differ in height, else `far_end` where it decides one (what a fit writes there). Its frames
(`SceneObject.frames`, by kind) are metric matrices in the object frame, set by `with_frames`:
contact frames from a GraspSpec (`synthesis.grasp_candidates`) and a placement frame from a
PlaceSpec (`synthesis.placement_point_pose`), or frames of any other kind a caller carries; a
held object's pose is the TCP x the transform recorded at the close, as the registry reports it.

`from_geometry` builds it from an `ObjectGeometry` of the scene registry, what agents use; a
control-plane
tool builds the same from ground truth on the sim side (`roboscript.demos.gt_scene`).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.contracts.geometry import Pose
from roboscript.manipulation.specs import (
    END_BAND,
    END_HEIGHT_MIN_M,
    FAR_END,
    HIGH_END,
    LOW_END,
    SpecError,
)
from roboscript.perception import parts as geometric
from roboscript.perception.fit import Symmetry
from roboscript.perception.registry import WHOLE, ObjectGeometry
from roboscript.spatial.pose import as_matrix, pose_to_matrix, transform
from roboscript.spatial.rotation import yaw_matrix

Array = NDArray[np.float64]
ALIKE: Final[Mapping[str, tuple[str, ...]]] = {
    "neck": ("button",),
    "button": ("neck", "rocker"),
    "rocker": ("button",),
}
"""Part kinds the classifier also gives a part of the named kind: tried when none of it is."""
MOVING: Final = ("hinge", "slide")
"""A moving link's kinds: never what fixes the object's own front."""


@dataclass(frozen=True, eq=False)
class Part:
    """A part: its frame (4x4, object frame; origin at its box center, z up), the box's extents
    along that frame's axes, whether x is decided (else its `turns`, deg about z, are equally
    good frames), and points on its surface in its frame (what an `apex` anchor searches; may
    be empty)."""

    frame: Array
    size: Array
    x_known: bool
    points: Array = field(default_factory=lambda: np.zeros((0, 3)))
    kinds: tuple[str, ...] = ()
    turns: tuple[float, ...] = (0.0, 180.0)

    @property
    def pivot(self) -> Array:
        """The frame moved to the part's root end along a known x (else its center) and to its
        bottom (as `perception.parts.Part.pivot`)."""
        shift = np.array([-self.size[0] / 2 if self.x_known else 0.0, 0.0, -self.size[2] / 2])
        out = self.frame.copy()
        out[:3, 3] = self.frame[:3, 3] + self.frame[:3, :3] @ shift
        return out

    def turned(self, yaw: float) -> "Part":
        """The part with its frame turned by `yaw` about its z and x now decided, its origin and
        box the center and extents of its points in the turned frame (without points: the box
        turned)."""
        rotation = yaw_matrix(yaw)
        points = self.points @ rotation
        if len(points) == 0:
            size = np.abs(rotation).T @ self.size
            return replace(self, frame=self.frame @ transform(rotation), size=size, x_known=True)
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) / 2
        frame = self.frame @ transform(rotation, rotation @ center)
        return replace(self, frame=frame, size=high - low, x_known=True, points=points - center)


@dataclass(frozen=True, eq=False)
class ObjectProxy:
    """See the module docstring. `pose` is the object frame in the world (4x4); `points` its
    measured points in the object frame (may be empty); `local_frames` its frames by kind, 4x4
    matrices in the object frame (`frames` gives them in the world); `shape_symmetry` the yaw
    turns its shape looks the same under (None: `symmetry`); `chosen`: its frame's x was decided
    by a choice (a pose grasp's pin, an `orient`), not by a cue its shape or a handle gave;
    `far_end`: its far end (object frame) as its first measurement found it
    (`ObjectGeometry.far_end`; None: from its points now)."""

    name: str
    pose: Array
    symmetry: Symmetry
    parts: Mapping[str, Part]
    points: Array = field(default_factory=lambda: np.zeros((0, 3)))
    local_frames: Mapping[str, tuple[Array, ...]] = field(default_factory=dict)
    shape_symmetry: Symmetry | None = None
    chosen: bool = False
    far_end: Array | None = None

    @property
    def symmetric_shape(self) -> bool:
        """Whether its measured shape looks the same turned half a turn about z (its fit's
        symmetry, `ObjectGeometry.shape_symmetry`), whatever decided its frame's x."""
        return (self.shape_symmetry or self.symmetry) != "none"

    @property
    def whole(self) -> Part:
        return self.parts[WHOLE]

    def part(self, name: str) -> Part:
        """The first part `name` gives (`parts_of`); a clear error when it gives none."""
        found = self.parts_of(name)
        if not found:
            raise SpecError(f"{self.name} has no part {name!r}; its parts: {sorted(self.parts)}")
        return found[0][1]

    def parts_of(self, name: str) -> list[tuple[str, Part]]:
        """The parts a spec's part name gives, by name: the part of that name, else every part
        of that kind, else every part of a kind `ALIKE` it (none if there is none)."""
        if name in self.parts:
            return [(name, self.parts[name])]
        for kinds in ((name,), ALIKE.get(name, ())):
            found = [(n, p) for n, p in self.parts.items() if set(kinds) & set(p.kinds)]
            if found:
                return found
        return []

    def decided_kind(self) -> str | None:
        """The kind of the object's single fixed part whose x the geometry decides (one handle);
        None when there is none or more than one (two handles, a rod's ends)."""
        decided = [p for n, p in self.parts.items() if n != WHOLE and p.x_known and p.kinds]
        fixed = [p for p in decided if p.kinds[0] not in MOVING]
        return fixed[0].kinds[0] if len(fixed) == 1 else None

    def front(self, by_height: bool = True) -> str | None:
        """What fixes the object's own front: its `decided_kind`, else (`by_height`) `low_end`
        where the ends of its long side stand `END_HEIGHT_MIN_M` apart in height (a toe),
        else `far_end` for an asymmetric object whose footprint decides one; None when none
        does. Height comes before the footprint's far end: every measurement finds the lower end
        again, where a near-symmetric footprint's far end (a few mm off its box center) may come
        out at either end from one measurement to the next."""
        kind = self.decided_kind()
        if kind is not None:
            return kind
        if by_height and self.aim(LOW_END) is not None:
            return LOW_END
        return FAR_END if self.symmetry == "none" and self._far_end() is not None else None

    def aim(self, name: str) -> Array | None:
        """The point (object frame) a frame's x turned toward `name` points to: for `far_end` the
        end of its footprint farthest from the footprint's center (`front`); else the center of
        the first part `name` gives, else of the single decided part candidate of that kind the
        object's points show, else of its single decided candidate of any kind (its one
        asymmetric protrusion); None if there is none."""
        if name == FAR_END:
            return self._far_end()
        if name in (LOW_END, HIGH_END):
            ends = self.end_heights()
            if ends is None or abs(ends[0][1] - ends[1][1]) < END_HEIGHT_MIN_M:
                return None
            low, high = sorted(ends, key=lambda end: end[1])
            return (low if name == LOW_END else high)[0]
        found = self.parts_of(name)
        if found:
            return found[0][1].frame[:3, 3].copy()
        points = self.surface()
        if len(points) == 0:
            return None
        decided = [c for c in geometric.candidates(points, np.eye(4)) if c.decided]
        for kept in ([c for c in decided if name in c.kinds], decided):
            if len(kept) == 1:
                center: Array = kept[0].frame[:3, 3].copy()
                return center
        return None

    def end_heights(self) -> tuple[tuple[Array, float], tuple[Array, float]] | None:
        """The two ends of the whole's long side (its box's x): each one's point (object frame,
        on the box's axis at its center's height, in the middle of the end) and how tall the
        object stands there (the top of its points in the outer `END_BAND` of the length, a
        high percentile against stray points, above the object's lowest point); None for a
        footprint no longer than wide or too few points at an end."""
        whole = self.whole
        points = self.surface()
        if len(points) < 20 or whole.size[0] <= whole.size[1]:
            return None
        local = (points - whole.frame[:3, 3]) @ whole.frame[:3, :3]
        half = whole.size[0] / 2
        bottom = float(local[:, 2].min())
        ends = []
        for sign in (1.0, -1.0):
            band = local[sign * local[:, 0] >= half * (1 - 2 * END_BAND)]
            if len(band) < 5:
                return None
            top = float(np.percentile(band[:, 2], 95))
            at = whole.frame[:3, 3] + whole.frame[:3, :3] @ np.array(
                [sign * half * (1 - END_BAND), 0, 0]
            )
            ends.append((at, top - bottom))
        return ends[0], ends[1]

    def surface(self) -> Array:
        """Its surface points (object frame): the whole's if any, since the measured ones may be
        only a hull's."""
        whole = self.whole
        points = whole.points @ whole.frame[:3, :3].T + whole.frame[:3, 3]
        return points if len(points) else self.points

    def _far_end(self) -> Array | None:
        """`perception.parts.far_end` of its footprint (object frame), as its first measurement
        found it where the registry kept that (`far_end`): a near-symmetric footprint's far end
        may come out at the other end from the next view's points, and a frame turned toward it
        must not turn with it."""
        center = self.whole.frame[:3, 3]
        if self.far_end is not None:
            return np.array([self.far_end[0], self.far_end[1], center[2]])
        return geometric.far_end(self.surface(), center)

    def facing(self, name: str) -> "ObjectProxy":
        """The same object with its frame's x (and its whole's box) turned toward `name` (`aim`),
        so that orientations written in its frame (a placement's `relative`, the
        placement's object axis) mean the same on a model whose front lies elsewhere from its
        footprint's sides; itself when there is nothing to aim at."""
        aimed = self.aim(name)
        whole = self.whole
        toward = aimed - whole.frame[:3, 3] if aimed is not None else np.zeros(3)
        if np.hypot(toward[0], toward[1]) < 1e-9:
            return self
        return self.turned(math.atan2(toward[1], toward[0]))

    def turned(self, yaw: float) -> "ObjectProxy":
        """The same object with its frame (and its whole's box) turned by `yaw` (rad) about its
        z: what is written in its frame then means the turned frame (`facing`; a co-grasp's pair
        of grasps tried on the object's other end, `ActionApi.co_grasp`)."""
        turn = transform(yaw_matrix(yaw))
        back = np.linalg.inv(turn)
        parts = {n: replace(p, frame=back @ p.frame) for n, p in self.parts.items()}
        parts[WHOLE] = parts[WHOLE].turned(yaw)  # its axes back along the object's
        return replace(
            self,
            pose=self.pose @ turn,
            parts=parts,
            points=self.points @ turn[:3, :3],
            far_end=None if self.far_end is None else back[:3, :3] @ self.far_end,
            local_frames={
                kind: tuple(back @ m for m in frames) for kind, frames in self.local_frames.items()
            },
        )

    def get_pose(self) -> Pose:
        return Pose.from_matrix(self.pose)

    def frames(self, kind: str) -> tuple[Array, ...]:
        """Its frames of `kind` in the world (`SceneObject.frames`); none for a kind it lacks."""
        return tuple(self.pose @ m for m in self.local_frames.get(kind, ()))

    def with_frames(self, frames: Mapping[str, Sequence[Array]]) -> "ObjectProxy":
        """The same object carrying these frames by kind (object frame) and no others."""
        return replace(self, local_frames={k: tuple(v) for k, v in frames.items()})

    def at(self, pose: ArrayLike) -> "ObjectProxy":
        """The same object with its frame at `pose` (a 4x4 or 7-vector world pose)."""
        return replace(self, pose=as_matrix(pose))

    @classmethod
    def from_geometry(cls, geometry: ObjectGeometry) -> "ObjectProxy":
        """The proxy of a registered object as it is now (a held one where its grasp holds it),
        named by its role (what a placement's target names)."""
        pose = pose_to_matrix(geometry.pose)
        inverse = np.linalg.inv(pose)
        parts = {}
        direction = geometry.direction
        whole_turns = direction.turns if direction is not None and direction.weak else (0.0, 180.0)
        for name, part in geometry.parts.items():
            world = pose_to_matrix(part.pose)
            back = np.linalg.inv(world)
            parts[name] = Part(
                frame=inverse @ world,
                size=np.asarray(part.size, dtype=float),
                x_known=len(part.candidates) == 1,
                points=np.asarray(part.points, dtype=float) @ back[:3, :3].T + back[:3, 3],
                kinds=()
                if name == WHOLE
                else (name,),  # a registered part is named as specs name it
                turns=whole_turns if name == WHOLE else (0.0, 180.0),
            )
        points = geometry.points @ inverse[:3, :3].T + inverse[:3, 3]
        return cls(
            geometry.name,
            pose,
            geometry.symmetry,
            parts,
            points,
            shape_symmetry=geometry.shape_symmetry,
            chosen=direction is not None and direction.cue == "chosen",
            far_end=None
            if geometry.far_end is None
            else inverse[:3, :3] @ geometry.far_end + inverse[:3, 3],
        )
