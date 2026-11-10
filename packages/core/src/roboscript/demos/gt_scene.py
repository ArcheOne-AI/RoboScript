"""A ground-truth scene (control plane only): the perceived-object proxies a control-plane tool
fits and rebuilds specs on, built from ground truth instead of perception. No asset annotation is
used.

Every task object's box is measured once per episode from its hulls' vertices (`GtSnapshot.hulls`:
collision hulls, or a visual-only object's render shapes) as perception fits a point cloud
(`perception.fit`, shape `auto`), and carried by the object's true pose. The fit's yaw symmetry is
a guess from the box; the points decide it: a turn counts only if it leaves the points' convex
hull in place, so an object with distinct ends in a two-fold box is placed the right way round,
though
either end may still be grasped. An articulated object's moving links become parts: each link's
box measured once along the link's own axes, carried by the link's true pose, named by its joint
(`hinge0`, `hinge1`, ... for revolute joints, `slide0`, ... for prismatic ones, largest first); a
part's hinge is its joint's axis, fixed in the object's root frame, and a hinged part's x follows
its hinge: along a level one, the way the joint's axis points (a lid, a switch's rocker, however
its sides compare), across an upright one, away from it (a door).

Functional parts come from geometry alone, as perception's do: each object's and each moving
link's hulls sampled over their surfaces (`perception.parts.sample_hulls`), and the part
candidates found on them (`perception.parts.candidates`: handles, rims, necks, rockers, ends, ...)
in the object's (the link's) frame, named by kind (`handle0`, `end1`; on a moving link after it,
`slide0_handle0`) and carried with it. A moving link is a part of its joint's kind (`hinge`,
`slide`).
"""

import math
import re
from dataclasses import dataclass, field, replace
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull

from roboscript.contracts.env import GtSnapshot
from roboscript.contracts.geometry import OrientedBox, Pose, Vec3
from roboscript.contracts.robot import RobotState
from roboscript.manipulation import ObjectProxy, Part
from roboscript.perception import parts as geometric
from roboscript.perception.fit import Symmetry, fit
from roboscript.perception.registry import WHOLE
from roboscript.spatial.pose import as_matrix, transform
from roboscript.spatial.rotation import quat_from_matrix, quat_to_matrix, yaw_matrix

Array = NDArray[np.float64]
_ROUND: Final = "continuous"
_PREFIX: Final = {"revolute": "hinge", "prismatic": "slide"}
_TURNS: Final[dict[Symmetry, tuple[float, ...]]] = {
    "continuous": (45.0, 90.0, 180.0),
    "half_pi": (90.0, 180.0),
    "pi": (180.0,),
    "none": (),
}
_WEAKER: Final[dict[Symmetry, Symmetry]] = {"continuous": "half_pi", "half_pi": "pi", "pi": "none"}
SPACING_M: Final = 0.002
"""Surface points are sampled this far apart on each hull's faces (`sample_hulls`): what part
candidates are found on and an `apex` anchor searches."""
COVERS: Final = 0.8
"""A candidate on a moving link spanning this share of the link's length and width is the link
itself (a lid's whole top), not a part of it."""
CANDIDATE_M: Final = 0.005
"""A point this close to a part candidate's box counts as on it (`Scene.part_of`)."""
_OUTSIDE_M: Final = 0.004
"""A turned hull point this far outside the hull (or 5% of the footprint's short side, if more)
breaks the turn's symmetry: hull vertices are sparse and tessellated (0.8 mm for a 4 cm cylinder
of 16 sides) where an end or a handle stands out by centimeters."""


@dataclass(frozen=True, eq=False)
class MeasuredBox:
    """An object's measured box: its frame in the asset frame (`local`, 4x4), the box center and
    extents in that frame, the yaw symmetry the points bear out, whether the fit decided x's
    direction (`x_known`: a single handle), the points it was measured from and points on its
    surface (both in that frame; `surface` empty until sampled), and its part candidates (their
    frames and points in that frame)."""

    local: Array
    center: Array
    size: Array
    symmetry: Symmetry
    x_known: bool
    points: Array
    surface: Array = field(default_factory=lambda: np.zeros((0, 3)))
    candidates: tuple[geometric.Part, ...] = ()
    kinds: tuple[str, ...] = ()


def measure(true_pose: ArrayLike, points: ArrayLike, table_z: float | None = None) -> MeasuredBox:
    """Fit an object's world points (e.g. its collision hulls at `true_pose`) as perception fits a
    point cloud (`perception.fit`, shape `auto`): an upright box, its frame's x along the
    footprint's long side (or toward a single handle). With the true pose a box's x also points
    the asset's own way, so that specs fitted on one seed read the same on the next (a task's
    models share their asset frame's convention, which its judge reads): along the asset axis
    nearest to it, positive, and on a square footprint (either side) along the asset's first level
    axis; a handle keeps x. Only a footprint the points show round keeps the fit's arbitrary yaw
    (every yaw is equivalent); the fit's symmetry is weakened until its turns leave the points'
    hull in place."""
    world = np.asarray(points, dtype=float)
    fitted = fit(world, "auto", table_z)
    frame = fitted.frame.matrix()
    asset = as_matrix(true_pose)
    inverse = np.linalg.inv(frame)
    symmetry = _borne_out(
        world @ inverse[:3, :3].T + inverse[:3, 3],
        inverse[:3, :3] @ np.asarray(fitted.obb.center) + inverse[:3, 3],
        fitted.symmetry,
    )
    axes = asset[:3, :3]
    level = axes[:, 1 if int(np.argmax(np.abs(axes[2]))) == 0 else 0]
    if fitted.symmetry == _ROUND and symmetry != _ROUND:  # x along the asset's first level axis
        x = np.array([level[0], level[1], 0.0]) / np.linalg.norm(level[:2])
        frame[:3, :3] = np.column_stack([x, np.cross([0.0, 0.0, 1.0], x), [0.0, 0.0, 1.0]])
    elif fitted.symmetry == "half_pi":
        turns = (frame @ transform(yaw_matrix(math.pi * k / 2)) for k in range(4))
        frame = max(turns, key=lambda m: float(m[:3, 0] @ level))
    elif fitted.symmetry == "pi" or fitted.shape != "cylinder":  # a two-fold or irregular box
        toward = axes[:, int(np.argmax(np.abs(axes.T @ frame[:3, 0])))]
        frame = frame if frame[:3, 0] @ toward >= 0 else frame @ transform(yaw_matrix(math.pi))
    inverse = np.linalg.inv(frame)
    obb = fitted.obb
    obb_rotation = inverse[:3, :3] @ quat_to_matrix(obb.q)
    size = np.abs(obb_rotation) @ np.asarray(obb.size)
    center = inverse[:3, :3] @ np.asarray(obb.center) + inverse[:3, 3]
    local_points = world @ inverse[:3, :3].T + inverse[:3, 3]
    return MeasuredBox(
        np.linalg.inv(asset) @ frame,
        center,
        size,
        symmetry,
        fitted.symmetry == "none",
        local_points,
    )


def _borne_out(local: Array, center: Array, symmetry: Symmetry) -> Symmetry:
    """The strongest of `symmetry` and its weakenings whose turns about the vertical through
    `center` (in the box frame, as the points `local`) keep every point within `_OUTSIDE_M` of
    the points' convex hull (joggled: a flat object's points may be coplanar)."""
    hull = ConvexHull(local, qhull_options="QJ").equations
    allowed = max(_OUTSIDE_M, 0.05 * min(float(np.ptp(local[:, 0])), float(np.ptp(local[:, 1]))))

    def keeps(turn: float) -> bool:
        turned = (local - center) @ yaw_matrix(math.radians(turn)).T + center
        return float((turned @ hull[:, :3].T + hull[:, 3]).max()) <= allowed

    while symmetry != "none" and not all(keeps(turn) for turn in _TURNS[symmetry]):
        symmetry = _WEAKER[symmetry]
    return symmetry


@dataclass(frozen=True, eq=False)
class PartBox:
    """A moving link as a part: the link's name, its box in the link's own frame (`local`, the
    part frame there, origin at the box center; `size` along its axes), its joint's kind, the
    joint's axis in the object's root frame (a point on it and its direction), whether the
    part's x has a direction (`hinge_frame`), points on its surface (link frame), its part
    candidates (a drawer's handle; frames and points in the part frame, `local`) and the kinds of
    a candidate that is the link itself (a switch's rocker: what perception would find it as)."""

    link: str
    local: Array
    size: Array
    kind: Literal["revolute", "prismatic"]
    hinge: tuple[Array, Array]
    x_known: bool = False
    surface: Array = field(default_factory=lambda: np.zeros((0, 3)))
    candidates: tuple[geometric.Part, ...] = ()
    kinds: tuple[str, ...] = ()


def measure_part(link_pose: ArrayLike, points: ArrayLike) -> tuple[Array, Array]:
    """The box of a link's world points (its collision hulls at `link_pose`) along the link's
    axes, which it keeps as the link moves: the part frame in the link frame (z the link axis
    nearest the world's up when measured, pointing up; x the longer of the other two) and the
    box's extents along it."""
    link = as_matrix(link_pose)
    local = (np.asarray(points, dtype=float) - link[:3, 3]) @ link[:3, :3]
    low, high = local.min(axis=0), local.max(axis=0)
    up = int(np.argmax(np.abs(link[2, :3])))
    z = np.eye(3)[up] * np.sign(link[2, up])
    long = max((k for k in range(3) if k != up), key=lambda k: high[k] - low[k])
    x = np.eye(3)[long]
    rotation = np.column_stack([x, np.cross(z, x), z])
    return transform(rotation, (low + high) / 2), np.abs(rotation.T) @ (high - low)


def truth_proxy(
    name: str,
    true_pose: ArrayLike,
    box: MeasuredBox,
    parts: dict[str, tuple[ArrayLike, PartBox]] | None = None,
) -> ObjectProxy:
    """The proxy of an object at its true pose (the asset frame's, 7-vector or 4x4), from its box
    measured once, with parts: its part candidates, and its moving links (each a link's true pose
    now, world, and its box) with theirs. x is known only where the fit decided it (a single
    handle): an asymmetric object in a symmetric box keeps both directions, as it would from
    perception."""
    pose = as_matrix(true_pose) @ box.local
    inverse = np.linalg.inv(pose)
    found = {
        WHOLE: Part(transform(position=box.center), box.size, box.x_known, box.surface - box.center)
    }
    counts: dict[str, int] = {}

    def add(candidate: geometric.Part, frame: Array, on: str = "") -> None:
        name = f"{on}{candidate.kinds[0]}"
        counts[name] = counts.get(name, 0) + 1
        back = np.linalg.inv(candidate.frame)
        points = candidate.points @ back[:3, :3].T + back[:3, 3]
        found[f"{name}{counts[name] - 1}"] = Part(
            frame, candidate.extent, candidate.decided, points, candidate.kinds
        )

    for candidate in box.candidates:
        add(candidate, candidate.frame)
    for part, (link_pose, measured) in (parts or {}).items():
        frame = inverse @ as_matrix(link_pose) @ measured.local
        back = np.linalg.inv(measured.local)
        surface = measured.surface @ back[:3, :3].T + back[:3, 3]
        kind = (_PREFIX[measured.kind], *measured.kinds)
        found[part] = Part(frame, measured.size, measured.x_known, surface, kind)
        for candidate in measured.candidates:
            add(candidate, frame @ candidate.frame, f"{part}_")
    return ObjectProxy(name, pose, box.symmetry, found, box.points)


@dataclass(frozen=True)
class Scene:
    """An episode's objects: boxes and parts measured once, true poses (objects' and their moving
    links') at one moment."""

    boxes: dict[str, MeasuredBox]
    parts: dict[str, dict[str, PartBox]]
    poses: dict[str, Pose]
    links: dict[str, dict[str, Pose]]

    def proxy(self, name: str) -> ObjectProxy:
        parts = {
            part: (self.links[name][box.link].as_list(), box)
            for part, box in self.parts.get(name, {}).items()
            if box.link in self.links.get(name, {})
        }
        return truth_proxy(role(name), self.poses[name].as_list(), self.boxes[name], parts)

    def at(self, poses: dict[str, Pose], links: dict[str, dict[str, Pose]]) -> "Scene":
        """The measured objects at `poses`, their links at `links` (a pre-run's may name an object
        since replaced)."""
        return replace(
            self,
            poses={name: poses[name] for name in self.boxes if name in poses},
            links={name: dict(links.get(name, {})) for name in self.parts},
        )

    def hinge(self, name: str, part: str) -> tuple[Array, Array]:
        """A part's joint axis in the world now: a point on it and its direction."""
        root = self.poses[name].matrix()
        point, direction = self.parts[name][part].hinge
        return root[:3, :3] @ point + root[:3, 3], root[:3, :3] @ direction

    def part_of(
        self, name: str, point: ArrayLike, margin: float, links: bool = False
    ) -> str | None:
        """The part of `name` whose box holds the world `point`, or None: a moving link's box
        widened by `margin`, a part candidate's by `CANDIDATE_M` (it is measured on the part's
        own points); parts that move first (a lid lying on the base's top is the lid), then the
        smallest; with `links`, moving links only."""
        proxy = self.proxy(name)
        found = []
        for part, shape in proxy.parts.items():
            link = part in self.parts.get(name, {})
            if part == WHOLE or (links and not link):
                continue
            box = proxy.pose @ shape.frame
            local = np.linalg.inv(box) @ np.array([*np.asarray(point, dtype=float), 1.0])
            if np.all(np.abs(local[:3]) <= shape.size / 2 + (margin if link else CANDIDATE_M)):
                found.append((not self.moves(name, part), float(np.prod(shape.size)), part))
        return min(found)[2] if found else None

    def moves(self, name: str, part: str) -> bool:
        """Whether a part of `name` moves on a joint: a moving link, or a candidate on one."""
        return part.split("_")[0] in self.parts.get(name, {})


def measured(snapshot: GtSnapshot) -> Scene:
    """The scene of a snapshot with points: every object's box measured from its hull points, an
    articulated object's moving links as parts."""
    boxes = {}
    for name, hulls in snapshot.hulls.items():
        pose = snapshot.objects[name].as_list()
        box = measure(pose, vertices(hulls), snapshot.table_top_z)
        world = as_matrix(pose) @ box.local
        frame = np.linalg.inv(world)
        sampled = geometric.sample_hulls(hulls, SPACING_M)
        # Part candidates on what stays with the object's root: its moving links have their own.
        moving = {
            tuple(map(tuple, h))
            for link in snapshot.links.get(name, {}).values()
            for h in link.hulls
        }
        fixed = [h for h in hulls if tuple(map(tuple, h)) not in moving]
        found = geometric.candidates(
            geometric.sample_hulls(fixed, SPACING_M), world, snapshot.table_top_z
        )
        surface = sampled @ frame[:3, :3].T + frame[:3, 3]
        boxes[name] = replace(box, surface=surface, candidates=tuple(found))
    parts: dict[str, dict[str, PartBox]] = {}
    for name, links in snapshot.links.items():
        root = snapshot.objects[name].matrix()
        inverse = np.linalg.inv(root)
        found = []
        for link, truth in links.items():
            local, size = measure_part(truth.pose.as_list(), vertices(truth.hulls))
            origin = inverse[:3, :3] @ np.asarray(truth.joint.origin) + inverse[:3, 3]
            axis = inverse[:3, :3] @ np.asarray(truth.joint.axis)
            back = np.linalg.inv(truth.pose.matrix())
            sampled = geometric.sample_hulls(truth.hulls, SPACING_M)
            box = PartBox(
                link,
                local,
                size,
                truth.joint.kind,
                (origin, axis),
                surface=sampled @ back[:3, :3].T + back[:3, 3],
            )
            if box.kind == "revolute":
                box = hinge_frame(box, inverse @ truth.pose.matrix())
            on_link = geometric.candidates(sampled, truth.pose.matrix() @ box.local)
            small = [bool(np.any(c.extent[:2] < COVERS * box.size[:2])) for c in on_link]
            smaller = tuple(c for c, keep in zip(on_link, small, strict=True) if keep)
            itself = tuple(
                k for c, keep in zip(on_link, small, strict=True) if not keep for k in c.kinds
            )
            found.append(replace(box, candidates=smaller, kinds=itself))
        found.sort(key=lambda box: -float(np.prod(box.size)))
        counts: dict[str, int] = {}
        named = {}
        for box in found:
            k = counts.get(box.kind, 0)
            counts[box.kind] = k + 1
            named[f"{_PREFIX[box.kind]}{k}"] = box
        parts[name] = named
    links = {name: {link: t.pose for link, t in ls.items()} for name, ls in snapshot.links.items()}
    return Scene(boxes, parts, dict(snapshot.objects), links)


def hinge_frame(box: PartBox, link: ArrayLike) -> PartBox:
    """A hinged part's frame with x decided by its hinge, where the hinge decides it: along a
    level hinge, the way the joint's axis points (a lid, a switch's rocker: the part axis nearest
    the hinge, whichever of its sides is longer); across an upright hinge, away from it where the
    hinge lies off the part's center by over a quarter of its length (a door); else as it is.
    `link` is the link's pose in the object's root frame."""
    frame = as_matrix(link) @ box.local
    origin, axis = box.hinge
    if abs(float(axis @ frame[:3, 2])) < 0.5:  # a level hinge
        along = frame[:3, :2].T @ axis
        k = int(np.argmax(np.abs(along)))
        turn = (0.0, math.pi / 2)[k] if along[k] > 0 else (math.pi, -math.pi / 2)[k]
        size = box.size[[1, 0, 2]] if k else box.size
        return replace(box, local=box.local @ transform(yaw_matrix(turn)), size=size, x_known=True)
    offset = frame[:3, 3] - origin
    away = offset - (offset @ axis) * axis
    across = float(frame[:3, 0] @ away)
    if np.linalg.norm(away) <= 0.25 * box.size[0] or abs(across) <= 0.5 * np.linalg.norm(away):
        return box
    local = box.local if across > 0 else box.local @ transform(yaw_matrix(math.pi))
    return replace(box, local=local, x_known=True)


def vertices(hulls: list[list[Vec3]]) -> Array:
    """All vertices of an object's convex hulls, (N, 3)."""
    return np.concatenate([np.asarray(hull, dtype=float).reshape(-1, 3) for hull in hulls])


def role(name: str) -> str:
    """A task object's name as an object role (`item[0]` -> `item_0`)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def obstacles(scene: Scene, robot: RobotState, margin: float) -> list[OrientedBox]:
    """The boxes of `scene`'s objects but those an arm holds: a closed gripper whose TCP is in
    the object's box, or in one of its moving links' boxes (a door swung open), grown by
    `margin`."""
    boxes = []
    for name in scene.boxes:
        proxy = scene.proxy(name)
        world = proxy.pose @ proxy.whole.frame
        half = proxy.whole.size / 2 + margin
        held = any(
            arm.gripper_cmd < 0.5
            and (
                np.all(np.abs((np.linalg.inv(world) @ [*arm.tcp.p, 1.0])[:3]) <= half)
                or scene.part_of(name, arm.tcp.p, margin, links=True) is not None
            )
            for arm in robot.arms.values()
        )
        if not held:
            q = quat_from_matrix(world[:3, :3])
            boxes.append(
                OrientedBox(center=tuple(world[:3, 3]), size=tuple(proxy.whole.size), q=tuple(q))
            )
    return boxes
