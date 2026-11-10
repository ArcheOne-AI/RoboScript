"""The robot's own geometry at its measured joints, for removing its points: no privileged data.

Each arm's links carry their URDF collision geometry (meshes, boxes, cylinders, spheres), sampled
densely over its surface in the link's frame, and the cuRobo collision spheres the embodiment
ships (they reach a little beyond some visual meshes); fingers keep their surface samples only,
since their spheres bulge into what they hold. Forward kinematics at the measured joints (the
motion planner's `fk`: the ee pose, and the other links by name) places them in the world
(`SelfFilter`), giving a `RobotShape`: a point within the margin of a surface sample, or inside a
sphere grown by it, is the robot's. The margin shrinks with the point's camera depth, so a wrist
camera a few centimeters from the fingers keeps what they hold. FK reports the fingers at the
opening the planner locks them at, so a finger slides along its joint axis to the measured
opening, or, when the env measures none, is repeated over its travel. Every finger follows the
gripper joint one to one (mimic joints). A cuRobo yml of format 2 names
its ee link as its one tool frame and locks no finger: its fingers are the prismatic joints of its
cspace, on links the planner does not model, so each is placed from its parent link's pose at the
joint's URDF origin (`Finger.mount`). Links FK does not report (the robot's body beyond its arms)
are not covered.
"""

import itertools
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import FKRequest, Planner
from roboscript.contracts.robot import Arm, RobotState
from roboscript.embodiments import EmbodimentProfile
from roboscript.perception.config import DEFAULTS, SelfFilterConfig

SPACING = 0.004
"""Spacing of the samples over a link's collision geometry (meters)."""


@dataclass(frozen=True)
class Finger:
    """A finger link that FK reports at joint value `locked`; it slides along `axis` (a unit
    vector in its own frame) as its joint moves within [`lower`, `upper`] (meters)."""

    axis: tuple[float, float, float]
    locked: float
    lower: float
    upper: float
    mount: tuple[str, NDArray[np.float64]] | None = None
    """For a finger FK does not report: its parent link and the joint's origin in it (4x4); the
    finger is there at joint value `locked`."""


@dataclass(frozen=True, eq=False)
class ArmBody:
    """One arm, in its links' frames (meters): `surfaces[link]` (K, 3) samples of the link's
    surface, `spheres[link]` (N, 4) centers and radii; the `ee_link` is placed by FK's ee pose.
    `base` is the world pose of the robot root the arm's FK hangs off; `gripper_scale` maps the
    normalized opening (0 closed, 1 open) linearly onto the finger joint."""

    base: Pose
    ee_link: str
    fingers: Mapping[str, Finger]
    gripper_scale: tuple[float, float]
    spheres: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    surfaces: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def links(self) -> list[str]:
        return list(dict.fromkeys([*self.surfaces, *self.spheres]))


@dataclass(frozen=True, eq=False)
class RobotShape:
    """The robot in the world: points within the margin of a `surface` sample (K, 3), or inside a
    sphere (M, 4) grown by it, are the robot's. The margin is `margin`, or with `per_meter` that
    much per meter of the point's camera depth, kept within [`min_margin`, `margin`]."""

    spheres: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 4)))
    surface: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    margin: float = 0.0
    per_meter: float | None = None
    min_margin: float = 0.0

    def margins(self, n: int, depth: ArrayLike | None = None) -> NDArray[np.float64]:
        """The margin of each of `n` points at camera `depth` (meters)."""
        if depth is None or self.per_meter is None:
            return np.full(n, self.margin)
        return np.clip(
            self.per_meter * np.asarray(depth, dtype=float), self.min_margin, self.margin
        )

    def contains(self, points: ArrayLike, depth: ArrayLike | None = None) -> NDArray[np.bool_]:
        """Which points (N, 3), seen at camera `depth` (N,), are the robot's."""
        p = np.asarray(points, dtype=float)
        margin = self.margins(len(p), depth)
        inside = inside_spheres(p, self.spheres, margin)
        if len(self.surface) and len(p):
            reach = float(margin.max())
            lo, hi = self.surface.min(axis=0) - reach, self.surface.max(axis=0) + reach
            near = np.flatnonzero(np.all((p >= lo) & (p <= hi), axis=1) & ~inside)
            if len(near):
                gap, _ = cKDTree(self.surface).query(p[near], distance_upper_bound=reach)
                inside[near[gap <= margin[near]]] = True
        return inside


def join(shapes: Iterable[RobotShape]) -> RobotShape:
    """One shape of several with the same margins (the arms of one robot)."""
    shapes = list(shapes)
    if len({(s.margin, s.per_meter, s.min_margin) for s in shapes}) > 1:
        raise ValueError("robot shapes to join need one margin")
    if not shapes:
        return RobotShape()
    return replace(
        shapes[0],
        spheres=np.vstack([s.spheres for s in shapes]),
        surface=np.vstack([s.surface for s in shapes]),
    )


@dataclass(frozen=True, eq=False)
class RobotBody:
    arms: Mapping[Arm, ArmBody]

    @property
    def origin(self) -> NDArray[np.float64]:
        """The mean world position of the arms' bases: where the robot-base midline runs."""
        return np.mean([arm.base.p for arm in self.arms.values()], axis=0)


def load_arm_body(
    curobo_yml: Path,
    urdf: Path,
    base: Pose,
    gripper_scale: tuple[float, float],
    spacing: float = SPACING,
) -> ArmBody:
    """An arm from its cuRobo robot config (the links FK reports, spheres, locked finger joints)
    and the URDF (collision geometry sampled every `spacing` meters; the finger joints' child
    links, axes and limits)."""
    kinematics = yaml.safe_load(curobo_yml.read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    spheres = kinematics["collision_spheres"]
    if isinstance(spheres, str):
        path = rebase(spheres, curobo_yml.parent)
        spheres = yaml.safe_load(path.read_text(encoding="utf-8"))["collision_spheres"]
    ee = kinematics.get("ee_link") or _tool_frame(kinematics, curobo_yml)
    links = [ee, *(link for link in kinematics["collision_link_names"] if link != ee)]
    joints = _urdf_joints(urdf)
    fingers = {}
    locks = kinematics.get("lock_joints") or {}
    for name, value in locks.items():
        joint = joints.get(name)
        if joint is None or joint.type != "prismatic":
            raise ValueError(f"{curobo_yml.name} locks {name!r}, not a prismatic joint in {urdf}")
        fingers[joint.child] = Finger(joint.axis, float(value), joint.lower, joint.upper)
    if not locks and float(kinematics.get("format_version", 1)) >= 2:
        for name in kinematics["cspace"]["joint_names"]:
            joint = joints.get(name)
            if (
                joint is not None
                and joint.type == "prismatic"
                and joint.child not in links
                and joint.parent in links
            ):
                mount = (joint.parent, joint.origin)
                fingers[joint.child] = Finger(joint.axis, 0.0, joint.lower, joint.upper, mount)
        links += [link for link in fingers if link not in links]
    surfaces = link_surfaces(urdf, links, spacing)
    return ArmBody(
        base=base,
        ee_link=ee,
        fingers=fingers,
        gripper_scale=gripper_scale,
        spheres={
            link: np.array([[*s["center"], s["radius"]] for s in spheres[link]], dtype=float)
            for link in links
            if spheres.get(link)
        },
        surfaces=surfaces,
    )


def _tool_frame(kinematics: dict[str, object], curobo_yml: Path) -> str:
    """A format-2 yml's ee link: its one tool frame."""
    frames = kinematics.get("tool_frames")
    if not isinstance(frames, list) or len(frames) != 1:
        raise ValueError(f"{curobo_yml.name} names no ee_link and not exactly one tool frame")
    return str(frames[0])


def robot_body(profile: EmbodimentProfile) -> RobotBody:
    """The bodies of a profile's arms from their planner assets (cuRobo config and URDF)."""
    arms = {}
    for arm, spec in profile.arms.items():
        planner = spec.planner
        if planner is None or planner.curobo is None:
            raise ValueError(
                f"embodiment {profile.name!r}: the {arm} arm has no cuRobo config to take "
                "collision spheres from; build its ArmBody by hand"
            )
        curobo, urdf = profile.asset(planner.curobo), profile.asset(planner.urdf)
        arms[arm] = load_arm_body(curobo, urdf, planner.base, spec.gripper.scale)
    return RobotBody(arms=arms)


class SelfFilter:
    """The robot's shape for a robot state: FK at its measured joints and openings."""

    def __init__(
        self, planner: Planner, body: RobotBody, config: SelfFilterConfig = DEFAULTS.self_filter
    ) -> None:
        self.planner = planner
        self.body = body
        self.config = config

    async def shape(self, robot: RobotState) -> RobotShape:
        poses: dict[Arm, dict[str, Pose]] = {}
        for arm, body in self.body.arms.items():
            state = robot.arms[arm]
            if state.joints is None:
                raise ValueError(f"the self-filter needs the {arm} arm's measured joints")
            mounted = {n: f.mount for n, f in body.fingers.items() if f.mount is not None}
            links = [name for name in body.links if name != body.ee_link and name not in mounted]
            request = FKRequest(arm=arm, base=body.base, joints=[state.joints], links=links)
            fk = await self.planner.fk(request)
            poses[arm] = {name: fk.links[name][0] for name in links}
            poses[arm][body.ee_link] = fk.ee[0]
            for name, (parent, origin) in mounted.items():
                poses[arm][name] = Pose.from_matrix(poses[arm][parent].matrix() @ origin)
        openings = {arm: robot.arms[arm].gripper for arm in self.body.arms}
        return robot_shape(self.body, poses, openings, self.config)


def robot_shape(
    body: RobotBody,
    links: Mapping[Arm, Mapping[str, Pose]],
    openings: Mapping[Arm, float | None],
    config: SelfFilterConfig = DEFAULTS.self_filter,
) -> RobotShape:
    """The robot's shape from its arms' link poses (world, from FK) and measured openings."""
    arms = [
        arm_shape(b, links[arm], openings[arm], config.margin, config.finger_samples)
        for arm, b in body.arms.items()
    ]
    return replace(join(arms), per_meter=config.per_meter, min_margin=config.min_margin)


def rebase(asset: str, directory: Path) -> Path:
    """An asset path from an embodiment file, which may point into another checkout (or at
    a `${ASSETS_PATH}` placeholder): only what follows the embodiment directory's name is kept."""
    marker = f"/{directory.name}/"
    if marker not in asset:
        raise ValueError(
            f"{asset!r} does not lie in an embodiment directory named {directory.name}"
        )
    path = directory / asset.rsplit(marker, 1)[1]
    if not path.is_file():
        raise FileNotFoundError(f"{asset!r} rebased to {path}, which does not exist")
    return path


def link_surfaces(
    urdf: Path, links: Iterable[str], spacing: float = SPACING
) -> dict[str, NDArray[np.float64]]:
    """Samples, about `spacing` apart, of the URDF collision geometry of `links`, each in its
    link's frame: meshes (STL, relative to the URDF's directory; `scale` applied), boxes,
    cylinders and spheres."""
    wanted = set(links)
    out = {}
    for link in ET.parse(urdf).getroot().iter("link"):
        if link.get("name") not in wanted:
            continue
        samples = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None or not len(geometry):
                continue
            m = _origin(collision.find("origin"))
            local = _sample(_triangles(geometry[0], urdf.parent), spacing)
            samples.append(local @ m[:3, :3].T + m[:3, 3])
        if samples:
            out[link.get("name", "")] = np.vstack(samples)
    return out


def read_stl(path: Path) -> NDArray[np.float64]:
    """Triangles (T, 3, 3) of a binary or ASCII STL file."""
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], "little") if len(data) >= 84 else -1
    if len(data) == 84 + 50 * count:
        record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        return np.frombuffer(data, record, count=count, offset=84)["vertices"].astype(float)
    numbers = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
    if not numbers:
        raise ValueError(f"{path} is neither a binary nor an ASCII STL file")
    return np.array(numbers, dtype=float).reshape(-1, 3, 3)


def _triangles(geometry: ET.Element, directory: Path) -> NDArray[np.float64]:
    """Triangles of one URDF geometry element, in its collision frame."""
    kind = geometry.tag
    if kind == "mesh":
        path = directory / geometry.get("filename", "")
        if path.suffix.lower() != ".stl":
            raise ValueError(f"{path}: only STL collision meshes are read")
        scale = np.array(geometry.get("scale", "1 1 1").split(), dtype=float)
        return read_stl(path) * scale
    if kind == "box":
        half = np.array(geometry.get("size", "0 0 0").split(), dtype=float) / 2
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
        faces = [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
        return corners[np.array(faces)]
    if kind in ("cylinder", "sphere"):
        r = float(geometry.get("radius", "0"))
        if kind == "cylinder":
            half = float(geometry.get("length", "0")) / 2
            heights, radii = np.array([-half, half]), np.array([r, r])
        else:  # 16 bands of latitude
            polar = np.linspace(0, np.pi, 17)
            heights, radii = r * np.cos(polar), r * np.sin(polar)
        a = np.linspace(0, 2 * np.pi, 33)
        rings = [
            np.column_stack([q * np.cos(a), q * np.sin(a), np.full(33, z)])
            for z, q in zip(heights, radii, strict=True)
        ]
        triangles = []
        for lo, hi in itertools.pairwise(rings):
            triangles += [
                np.stack([lo[:-1], lo[1:], hi[1:]], 1),
                np.stack([lo[:-1], hi[1:], hi[:-1]], 1),
            ]
        if kind == "cylinder":  # the caps, as fans from their centers
            for ring in (rings[0], rings[-1]):
                center = np.broadcast_to([0.0, 0.0, ring[0, 2]], ring[:-1].shape)
                triangles.append(np.stack([center, ring[:-1], ring[1:]], 1))
        return np.concatenate(triangles)
    raise ValueError(f"URDF collision geometry {kind!r} is not supported")


def _sample(triangles: NDArray[np.float64], spacing: float) -> NDArray[np.float64]:
    """Points over the triangles, about `spacing` apart (area-weighted, a fixed seed)."""
    edge_a, edge_b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    if area.sum() <= 0:
        return triangles.reshape(-1, 3)
    rng = np.random.default_rng(0)
    n = int(np.ceil(area.sum() / spacing**2))
    pick = rng.choice(len(triangles), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return triangles[pick, 0] + u[:, None] * edge_a[pick] + v[:, None] * edge_b[pick]


def _origin(element: ET.Element | None) -> NDArray[np.float64]:
    """The 4x4 transform of a URDF `origin` (xyz, then fixed-axis roll, pitch, yaw)."""
    m = np.eye(4)
    if element is None:
        return m
    roll, pitch, yaw = np.array(element.get("rpy", "0 0 0").split(), dtype=float)
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = np.array(element.get("xyz", "0 0 0").split(), dtype=float)
    return m


class _Joint(NamedTuple):
    type: str
    child: str
    axis: tuple[float, float, float]
    lower: float
    upper: float
    parent: str = ""
    origin: NDArray[np.float64] = np.eye(4)


def _urdf_joints(urdf: Path) -> dict[str, _Joint]:
    def attr(joint: ET.Element, tag: str, key: str, default: str) -> str:
        element = joint.find(tag)
        return default if element is None else element.get(key, default)

    joints = {}
    for joint in ET.parse(urdf).getroot().iter("joint"):
        x, y, z = np.array(attr(joint, "axis", "xyz", "1 0 0").split(), dtype=float)
        norm = float(np.linalg.norm([x, y, z])) or 1.0  # fixed joints may give "0 0 0"
        joints[joint.get("name", "")] = _Joint(
            type=joint.get("type", ""),
            child=attr(joint, "child", "link", ""),
            axis=(float(x) / norm, float(y) / norm, float(z) / norm),
            lower=float(attr(joint, "limit", "lower", "0")),
            upper=float(attr(joint, "limit", "upper", "0")),
            parent=attr(joint, "parent", "link", ""),
            origin=_origin(joint.find("origin")),
        )
    return joints


def arm_shape(
    body: ArmBody,
    links: Mapping[str, Pose],
    opening: float | None,
    margin: float,
    samples: int = 5,
) -> RobotShape:
    """The world shape of an arm whose links are at `links` (world poses from FK, fingers at their
    locked value), fingers moved to the measured `opening` (None: over their travel); a finger
    with surface samples leaves its spheres out."""
    spheres, surface = [], []
    low, high = body.gripper_scale
    for name in body.links:
        m = links[name].matrix()
        finger = body.fingers.get(name)
        if finger is None:
            shifts = [np.zeros(3)]
        else:
            travel = np.linspace(finger.lower, finger.upper, samples)
            values = travel if opening is None else [low + opening * (high - low)]
            axis = m[:3, :3] @ np.asarray(finger.axis)
            shifts = [
                (q - finger.locked) * axis for q in np.clip(values, finger.lower, finger.upper)
            ]
        if name in body.surfaces:
            world = body.surfaces[name] @ m[:3, :3].T + m[:3, 3]
            surface += [world + shift for shift in shifts]
        if name in body.spheres and (finger is None or name not in body.surfaces):
            local = body.spheres[name]
            centers = local[:, :3] @ m[:3, :3].T + m[:3, 3]
            spheres += [np.column_stack([centers + shift, local[:, 3]]) for shift in shifts]
    return RobotShape(
        spheres=np.vstack(spheres) if spheres else np.zeros((0, 4)),
        surface=np.vstack(surface) if surface else np.zeros((0, 3)),
        margin=margin,
    )


def inside_spheres(
    points: ArrayLike, spheres: ArrayLike, grow: ArrayLike = 0.0
) -> NDArray[np.bool_]:
    """Which points (N, 3) lie inside any of the spheres (M, 4) grown by `grow` (per point)."""
    points = np.asarray(points, dtype=float)
    spheres = np.asarray(spheres, dtype=float)
    inside = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(spheres):
        return inside
    grow = np.broadcast_to(np.asarray(grow, dtype=float), (len(points),))
    reach = spheres[:, 3] + float(grow.max())
    lo = (spheres[:, :3] - reach[:, None]).min(axis=0)
    hi = (spheres[:, :3] + reach[:, None]).max(axis=0)
    near = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
    if not len(near):
        return inside
    tree = cKDTree(points[near])
    hits = tree.query_ball_point(spheres[:, :3], reach, return_sorted=False)
    for sphere, found in zip(spheres, hits, strict=True):
        index = near[found]
        distance = np.linalg.norm(points[index] - sphere[:3], axis=1)
        inside[index[distance <= sphere[3] + grow[index]]] = True
    return inside
