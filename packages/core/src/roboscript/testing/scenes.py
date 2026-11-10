"""Synthetic RGB-D scenes for perception tests: a table, boxes, cylinders and spheres, ray-cast
through pinhole cameras, with ground-truth surface samples.

`render` gives RGB, depth (meters along the optical axis, 0 where a ray hits nothing) and the
index of the shape each pixel shows (-1 the table, -2 the floor, -3 nothing); `surface` samples a
shape's whole surface (every side) for like-for-like ground truth;
`observation` stores rendered views in a blob store as an `Observation`; `table_model` is the
scenes' table as perception models it. Colors are flat.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from roboscript.blobs import BlobStore
from roboscript.contracts.camera import CameraFrame, CameraModel
from roboscript.contracts.geometry import Pose
from roboscript.contracts.ids import FrameId, new_id
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import Arm, ArmState, RobotState
from roboscript.perception.config import PerceptionConfig, TableConfig
from roboscript.perception.frames import CameraImage
from roboscript.perception.table import EDGES, Table, landmarks
from roboscript.spatial.rotation import quat_from_axis_angle

TABLE_Z = 0.70
TABLE_HALF = (0.5, 0.4)
TABLE_RGB = (205, 190, 165)
FLOOR_RGB = (90, 90, 90)
HEAD = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -0.8, 0.6, -0.45], [0.0, -0.6, -0.8, 1.35], [0.0, 0.0, 0.0, 1.0]]
)
"""An overhead camera behind the robot (at y = -0.45 m, 1.35 m up) looking down at the table."""
PERCEPTION = PerceptionConfig(
    anchor="overhead",
    camera_words={
        "wrist_left": "a camera on the robot's left wrist",
        "wrist_right": "a camera on the robot's right wrist",
    },
    table=TableConfig(xy_box=(-0.52, 0.52, -0.42, 0.42), z_range=(0.56, 0.76)),
)
"""The perception binding of these scenes and of MockEnv's (the same table): frames are anchored
on the overhead camera, and the table (top at `TABLE_Z`, half sizes `TABLE_HALF`) is searched for a
little beyond its sides."""


def perception(**fields: Any) -> PerceptionConfig:
    """`PERCEPTION` with `fields` (`PerceptionConfig`'s) replaced; a `table` given as a dict
    replaces those keys of `PERCEPTION.table` only."""
    table = fields.get("table")
    if isinstance(table, dict):
        fields["table"] = PERCEPTION.table.model_copy(update=table)
    return PERCEPTION.model_validate(
        {
            **PERCEPTION.model_dump(),
            **{
                key: value.model_dump() if hasattr(value, "model_dump") else value
                for key, value in fields.items()
            },
        }
    )


@dataclass(frozen=True)
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_deg: float = 0.0
    rgb: tuple[int, int, int] = (200, 40, 40)


@dataclass(frozen=True)
class Cylinder:
    """Standing (axis z), or lying along the yaw direction when `lying`."""

    center: tuple[float, float, float]
    radius: float
    length: float
    yaw_deg: float = 0.0
    lying: bool = False
    rgb: tuple[int, int, int] = (40, 80, 210)


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float
    rgb: tuple[int, int, int] = (60, 60, 60)


Shape = Box | Cylinder | Sphere


@dataclass(frozen=True, eq=False)
class Camera:
    name: str
    K: NDArray[np.float64]
    cam2world: NDArray[np.float64]
    width: int
    height: int


def pinhole(
    name: str, cam2world: NDArray[np.float64], width: int = 512, height: int = 384, fovy: float = 45
) -> Camera:
    """A camera of `width` x `height` pixels with a vertical field of view of `fovy` degrees."""
    f = height / 2 / math.tan(math.radians(fovy) / 2)
    K = np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])
    return Camera(name, K, np.asarray(cam2world, dtype=float), width, height)


def look_at(eye: Sequence[float], target: Sequence[float]) -> NDArray[np.float64]:
    """cam2world of a camera at `eye` looking at `target`, image rows running downward."""
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    m[:3, 3] = eye
    return m


def _rotation(shape: Shape) -> NDArray[np.float64]:
    """Shape frame -> world: turned by the yaw; a lying cylinder's axis (local z) runs along the
    yaw direction."""
    yaw = math.radians(getattr(shape, "yaw_deg", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if isinstance(shape, Cylinder) and shape.lying:  # the local z axis along the yaw direction
        return rz @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    return rz


def render(
    shapes: Sequence[Shape], camera: Camera, table: bool = True, floor: bool = True
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.int64]]:
    """RGB (H, W, 3), depth (H, W) float32 in meters and the shape index under each pixel."""
    v, u = np.mgrid[0 : camera.height, 0 : camera.width]
    rays = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3) @ np.linalg.inv(camera.K).T
    origin, direction = camera.cam2world[:3, 3], rays @ camera.cam2world[:3, :3].T
    depth = np.full(len(rays), np.inf)  # ray parameter = camera z, since rays have camera z 1
    index = np.full(len(rays), -3)
    with np.errstate(divide="ignore", invalid="ignore"):
        for level, code, half in ((TABLE_Z, -1, TABLE_HALF), (0.0, -2, None)):
            if (code == -1 and not table) or (code == -2 and not floor):
                continue
            t = (level - origin[2]) / direction[:, 2]
            hit = origin[:2] + t[:, None] * direction[:, :2]
            ok = (t > 0) & (t < depth)
            if half is not None:
                ok &= (np.abs(hit[:, 0]) <= half[0]) & (np.abs(hit[:, 1]) <= half[1])
            depth[ok], index[ok] = t[ok], code
        for i, shape in enumerate(shapes):
            rotation = _rotation(shape)
            d = direction @ rotation
            o = np.broadcast_to((origin - np.asarray(shape.center)) @ rotation, d.shape)
            t = _hit(shape, o, d)
            closer = np.isfinite(t) & (t > 0) & (t < depth)
            depth[closer], index[closer] = t[closer], i
    colors = np.array([*(s.rgb for s in shapes), (0, 0, 0), FLOOR_RGB, TABLE_RGB], dtype=np.uint8)
    rgb = colors[index]  # -1: the table, -2: the floor, -3: black
    depth[np.isinf(depth)] = 0.0
    shape_hw = (camera.height, camera.width)
    return (
        rgb.reshape(*shape_hw, 3),
        depth.reshape(shape_hw).astype(np.float32),
        index.reshape(shape_hw),
    )


def _hit(shape: Shape, o: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ray parameter of the first entry into `shape` (local rays), NaN on a miss."""
    if isinstance(shape, Box):
        half = np.asarray(shape.size) / 2
        near, far = (-half - o) / d, (half - o) / d
        t_in = np.nanmax(np.minimum(near, far), axis=1)
        t_out = np.nanmin(np.maximum(near, far), axis=1)
        return np.where(t_in <= t_out, t_in, np.nan)
    if isinstance(shape, Sphere):
        b = np.einsum("ij,ij->i", o, d)
        a = np.einsum("ij,ij->i", d, d)
        disc = b * b - a * (np.einsum("ij,ij->i", o, o) - shape.radius**2)
        return np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.nan)
    r, half = shape.radius, shape.length / 2
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1]
    disc = b * b - a * (o[:, 0] ** 2 + o[:, 1] ** 2 - r * r)
    side = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.nan)
    side = np.where(np.abs(o[:, 2] + side * d[:, 2]) <= half, side, np.nan)
    best = side
    for level in (half, -half):
        t = (level - o[:, 2]) / d[:, 2]
        p = o[:, :2] + t[:, None] * d[:, :2]
        cap = np.where((p**2).sum(axis=1) <= r * r, t, np.nan)
        best = np.fmin(best, np.where(cap > 0, cap, np.nan))
    return best


def surface(shape: Shape, spacing: float = 0.002) -> NDArray[np.float64]:
    """World points over the whole surface of a shape, about `spacing` apart."""
    if isinstance(shape, Box):
        sx, sy, sz = shape.size
        faces = []
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            a, b = (
                np.linspace(-0.5, 0.5, max(2, round(shape.size[i] / spacing) + 1)) for i in others
            )
            grid = np.stack(np.meshgrid(a, b, indexing="ij"), axis=-1).reshape(-1, 2)
            for sign in (-0.5, 0.5):
                face = np.zeros((len(grid), 3))
                face[:, others] = grid
                face[:, axis] = sign
                faces.append(face * [sx, sy, sz])
        local = np.concatenate(faces)
    elif isinstance(shape, Sphere):
        n = max(200, int(4 * math.pi * shape.radius**2 / spacing**2))
        k = np.arange(n) + 0.5
        polar = np.arccos(1 - 2 * k / n)
        azimuth = math.pi * (1 + 5**0.5) * k
        local = shape.radius * np.stack(
            [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], 1
        )
    else:
        r, half = shape.radius, shape.length / 2
        angles = np.arange(0, 2 * math.pi, spacing / r)
        heights = np.arange(-half, half + 1e-9, spacing)
        aa, hh = np.meshgrid(angles, heights, indexing="ij")
        side = np.stack([r * np.cos(aa), r * np.sin(aa), hh], axis=-1).reshape(-1, 3)
        radii = np.arange(0, r + 1e-9, spacing)
        caps = [
            [rr * math.cos(t), rr * math.sin(t), z]
            for z in (-half, half)
            for rr in radii
            for t in np.arange(0, 2 * math.pi, spacing / max(rr, spacing))
        ]
        local = np.concatenate([side, np.asarray(caps)])
    return local @ _rotation(shape).T + np.asarray(shape.center)


def capture(
    shapes: Sequence[Shape], cameras: Sequence[Camera], table: bool = True
) -> tuple[dict[str, CameraImage], dict[str, NDArray[np.int64]]]:
    """Each camera's image and its shape-index map (which shape each pixel shows); without the
    built-in `table`, a slab among the shapes can stand in for it."""
    images, indices = {}, {}
    for camera in cameras:
        rgb, depth, index = render(shapes, camera, table=table)
        images[camera.name] = CameraImage(camera.name, camera.K, camera.cam2world, rgb, depth)
        indices[camera.name] = index
    return images, indices


def robot_state(
    step: int = 0,
    gripper: float | None = 1.0,
    tcp: Mapping[str, Sequence[float]] | None = None,
    joints: Sequence[float] = (0.0,) * 6,
) -> RobotState:
    """Two arms at rest; `tcp` maps an arm to its TCP position (grippers pointing down)."""
    w, qx, qy, qz = quat_from_axis_angle([0.0, 1.0, 0.0], math.pi / 2).tolist()
    homes: tuple[tuple[Arm, float], ...] = (("left", -0.25), ("right", 0.25))
    arms: dict[Arm, ArmState] = {}
    for arm, x in homes:
        px, py, pz = (tcp or {}).get(arm, (x, -0.2, 0.9))
        arms[arm] = ArmState(
            ee=Pose(p=(px, py, pz + 0.12), q=(w, qx, qy, qz)),
            tcp=Pose(p=(px, py, pz), q=(w, qx, qy, qz)),
            joints=list(joints),
            joints_cmd=list(joints),
            gripper=gripper,
            gripper_cmd=1.0 if gripper is None or gripper > 0.5 else 0.0,
        )
    return RobotState(step=step, step_limit=100, arms=arms, qpos_cmd=[0.0] * 14)


async def observation(
    blobs: BlobStore,
    shapes: Sequence[Shape],
    cameras: Sequence[Camera],
    robot: RobotState | None = None,
    step: int = 0,
    instruction: str | None = None,
) -> Observation:
    """Render every camera and store RGB (PNG) and depth (.npy) in the blob store."""
    frames = {}
    for camera in cameras:
        rgb, depth, _ = render(shapes, camera)
        model = CameraModel(
            name=camera.name,
            width=camera.width,
            height=camera.height,
            K=tuple(map(tuple, camera.K.tolist())),  # type: ignore[arg-type]
            cam2world=tuple(map(tuple, camera.cam2world.tolist())),  # type: ignore[arg-type]
        )
        frames[camera.name] = CameraFrame(
            camera=model, rgb=await blobs.aput_image(rgb), depth=await blobs.aput_array(depth)
        )
    return Observation(
        frame_id=FrameId(new_id("frame")),
        step=step,
        cameras=frames,
        robot=robot or robot_state(step),
        instruction=instruction,
    )


def table_model(edges: bool = True) -> Table:
    """The scenes' table as perception models it: level at `TABLE_Z`, its edges known (or all
    unknown, as when no view shows them)."""
    half_x, half_y = TABLE_HALF
    extent = (-half_x, half_x, -half_y, half_y)
    known = dict(zip(EDGES, extent, strict=True)) if edges else dict.fromkeys(EDGES)
    return Table(
        z=TABLE_Z,
        extent=extent,
        edges=known,
        normal=(0.0, 0.0, 1.0),
        method="histogram",
        inliers=1,
        rms=0.0,
        landmarks=landmarks(TABLE_Z, known),
    )
