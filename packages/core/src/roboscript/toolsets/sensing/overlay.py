"""Overlays on camera images, drawn from calibration and proprioception only
(docs/concepts/learning.md): TCP markers with a line from the ee and the TCP axes, a table grid
once a table estimate exists, labeled points and boxes a caller asks for, and frames a caller gives
(a registered object's or part's axes as arrows, x red, y green, z blue; where its x is open, the
candidate x arrows labeled A, B, ...; a small gizmo of the world axes at the robot base), and a
close-up of an overlay around an object's box (`close_up`). Nothing here knows the scene.

Geometry is projected with the image's own camera (`spatial.camera` conventions); segments are
clipped to 2 cm in front of the camera and to the image, so wrist cameras close to the arms draw
sensibly.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw, ImageFont

from roboscript.contracts.camera import CameraModel
from roboscript.contracts.geometry import OrientedBox
from roboscript.contracts.robot import Arm, RobotState
from roboscript.spatial.camera import project
from roboscript.spatial.pose import transform_points
from roboscript.spatial.rotation import quat_to_matrix

ARM_COLORS: Final[Mapping[Arm, tuple[int, int, int]]] = {
    "left": (0, 225, 255),  # cyan
    "right": (255, 140, 0),  # orange
}
AXIS_COLORS: Final = ((235, 45, 45), (40, 205, 70), (60, 110, 255))  # x red, y green, z blue
MARK_COLOR: Final = (255, 0, 200)  # magenta: points and boxes a caller asked for
_GRID: Final = (20, 20, 20, 120)
_NEAR: Final = 0.02  # meters in front of the camera where segments are cut


@dataclass(frozen=True)
class TableGrid:
    """A table estimate to draw the grid on: the plane's height `z` and its extent
    (x0, x1, y0, y1), in meters in the world frame."""

    z: float
    extent: tuple[float, float, float, float]


@dataclass(frozen=True)
class Mark:
    """A labeled point: world coordinates (projected into the image) or a pixel (u, v)."""

    label: str
    world: tuple[float, float, float] | None = None
    pixel: tuple[float, float] | None = None


@dataclass(frozen=True)
class Axes:
    """A frame to draw: `label`, its world pose (4x4), arrow `length` (m); `candidates`: the
    turns (deg, about its z) of the x arrows to draw labeled A, B, ... instead of its x and y."""

    label: str
    pose: NDArray[np.float64]
    length: float
    candidates: tuple[float, ...] = ()


CANDIDATE_LABELS: Final = "ABCD"


@dataclass(frozen=True)
class Box:
    """A labeled pixel box (u0, v0, u1, v1)."""

    label: str
    box: tuple[float, float, float, float]


def draw(
    rgb: NDArray[np.uint8],
    camera: CameraModel,
    *,
    robot: RobotState | None = None,
    axes_m: float | None = None,
    grid: TableGrid | None = None,
    grid_m: float = 0.1,
    marks: Sequence[Mark] = (),
    boxes: Sequence[Box] = (),
    axes: Sequence[Axes] = (),
    base: ArrayLike | None = None,
) -> NDArray[np.uint8]:
    """`rgb` (H, W, 3) seen by `camera` with the overlays asked for: `robot` draws each arm's TCP
    as a dot (`ARM_COLORS`), and with `axes_m` also a line from the ee to the TCP and the TCP's
    axes that long; `grid` draws lines every `grid_m` meters on the table plane, labeled in
    meters; `marks` and `boxes` are drawn in `MARK_COLOR`; `axes` as arrows (`Axes`); `base`, a
    world point (the robot base at the table's height), gets the world's x and y as a gizmo."""
    canvas = _Canvas(rgb, camera)
    if grid is not None:
        _grid(canvas, grid, grid_m)
    if robot is not None:
        _robot(canvas, robot, axes_m)
    for mark in marks:
        at = canvas.pixel(mark.world) if mark.world is not None else mark.pixel
        if at is not None and canvas.inside(at):
            canvas.dot(at, MARK_COLOR, mark.label)
    for box in boxes:
        canvas.box(box.box, MARK_COLOR, box.label)
    for frame in axes:
        _axes(canvas, frame)
    if base is not None:
        at = np.asarray(base, dtype=float)
        _axes(canvas, Axes("base", _translation(at), _BASE_M), arrows=(0, 1))
    return canvas.done()


_BASE_M: Final = 0.08
"""The base gizmo's arrow length (m)."""


def _axes(canvas: "_Canvas", frame: Axes, arrows: Sequence[int] = (0, 1, 2)) -> None:
    """A frame's arrows from its origin, labeled; `candidates`: labeled x arrows and z only."""
    rotation, origin = frame.pose[:3, :3], frame.pose[:3, 3]
    tips: list[tuple[NDArray[np.float64], tuple[int, int, int], str]] = []
    if frame.candidates:
        for label, turn in zip(CANDIDATE_LABELS, frame.candidates, strict=False):
            a = math.radians(turn)
            x = rotation[:, 0] * math.cos(a) + rotation[:, 1] * math.sin(a)
            tips.append((x, AXIS_COLORS[0], label))
        tips.append((rotation[:, 2], AXIS_COLORS[2], "z"))
    else:
        tips = [(rotation[:, k], AXIS_COLORS[k], "xyz"[k]) for k in arrows]
    for direction, color, label in tips:
        canvas.arrow(origin, origin + direction * frame.length, color, label)
    at = canvas.pixel(origin)
    if at is not None and canvas.inside(at) and frame.label:
        canvas.text((at[0] - 2 * canvas.radius, at[1] + canvas.radius), frame.label)


CLOSE_UP_SIDE: Final = 448
"""Pixels of a close-up's longer side."""


def close_up(
    overlay: NDArray[np.uint8],
    camera: CameraModel,
    box: OrientedBox,
    origin: ArrayLike,
    reach: float,
) -> NDArray[np.uint8]:
    """`overlay` cut around an object's `box` and `reach` meters around its frame's `origin` (its
    arrows' length), with a margin, enlarged to `CLOSE_UP_SIDE`: what a frame's arrows point
    at, big enough to read. The whole overlay when none of it is in front of the camera."""
    rotation = quat_to_matrix(box.q)
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    corners = np.asarray(box.center) + (signs * np.asarray(box.size) / 2) @ rotation.T
    around = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    ring = np.asarray(origin, dtype=float)[:3] + reach * around
    uv, _, front = project(np.vstack([corners, ring]), camera.K, camera.cam2world)
    h, w = overlay.shape[:2]
    if not front.any():
        return overlay
    (u0, v0), (u1, v1) = np.nanmin(uv[front], axis=0), np.nanmax(uv[front], axis=0)
    half = max(u1 - u0, v1 - v0) * 0.6 + 12
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    left, right = int(max(0, cu - half)), int(min(w, cu + half))
    top, bottom = int(max(0, cv - half)), int(min(h, cv + half))
    if right - left < 2 or bottom - top < 2:
        return overlay
    crop = Image.fromarray(np.ascontiguousarray(overlay[top:bottom, left:right, :3]))
    scale = CLOSE_UP_SIDE / max(crop.size)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    return np.asarray(crop.resize(size, Image.Resampling.LANCZOS))


def _translation(at: NDArray[np.float64]) -> NDArray[np.float64]:
    m = np.eye(4)
    m[:3, 3] = at
    return m


def grid_values(low: float, high: float, spacing: float) -> list[float]:
    """The multiples of `spacing` within [low, high]."""
    first, last = math.ceil(low / spacing - 1e-9), math.floor(high / spacing + 1e-9)
    return [k * spacing for k in range(first, last + 1)]


def _robot(canvas: "_Canvas", robot: RobotState, axes_m: float | None) -> None:
    for arm, state in robot.arms.items():
        color = ARM_COLORS[arm]
        tcp = np.array(state.tcp.p)
        if axes_m is not None:
            canvas.segment(state.ee.p, tcp, color, 2)
            rotation = quat_to_matrix(state.tcp.q)
            for axis, axis_color in enumerate(AXIS_COLORS):
                canvas.segment(tcp, tcp + rotation[:, axis] * axes_m, axis_color, 2)
        at = canvas.pixel(tcp)
        if at is not None and canvas.inside(at):
            canvas.dot(at, color, arm[0].upper())


def _grid(canvas: "_Canvas", grid: TableGrid, spacing: float) -> None:
    """Lines of constant x labeled at their lowest visible point, lines of constant y at their
    leftmost one."""
    x0, x1, y0, y1 = grid.extent
    z, digits = grid.z, max(0, -math.floor(math.log10(spacing) + 1e-9))
    for x in grid_values(x0, x1, spacing):
        if shown := canvas.segment((x, y0, z), (x, y1, z), _GRID, 1):
            canvas.text(max(shown, key=lambda p: p[1]), f"x={x:.{digits}f}")
    for y in grid_values(y0, y1, spacing):
        if shown := canvas.segment((x0, y, z), (x1, y, z), _GRID, 1):
            canvas.text(min(shown, key=lambda p: p[0]), f"y={y:.{digits}f}")


class _Canvas:
    """A transparent layer over the image with drawing in world or pixel coordinates."""

    def __init__(self, rgb: NDArray[np.uint8], camera: CameraModel) -> None:
        self.image = Image.fromarray(np.ascontiguousarray(rgb[..., :3])).convert("RGBA")
        self.layer = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        self.pen = ImageDraw.Draw(self.layer)
        self.K = np.array(camera.K, dtype=float)
        self.world_to_camera = np.linalg.inv(np.array(camera.cam2world, dtype=float))
        self.size = (camera.width, camera.height)
        scale = max(self.size) / 640
        self.radius = max(3, round(6 * scale))
        self.font = ImageFont.load_default(size=max(12, round(13 * scale)))

    def done(self) -> NDArray[np.uint8]:
        return np.asarray(Image.alpha_composite(self.image, self.layer).convert("RGB"))

    def inside(self, at: tuple[float, float]) -> bool:
        return 0 <= at[0] < self.size[0] and 0 <= at[1] < self.size[1]

    def pixel(self, point: ArrayLike) -> tuple[float, float] | None:
        """Where a world point appears; None if it is not in front of the camera."""
        camera = transform_points(self.world_to_camera, point)
        return None if camera[2] <= _NEAR else self._project(camera)

    def segment(
        self, a: ArrayLike, b: ArrayLike, color: Iterable[int], width: int
    ) -> list[tuple[float, float]] | None:
        """Draw a world segment cut at the near plane and at the image border; the pixels of
        its ends as drawn, or None if nothing of it is in view."""
        ca, cb = transform_points(self.world_to_camera, np.array([a, b], dtype=float))
        if ca[2] < _NEAR and cb[2] < _NEAR:
            return None
        if ca[2] < _NEAR:
            ca = ca + (cb - ca) * (_NEAR - ca[2]) / (cb[2] - ca[2])
        elif cb[2] < _NEAR:
            cb = cb + (ca - cb) * (_NEAR - cb[2]) / (ca[2] - cb[2])
        clipped = _clip(self._project(ca), self._project(cb), self.size)
        if clipped is not None:
            self.pen.line(clipped, fill=_rgba(color), width=width)
        return clipped

    def dot(self, at: tuple[float, float], color: Iterable[int], label: str) -> None:
        u, v = at
        r = self.radius
        self.pen.ellipse((u - r, v - r, u + r, v + r), fill=_rgba(color), outline=(0, 0, 0, 255))
        if label:
            self.text((u + r + 2, v - 2 * r), label)

    def arrow(self, a: ArrayLike, b: ArrayLike, color: Iterable[int], label: str) -> None:
        """A world arrow from `a` to `b` with its head and `label` at the tip, as far as it is
        in view."""
        shown = self.segment(a, b, color, 3)
        if shown is None:
            return
        (u0, v0), (u1, v1) = shown
        length = math.hypot(u1 - u0, v1 - v0)
        if length < 1:
            return
        du, dv = (u1 - u0) / length, (v1 - v0) / length
        head = min(3 * self.radius, 0.4 * length)
        for side in (1, -1):
            wing = (u1 - head * (du + side * 0.5 * dv), v1 - head * (dv - side * 0.5 * du))
            self.pen.line([(u1, v1), wing], fill=_rgba(color), width=3)
        if label:
            self.text((u1 + du * self.radius, v1 + dv * self.radius - self.radius), label)

    def box(self, box: tuple[float, float, float, float], color: Iterable[int], label: str) -> None:
        u0, v0, u1, v1 = box
        left, top = min(u0, u1), min(v0, v1)
        self.pen.rectangle((left, top, max(u0, u1), max(v0, v1)), outline=_rgba(color), width=2)
        if label:
            self.text((left, top - 3 * self.radius), label)

    def text(self, at: tuple[float, float], text: str) -> None:
        """Dark text on a light patch, so it reads on any background, moved into the image."""
        left, top, right, bottom = self.pen.textbbox(at, text, font=self.font)
        width, height = self.size
        dx = min(max(0.0, 2 - left), width - 3 - right)
        dy = min(max(0.0, 1 - top), height - 2 - bottom)
        at = (at[0] + dx, at[1] + dy)
        patch = (left + dx - 2, top + dy - 1, right + dx + 2, bottom + dy + 1)
        self.pen.rectangle(patch, fill=(255, 255, 255, 170))
        self.pen.text(at, text, fill=(0, 0, 0, 255), font=self.font)

    def _project(self, point: NDArray[np.float64]) -> tuple[float, float]:
        u, v, w = self.K @ point
        return float(u / w), float(v / w)


def _clip(
    a: tuple[float, float], b: tuple[float, float], size: tuple[int, int]
) -> list[tuple[float, float]] | None:
    """Segment a-b cut to the image (Liang-Barsky), or None if it misses it."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    low, high = 0.0, 1.0
    width, height = size
    for p, q in ((-dx, x0 + 1), (dx, width + 1 - x0), (-dy, y0 + 1), (dy, height + 1 - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            low = max(low, t)
        else:
            high = min(high, t)
    if low > high:
        return None
    return [(x0 + low * dx, y0 + low * dy), (x0 + high * dx, y0 + high * dy)]


def _rgba(color: Iterable[int]) -> tuple[int, ...]:
    values = tuple(color)
    return values if len(values) == 4 else (*values, 255)
