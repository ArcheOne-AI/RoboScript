"""What every perception step starts from: per camera, the world point under each pixel and the
pixels on the robot; per frame, the table (its plane from the anchor camera, its edges from every
view).

Build it once per frame (`analyze`, or `frame_geometry` from loaded images); it is immutable. Under
the identify-once policy an episode finds the table on its initial frame and passes that `Table`
for every later frame, so the table and its landmarks stay fixed.
"""

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import anyio
import numpy as np
from numpy.typing import ArrayLike, NDArray

from roboscript.contracts.ids import FrameId
from roboscript.perception.config import DEFAULTS, PerceptionConfig
from roboscript.perception.frames import CameraImage, Frame
from roboscript.perception.robot import RobotShape
from roboscript.perception.table import Table, edge_pixels, fit_table, with_edges
from roboscript.spatial.camera import backproject_world


@dataclass(frozen=True, eq=False)
class View:
    """One camera of a frame: its image and depth (meters), `xyz` (H, W, 3) the world point
    under each pixel (NaN without depth), `robot` (H, W) the pixels the robot's shape covers."""

    image: CameraImage
    depth: NDArray[np.float32]
    xyz: NDArray[np.float64]
    robot: NDArray[np.bool_]

    @property
    def name(self) -> str:
        return self.image.name

    @property
    def usable(self) -> NDArray[np.bool_]:
        """Pixels with depth that are not on the robot."""
        return ~np.isnan(self.xyz[..., 0]) & ~self.robot


@dataclass(frozen=True, eq=False)
class FrameGeometry:
    """A frame's views, the table, the robot shape the views were filtered with and where the
    robot's base is (world x, y; None: unknown), which places what the robot sees as left and
    right of its midline."""

    frame: FrameId
    step: int
    views: Mapping[str, View]
    anchor: str
    table: Table
    robot: RobotShape | None
    base: tuple[float, float] | None = None

    def view(self, camera: str) -> View:
        if camera not in self.views:
            raise LookupError(f"no camera {camera!r} in frame {self.frame}: {sorted(self.views)}")
        return self.views[camera]


def view_of(image: CameraImage, robot: RobotShape | None = None) -> View:
    if image.depth is None:
        raise ValueError(f"camera {image.name!r} has no depth: perception needs depth images")
    points, pixels = backproject_world(image.depth, image.K, image.cam2world)
    xyz = np.full((*image.shape, 3), np.nan)
    xyz[pixels[:, 1], pixels[:, 0]] = points
    mask = np.zeros(image.shape, dtype=bool)
    if robot is not None:
        on_robot = pixels[robot.contains(points, image.depth[pixels[:, 1], pixels[:, 0]])]
        mask[on_robot[:, 1], on_robot[:, 0]] = True
    return View(image=image, depth=image.depth, xyz=xyz, robot=mask)


def frame_geometry(
    frame: FrameId,
    step: int,
    images: Mapping[str, CameraImage],
    config: PerceptionConfig = DEFAULTS,
    robot: RobotShape | None = None,
    robot_base: ArrayLike | None = None,
    table: Table | None = None,
) -> FrameGeometry:
    """Views of the images, robot pixels where the `robot` shape covers them (`SelfFilter`), and
    the `table` (None: found in these views; `robot_base` places the robot-midline landmark)."""
    if config.anchor_camera not in images:
        raise LookupError(f"the anchor camera {config.anchor!r} is not among {sorted(images)}")
    views = {name: view_of(image, robot) for name, image in images.items()}
    if table is None:
        table = find_table(views, config, robot_base)
    base = None if robot_base is None else np.asarray(robot_base, dtype=float)[:2]
    return FrameGeometry(
        frame=frame,
        step=step,
        views=views,
        anchor=config.anchor_camera,
        table=table,
        robot=robot,
        base=None if base is None else (float(base[0]), float(base[1])),
    )


def find_table(
    views: Mapping[str, View],
    config: PerceptionConfig = DEFAULTS,
    robot_base: ArrayLike | None = None,
) -> Table:
    """The table plane in the anchor view, with the edges all the views show."""
    anchor = views[config.anchor_camera]
    table = fit_table(anchor.xyz[anchor.usable], config.table)
    found = [edge_pixels(v.xyz, v.usable, table, config.table) for v in views.values()]
    edge = np.concatenate([e for e, _ in found])
    seen = np.concatenate([s for _, s in found])
    return with_edges(table, edge, seen, config.table, robot_base)


async def analyze(
    frame: Frame,
    config: PerceptionConfig = DEFAULTS,
    robot: RobotShape | None = None,
    robot_base: ArrayLike | None = None,
    cameras: Iterable[str] | None = None,
    table: Table | None = None,
) -> FrameGeometry:
    """`frame_geometry` of a frame's images (all cameras unless `cameras`), off the event loop."""
    images = await frame.images(cameras)
    build = functools.partial(
        frame_geometry, frame.id, frame.step, images, config, robot, robot_base, table
    )
    return await anyio.to_thread.run_sync(build)
