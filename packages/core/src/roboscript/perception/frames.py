"""Frames: immutable observations whose images load on first use, addressed by reference.

Perception functions take an explicit frame, never "the latest", so they cannot race an action.
An episode's observations live in its observation store (the Hub's per-episode frame store,
`roboscript.tools.frames`, which also seeds a reviewing role's episode with the Actor's frames under
the same aliases); `Frames` resolves a reference there: `live` (the newest observation), a frame
id or an alias (`frame_3`).
"""

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Protocol

import anyio
import numpy as np
from numpy.typing import NDArray

from roboscript.blobs import NPY, BlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraFrame
from roboscript.contracts.ids import FrameId
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.contracts.robot import RobotState

KEEP: Final = 8
"""Frames whose loaded images `Frames` keeps."""


@dataclass(frozen=True, eq=False)
class CameraImage:
    """One camera's capture: RGB (H, W, 3) uint8, depth (H, W) float32 in meters (0: no return),
    intrinsics `K` and the camera-to-world transform `cam2world` (OpenCV axes)."""

    name: str
    K: NDArray[np.float64]
    cam2world: NDArray[np.float64]
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32] | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[0], self.rgb.shape[1]


class Frame:
    """An `Observation` plus its images, loaded from the blob store once, on first use."""

    def __init__(self, observation: Observation, blobs: BlobStore) -> None:
        self.observation = observation
        self._blobs = blobs
        self._images: dict[str, CameraImage] = {}
        self._lock = anyio.Lock()

    @property
    def id(self) -> FrameId:
        return self.observation.frame_id

    @property
    def step(self) -> int:
        return self.observation.step

    @property
    def robot(self) -> RobotState:
        return self.observation.robot

    @property
    def cameras(self) -> list[str]:
        return list(self.observation.cameras)

    async def image(self, camera: str) -> CameraImage:
        if camera not in self.observation.cameras:
            raise LookupError(f"frame {self.id} has no camera {camera!r}; cameras: {self.cameras}")
        async with self._lock:
            if camera not in self._images:
                self._images[camera] = await load_image(
                    self._blobs, self.observation.cameras[camera]
                )
        return self._images[camera]

    async def images(self, cameras: Iterable[str] | None = None) -> dict[str, CameraImage]:
        names = self.cameras if cameras is None else list(cameras)
        return {name: await self.image(name) for name in names}


async def load_image(blobs: BlobStore, frame: CameraFrame) -> CameraImage:
    camera = frame.camera
    rgb = await read_rgb(blobs, frame.rgb)
    depth = None if frame.depth is None else await blobs.aget_array(frame.depth)
    return CameraImage(
        name=camera.name,
        K=np.array(camera.K, dtype=float),
        cam2world=np.array(camera.cam2world, dtype=float),
        rgb=rgb,
        depth=depth,
    )


async def read_rgb(blobs: BlobStore, ref: BlobRef) -> NDArray[np.uint8]:
    rgb = await (blobs.aget_array(ref) if ref.media_type == NPY else blobs.aget_image(ref))
    return rgb[..., :3]  # drop alpha


class ObservationStore(Protocol):
    """An episode's observations (`roboscript.tools.frames.FrameStore` is one): `get` by frame
    id or alias, `entries` in the order they came, `live` the newest one the episode took."""

    async def get(self, frame: str) -> Observation: ...

    def entries(self) -> list[FrameEntry]: ...

    def live(self) -> FrameEntry | None: ...


class Frames:
    """`Frame`s of an episode's observations: `live` is the newest the episode took (never a
    seeded frame, such as a demonstration's), a frame id or alias names any; the images of the
    last `KEEP` frames stay loaded."""

    def __init__(self, store: ObservationStore, blobs: BlobStore) -> None:
        self._store = store
        self._blobs = blobs
        self._frames: OrderedDict[FrameId, Frame] = OrderedDict()

    async def frame(self, ref: str) -> Frame:
        if ref == "live":
            entry = self._store.live()
            if entry is None:
                raise LookupError("no observation yet: observe first")
            ref = entry.frame_id
        observation = await self._store.get(ref)
        frame = self._frames.pop(observation.frame_id, None) or Frame(observation, self._blobs)
        self._frames[observation.frame_id] = frame
        while len(self._frames) > KEEP:
            self._frames.popitem(last=False)
        return frame
