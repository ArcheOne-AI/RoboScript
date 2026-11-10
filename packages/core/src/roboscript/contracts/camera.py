"""Cameras and captured frames, OpenCV convention: x right, y down, z forward (the optical axis)."""

from pydantic import BaseModel, ConfigDict, FiniteFloat, PositiveInt, model_validator
from typing_extensions import Self

from roboscript.contracts.blob import BlobRef

_Row3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
_Row4 = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]


class CameraModel(BaseModel):
    """A pinhole camera: intrinsics `K` (pixels) and the camera-to-world transform (meters)."""

    model_config = ConfigDict(frozen=True)

    name: str
    width: PositiveInt
    height: PositiveInt
    K: tuple[_Row3, _Row3, _Row3]
    cam2world: tuple[_Row4, _Row4, _Row4, _Row4]


class CameraFrame(BaseModel):
    """One capture: RGB (H, W, 3) uint8 and optional depth (H, W) float32 in meters, where 0
    means no return."""

    model_config = ConfigDict(frozen=True)

    camera: CameraModel
    rgb: BlobRef
    depth: BlobRef | None = None

    @model_validator(mode="after")
    def _images_match_camera(self) -> Self:
        size = (self.camera.height, self.camera.width)
        for blob in (self.rgb, self.depth):
            if blob is not None and blob.shape is not None and blob.shape[:2] != size:
                raise ValueError(
                    f"camera {self.camera.name!r} is {size[1]}x{size[0]} but an image has shape "
                    f"{blob.shape}"
                )
        return self
