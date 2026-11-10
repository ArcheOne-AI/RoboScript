"""Blob encodings shared by the stores: raw bytes, `.npy` arrays, PNG/JPEG images, and depth
images as 16-bit PNGs of millimetres (as a depth camera delivers them). No pickle. Readers decode
by media type (`decode_pixels`), so `.npy` blobs stay readable. Images
are decoded bounded: only the format their media type names, and only once the header agrees
with the ref's shape."""

import hashlib
import io
import math
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from PIL import Image

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.ids import BlobId

NPY: Final = "application/x-npy"
ImageFormat = Literal["png", "jpeg"]
IMAGE_TYPES: Final[dict[str, str]] = {"png": "image/png", "jpeg": "image/jpeg"}
_PILLOW_FORMATS: Final = {"image/png": "PNG", "image/jpeg": "JPEG"}
DEPTH_SCALE: Final = "depth_scale_m"
"""The meta key of a depth PNG: metres per stored unit."""
DEPTH_SCALE_M: Final = 0.001  # millimetres


def blob_ref(
    data: bytes,
    media_type: str,
    meta: dict[str, Any] | None = None,
    dtype: str | None = None,
    shape: tuple[int, ...] | None = None,
) -> BlobRef:
    return BlobRef(
        id=BlobId(hashlib.sha256(data).hexdigest()),
        size=len(data),
        media_type=media_type,
        dtype=dtype,
        shape=shape,
        meta=dict(meta or {}),
    )


def encode_array(arr: np.ndarray, meta: dict[str, Any] | None = None) -> tuple[bytes, BlobRef]:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    data = buffer.getvalue()
    return data, blob_ref(data, NPY, meta, str(arr.dtype), tuple(arr.shape))


def decode_array(data: bytes) -> np.ndarray:
    """The array, read-only like a memory-mapped blob."""
    arr: np.ndarray = np.load(io.BytesIO(data), allow_pickle=False)
    arr.setflags(write=False)
    return arr


def encode_image(
    arr: np.ndarray, fmt: ImageFormat, meta: dict[str, Any] | None = None
) -> tuple[bytes, BlobRef]:
    """Encode an HxWx3 (RGB) or HxWx4 (RGBA) uint8 image."""
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"an image is HxWx3 or HxWx4 uint8, got {arr.dtype} {arr.shape}")
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format=fmt.upper())
    data = buffer.getvalue()
    return data, blob_ref(data, IMAGE_TYPES[fmt], meta, "uint8", tuple(arr.shape))


def decode_image(source: Path | bytes, ref: BlobRef) -> np.ndarray:
    """An image blob's pixels. Pillow reads only the format the media type names, and decodes
    nothing before the header's size and bands agree with the ref's shape."""
    with io.BytesIO(source) if isinstance(source, bytes) else source.open("rb") as stream:
        if ref.media_type not in _PILLOW_FORMATS:
            raise ValueError(f"blob {ref.id} is {ref.media_type}, not a PNG or JPEG image")
        with Image.open(stream, formats=[_PILLOW_FORMATS[ref.media_type]]) as image:
            bands = len(image.getbands())
            found = (image.height, image.width) + ((bands,) if bands > 1 else ())
            if ref.shape is None or tuple(ref.shape) != found:
                raise ValueError(f"blob {ref.id} is a {found} image; its ref says {ref.shape}")
            return np.array(image)


def encode_depth(depth: np.ndarray, meta: dict[str, Any] | None = None) -> tuple[bytes, BlobRef]:
    """Encode an HxW depth image in metres (0 where nothing was hit) as a 16-bit PNG of
    millimetres: within 0.5 mm, up to 65.535 m (a value under 0.5 mm reads as no return). The ref
    describes what readers get back."""
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"a depth image is HxW float metres, got {depth.dtype} {depth.shape}")
    units = np.rint(depth.astype(np.float64) / DEPTH_SCALE_M)
    top = np.iinfo(np.uint16).max
    if not np.isfinite(units).all() or units.min(initial=0) < 0 or units.max(initial=0) > top:
        raise ValueError(f"depth must be finite and within 0-{top * DEPTH_SCALE_M:g} m")
    buffer = io.BytesIO()
    Image.fromarray(units.astype(np.uint16)).save(buffer, format="PNG")
    data = buffer.getvalue()
    info = {**(meta or {}), DEPTH_SCALE: DEPTH_SCALE_M}
    return data, blob_ref(data, IMAGE_TYPES["png"], info, "float32", tuple(depth.shape))


def decode_pixels(source: Path | bytes, ref: BlobRef) -> np.ndarray:
    """An image blob as an array, read-only: its pixels, or a depth PNG's float32 metres (0 still
    meaning no return; Pillow reads 16-bit PNGs as uint16 or int32, by version). The metres are
    computed in float64 and rounded once, so they stay within 0.5 mm and one float32 step."""
    pixels = decode_image(source, ref)
    scale = ref.meta.get(DEPTH_SCALE)
    if scale is not None and not (
        isinstance(scale, int | float) and not isinstance(scale, bool) and 0 < scale < math.inf
    ):
        raise ValueError(f"blob {ref.id}: {DEPTH_SCALE} must be a positive number, not {scale!r}")
    arr = pixels if scale is None else (pixels * float(scale)).astype(np.float32)
    arr.setflags(write=False)
    return arr
