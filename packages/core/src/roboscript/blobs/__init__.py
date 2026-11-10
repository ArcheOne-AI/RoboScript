"""The content-addressed blob store for bulk data (docs/architecture.md).

Only `BlobRef`s travel between services; the bytes live in a directory shared by the services of
a machine (`LocalBlobStore`; a service shared by the callers of several reads and writes through
`BlobRoots`) or behind `GET/HEAD/PUT /blobs/sha256/<hex>` across machines (`blobs_routes`,
`HttpBlobStore`). Arrays are `.npy`, images PNG or JPEG, depth images 16-bit
PNGs of millimetres; nothing is pickled. `put_frame` is the one place a camera capture's
encodings are decided (RGB as PNG, depth as millimetres), for every observation of every env.
Service code uses the async API (`BlobStore`), which both stores share: `aget_array` decodes by
media type (a depth PNG comes back as float32 metres), arrays come back read-only, and `aclose()`
releases connections. Garbage collection is out of scope: blobs stay until their directory is
removed.
"""

from pathlib import Path
from typing import Any, Protocol

import anyio
import numpy as np
from pydantic import SecretStr

from roboscript.blobs.codec import DEPTH_SCALE, IMAGE_TYPES, NPY, ImageFormat
from roboscript.blobs.http import HttpBlobStore, blobs_routes
from roboscript.blobs.local import BlobRoots, LocalBlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraFrame, CameraModel


class BlobStore(Protocol):
    async def aput_bytes(
        self, data: bytes, media_type: str, meta: dict[str, Any] | None = None
    ) -> BlobRef: ...

    async def aput_array(self, arr: np.ndarray, meta: dict[str, Any] | None = None) -> BlobRef: ...

    async def aput_image(
        self, arr: np.ndarray, fmt: ImageFormat = "png", meta: dict[str, Any] | None = None
    ) -> BlobRef: ...

    async def aput_depth(
        self, depth: np.ndarray, meta: dict[str, Any] | None = None
    ) -> BlobRef: ...

    async def aget_bytes(self, ref: BlobRef) -> bytes: ...

    async def aget_array(self, ref: BlobRef) -> np.ndarray: ...

    async def aget_image(self, ref: BlobRef) -> np.ndarray: ...

    async def aexists(self, ref: BlobRef) -> bool: ...

    async def aclose(self) -> None: ...


async def put_frame(
    store: BlobStore, camera: CameraModel, rgb: np.ndarray, depth: np.ndarray | None
) -> CameraFrame:
    """A camera's capture as observations keep it: the RGB as PNG and the depth (HxW metres, 0
    where nothing was hit) as a 16-bit PNG of millimetres: read back within 0.5 mm (and one
    float32 step) of what was measured. A local store does it in a worker thread, through its
    blocking twin `LocalBlobStore.put_frame`."""
    if isinstance(store, LocalBlobStore):
        return await anyio.to_thread.run_sync(store.put_frame, camera, rgb, depth)
    meta = {"camera": camera.name}
    return CameraFrame(
        camera=camera,
        rgb=await store.aput_image(rgb, "png", meta),
        depth=None if depth is None else await store.aput_depth(depth, meta),
    )


def open_store(uri_or_path: str | Path, token: SecretStr | None = None) -> BlobStore:
    """An `HttpBlobStore` for an http(s) URL, else a `LocalBlobStore` for the directory."""
    if isinstance(uri_or_path, str) and uri_or_path.startswith(("http://", "https://")):
        return HttpBlobStore(uri_or_path, token)
    return LocalBlobStore(Path(uri_or_path))


__all__ = [
    "DEPTH_SCALE",
    "IMAGE_TYPES",
    "NPY",
    "BlobRoots",
    "BlobStore",
    "HttpBlobStore",
    "ImageFormat",
    "LocalBlobStore",
    "blobs_routes",
    "open_store",
    "put_frame",
]
