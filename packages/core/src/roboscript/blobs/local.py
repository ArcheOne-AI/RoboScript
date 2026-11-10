"""The blob store as a local directory, shared by the services of one machine; `BlobRoots` for a
service that serves the callers of several such stores."""

import os
import tempfile
from pathlib import Path
from typing import Any

import anyio
import numpy as np

from roboscript.blobs import codec
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraFrame, CameraModel
from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.ids import BlobId


class LocalBlobStore:
    """Content-addressed files `<root>/<hex[:2]>/<hex>`, written atomically and read-only.

    The plain methods block (for worker threads); the `a`-prefixed ones run them in a thread.
    Putting the same bytes twice is a no-op. A missing blob is `ProtocolError("not_found")`.
    Arrays come back read-only (copy one to change it); images come back as new arrays.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, ref: BlobRef | BlobId) -> Path:
        blob_id = ref.id if isinstance(ref, BlobRef) else ref
        return self.root / blob_id[:2] / blob_id

    def exists(self, ref: BlobRef) -> bool:
        return self.path(ref).is_file()

    def put_bytes(
        self, data: bytes, media_type: str, meta: dict[str, Any] | None = None
    ) -> BlobRef:
        ref = codec.blob_ref(data, media_type, meta)
        self._write(ref, data)
        return ref

    def put_array(self, arr: np.ndarray, meta: dict[str, Any] | None = None) -> BlobRef:
        """Store as `.npy`; the ref records dtype and shape."""
        data, ref = codec.encode_array(arr, meta)
        self._write(ref, data)
        return ref

    def put_image(
        self, arr: np.ndarray, fmt: codec.ImageFormat = "png", meta: dict[str, Any] | None = None
    ) -> BlobRef:
        """Store an HxWx3 (RGB) or HxWx4 (RGBA) uint8 image as PNG or JPEG."""
        data, ref = codec.encode_image(arr, fmt, meta)
        self._write(ref, data)
        return ref

    def put_depth(self, depth: np.ndarray, meta: dict[str, Any] | None = None) -> BlobRef:
        """Store an HxW depth image in metres as a 16-bit PNG of millimetres."""
        data, ref = codec.encode_depth(depth, meta)
        self._write(ref, data)
        return ref

    def put_frame(
        self, camera: CameraModel, rgb: np.ndarray, depth: np.ndarray | None
    ) -> CameraFrame:
        """`roboscript.blobs.put_frame`, blocking: a camera's capture as observations keep it."""
        meta = {"camera": camera.name}
        return CameraFrame(
            camera=camera,
            rgb=self.put_image(rgb, "png", meta),
            depth=None if depth is None else self.put_depth(depth, meta),
        )

    def get_bytes(self, ref: BlobRef) -> bytes:
        try:
            return self.path(ref).read_bytes()
        except FileNotFoundError:
            raise _missing(ref) from None

    def get_array(self, ref: BlobRef, mmap: bool = True) -> np.ndarray:
        """The stored array, read-only: an `.npy` memory-mapped (or loaded if `mmap` is false),
        an image's pixels, a depth PNG as float32 metres."""
        try:
            if ref.media_type != codec.NPY:
                return codec.decode_pixels(self.path(ref), ref)
            arr: np.ndarray = np.load(
                self.path(ref), mmap_mode="r" if mmap else None, allow_pickle=False
            )
        except FileNotFoundError:
            raise _missing(ref) from None
        arr.setflags(write=False)
        return arr

    def get_image(self, ref: BlobRef) -> np.ndarray:
        try:
            return codec.decode_image(self.path(ref), ref)
        except FileNotFoundError:
            raise _missing(ref) from None

    def _write(self, ref: BlobRef, data: bytes) -> None:
        path = self.path(ref)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(data)
            os.chmod(tmp, 0o444)  # blobs are immutable
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise

    async def aput_bytes(
        self, data: bytes, media_type: str, meta: dict[str, Any] | None = None
    ) -> BlobRef:
        return await anyio.to_thread.run_sync(self.put_bytes, data, media_type, meta)

    async def aput_array(self, arr: np.ndarray, meta: dict[str, Any] | None = None) -> BlobRef:
        return await anyio.to_thread.run_sync(self.put_array, arr, meta)

    async def aput_image(
        self, arr: np.ndarray, fmt: codec.ImageFormat = "png", meta: dict[str, Any] | None = None
    ) -> BlobRef:
        return await anyio.to_thread.run_sync(self.put_image, arr, fmt, meta)

    async def aput_depth(self, depth: np.ndarray, meta: dict[str, Any] | None = None) -> BlobRef:
        return await anyio.to_thread.run_sync(self.put_depth, depth, meta)

    async def aget_bytes(self, ref: BlobRef) -> bytes:
        return await anyio.to_thread.run_sync(self.get_bytes, ref)

    async def aget_array(self, ref: BlobRef, mmap: bool = True) -> np.ndarray:
        return await anyio.to_thread.run_sync(self.get_array, ref, mmap)

    async def aget_image(self, ref: BlobRef) -> np.ndarray:
        return await anyio.to_thread.run_sync(self.get_image, ref)

    async def aexists(self, ref: BlobRef) -> bool:
        return await anyio.to_thread.run_sync(self.exists, ref)

    async def aclose(self) -> None:
        """Nothing to release; here so that any `BlobStore` can be closed alike."""


class BlobRoots:
    """The blob stores a service shares with its callers: its own (`primary`) and those named in
    `listing`, a directory with a file per store whose content is the store's directory (a stack
    shared by several tenants lists each tenant's store there, `roboscript.stack.shared`). A
    request's input is read from the store that holds it, and its outputs go to that store, where
    its caller reads them (every holder of the input gets the outputs). The listing is read again
    when its directory changed or a blob is in none of the stores known."""

    def __init__(self, primary: Path, listing: Path | None = None) -> None:
        self.primary, self.listing = LocalBlobStore(primary), listing
        self._stores: dict[Path, LocalBlobStore] = {}
        self._listed: int | None = None  # the listing directory's mtime when last read
        self._read_listing()

    def store_of(self, ref: BlobRef) -> LocalBlobStore:
        """The first store holding `ref`; `ProtocolError("not_found")` if none does."""
        return self.stores_of(ref)[0]

    def stores_of(self, ref: BlobRef) -> list[LocalBlobStore]:
        """Every store holding `ref` (blobs are named by their content, so callers that sent the
        same image each hold it); `ProtocolError("not_found")` if none does."""
        self._read_listing_if_changed()  # a tenant that came since holds the same image too
        for again in (False, True):
            if again:
                self._read_listing()
            if held := [s for s in (self.primary, *self._stores.values()) if s.exists(ref)]:
                return held
        raise _missing(ref)

    def _read_listing_if_changed(self) -> None:
        try:
            stamp = self.listing.stat().st_mtime_ns if self.listing is not None else None
        except OSError:
            return
        if stamp != self._listed:
            self._read_listing()

    def _read_listing(self) -> None:
        if self.listing is None or not self.listing.is_dir():
            return
        self._listed = self.listing.stat().st_mtime_ns
        found = {}
        for entry in sorted(self.listing.iterdir()):
            try:
                root = Path(entry.read_text(encoding="utf-8").strip())
            except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
                continue  # removed as we looked, or not a listing file
            if root.is_absolute():
                found[root] = self._stores.get(root) or LocalBlobStore(root)
        self._stores = found


def _missing(ref: BlobRef) -> ProtocolError:
    return ProtocolError("not_found", f"no blob {ref.id}")
