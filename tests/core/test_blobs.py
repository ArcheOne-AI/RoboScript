import hashlib
import os
import stat
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from roboscript.blobs import (
    DEPTH_SCALE,
    HttpBlobStore,
    LocalBlobStore,
    blobs_routes,
)
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraModel
from roboscript.contracts.errors import RSError
from roboscript.rpc import make_app

TOKEN = SecretStr("blob-token")
RNG = np.random.default_rng(0)
RGB = RNG.integers(0, 256, (48, 64, 3), dtype=np.uint8)
RGBA = RNG.integers(0, 256, (48, 64, 4), dtype=np.uint8)
METRES = (RNG.random((48, 64)) * 3).astype(np.float32)  # a depth image, 0 where nothing was hit
METRES[0, :5] = 0.0
CAMERA = CameraModel(
    name="overhead", width=64, height=48, K=[[50, 0, 32], [0, 50, 24], [0, 0, 1]],
    cam2world=np.eye(4).tolist(),
)  # fmt: skip


def test_local_layout_and_idempotent_atomic_puts(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    ref = store.put_bytes(b"hello", "text/plain", {"camera": "head"})
    digest = hashlib.sha256(b"hello").hexdigest()
    assert ref == BlobRef(id=digest, size=5, media_type="text/plain", meta={"camera": "head"})
    path = tmp_path / digest[:2] / digest
    assert store.path(ref) == path and path.read_bytes() == b"hello"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    mtime = path.stat().st_mtime_ns
    assert store.put_bytes(b"hello", "application/octet-stream").id == ref.id
    assert path.stat().st_mtime_ns == mtime  # not rewritten
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == [digest]  # no temp files
    assert store.exists(ref) and store.get_bytes(ref) == b"hello"


def test_depth_is_a_png_of_millimetres(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path)
    ref = store.put_depth(METRES, {"camera": "overhead"})
    assert (ref.media_type, ref.dtype, ref.shape) == ("image/png", "float32", (48, 64))
    assert ref.meta == {"camera": "overhead", DEPTH_SCALE: 0.001}
    depth = store.get_array(ref)  # readers decode by media type: float32 metres again
    assert depth.dtype == np.float32 and not depth.flags.writeable
    assert np.abs(depth - METRES).max() <= 0.0005 + 1e-6 and (depth[0, :5] == 0).all()
    npy = store.put_array(METRES)  # a raw array: read as written
    assert np.array_equal(store.get_array(npy), METRES)
    assert np.array_equal(store.get_array(store.put_image(RGB)), RGB)  # any image, as pixels
    over = np.full((2, 2), 70.0, np.float32)  # beyond 65.535 m
    for bad in (METRES[None], METRES.astype(np.uint16), METRES * np.nan, -METRES - 1, over):
        with pytest.raises(ValueError, match="depth"):
            store.put_depth(bad)


@pytest.fixture
async def served(tmp_path: Path) -> AsyncIterator[tuple[LocalBlobStore, httpx.AsyncClient]]:
    store = LocalBlobStore(tmp_path / "server")
    app = make_app(blobs_routes(store, TOKEN))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://b") as http:
        yield store, http


async def test_fetched_bytes_must_match_the_ref(
    served: tuple[LocalBlobStore, httpx.AsyncClient],
) -> None:
    server, http = served
    ref = server.put_bytes(b"genuine", "text/plain")
    path = server.path(ref)
    os.chmod(path, 0o644)
    path.write_bytes(b"tampered")  # a corrupted store
    with pytest.raises(RSError) as caught:
        await HttpBlobStore("http://b", TOKEN, http=http).aget_bytes(ref)
    assert caught.value.code == "internal" and "hash to" in str(caught.value.detail)
