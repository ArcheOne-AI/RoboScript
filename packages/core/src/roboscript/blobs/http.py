"""The blob store across machines: `GET/HEAD/PUT /blobs/sha256/<hex>` and its client."""

import hashlib
import re
from typing import Any, Final

import anyio
import httpx
import numpy as np
from pydantic import SecretStr
from starlette.requests import Request
from starlette.responses import FileResponse, Response
from starlette.routing import Route

from roboscript.blobs import codec
from roboscript.blobs.local import LocalBlobStore
from roboscript.contracts import versions
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import ProtocolError, RSError
from roboscript.contracts.ids import BlobId
from roboscript.rpc import (
    Transport,
    authorize,
    error_from_response,
    problem_response,
    require_contract,
)

CONTRACT: Final = versions.parse(f"blobs/{versions.VERSIONS['blobs']}")
_BLOB_ID = re.compile(r"[0-9a-f]{64}")


def blobs_routes(store: LocalBlobStore, token: SecretStr | None) -> list[Route]:
    """Serve `store`: GET and HEAD read a blob, PUT stores one after checking its sha256.

    PUT takes the media type from `Content-Type` and answers 201 without a body.
    """

    async def blob(request: Request) -> Response:
        try:
            authorize(request, token)
            require_contract(request, CONTRACT)
            blob_id = BlobId(request.path_params["hex"])
            if not _BLOB_ID.fullmatch(blob_id):
                raise ProtocolError("invalid_request", "a blob id is 64 lowercase hex digits")
            if request.method == "PUT":
                return await _put(store, blob_id, request)
            path = store.path(blob_id)
            if not await anyio.Path(path).is_file():
                raise ProtocolError("not_found", f"no blob {blob_id}")
        except RSError as error:
            return problem_response(error)
        return FileResponse(path, media_type="application/octet-stream")

    return [Route("/blobs/sha256/{hex}", blob, methods=["GET", "PUT"])]  # GET implies HEAD


async def _put(store: LocalBlobStore, blob_id: BlobId, request: Request) -> Response:
    media_type = request.headers.get("content-type")
    if media_type is None:
        raise ProtocolError("invalid_request", "send the blob's media type as Content-Type")
    data = await request.body()
    await anyio.to_thread.run_sync(_put_checked, store, blob_id, data, media_type)
    return Response(status_code=201)


def _put_checked(store: LocalBlobStore, blob_id: BlobId, data: bytes, media_type: str) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != blob_id:
        raise ProtocolError("invalid_request", f"the body's sha256 is {digest}, not {blob_id}")
    store.put_bytes(data, media_type)


class HttpBlobStore:
    """The async API of a blob store served by `blobs_routes` at `url`.

    Each request must finish within `timeout` seconds. Without `http` the store opens its own
    connections; release them with `aclose()`. Refs are built client-side, so they carry dtype,
    shape and meta as `LocalBlobStore`'s do. Fetched bytes are checked against the ref's sha256;
    arrays come back read-only, images as new arrays, as from `LocalBlobStore`.
    """

    def __init__(
        self,
        url: str,
        token: SecretStr | None,
        *,
        http: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.url = url.rstrip("/")
        self._headers = {versions.HEADER: str(CONTRACT)}
        if token is not None:
            self._headers["Authorization"] = f"Bearer {token.get_secret_value()}"
        self._transport = Transport(http, timeout)

    def __repr__(self) -> str:
        return f"HttpBlobStore({self.url!r})"

    async def aput_bytes(
        self, data: bytes, media_type: str, meta: dict[str, Any] | None = None
    ) -> BlobRef:
        ref = await anyio.to_thread.run_sync(codec.blob_ref, data, media_type, meta)
        return await self._put(data, ref)

    async def aput_array(self, arr: np.ndarray, meta: dict[str, Any] | None = None) -> BlobRef:
        return await self._put(*await anyio.to_thread.run_sync(codec.encode_array, arr, meta))

    async def aput_image(
        self, arr: np.ndarray, fmt: codec.ImageFormat = "png", meta: dict[str, Any] | None = None
    ) -> BlobRef:
        encoded = await anyio.to_thread.run_sync(codec.encode_image, arr, fmt, meta)
        return await self._put(*encoded)

    async def aput_depth(self, depth: np.ndarray, meta: dict[str, Any] | None = None) -> BlobRef:
        return await self._put(*await anyio.to_thread.run_sync(codec.encode_depth, depth, meta))

    async def aget_bytes(self, ref: BlobRef) -> bytes:
        data = (await self._send("GET", ref)).content
        await anyio.to_thread.run_sync(_check, ref, data)
        return data

    async def aget_array(self, ref: BlobRef) -> np.ndarray:
        data = await self.aget_bytes(ref)
        if ref.media_type == codec.NPY:
            return await anyio.to_thread.run_sync(codec.decode_array, data)
        return await anyio.to_thread.run_sync(codec.decode_pixels, data, ref)

    async def aget_image(self, ref: BlobRef) -> np.ndarray:
        return await anyio.to_thread.run_sync(codec.decode_image, await self.aget_bytes(ref), ref)

    async def aexists(self, ref: BlobRef) -> bool:
        try:
            await self._send("HEAD", ref)
        except ProtocolError as error:
            if error.code == "not_found":
                return False
            raise
        return True

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def _put(self, data: bytes, ref: BlobRef) -> BlobRef:
        await self._send("PUT", ref, content=data, headers={"Content-Type": ref.media_type})
        return ref

    async def _send(self, method: str, ref: BlobRef, **kwargs: Any) -> httpx.Response:
        headers = {**self._headers, **kwargs.pop("headers", {})}
        url = f"{self.url}/blobs/sha256/{ref.id}"
        response = await self._transport.send(method, url, headers=headers, **kwargs)
        if not response.is_success:
            raise error_from_response(response)
        return response


def _check(ref: BlobRef, data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != ref.id:
        raise RSError("internal", f"blob {ref.id} was served with bytes that hash to {digest}")
