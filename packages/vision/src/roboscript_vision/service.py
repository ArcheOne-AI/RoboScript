"""The vision service: `python -m roboscript_vision --config <file or name> [--ready-file PATH]`.

It serves `Vision` over the typed RPC on 127.0.0.1 (port 0 unless `--port`) and writes its ready
file as soon as it listens, then loads and warms up its models; `/readyz` answers 200 once they
are warm, and calls before that fail as `not_ready`; a model that fails to load ends the service
with its error in the log. It refuses to start unless `CUDA_VISIBLE_DEVICES` names a GPU, and uses
the first one. It serves any number of callers: one GPU thread runs their model calls one at a
time (at most `max_pending` wait; more fail as `busy`), while images are read from and masks
written to the shared blob store off that thread. An unexpected error on the GPU thread leaves
the service not ready (its `failure`): its call fails as `not_ready`, so do those queued behind
it, and the service stops serving and exits (`rpc.ServiceFault`), so that its supervisor
restarts it. Clients: `rpc.client(Vision, url, token=None,
timeout=10, timeouts=contracts.vision.TIMEOUTS)`.
"""

import argparse
import logging
import os
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Final, TypeVar

import anyio
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from roboscript.blobs import NPY, BlobRoots, LocalBlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import InfraError, ProtocolError, RSError
from roboscript.contracts.vision import (
    MAX_IMAGE_PIXELS,
    DetectResult,
    Mask,
    ModelInfo,
    Queries,
    QueryDetections,
    SegmentPrompt,
    SegmentResult,
    TopK,
    Vision,
    VisionInfo,
)
from roboscript.rpc import Worker, make_app, routes, serve_while
from roboscript.settings import load_config_ref
from roboscript_vision import __version__
from roboscript_vision.backends import Detector, Image, Segmenter, from_config
from roboscript_vision.config import VisionConfig

log = logging.getLogger(__name__)
T = TypeVar("T")

_IMAGE_TYPES: Final = (NPY, "image/png", "image/jpeg")
_WARMUP: Final = (
    SegmentPrompt(text="object"),
    SegmentPrompt(boxes=[(200.0, 150.0, 440.0, 330.0)]),
    SegmentPrompt(points=[(320.0, 240.0)]),
)


class VisionService:
    """`Vision` over a detector and a segmenter (either may be None) on one GPU thread."""

    def __init__(
        self, config: VisionConfig, detector: Detector | None, segmenter: Segmenter | None
    ) -> None:
        self.config = config
        self.blobs = LocalBlobStore(config.shared_blob_dir())
        self.roots = BlobRoots(self.blobs.root, config.caller_blob_roots())
        self.detector = detector
        self.segmenter = segmenter
        self._gpu = Worker("vision-gpu")
        self._pending = 0
        self._info: VisionInfo | None = None
        self._failure: str | None = None

    @property
    def ready(self) -> bool:
        return self._info is not None and self._failure is None

    @property
    def failure(self) -> str | None:
        """What failed on the GPU thread, leaving the models suspect; None while they are sound."""
        return self._failure

    async def start(self) -> None:
        """Load (and warm up) every model on the GPU thread; then `ready` is true."""
        info, _ = await self._run(self._load)
        names = ", ".join(f"{m.name} ({m.revision})" for m in info.models) or "no models"
        log.info("loaded %s on %s", names, info.runtime.get("gpu", "no GPU"))
        self._info = info

    async def info(self) -> VisionInfo:
        return self._ready()

    async def detect(self, image: BlobRef, queries: Queries, top_k: TopK = 5) -> DetectResult:
        self._ready()
        detector = _serving(self.detector, "grounding_dino")
        rgb, _ = await self._image(image)
        found, seconds = await self._run(lambda: [detector.detect(rgb, q, top_k) for q in queries])
        return DetectResult(
            model=detector.name,
            queries=[
                QueryDetections(query=q, detections=d) for q, d in zip(queries, found, strict=True)
            ],
            time_s=seconds,
        )

    async def segment(
        self, image: BlobRef, prompt: SegmentPrompt, top_k: TopK = 5
    ) -> SegmentResult:
        self._ready()
        segmenter = _serving(self.segmenter, "sam3")
        rgb, stores = await self._image(image)  # the masks go to the callers' stores too
        _check_inside(prompt, *rgb.shape[:2])
        found, seconds = await self._run(
            lambda: [(m, s, _bounds(m)) for m, s in segmenter.segment(rgb, prompt, top_k)]
        )
        masks = [
            Mask(mask=await _put_everywhere(stores, mask), score=score, box=box)
            for mask, score, box in found
        ]
        return SegmentResult(model=segmenter.name, masks=masks, time_s=seconds)

    def _ready(self) -> VisionInfo:
        if self._failure is not None:
            raise self._refusal()
        if self._info is None:
            raise InfraError("not_ready", "the vision service is still loading its models")
        return self._info

    async def _run(self, fn: Callable[[], T]) -> tuple[T, float]:
        """`fn()` on the GPU thread and how long it ran there. An unexpected error there takes
        the service out of service (`_latched`) and fails the call as an infrastructure error,
        chained to it; one while the models load ends the service as it is."""
        if self._pending >= self.config.max_pending:
            raise InfraError("busy", f"{self._pending} calls are waiting for the GPU")
        self._pending += 1
        try:
            return await self._gpu.run(self._latched, fn)
        except RSError:  # a request the models cannot serve (too many tokens), or one refused
            raise
        except Exception as error:
            if self._info is None:
                raise
            failed = "the vision service failed and must be restarted: %s"
            log.error(failed, self._failure, exc_info=error)
            raise self._refusal() from error
        finally:
            self._pending -= 1

    def _refusal(self) -> InfraError:
        return InfraError("not_ready", f"the vision service failed ({self._failure}); restart it")

    def _latched(self, fn: Callable[[], T]) -> tuple[T, float]:
        """`_timed(fn)` on the GPU thread, unless the service failed: the failure is recorded
        here, on that thread, so that the calls queued behind the one that failed are refused,
        not run on models whose state is suspect. The first failure is the one it names."""
        if self._failure is not None:
            raise self._refusal()
        try:
            return _timed(fn)
        except RSError:
            raise
        except Exception as error:
            if self._failure is None:
                self._failure = f"{type(error).__name__} on the GPU thread"
            raise

    async def _image(self, ref: BlobRef) -> tuple[Image, list[LocalBlobStore]]:
        """The image in `ref` and the blob stores that hold it: its callers'."""
        if ref.media_type not in _IMAGE_TYPES:
            detail = f"blob {ref.id} is {ref.media_type}; an image is one of {_IMAGE_TYPES}"
            raise ProtocolError("invalid_request", detail)
        stores = await anyio.to_thread.run_sync(self.roots.stores_of, ref)
        return await anyio.to_thread.run_sync(_read_image, stores[0].path(ref), ref), stores

    def _load(self) -> VisionInfo:
        runtime = {"roboscript_vision": __version__}
        models = []
        blank = np.full((480, 640, 3), 128, np.uint8)
        if self.detector is not None:
            runtime |= self.detector.load()
            if self.config.warmup:
                self.detector.detect(blank, "object", 1)
            models.append(_model_info(self.detector, "detect"))
        if self.segmenter is not None:
            runtime |= self.segmenter.load()
            if self.config.warmup:
                for prompt in _WARMUP:
                    self.segmenter.segment(blank, prompt, 1)
            models.append(_model_info(self.segmenter, "segment"))
        return VisionInfo(models=models, runtime=runtime)


async def _put_everywhere(stores: list[LocalBlobStore], mask: Image) -> BlobRef:
    refs = [await store.aput_array(mask) for store in stores]
    return refs[0]  # content-addressed: the same in every store


def _timed(fn: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    return fn(), time.perf_counter() - started


def _model_info(model: Detector | Segmenter, task: str) -> ModelInfo:
    return ModelInfo(name=model.name, source=model.source, revision=model.revision, task=task)


def _serving(backend: T | None, key: str) -> T:
    if backend is None:
        raise ProtocolError("invalid_request", f"this vision service has no {key} model configured")
    return backend


def _read_image(path: Path, ref: BlobRef) -> Image:
    """The RGB image in blob `ref`, its size checked before it is decoded in full."""
    try:
        if ref.media_type == NPY:
            image = np.load(path, mmap_mode="r", allow_pickle=False)
            _check_image(ref, image.shape, str(image.dtype))
            return np.ascontiguousarray(image)
        with PILImage.open(path) as pil:
            dtype = "uint8" if pil.mode == "RGB" else pil.mode
            _check_image(ref, (pil.height, pil.width, len(pil.getbands())), dtype)
            return np.asarray(pil)
    except FileNotFoundError:
        raise ProtocolError("not_found", f"no blob {ref.id}") from None
    except (ValueError, OSError) as error:  # the bytes are not what the ref says
        detail = f"blob {ref.id} is not a readable {ref.media_type}: {error}"
        raise ProtocolError("invalid_request", detail) from None


def _check_image(ref: BlobRef, shape: tuple[int, ...], dtype: str) -> None:
    if dtype != "uint8" or len(shape) != 3 or shape[2] != 3:
        detail = f"an image is RGB (H, W, 3) uint8, blob {ref.id} is {dtype} {shape}"
        raise ProtocolError("invalid_request", detail)
    height, width = shape[:2]
    if not (height >= 1 and width >= 1 and height * width <= MAX_IMAGE_PIXELS):
        detail = f"blob {ref.id} is {width}x{height}; an image holds 1 to {MAX_IMAGE_PIXELS} pixels"
        raise ProtocolError("invalid_request", detail)


def _check_inside(prompt: SegmentPrompt, height: int, width: int) -> None:
    for box in prompt.boxes or []:
        u0, v0, u1, v1 = box
        if u0 < 0 or v0 < 0 or u1 > width or v1 > height:
            raise ProtocolError("invalid_request", f"box {box} leaves the {width}x{height} image")
    for point in prompt.points or []:
        u, v = point
        if not (0 <= u < width and 0 <= v < height):
            raise ProtocolError(
                "invalid_request", f"point {point} is outside the {width}x{height} image"
            )


def _bounds(mask: NDArray[np.bool_]) -> tuple[float, float, float, float] | None:
    """The smallest box holding the mask's pixels (exclusive maximum), or None if it is empty."""
    columns, rows = np.flatnonzero(mask.any(axis=0)), np.flatnonzero(mask.any(axis=1))
    if not len(columns):
        return None
    return float(columns[0]), float(rows[0]), float(columns[-1] + 1), float(rows[-1] + 1)


async def run(service: VisionService, *, port: int = 0, ready_file: Path | None = None) -> None:
    """Serve `service` until SIGINT or SIGTERM, loading its models meanwhile, or until it fails
    for good (`rpc.ServiceFault`)."""
    app = make_app(routes(Vision, service, token=None), ready=lambda: service.ready)
    await serve_while(
        app,
        service.start,
        name="the vision service",
        port=port,
        ready_file=ready_file,
        fault=lambda: service.failure,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m roboscript_vision", description=__doc__)
    parser.add_argument("--config", required=True, help="a YAML file or a name under configs/")
    parser.add_argument("--ready-file", type=Path, help="written once the service listens")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--set", action="append", default=[], help="a dotted override, key.path=value (repeat)"
    )
    args = parser.parse_args(argv)
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        parser.error("set CUDA_VISIBLE_DEVICES to the GPU this service may use")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config_ref(args.config, VisionConfig, overrides=args.set)
    config.apply_env()  # before torch and transformers are imported, on the GPU thread
    service = VisionService(config, *from_config(config))
    anyio.run(partial(run, service, port=args.port, ready_file=args.ready_file))
