"""GroundingDINO and SAM 3 through the vision service (`roboscript.contracts.vision`).

Both take a `Vision` client (`rpc.client(Vision, url, token=None, timeout=10,
timeouts=contracts.vision.TIMEOUTS)`, or the service object itself) and the blob store the service
shares with its callers: images go there as `.npy` (content-addressed, so an image is stored
once), masks come back from there. Scores rank; they are never thresholded.

- `VisionDetector` (`grounding_dino`): boxes for text queries, best first. The identification
  chain takes the top box per description as its cross-check.
- `VisionSegmenter` (`sam3`): the top mask for a text, the mask of the object in a box or under a
  point, or of what a region shows (prompted with the region's box).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from roboscript.blobs import BlobStore
from roboscript.contracts.vision import MAX_QUERIES, Detection, Vision
from roboscript.contracts.vision import SegmentPrompt as VisionPrompt
from roboscript.perception.grounding import (
    Candidate,
    Grounding,
    Mode,
    Resources,
    Segmentation,
    SegmentPrompt,
    mask_box,
)
from roboscript.perception.overlay import box_overlay
from roboscript.perception.views import FrameGeometry

TOP_K = 3
"""Boxes `ground` returns per query."""


@dataclass(frozen=True)
class Detected:
    """Boxes per query, best first (none: the model proposed nothing), the model that answered,
    its GPU time and the round trip's seconds."""

    boxes: dict[str, list[Detection]]
    model: str
    time_s: float
    latency_s: float


class VisionDetector:
    """GroundingDINO (base by default: the service's configuration decides)."""

    name: ClassVar[str] = "grounding_dino"

    def __init__(self, vision: Vision, blobs: BlobStore) -> None:
        self.vision = vision
        self.blobs = blobs

    @classmethod
    def build(cls, resources: Resources) -> "VisionDetector":
        return cls(*_service(resources, "GroundingDINO"))

    async def detect(
        self, rgb: NDArray[np.uint8], queries: Sequence[str], top_k: int = 1
    ) -> Detected:
        """At most `top_k` boxes per query in the image `rgb` (H, W, 3)."""
        started = time.monotonic()
        image = await self.blobs.aput_array(np.ascontiguousarray(rgb[..., :3]))
        boxes: dict[str, list[Detection]] = {}
        model, gpu = "", 0.0
        unique = list(dict.fromkeys(queries))
        for start in range(0, len(unique), MAX_QUERIES):
            result = await self.vision.detect(image, unique[start : start + MAX_QUERIES], top_k)
            boxes |= {q.query: q.detections for q in result.queries}
            model, gpu = result.model, gpu + result.time_s
        return Detected(boxes, model, gpu, time.monotonic() - started)

    async def ground(
        self, geometry: FrameGeometry, camera: str, query: str, mode: Mode = "box"
    ) -> Grounding:
        """The top `TOP_K` boxes for `query`, as boxes or as their centers."""
        started = time.monotonic()
        image = geometry.view(camera).image
        found = (await self.detect(image.rgb, [query], TOP_K)).boxes[query]
        candidates = [
            Candidate(box=d.box, score=d.score, label=d.label)
            if mode == "box"
            else Candidate(point=_center(d.box), score=d.score, label=d.label)
            for d in found
        ]
        boxes = {i: c.box for i, c in enumerate(candidates, 1) if c.box is not None}
        points = {i: c.point for i, c in enumerate(candidates, 1) if c.point is not None}
        overlay = box_overlay(image.rgb, boxes, points)
        elapsed = time.monotonic() - started
        return Grounding(geometry.frame, camera, query, candidates, overlay, self.name, elapsed)


class VisionSegmenter:
    """SAM 3: text through its concept detector, boxes and points through its tracker."""

    name: ClassVar[str] = "sam3"

    def __init__(self, vision: Vision, blobs: BlobStore) -> None:
        self.vision = vision
        self.blobs = blobs

    @classmethod
    def build(cls, resources: Resources) -> "VisionSegmenter":
        return cls(*_service(resources, "SAM 3"))

    async def segment(
        self, geometry: FrameGeometry, camera: str, prompt: SegmentPrompt
    ) -> Segmentation:
        """The mask for `prompt` in `camera`'s image; empty (score None) when a text matches
        nothing."""
        started = time.monotonic()
        image = geometry.view(camera).image
        blob = await self.blobs.aput_array(np.ascontiguousarray(image.rgb[..., :3]))
        result = await self.vision.segment(blob, _prompt(prompt), 1)
        if not result.masks:
            empty = np.zeros(image.shape, dtype=bool)
            return Segmentation(empty, None, self.name, time.monotonic() - started)
        top = result.masks[0]
        mask = np.asarray(await self.blobs.aget_array(top.mask), dtype=bool)
        return Segmentation(mask, top.score, self.name, time.monotonic() - started)


def _service(resources: Resources, what: str) -> tuple[Vision, BlobStore]:
    if resources.vision is None or resources.blobs is None:
        raise LookupError(f"{what} needs the vision service and the blob store it shares")
    return resources.vision, resources.blobs


def _prompt(prompt: SegmentPrompt) -> VisionPrompt:
    if prompt.text is not None:
        return VisionPrompt(text=prompt.text)
    if prompt.box is not None:
        return VisionPrompt(boxes=[prompt.box])
    if prompt.point is not None:
        return VisionPrompt(points=[prompt.point])
    if prompt.region is not None and prompt.region.any():
        return VisionPrompt(boxes=[mask_box(prompt.region)])
    raise ValueError("a segment prompt needs a text, a box, a point or a non-empty region")


def _center(box: Sequence[float]) -> tuple[float, float]:
    u0, v0, u1, v1 = box
    return (u0 + u1) / 2, (v0 + v1) / 2
