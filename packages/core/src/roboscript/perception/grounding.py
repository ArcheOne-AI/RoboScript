"""Grounding and segmentation backends: one output format whatever the backend.

A `Grounder` answers "where is <query>" in one camera of a frame with candidates (boxes or
points, scores, region ids) and a numbered overlay; a `Segmenter` turns a prompt (text, a box, a
point or a region) into a mask. Backends register in the `roboscript.perception` entry-point
group, and each builds itself from `Resources` (`build`): a model client, the vision service, the
blob store, and the frame's proposals as its caller computes them (`ProposalSource`), so that a
region id a backend names is the caller's proposal. The core ships `geometric`: every proposal of
the frame is a candidate, drawn as a Set-of-Mark overlay with attribute cards, for the agent (or
a model) to choose from; the VLM (`som`, `vlm`), GroundingDINO and SAM 3 backends reach their
models through the gateway and the vision service.

Model answers are 2D; positions come from geometry. A point lifted on its own lands wherever
the model's pixel fell, so
`snap` hands back the proposal under a point, or the one a box fits best, instead. Boxes and points
use the vision plane's continuous pixel coordinates (`roboscript.contracts.vision`): pixel (c, r)
covers [c, c + 1) x [r, r + 1), so a mask's box is (c0, r0, c1 + 1, r1 + 1) and a point lies in
pixel (floor(u), floor(v)).
"""

import time
import weakref
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

import anyio
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from roboscript.contracts.ids import FrameId
from roboscript.perception.config import DEFAULTS, PerceptionConfig
from roboscript.perception.lift import box_mask
from roboscript.perception.overlay import attribute_card, som_overlay
from roboscript.perception.proposals import Proposal, propose
from roboscript.perception.views import FrameGeometry

Mode = Literal["box", "point"]
ProposalSource = Callable[[FrameGeometry], Awaitable[list[Proposal]]]
"""A frame's proposals, as the caller computes them (its pipeline config, its cache)."""


def proposal_source(config: PerceptionConfig = DEFAULTS) -> ProposalSource:
    """Proposals computed with `config` in a worker thread, once per analysis while it lives."""
    kept: weakref.WeakKeyDictionary[FrameGeometry, list[Proposal]] = weakref.WeakKeyDictionary()

    async def proposals(geometry: FrameGeometry) -> list[Proposal]:
        if geometry not in kept:
            kept[geometry] = await anyio.to_thread.run_sync(propose, geometry, config)
        return kept[geometry]

    return proposals


@dataclass(frozen=True)
class Resources:
    """What a backend may build itself from (`build`): the pipeline config and the caller's
    proposals, a model `client` of the gateway (an `openai.AsyncOpenAI`) with the VLM settings
    (`roboscript.perception.vlm.VlmConfig`), the blob store and the `vision` service."""

    config: PerceptionConfig = DEFAULTS
    proposals: ProposalSource | None = None
    client: Any = None
    vlm: Any = None
    blobs: Any = None
    vision: Any = None

    def source(self) -> ProposalSource:
        return self.proposals if self.proposals is not None else proposal_source(self.config)


class Candidate(BaseModel):
    """One answer: a `region` (a proposal id of the frame) and/or a `box` (u0, v0, u1, v1) or
    `point` (u, v) in continuous pixel coordinates, with the backend's `score` (it ranks, it is
    never thresholded) and a text `label`."""

    model_config = ConfigDict(frozen=True)

    region: int | None = None
    box: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None
    score: float | None = None
    label: str | None = None


@dataclass(frozen=True, eq=False)
class Grounding:
    """Candidates for `query` in `camera`, best first, the overlay showing them (H, W, 3), and
    the model call behind them, if one was made (`vlm.Asked`, for the log and the cost)."""

    frame: FrameId
    camera: str
    query: str
    candidates: list[Candidate]
    overlay: NDArray[np.uint8]
    backend: str
    latency_s: float
    asked: Any = None


@dataclass(frozen=True)
class SegmentPrompt:
    """Exactly one of: text, a box (u0, v0, u1, v1), a point (u, v), a region mask (H, W)."""

    text: str | None = None
    box: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None
    region: NDArray[np.bool_] | None = None


@dataclass(frozen=True, eq=False)
class Segmentation:
    mask: NDArray[np.bool_]
    score: float | None
    backend: str
    latency_s: float


class Grounder(Protocol):
    name: ClassVar[str]

    async def ground(
        self, geometry: FrameGeometry, camera: str, query: str, mode: Mode = "box"
    ) -> Grounding: ...


class Segmenter(Protocol):
    name: ClassVar[str]

    async def segment(
        self, geometry: FrameGeometry, camera: str, prompt: SegmentPrompt
    ) -> Segmentation: ...


class GeometricGrounder:
    """Every proposal visible in the camera, as a numbered region with its box, anchor point and
    attribute card, in either mode; the query is not interpreted (no model). The proposals come
    from `proposals` (by default computed with `config`, once per analysis)."""

    name: ClassVar[str] = "geometric"

    def __init__(
        self, config: PerceptionConfig = DEFAULTS, proposals: ProposalSource | None = None
    ) -> None:
        self.config = config
        self.proposals = proposal_source(config) if proposals is None else proposals

    @classmethod
    def build(cls, resources: Resources) -> "GeometricGrounder":
        return cls(resources.config, resources.source())

    async def ground(
        self, geometry: FrameGeometry, camera: str, query: str, mode: Mode = "box"
    ) -> Grounding:
        started = time.perf_counter()
        proposals = await self.proposals(geometry)
        image = geometry.view(camera).image
        shown = [p for p in proposals if p.masks[camera].any()]
        overlay, anchors = await anyio.to_thread.run_sync(
            som_overlay, image.rgb, {p.id: p.masks[camera] for p in shown}
        )
        cards = await anyio.to_thread.run_sync(
            attribute_card, image.rgb, shown, geometry.table, camera, geometry.base
        )
        candidates = [
            Candidate(
                region=p.id, box=mask_box(p.masks[camera]), point=_center(anchors[p.id]), label=line
            )
            for p, line in zip(shown, cards.splitlines(), strict=True)
        ]
        return Grounding(
            frame=geometry.frame,
            camera=camera,
            query=query,
            candidates=candidates,
            overlay=overlay,
            backend=self.name,
            latency_s=time.perf_counter() - started,
        )


def snap(
    proposals: Sequence[Proposal],
    camera: str,
    point: Sequence[float] | None = None,
    box: Sequence[float] | None = None,
    radius_px: float = 6.0,
) -> Proposal | None:
    """The proposal a 2D answer refers to: the one under `point` (or within `radius_px` of it),
    else the one whose mask best overlaps `box` (IoU), so a big object behind a loosely boxed
    small one does not win; None when none does."""
    if point is not None:
        u, v = np.floor(np.asarray(point, dtype=float))  # the pixel it lies in
        best, best_distance = None, radius_px
        for p in proposals:
            vv, uu = np.nonzero(p.masks[camera])
            if not len(uu):
                continue
            distance = float(np.min(np.maximum(np.abs(uu - u), np.abs(vv - v))))
            if distance < best_distance:
                best, best_distance = p, distance
        return best
    if box is not None:
        best_box, best_iou = None, 0.0
        for p in proposals:
            region = box_mask(box, p.masks[camera].shape)
            iou = (p.masks[camera] & region).sum() / max((p.masks[camera] | region).sum(), 1)
            if iou > best_iou:
                best_box, best_iou = p, float(iou)
        return best_box
    raise ValueError("give a point or a box")


def mask_box(mask: NDArray[np.bool_]) -> tuple[float, float, float, float]:
    """The smallest box holding a (non-empty) mask's pixels, (c0, r0, c1 + 1, r1 + 1)."""
    v, u = np.nonzero(mask)
    return float(u.min()), float(v.min()), float(u.max() + 1), float(v.max() + 1)


def _center(pixel: tuple[int, int]) -> tuple[float, float]:
    return pixel[0] + 0.5, pixel[1] + 0.5
