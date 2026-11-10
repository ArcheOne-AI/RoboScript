"""Conformance checks for grounding and segmentation backends (docs/concepts/perception.md),
and stand-ins for their expensive edges.

A grounder conforms when it answers in the one format every backend shares: candidates in the
camera's continuous pixel coordinates (boxes with positive area inside the image, points inside
it, region ids that name the frame's proposals), an overlay of the camera's size, its own backend
name and a latency, for any camera of the frame and in both modes. A segmenter conforms when its
mask has the camera's size, is not empty for a box around an object, and names its backend. The
checks run on a synthetic scene (`scene_geometry`), so no benchmark is needed.

Use them from pytest by subclassing `GrounderConformance` (or `SegmenterConformance`) as a
`Test*` class and providing a `grounder_target` (`segmenter_target`) fixture, or without pytest
via `roboscript.testing.precheck`.

`ScriptedModel` stands in for the model gateway (OpenAI clients whose Responses calls a function
answers) and `FakeVision` for the vision service (`Vision`, in-process, with scripted boxes and
masks), for tests of the VLM and vision backends and of the perception toolsets.
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import openai
from numpy.typing import NDArray

from roboscript.blobs import BlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.ids import FrameId
from roboscript.contracts.vision import (
    Detection,
    DetectResult,
    Mask,
    ModelInfo,
    QueryDetections,
    SegmentPrompt,
    SegmentResult,
    VisionInfo,
)
from roboscript.perception.grounding import Grounder, Grounding, Segmenter
from roboscript.perception.grounding import SegmentPrompt as MaskPrompt
from roboscript.perception.proposals import propose
from roboscript.perception.views import FrameGeometry, frame_geometry
from roboscript.testing import unit
from roboscript.testing.scenes import HEAD, PERCEPTION, TABLE_Z, Box, capture, look_at, pinhole

Answer = Callable[[dict[str, Any]], Any]
"""A request body -> the JSON the model answers (a string is sent as it is, an `httpx.Response`
instead of the answer, e.g. an error status)."""


class ScriptedModel:
    """Model clients whose Responses calls `answer` answers; every request body is kept in
    `requests`. Use `client` as a Hub's or a backend's model factory."""

    def __init__(self, answer: Answer) -> None:
        self.answer = answer
        self.requests: list[dict[str, Any]] = []

    def client(self, session_id: str = "test") -> openai.AsyncOpenAI:
        http = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        return openai.AsyncOpenAI(
            api_key=f"token:{session_id}",
            base_url="http://model.test/v1",
            max_retries=0,
            http_client=http,  # type: ignore[arg-type]  # the client accepts any httpx client
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        answer = self.answer(body)
        if isinstance(answer, httpx.Response):
            return answer
        text = answer if isinstance(answer, str) else json.dumps(answer)
        headers = {"x-rs-deployment": "scripted"}
        return httpx.Response(200, json=_response(body["model"], text), headers=headers)


def asked_for(body: Mapping[str, Any]) -> str:
    """The JSON schema's name of a request: `set_of_mark`, `box`, `boxes` or `roles`."""
    return str(body["text"]["format"]["name"])


def prompt_of(body: Mapping[str, Any]) -> str:
    return str(body["input"][0]["content"][0]["text"])


def som_queries(body: Mapping[str, Any]) -> dict[str, str]:
    """A Set-of-Mark or boxes request's answer keys and the text each role comes with (its
    description, or what an abstract role does and is)."""
    found = re.findall(r'^- "(r\d+)"(?: [a-z0-9_]+)?: (.*)$', prompt_of(body), re.MULTILINE)
    return dict(found)


def query_roles(body: Mapping[str, Any]) -> dict[str, str]:
    """A request's answer keys and the abstract roles they stand for (none without roles)."""
    return dict(re.findall(r'^- "(r\d+)" ([a-z0-9_]+): ', prompt_of(body), re.MULTILINE))


def instruction_of(body: Mapping[str, Any]) -> str | None:
    """The live instruction a request carries, if any."""
    found = re.search(r'^The robot\'s instruction: "(.*)"$', prompt_of(body), re.MULTILINE)
    return None if found is None else found[1]


def som_cards(body: Mapping[str, Any]) -> dict[int, str]:
    """A Set-of-Mark request's attribute cards: mark -> color word."""
    found = re.findall(r"^mark (\d+): (\w+),", prompt_of(body), re.MULTILINE)
    return {int(mark): color for mark, color in found}


def box_query(body: Mapping[str, Any]) -> str:
    """What a box request asks to locate."""
    match = re.search(r'Locate: "(.*?)"\.', prompt_of(body))
    assert match is not None, prompt_of(body)
    return match[1]


def _response(model: str, text: str) -> dict[str, Any]:
    content = {"type": "output_text", "text": text, "annotations": []}
    message = {"id": "msg_1", "type": "message", "status": "completed", "role": "assistant"}
    usage = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 20,
        "output_tokens_details": {"reasoning_tokens": 10},
        "total_tokens": 1020,
    }
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [{**message, "content": [content]}],
        "usage": usage,
    }


@dataclass
class FakeVision:
    """The vision service in-process: `boxes(query)` gives a query's detections (best first),
    `masks(image, prompt)` a prompt's masks; images and masks go through `blobs`."""

    blobs: BlobStore
    boxes: Callable[[str], list[Detection]] = lambda query: []
    masks: Callable[[NDArray[np.uint8], SegmentPrompt], list[NDArray[np.bool_]]] = (
        lambda image, prompt: []
    )
    calls: list[tuple[str, Any]] = field(default_factory=list)

    async def info(self) -> VisionInfo:
        models = [
            ModelInfo(name="fake-dino", source="fake", revision="0", task="detect"),
            ModelInfo(name="fake-sam", source="fake", revision="0", task="segment"),
        ]
        return VisionInfo(models=models, runtime={})

    async def detect(self, image: BlobRef, queries: list[str], top_k: int = 5) -> DetectResult:
        self.calls.append(("detect", list(queries)))
        found = [QueryDetections(query=q, detections=self.boxes(q)[:top_k]) for q in queries]
        return DetectResult(model="fake-dino", queries=found, time_s=0.01)

    async def segment(self, image: BlobRef, prompt: SegmentPrompt, top_k: int = 5) -> SegmentResult:
        self.calls.append(("segment", prompt))
        rgb = await self.blobs.aget_array(image)
        masks = []
        for mask in self.masks(np.asarray(rgb), prompt)[:top_k]:
            v, u = np.nonzero(mask)
            box = (float(u.min()), float(v.min()), float(u.max() + 1), float(v.max() + 1))
            masks.append(Mask(mask=await self.blobs.aput_array(mask), score=0.9, box=box))
        return SegmentResult(model="fake-sam", masks=masks, time_s=0.01)


def scene_geometry() -> FrameGeometry:
    """Three colored blocks on the table, seen by an overhead and a side camera."""
    shapes = [
        Box((-0.15, 0.05, TABLE_Z + 0.025), (0.05, 0.05, 0.05), yaw_deg=20, rgb=(200, 30, 30)),
        Box((0.1, -0.05, TABLE_Z + 0.02), (0.12, 0.04, 0.04), yaw_deg=-35, rgb=(40, 170, 60)),
        Box((0.05, 0.15, TABLE_Z + 0.03), (0.06, 0.06, 0.06), rgb=(40, 60, 200)),
    ]
    side = look_at([-0.45, 0.0, 0.95], [0.0, 0.05, TABLE_Z])
    images, _ = capture(shapes, [pinhole("overhead", HEAD), pinhole("side", side, 256, 192)])
    return frame_geometry(FrameId("frame-conformance"), 0, images, PERCEPTION)


@dataclass(frozen=True)
class GrounderTarget:
    grounder: Grounder
    geometry: FrameGeometry
    query: str = "the red block"


class GrounderConformance:
    """The checks, as `test_*` methods taking a `GrounderTarget`."""

    @unit
    async def test_the_answer_has_the_shared_format(self, grounder_target: GrounderTarget) -> None:
        t = grounder_target
        for camera in t.geometry.views:
            answer = await t.grounder.ground(t.geometry, camera, t.query)
            _check(answer, t, camera)

    @unit
    async def test_both_modes_answer(self, grounder_target: GrounderTarget) -> None:
        t = grounder_target
        boxes = await t.grounder.ground(t.geometry, "overhead", t.query, mode="box")
        points = await t.grounder.ground(t.geometry, "overhead", t.query, mode="point")
        assert all(c.box is not None or c.region is not None for c in boxes.candidates)
        assert all(c.point is not None or c.region is not None for c in points.candidates)
        _check(boxes, t, "overhead")
        _check(points, t, "overhead")


def _check(answer: Grounding, target: GrounderTarget, camera: str) -> None:
    h, w = target.geometry.view(camera).image.shape
    assert answer.backend == type(target.grounder).name, answer.backend
    assert (answer.frame, answer.camera, answer.query) == (
        target.geometry.frame,
        camera,
        target.query,
    )
    assert answer.overlay.shape == (h, w, 3) and answer.overlay.dtype == np.uint8
    assert answer.latency_s >= 0
    regions = {p.id for p in propose(target.geometry, PERCEPTION)}
    for c in answer.candidates:
        if c.box is not None:  # continuous pixel coordinates (contracts.vision)
            u0, v0, u1, v1 = c.box
            assert 0 <= u0 < u1 <= w and 0 <= v0 < v1 <= h, (camera, c.box)
        if c.point is not None:
            assert 0 <= c.point[0] < w and 0 <= c.point[1] < h, (camera, c.point)
        assert c.region is None or c.region in regions, (c.region, sorted(regions))


@dataclass(frozen=True)
class SegmenterTarget:
    """A segmenter and a box (continuous pixels of `camera`) around an object of the scene."""

    segmenter: Segmenter
    geometry: FrameGeometry
    box: tuple[float, float, float, float]
    camera: str = "overhead"


class SegmenterConformance:
    """The checks, as `test_*` methods taking a `SegmenterTarget`."""

    @unit
    async def test_a_box_gives_a_mask_of_the_image(self, segmenter_target: SegmenterTarget) -> None:
        t = segmenter_target
        found = await t.segmenter.segment(t.geometry, t.camera, MaskPrompt(box=t.box))
        h, w = t.geometry.view(t.camera).image.shape
        assert found.mask.shape == (h, w) and found.mask.dtype == np.bool_
        assert found.mask.any(), "a box around an object gives a mask"
        assert found.backend == type(t.segmenter).name and found.latency_s >= 0

    @unit
    async def test_a_region_gives_a_mask_too(self, segmenter_target: SegmenterTarget) -> None:
        t = segmenter_target
        h, w = t.geometry.view(t.camera).image.shape
        region = np.zeros((h, w), dtype=bool)
        u0, v0, u1, v1 = (int(x) for x in t.box)
        region[v0:v1, u0:u1] = True
        found = await t.segmenter.segment(t.geometry, t.camera, MaskPrompt(region=region))
        assert found.mask.shape == (h, w) and found.mask.any()
