"""roboscript.testing.perception: the grounder and segmenter conformance suites, on every
backend (the VLM and the vision service through their stand-ins), and their entry points."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.vision import Detection
from roboscript.contracts.vision import SegmentPrompt as VisionPrompt
from roboscript.kernel import entrypoints
from roboscript.perception.grounding import (
    Candidate,
    GeometricGrounder,
    Grounding,
    Mode,
    mask_box,
)
from roboscript.perception.lift import box_mask
from roboscript.perception.proposals import propose
from roboscript.perception.views import FrameGeometry
from roboscript.perception.vision import VisionDetector, VisionSegmenter
from roboscript.perception.vlm import BoxGrounder, SomGrounder
from roboscript.testing import precheck
from roboscript.testing.perception import (
    FakeVision,
    GrounderConformance,
    GrounderTarget,
    ScriptedModel,
    SegmenterConformance,
    SegmenterTarget,
    scene_geometry,
)
from roboscript.testing.scenes import PERCEPTION


@pytest.fixture(scope="module")
def geometry() -> FrameGeometry:
    return scene_geometry()


@pytest.fixture
def grounder_target(geometry: FrameGeometry) -> GrounderTarget:
    return GrounderTarget(GeometricGrounder(PERCEPTION), geometry)


class TestGeometricGrounderConformance(GrounderConformance):
    pass


async def test_the_checks_run_without_pytest(grounder_target: GrounderTarget) -> None:
    results = await precheck(GrounderConformance(), grounder_target, tiers=["unit"])
    assert results == {
        "test_both_modes_answer": None,
        "test_the_answer_has_the_shared_format": None,
    }


async def test_conformance_catches_a_broken_grounder(geometry: FrameGeometry) -> None:
    class OffImage(GeometricGrounder):
        async def ground(
            self, geometry: FrameGeometry, camera: str, query: str, mode: Mode = "box"
        ) -> Grounding:
            answer = await super().ground(geometry, camera, query, mode)
            box = Candidate(box=(0.0, 0.0, 10_000.0, 5.0))
            return Grounding(answer.frame, camera, query, [box], answer.overlay, self.name, 0.1)

    results = await precheck(GrounderConformance(), GrounderTarget(OffImage(), geometry))
    assert results["test_the_answer_has_the_shared_format"] is not None


def first_mark(body: dict[str, Any]) -> dict[str, Any]:
    """Set-of-Mark: the first mark shown, for every key."""
    schema = body["text"]["format"]["schema"]["properties"]
    return {key: prop["anyOf"][0]["enum"][0] for key, prop in schema.items()}


class TestBoxGrounderConformance(GrounderConformance):
    @pytest.fixture
    def grounder_target(self, geometry: FrameGeometry) -> GrounderTarget:
        box = {"found": True, "box": [100, 100, 200, 180], "point": [150, 140]}
        return GrounderTarget(
            BoxGrounder(ScriptedModel(lambda body: box).client(), pipeline=PERCEPTION), geometry
        )


class TestSomGrounderConformance(GrounderConformance):
    @pytest.fixture
    def grounder_target(self, geometry: FrameGeometry) -> GrounderTarget:
        return GrounderTarget(
            SomGrounder(ScriptedModel(first_mark).client(), pipeline=PERCEPTION), geometry
        )


class TestVisionDetectorConformance(GrounderConformance):
    @pytest.fixture
    def grounder_target(self, geometry: FrameGeometry, tmp_path: Path) -> GrounderTarget:
        blobs = LocalBlobStore(tmp_path)
        found = [Detection(box=(50.0, 60.0, 150.0, 140.0), score=0.5, label="red block")]
        vision = FakeVision(blobs, boxes=lambda query: found)
        return GrounderTarget(VisionDetector(vision, blobs), geometry)


class TestVisionSegmenterConformance(SegmenterConformance):
    @pytest.fixture
    def segmenter_target(self, geometry: FrameGeometry, tmp_path: Path) -> SegmenterTarget:
        blobs = LocalBlobStore(tmp_path)

        def masks(image: np.ndarray, prompt: VisionPrompt) -> list[np.ndarray]:
            assert prompt.boxes is not None  # a region is sent as its box
            return [box_mask(prompt.boxes[0], image.shape[:2])]

        segmenter = VisionSegmenter(FakeVision(blobs, masks=masks), blobs)
        box = mask_box(propose(geometry, PERCEPTION)[0].masks["overhead"])
        return SegmenterTarget(segmenter, geometry, box)


@pytest.mark.skipif(
    "geometric" not in entrypoints.names("roboscript.perception"),
    reason="the simulator env runs roboscript from source: no entry points",
)
def test_the_backends_are_registered() -> None:
    names = entrypoints.names("roboscript.perception")
    assert names == ["geometric", "grounding_dino", "sam3", "som", "vlm"]
    for name in names:
        assert entrypoints.load("roboscript.perception", name).name == name
