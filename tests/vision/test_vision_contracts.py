"""The vision plane's models and interface."""

import pytest
from pydantic import ValidationError

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.ids import BlobId
from roboscript.contracts.versions import VERSIONS
from roboscript.contracts.vision import (
    TIMEOUTS,
    Detection,
    DetectResult,
    Mask,
    QueryDetections,
    SegmentPrompt,
    SegmentResult,
    Vision,
)
from roboscript.rpc import describe

REF = BlobRef(id=BlobId("0" * 64), size=1, media_type="application/x-npy", dtype="bool")


def test_the_interface_and_its_timeouts() -> None:
    spec = describe(Vision)
    assert str(spec.contract) == f"vision/{VERSIONS['vision']}"
    assert set(spec.methods) == set(TIMEOUTS) == {"info", "detect", "segment"}


def test_a_segment_prompt_holds_exactly_one_kind() -> None:
    assert SegmentPrompt(text="red block").text == "red block"
    assert SegmentPrompt(boxes=[(1, 2, 3, 4)]).boxes == [(1.0, 2.0, 3.0, 4.0)]
    assert SegmentPrompt(points=[(5, 6)]).points == [(5.0, 6.0)]
    for bad in ({}, {"text": "a", "points": [(1, 1)]}, {"boxes": [], "text": "a"}):
        with pytest.raises(ValidationError):
            SegmentPrompt.model_validate(bad)
    with pytest.raises(ValidationError, match="exactly one of text, boxes and points"):
        SegmentPrompt(text="a", boxes=[(0, 0, 1, 1)])
    for box in [(10, 0, 5, 5), (3, 0, 3, 5), (0, 4, 5, 4)]:  # boxes need a positive area
        with pytest.raises(ValidationError, match="u0 < u1 and v0 < v1"):
            SegmentPrompt(boxes=[box])
    with pytest.raises(ValidationError, match="blank"):
        SegmentPrompt(text=" \t ")
    with pytest.raises(ValidationError):
        SegmentPrompt(text="")


def test_results_round_trip_through_json() -> None:
    detected = DetectResult(
        model="grounding-dino-base",
        queries=[
            QueryDetections(
                query="red block",
                detections=[Detection(box=(1.5, 2, 30, 40), score=0.63, label="red block")],
            ),
            QueryDetections(query="green block", detections=[]),
        ],
        time_s=0.2,
    )
    segmented = SegmentResult(
        model="sam3",
        masks=[Mask(mask=REF, score=0.9, box=(3, 4, 5, 6)), Mask(mask=REF, score=0.1, box=None)],
        time_s=0.3,
    )
    for result in (detected, segmented):
        assert type(result).model_validate_json(result.model_dump_json()) == result
