"""The vision plane: open-vocabulary detection and promptable segmentation on a GPU
(docs/concepts/perception.md), served by `roboscript_vision` (`python -m roboscript_vision`).

Images and masks travel by reference through the blob store the caller shares with the service
(`roboscript.blobs`): an image is RGB (H, W, 3) uint8 with 1 <= H, W and H * W <=
`MAX_IMAGE_PIXELS`, stored as `.npy` (as `observe` stores it) or PNG/JPEG; a mask comes back as a
bool (H, W) `.npy` of the image's size.

Pixel coordinates are continuous: u to the right, v down, from the image's top-left corner, and
the pixel in column c and row r covers [c, c + 1) x [r, r + 1). A box is (u0, v0, u1, v1) with
u0 < u1 and v0 < v1, so the whole image is (0, 0, W, H) and a mask's box, the smallest one holding
its pixels, is (c0, r0, c1 + 1, r1 + 1) for its first and last column and row. A point (u, v)
lies in pixel (floor(u), floor(v)).

Scores rank a model's answers and nothing more. They are not confidences: a model's usual threshold
keeps wrong answers along with right ones. Rank
by them and log them; never threshold them. `model` names the model that answered, e.g.
`grounding-dino-base`.
"""

from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal, Protocol

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeFloat,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.versions import VERSIONS

MAX_IMAGE_PIXELS: Final = 1920 * 1080
MAX_QUERIES: Final = 16
MAX_PROMPTS: Final = 32
MAX_TOP_K: Final = 50

TIMEOUTS: Final = MappingProxyType({"info": 10.0, "detect": 60.0, "segment": 60.0})
"""Client timeouts in seconds, e.g. `rpc.client(Vision, url, token=None, timeout=10,
timeouts=TIMEOUTS)`; calls wait for each other on the service's one GPU thread."""


def _positive_area(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    u0, v0, u1, v1 = box
    if not (u0 < u1 and v0 < v1):
        raise ValueError(f"a box is (u0, v0, u1, v1) with u0 < u1 and v0 < v1, got {box}")
    return box


def _not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("the text is blank")
    return text


Box = Annotated[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat], AfterValidator(_positive_area)
]
"""(u0, v0, u1, v1) in pixels, with positive area."""
Point = tuple[FiniteFloat, FiniteFloat]
"""(u, v) in pixels."""
Text = Annotated[str, Field(min_length=1, max_length=256), AfterValidator(_not_blank)]
Queries = Annotated[list[Text], Field(min_length=1, max_length=MAX_QUERIES)]
TopK = Annotated[int, Field(ge=1, le=MAX_TOP_K)]


class Detection(BaseModel):
    """A box for a query; `label` holds the words of the query the model matched it to."""

    model_config = ConfigDict(frozen=True)

    box: Box
    score: float
    label: str


class QueryDetections(BaseModel):
    """Best first, at most `top_k`; empty when the model proposes nothing for the query."""

    model_config = ConfigDict(frozen=True)

    query: str
    detections: list[Detection]


class DetectResult(BaseModel):
    """One entry per query, in the order asked. `time_s` is the model time of the call on the
    service's GPU thread, without waiting for other calls or blob IO."""

    model_config = ConfigDict(frozen=True)

    model: str
    queries: list[QueryDetections]
    time_s: NonNegativeFloat


class SegmentPrompt(BaseModel):
    """Exactly one of: `text`, every instance of a phrase (SAM 3's concept segmentation);
    `boxes`, the one object inside each box; `points`, the one object under each point (a single
    point often segments a part only, e.g. a mug's handle). Boxes and points lie in the image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: Text | None = None
    boxes: Annotated[list[Box], Field(min_length=1, max_length=MAX_PROMPTS)] | None = None
    points: Annotated[list[Point], Field(min_length=1, max_length=MAX_PROMPTS)] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        given = [name for name in ("text", "boxes", "points") if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError(f"give exactly one of text, boxes and points, got {given or 'none'}")
        return self


class Mask(BaseModel):
    """A bool (H, W) mask and its box (None when it is empty)."""

    model_config = ConfigDict(frozen=True)

    mask: BlobRef
    score: float
    box: Box | None


class SegmentResult(BaseModel):
    """For text, the instances best first, at most `top_k`; for boxes or points, one mask per
    prompt in their order, scored by the model's own estimate of its IoU. `time_s` as in
    `DetectResult`."""

    model_config = ConfigDict(frozen=True)

    model: str
    masks: list[Mask]
    time_s: NonNegativeFloat


class ModelInfo(BaseModel):
    """`source` is a hub repository or a local directory; `revision` the hub commit of the loaded
    files, or `sha256:<hex>` of the weights file loaded from a directory."""

    model_config = ConfigDict(frozen=True)

    name: str
    source: str
    revision: str
    task: Literal["detect", "segment"]


class VisionInfo(BaseModel):
    """The loaded models and `runtime`: library versions, the GPU and `CUDA_VISIBLE_DEVICES`."""

    model_config = ConfigDict(frozen=True)

    models: list[ModelInfo]
    runtime: dict[str, str]


class Vision(Protocol):
    """Detection and segmentation of images in a shared blob store, for any number of callers.
    Calls wait for one GPU; when too many wait, a call fails as `busy` (retryable)."""

    PLANE: ClassVar[str] = "vision"
    VERSION: ClassVar[str] = VERSIONS[PLANE]

    async def info(self) -> VisionInfo: ...

    async def detect(self, image: BlobRef, queries: Queries, top_k: TopK = 5) -> DetectResult:
        """Boxes for each text query (GroundingDINO). It reads '.' as the end of a phrase: a query
        holding one is several phrases, and a box may match any of them (its `label` says which).
        Overlapping boxes for one object are all kept (no non-maximum suppression). A query
        longer than 256 tokens is refused (`invalid_request`)."""
        ...

    async def segment(
        self, image: BlobRef, prompt: SegmentPrompt, top_k: TopK = 5
    ) -> SegmentResult:
        """Masks for a text, boxes or points (SAM 3); `top_k` bounds text instances only. A text
        longer than SAM 3's 32 tokens is refused (`invalid_request`)."""
        ...
