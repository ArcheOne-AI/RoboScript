"""The models behind the vision service.

GroundingDINO and SAM 3's text prompts run on transformers 5.17 in fp32, with the processors' own
pre- and post-processing, and candidates below `FLOOR` are dropped. SAM 3's box and point prompts
use its tracker's interactive decoder. A `Detector` or `Segmenter` is built from config without
touching torch. `load()` imports torch and transformers and loads the weights, and every method
runs on the service's GPU thread. Above `FLOOR` nothing is thresholded (see
`roboscript.contracts.vision` on scores).
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.vision import Box, Detection, Point, SegmentPrompt
from roboscript_vision.config import GroundingDinoConfig, Sam3Config, VisionConfig

Image = NDArray[np.uint8]
"""RGB (H, W, 3)."""
Scored = tuple[NDArray[np.bool_], float]
"""A mask (H, W) and its score."""

DEVICE: Final = "cuda:0"
"""The first GPU that `CUDA_VISIBLE_DEVICES` names, the one device the service uses."""
TRANSFORMERS: Final = "5.17"
"""The oldest transformers the models run with: older ones lack SAM 3 and name GroundingDINO's
thresholds differently."""
FLOOR: Final = 0.05
"""The post-processing threshold for both models: it drops empty slots, not doubtful answers."""
MASK_THRESHOLD: Final = 0.5
_LOADING_PROBLEMS: Final = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")


class Detector(Protocol):
    name: str
    source: str
    revision: str
    """Set by `load`: what `ModelInfo.revision` reports."""

    def load(self) -> dict[str, str]:
        """Load the weights; return runtime facts (library versions, GPU) for `info`."""
        ...

    def detect(self, image: Image, query: str, top_k: int) -> list[Detection]:
        """At most `top_k` boxes for `query`, best first, inside the image."""
        ...


class Segmenter(Protocol):
    name: str
    source: str
    revision: str

    def load(self) -> dict[str, str]: ...

    def segment(self, image: Image, prompt: SegmentPrompt, top_k: int) -> list[Scored]:
        """As `SegmentResult.masks` orders them; boxes and points lie inside the image."""
        ...


def model_name(source: str) -> str:
    """`IDEA-Research/grounding-dino-base` -> `grounding-dino-base`, `/weights/sam3` -> `sam3`."""
    return Path(source.rstrip("/")).name


def from_config(config: VisionConfig) -> tuple[Detector | None, Segmenter | None]:
    """The configured models, not loaded yet."""
    gdino, sam3 = config.grounding_dino, config.sam3
    return (
        None if gdino is None else GroundingDino(gdino),
        None if sam3 is None else Sam3(sam3),
    )


class GroundingDino:
    """GroundingDINO as a `Detector`: one forward pass per query, lower-cased and ending with a
    period (the phrase form it was trained on); boxes are clipped to the image, and one that
    lies wholly outside it is dropped. Overlapping boxes are all kept (no NMS)."""

    def __init__(self, config: GroundingDinoConfig) -> None:
        self.source = config.model
        self.name = model_name(config.model)
        self.revision = ""
        self._pinned = config.revision
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> dict[str, str]:
        transformers = _transformers()
        self._processor = transformers.AutoProcessor.from_pretrained(
            self.source, revision=self._pinned
        )
        self._model = _load_model(
            transformers.AutoModelForZeroShotObjectDetection, self.source, self._pinned
        )
        self.revision = _revision(self._model, self.source)
        return _runtime()

    def detect(self, image: Image, query: str, top_k: int) -> list[Detection]:
        import torch

        height, width = image.shape[:2]
        text = query.lower().strip().rstrip(".") + "."
        _check_tokens(self._processor, text, self._model.config.max_text_len, "GroundingDINO")
        inputs = self._processor(
            images=PILImage.fromarray(image), text=text, return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        found = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=FLOOR,
            text_threshold=FLOOR,
            target_sizes=[(height, width)],
        )[0]
        size = torch.tensor([width, height, width, height], device=found["boxes"].device)
        boxes = torch.minimum(found["boxes"].clamp(min=0), size).tolist()
        scores = found["scores"].tolist()
        order = torch.argsort(found["scores"], descending=True).tolist()
        inside = [i for i in order if boxes[i][0] < boxes[i][2] and boxes[i][1] < boxes[i][3]]
        return [
            Detection(box=tuple(boxes[i]), score=scores[i], label=found["text_labels"][i])
            for i in inside[:top_k]
        ]


class Sam3:
    """SAM 3 as a `Segmenter`. Text goes to its concept detector (`Sam3Model`: every instance of a
    phrase); boxes and points go to its tracker's interactive decoder (`Sam3TrackerModel`: the one
    object in a box or under a point, as SAM 2 does it, best of three candidates for a point).
    Both load from the same checkpoint. Only the masks returned are upsampled to the image."""

    def __init__(self, config: Sam3Config) -> None:
        self.source = config.model
        self.name = model_name(config.model)
        self.revision = ""
        self._pinned = config.revision
        self._processor: Any = None
        self._model: Any = None
        self._tracker_processor: Any = None
        self._tracker: Any = None

    def load(self) -> dict[str, str]:
        transformers = _transformers()
        self._processor = transformers.Sam3Processor.from_pretrained(
            self.source, revision=self._pinned
        )
        self._model = _load_model(transformers.Sam3Model, self.source, self._pinned)
        self._tracker_processor = transformers.Sam3TrackerProcessor.from_pretrained(
            self.source, revision=self._pinned
        )
        self._tracker = _load_model(transformers.Sam3TrackerModel, self.source, self._pinned)
        self.revision = _revision(self._model, self.source)
        return _runtime()

    def segment(self, image: Image, prompt: SegmentPrompt, top_k: int) -> list[Scored]:
        pil = PILImage.fromarray(image)
        if prompt.text is not None:
            return self._text(pil, prompt.text, top_k)
        if prompt.boxes is not None:
            return self._boxes(pil, prompt.boxes)
        assert prompt.points is not None  # SegmentPrompt holds exactly one kind
        return self._points(pil, prompt.points)

    def _text(self, pil: PILImage.Image, text: str, top_k: int) -> list[Scored]:
        import torch

        limit = self._model.config.text_config.max_position_embeddings
        _check_tokens(self._processor, text, limit, "SAM 3")
        inputs = self._processor(images=pil, text=text, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = self._model(**inputs)
            # The processor's post_process_instance_segmentation, for the top_k instances only.
            scores = outputs.pred_logits[0].sigmoid()
            if outputs.presence_logits is not None:
                scores = scores * outputs.presence_logits[0].sigmoid()
            order = torch.argsort(scores, descending=True)[:top_k]
            order = order[scores[order] > FLOOR]
            if not len(order):  # interpolate refuses an empty batch of masks
                return []
            masks = torch.nn.functional.interpolate(
                outputs.pred_masks[0][order].sigmoid().unsqueeze(0),
                size=(pil.height, pil.width),
                mode="bilinear",
                align_corners=False,
            )[0]
            binary = (masks > MASK_THRESHOLD).cpu().numpy()
        return list(zip(binary, scores[order].tolist(), strict=True))

    def _boxes(self, pil: PILImage.Image, boxes: list[Box]) -> list[Scored]:
        import torch

        inputs = self._tracker_processor(
            images=pil, input_boxes=[[list(box) for box in boxes]], return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            outputs = self._tracker(**inputs, multimask_output=False)
        masks = self._tracker_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]  # (boxes, 1, H, W)
        scores = outputs.iou_scores[0, :, 0].tolist()
        return [(masks[i, 0].numpy(), scores[i]) for i in range(len(boxes))]

    def _points(self, pil: PILImage.Image, points: list[Point]) -> list[Scored]:
        import torch

        inputs = self._tracker_processor(
            images=pil,
            input_points=[[[list(point)] for point in points]],  # one object per point
            input_labels=[[[1] for _ in points]],
            return_tensors="pt",
        ).to(DEVICE)
        with torch.inference_mode():
            outputs = self._tracker(**inputs, multimask_output=True)
        scores, best = outputs.iou_scores[0].max(dim=1)  # the best of three candidates
        chosen = outputs.pred_masks[0, torch.arange(len(points), device=best.device), best]
        masks = self._tracker_processor.post_process_masks(
            chosen[None, :, None].cpu(), inputs["original_sizes"]
        )[0]  # (points, 1, H, W)
        return [(masks[i, 0].numpy(), score) for i, score in enumerate(scores.tolist())]


def _check_tokens(processor: Any, text: str, limit: int, model: str) -> None:
    tokens = len(processor.tokenizer(text)["input_ids"])
    if tokens > limit:
        detail = f"{text!r} is {tokens} tokens; {model} reads at most {limit}"
        raise ProtocolError("invalid_request", detail)


def _transformers() -> Any:
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version(TRANSFORMERS):
        raise RuntimeError(
            f"transformers {transformers.__version__} is too old (need {TRANSFORMERS}): run the "
            "service in the vision env (packages/vision/README.md)"
        )
    # Silences e.g. the expected warning about loading the sam3_video checkpoint as a tracker.
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    return transformers


def _load_model(model_class: Any, source: str, revision: str | None) -> Any:
    """`from_pretrained` from safetensors only, refusing weights that do not fit the class."""
    model, loading = model_class.from_pretrained(
        source, revision=revision, use_safetensors=True, output_loading_info=True
    )
    if problems := {key: loading[key] for key in _LOADING_PROBLEMS if loading.get(key)}:
        raise RuntimeError(f"{source} does not fit {model_class.__name__}: {problems}")
    return model.to(DEVICE).eval()


def _revision(model: Any, source: str) -> str:
    """The hub commit the weights came from, or sha256 of the weights file in a directory."""
    if commit := model.config._commit_hash:
        return str(commit)
    digest = hashlib.sha256()
    with (Path(source) / "model.safetensors").open("rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _runtime() -> dict[str, str]:
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(DEVICE),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],  # checked at start
    }
