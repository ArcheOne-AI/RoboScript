"""Results as model content (docs/concepts/agent-runtime.md): a line of text with the key numbers,
then the images.

The Hub renders only after aliasing, so ids appear as `obj_3` and the like. Numbers print to the
millimeter (three decimals; SI units) and fields named `*_deg` to 0.1 degree; empty fields are
left out. Image blobs go out as `ImagePart`s, downscaled to `RenderConfig.image_max_side`. A
tool spec's own `render` replaces the text and picks the images; the limits apply to both.
"""

import json
from collections.abc import Sequence
from enum import Enum
from typing import Any

import anyio
import numpy as np
from PIL import Image
from pydantic import BaseModel

from roboscript.blobs import BlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.tools import ContentPart, ImagePart, TextPart
from roboscript.hub.config import RenderConfig
from roboscript.tools import ToolSpec

_LIST_LIMIT = 8  # longer lists show their first 3 items and a count


async def render_result(
    spec: ToolSpec, shown: BaseModel, blobs: BlobStore, config: RenderConfig
) -> list[ContentPart]:
    parts = spec.render(shown) if spec.render is not None else default_parts(spec.name, shown)
    return await egress(parts, blobs, config)


def default_parts(name: str, result: BaseModel) -> list[ContentPart]:
    """`<tool>: key=value, ...`, with `<image n>` where images are, then those images."""
    images: list[BlobRef] = []
    text = f"{name}: {_fields(result, images)}" if _items(result) else f"{name}: done"
    shown = (ImagePart(blob=ref, label=f"image {i}") for i, ref in enumerate(images, 1))
    return [TextPart(text=text), *shown]


async def egress(
    parts: Sequence[ContentPart], blobs: BlobStore, config: RenderConfig
) -> list[ContentPart]:
    """Apply the limits: cut long text, keep `max_images` images, downscale large ones."""
    out: list[ContentPart] = []
    images = left_out = 0
    for part in parts:
        if isinstance(part, TextPart):
            out.append(TextPart(text=cut(part.text, config.max_text_chars)))
        elif images == config.max_images:
            left_out += 1
        else:
            images += 1
            out.append(await _fit(part, blobs, config.image_max_side))
    if left_out:
        out.append(TextPart(text=f"({left_out} more images left out.)"))
    return out


def _items(model: BaseModel) -> list[tuple[str, Any]]:
    """Computed fields (summaries, like a receipt's `ok`) first, then fields; no empty ones."""
    computed = [(name, getattr(model, name)) for name in type(model).model_computed_fields]
    return [(key, value) for key, value in [*computed, *model] if value not in (None, "", [], {})]


def _fields(model: BaseModel, images: list[BlobRef]) -> str:
    return ", ".join(f"{key}={_format(value, key, images)}" for key, value in _items(model))


def _format(value: Any, key: str, images: list[BlobRef]) -> str:
    if isinstance(value, BlobRef):
        if value.media_type.startswith("image/"):
            if value not in images:
                images.append(value)
            return f"<image {images.index(value) + 1}>"
        shape = "x".join(map(str, value.shape or ()))
        return f"<{' '.join(filter(None, (value.media_type, value.dtype, shape)))}>"
    if isinstance(value, BaseModel):
        return "{" + _fields(value, images) + "}"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _number(value, key)
    if isinstance(value, (list, tuple)):
        head = value if len(value) <= _LIST_LIMIT else value[:3]
        text = ", ".join(_format(item, key, images) for item in head)
        return f"[{text}]" if len(value) <= _LIST_LIMIT else f"[{text}, ... {len(value)} items]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={_format(v, str(k), images)}" for k, v in value.items()) + "}"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _number(value: float, key: str) -> str:
    digits = 1 if key.endswith("_deg") else 3
    text = f"{round(value, digits) + 0.0:.{digits}f}"  # + 0.0 turns -0.0 into 0.0
    return text.rstrip("0").rstrip(".")


def cut(text: str, limit: int) -> str:
    """`text`, cut after `limit` characters."""
    return text if len(text) <= limit else text[:limit] + " ... (cut)"


async def _fit(part: ImagePart, blobs: BlobStore, max_side: int) -> ImagePart:
    shape = part.blob.shape
    if shape is not None and max(shape[:2]) <= max_side:
        return part
    image = await blobs.aget_image(part.blob)
    if max(image.shape[:2]) <= max_side:
        return part
    small = await anyio.to_thread.run_sync(_downscale, image, max_side)
    blob = await blobs.aput_image(small, meta={"resized_from": part.blob.id})
    return ImagePart(blob=blob, label=part.label)


def _downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
