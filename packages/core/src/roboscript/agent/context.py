"""The context budget (docs/concepts/learning.md): how big a request's prompt is estimated to
be, and how far its input shrinks when that is over the budget.

The estimate is deterministic from what the log keeps of a request: its canonical JSON less the
image URLs and the encrypted reasoning, at `chars_per_token`; the encrypted reasoning at
`encrypted_chars_per_token` (the upstream bills far less of it than its length); each image at
its billed cost for its `detail` and the shape its blob reference records. The defaults estimate
above what the upstreams bill, so the budget errs on the safe side.

Over the budget, the loop shrinks the input one step at a time down to `shrink_to` of it (so
that the next calls, which add a little each, do not shrink again at once and the prompt cache
holds meanwhile): first fewer tool images, halving until only the newest exchange's are left,
then more previous turns summarized, one at a time, then the current turn's oldest exchanges
folded, one at a time. The steps only go one way within an episode (a new turn starts folding
its own exchanges afresh), so the prompts of an episode are a function of its log.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from roboscript.agent.history import HistoryPolicy, ImageDelivery, Shrink, Turn, summarized_turns
from roboscript.agent.recording import BLOB_URL
from roboscript.contracts.blob import BlobRef


class ImageCost(BaseModel):
    """An image's billed tokens: `tokens_per_patch` for each `patch_px` square patch, at most
    `max_patches` (a bigger image is scaled down to fit), plus `base`; `low` for low detail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    patch_px: PositiveInt = 32
    tokens_per_patch: PositiveFloat = 1.0
    max_patches: PositiveInt = 1536
    base: int = Field(default=0, ge=0)
    low: PositiveInt = 85

    def tokens(self, ref: BlobRef | None, detail: str) -> int:
        """An image of unknown shape costs the most an image can."""
        if detail == "low":
            return self.low
        patches = self.max_patches
        if ref is not None and ref.shape is not None and len(ref.shape) >= 2:
            height, width = ref.shape[0], ref.shape[1]
            size = self.patch_px
            patches = min(patches, math.ceil(height / size) * math.ceil(width / size))
        return self.base + math.ceil(patches * self.tokens_per_patch)


class ContextBudget(BaseModel):
    """`windows`: each model's context window in tokens (a gateway model or alias, as the
    request names it), `default_window` for the others; the budget is `fraction` of the window;
    a shrink goes down to `shrink_to` of the budget. After the upstream refused a prompt as too
    long, the rest of the episode runs at `overflow_fraction` of the budget with tool images
    delivered as `overflow_delivery` (a user message: some upstreams bill images inside a
    `function_call_output` as text, which inflates the prompt)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    windows: dict[str, PositiveInt] = {}
    default_window: PositiveInt = 128_000
    fraction: float = Field(default=0.6, gt=0, le=1)
    shrink_to: float = Field(default=0.75, gt=0, le=1)
    overflow_fraction: float = Field(default=0.5, gt=0, le=1)
    overflow_delivery: ImageDelivery | None = "message"
    chars_per_token: PositiveFloat = 4.0
    encrypted_chars_per_token: PositiveFloat = 20.0
    images: ImageCost = ImageCost()

    def budget(self, model: str, overflowed: bool) -> int:
        window = self.windows.get(model, self.default_window)
        return int(window * self.fraction * (self.overflow_fraction if overflowed else 1.0))


def estimate(request: Mapping[str, Any], refs: Mapping[str, BlobRef], config: ContextBudget) -> int:
    """The prompt tokens a request is estimated to cost."""
    counts = {"text": 0, "encrypted": 0, "images": 0}
    stripped = _strip(request, refs, config, counts)
    text = len(json.dumps(stripped, ensure_ascii=False, separators=(",", ":")))
    return (
        math.ceil(text / config.chars_per_token)
        + math.ceil(counts["encrypted"] / config.encrypted_chars_per_token)
        + counts["images"]
    )


def _strip(
    value: Any, refs: Mapping[str, BlobRef], config: ContextBudget, counts: dict[str, int]
) -> Any:
    """`value` without image URLs and encrypted reasoning, whose cost goes to `counts`."""
    if isinstance(value, dict):
        if value.get("type") == "input_image":
            url = str(value.get("image_url", ""))
            ref = refs.get(url.removeprefix(BLOB_URL)) if url.startswith(BLOB_URL) else None
            counts["images"] += config.images.tokens(ref, str(value.get("detail", "auto")))
            return {"type": "input_image"}
        out = {}
        for key, item in value.items():
            if key == "encrypted_content" and isinstance(item, str):
                counts["encrypted"] += len(item)
            else:
                out[key] = _strip(item, refs, config, counts)
        return out
    if isinstance(value, list):
        return [_strip(item, refs, config, counts) for item in value]
    return value


def tighter(
    shrink: Shrink, turns: Sequence[Turn], policy: HistoryPolicy, shown: int
) -> Shrink | None:
    """The next shrink step after `shrink` for an input built from `turns` that shows `shown`
    tool images; None when nothing more can go."""
    images = min(policy.tool_images, shown) if shrink.tool_images is None else shrink.tool_images
    if images > 0:
        return replace(shrink, tool_images=images // 2)
    previous = len(turns) - 1
    summarized = min(
        previous, max(summarized_turns(previous, policy.text_turns), shrink.fold_turns)
    )
    if summarized < previous:
        return replace(shrink, fold_turns=summarized + 1)
    if shrink.fold_exchanges < len(turns[-1].exchanges) - 1:
        return replace(shrink, fold_exchanges=shrink.fold_exchanges + 1)
    return None
