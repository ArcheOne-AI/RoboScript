"""What the model is shown of an episode so far (docs/concepts/learning.md): the Responses
`input` built from the brief and the turns under a `HistoryPolicy`.

- The brief (the first user message, e.g. the Reviser's material with key images) in full; its
  images only in the first `brief_image_turns` turns (a role without actions has one long turn,
  so it keeps them), placeholders afterwards.
- The current turn in full, the model's reasoning items included, so that its encrypted
  reasoning carries across the calls of one turn; images returned by its tools only for its last
  `image_exchanges` exchanges (roles without actions live in one long turn).
- The previous turns in text: their observations, calls, outputs and text; images only for the
  previous `image_turns` turns (placeholders otherwise); no reasoning, and replayed items lose
  their ids (an item id would make the API ask for the reasoning item that went with it).
- Older turns as one summary line each. Turns are summarized `text_turns` at a time, so between
  `text_turns` and `2 * text_turns - 1` previous turns stay in full and the start of the input
  only changes once every `text_turns` turns (the prompt cache holds meanwhile).
- At most `tool_images` images from tool results, the newest first, a call's images all or none;
  the others as a stub that names them and the call that shows them again. Shown tool images go
  inside the outputs, or (`ImageDelivery`) in one user message after the exchange's outputs,
  numbered, each output naming its own.

A `Shrink` (the context budget's, `agent.context`) shows fewer tool images, summarizes more
previous turns and folds the current turn's oldest exchanges into one summary message; the brief
and the newest exchange stay as they are.

Images appear as `rs-blob:<sha256>` URLs; `recording.materialize` turns them into data URLs.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from roboscript.agent.recording import BLOB_URL
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.tools import ContentPart, ImagePart, TextPart

ImageDelivery = Literal["output", "message"]
"""Where images from tool results go: inside the `function_call_output` (the default), or in one
user message right after the reply's outputs, numbered in order, each output keeping its text and
naming its images ("image 2 attached below"). A user message ends the model's reasoning chain
(the API drops earlier reasoning items after one), so `message` loses the carry-over within a
turn."""

Detail = Literal["low", "high", "auto"]

_SUMMARY_ARGS: Final = 160  # characters of a call's arguments in the summary
_SUMMARY_OUTPUT: Final = 200  # characters of its output


class HistoryPolicy(BaseModel):
    """`text_turns`: previous turns kept in full text (at least that many once there are, at most
    twice that many less one); `image_turns`: previous turns whose images are shown;
    `image_exchanges`: exchanges of the current turn whose tool images are shown;
    `brief_image_turns`: the turns (from the first) that show the brief's images (None: all)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_turns: NonNegativeInt = 6
    image_turns: NonNegativeInt = 0
    image_exchanges: NonNegativeInt = 2
    brief_image_turns: PositiveInt | None = 3
    tool_images: NonNegativeInt = 8


@dataclass(frozen=True)
class Shrink:
    """What the context budget takes out beyond the policy: `tool_images`, the most tool images
    shown (None: the policy's; the newest exchange's always show, within the policy's);
    `fold_turns`, previous turns summarized at least; `fold_exchanges`, the current turn's oldest
    exchanges folded into one summary message (never its last); `delivery`, where tool images
    go instead of the configured delivery."""

    tool_images: int | None = None
    fold_turns: int = 0
    fold_exchanges: int = 0
    delivery: ImageDelivery | None = None


NO_SHRINK: Final = Shrink()


@dataclass
class CallRecord:
    """A function call the model made and the output it was shown (text first, then images).
    `status`: `executed` (sent to the Hub), `invalid` (arguments not a JSON object) or `refused`
    (after the turn's admitted action); `error` is the error code of an error output."""

    item: dict[str, Any]
    arguments: dict[str, Any] | None
    content: list[ContentPart]
    error: str | None = None
    status: Literal["executed", "invalid", "refused"] = "executed"

    @property
    def name(self) -> str:
        return str(self.item.get("name", ""))

    @property
    def call_id(self) -> str:
        return str(self.item.get("call_id", ""))


@dataclass
class Exchange:
    """One model call of a turn: its number in the episode, the model's output items, the
    outputs of its function calls and the nudge that answered a reply without one."""

    call: int
    output: list[dict[str, Any]]
    results: list[CallRecord] = field(default_factory=list)
    nudge: str | None = None


@dataclass
class Turn:
    """A turn: the observation it opened with (if any), its exchanges, and the turn-ending call
    the Hub admitted (`action`), which closed it."""

    index: int
    observation: list[ContentPart] | None = None
    exchanges: list[Exchange] = field(default_factory=list)
    action: CallRecord | None = None


def build_input(
    turns: Sequence[Turn],
    brief: Sequence[ContentPart] | None,
    policy: HistoryPolicy,
    delivery: ImageDelivery,
    detail: Detail,
    shrink: Shrink = NO_SHRINK,
) -> tuple[list[dict[str, Any]], dict[str, BlobRef]]:
    """The input items for the next model call in the last of `turns`, and the blobs its image
    URLs name."""
    shown = _Shown(detail)
    delivery = shrink.delivery or delivery
    items: list[dict[str, Any]] = []
    current = turns[-1].index
    if brief:
        limit = policy.brief_image_turns
        if limit is None or current <= limit:
            items.append(_user(shown.parts(brief, images=True)))
        else:
            note = _text(
                f"(The images above were shown in the first {limit} turns; get_image shows a "
                "frame again.)"
            )
            items.append(_user([*shown.parts(brief, images=False), note]))
    summarized, folded = _layout(turns, policy, shrink)
    old, kept = turns[:summarized], turns[summarized:]
    if old:
        items.append(_user([_text(summary(old))]))
    with_images = _with_images(kept, folded, policy, shrink)
    for turn in kept:
        age = current - turn.index
        if turn.observation is not None:
            items.append(_user(shown.parts(turn.observation, age <= policy.image_turns)))
        exchanges = turn.exchanges
        if age == 0 and folded:
            items.append(_user([_text(folded_summary(exchanges[:folded]))]))
            exchanges = exchanges[folded:]
        for exchange in exchanges:
            items += filter(None, (_replay(item, age == 0) for item in exchange.output))
            later: list[dict[str, Any]] = []
            for record in exchange.results:
                first = sum(part["type"] == "input_image" for part in later) + 1
                output, after = shown.output(record, id(record) in with_images, delivery, first)
                items.append(
                    {"type": "function_call_output", "call_id": record.call_id, "output": output}
                )
                later += after
            if later:
                items.append(_user([_text("Images the calls above returned:"), *later]))
            if exchange.nudge is not None:
                items.append(_user([_text(exchange.nudge)]))
    return items, shown.refs


def tool_images(turns: Sequence[Turn], policy: HistoryPolicy, shrink: Shrink) -> int:
    """How many tool images an input built from `turns` shows."""
    summarized, folded = _layout(turns, policy, shrink)
    chosen = _with_images(turns[summarized:], folded, policy, shrink)
    return sum(
        isinstance(part, ImagePart)
        for turn in turns[summarized:]
        for exchange in turn.exchanges
        for record in exchange.results
        if id(record) in chosen
        for part in record.content
    )


def _layout(turns: Sequence[Turn], policy: HistoryPolicy, shrink: Shrink) -> tuple[int, int]:
    """How many previous turns are summarized and how many of the current turn's exchanges
    folded."""
    previous = turns[-1].index - 1
    policed = summarized_turns(previous, policy.text_turns)
    folded = min(shrink.fold_exchanges, max(0, len(turns[-1].exchanges) - 1))
    return min(previous, max(policed, shrink.fold_turns)), folded


def _with_images(
    turns: Sequence[Turn], folded: int, policy: HistoryPolicy, shrink: Shrink
) -> set[int]:
    """The ids of the call records whose images are shown: newest first, all of a call's images
    or none, and none older than the first call that does not fit."""
    current = turns[-1].index
    cap = policy.tool_images
    limit = cap if shrink.tool_images is None else min(cap, shrink.tool_images)
    chosen: set[int] = set()
    count = 0
    for turn in reversed(turns):
        age = current - turn.index
        exchanges = turn.exchanges[folded:] if age == 0 else turn.exchanges
        for back, exchange in enumerate(reversed(exchanges)):
            past = back >= policy.image_exchanges if age == 0 else age > policy.image_turns
            if past:
                return chosen
            newest = age == 0 and back == 0
            for record in reversed(exchange.results):
                images = sum(isinstance(part, ImagePart) for part in record.content)
                if not images:
                    continue
                if count + images > (cap if newest else limit):
                    return chosen
                chosen.add(id(record))
                count += images
    return chosen


def summary(turns: Sequence[Turn]) -> str:
    """One line per turn: its calls and the start of their outputs."""
    lines = ["Earlier turns in brief (their observations and images are left out):"]
    for turn in turns:
        calls = [_line(record) for exchange in turn.exchanges for record in exchange.results]
        lines.append(f"turn {turn.index}: " + ("; ".join(calls) if calls else "no calls"))
    return "\n".join(lines)


def folded_summary(exchanges: Sequence[Exchange]) -> str:
    """One line per folded call of the current turn."""
    lines = ["Earlier calls of this turn in brief (folded to save context; images left out):"]
    for exchange in exchanges:
        text = "".join(
            part.get("text", "")
            for item in exchange.output
            if item.get("type") == "message"
            for part in item.get("content", [])
        )
        calls = [_line(record) for record in exchange.results]
        if text:
            calls.insert(0, "said: " + _cut(text, _SUMMARY_OUTPUT))
        lines.append(f"call {exchange.call}: " + ("; ".join(calls) if calls else "no calls"))
    return "\n".join(lines)


def _line(record: CallRecord) -> str:
    return f"{record.name}({_brief_arguments(record)}) -> " + _cut(
        text_of(record.content), _SUMMARY_OUTPUT
    )


def text_of(content: Sequence[ContentPart]) -> str:
    return "\n".join(part.text for part in content if isinstance(part, TextPart))


def summarized_turns(previous: int, block: int) -> int:
    """How many of the `previous` turns to summarize: whole blocks, keeping at least one block
    of recent turns in full."""
    if block == 0:
        return previous
    return max(0, (previous // block - 1) * block)


class _Shown:
    """Content parts as input items; collects the blobs the image URLs name."""

    def __init__(self, detail: Detail) -> None:
        self.detail = detail
        self.refs: dict[str, BlobRef] = {}

    def parts(self, content: Sequence[ContentPart], images: bool) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, TextPart):
                out.append(_text(part.text))
            elif images:
                out += self.image(part)
            else:
                out.append(_text(f"[{part.label or 'image'}: left out]"))
        return out

    def image(self, part: ImagePart, number: int | None = None) -> list[dict[str, Any]]:
        """The image, after its label (and its number among the images of a user message)."""
        self.refs[part.blob.id] = part.blob
        url = {"type": "input_image", "image_url": BLOB_URL + part.blob.id, "detail": self.detail}
        names = [f"image {number}"] if number is not None else []
        names += [part.label] if part.label else []
        return [_text(f"[{': '.join(names)}]"), url] if names else [url]

    def output(
        self, record: CallRecord, images: bool, delivery: ImageDelivery, first: int = 1
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]]]:
        """A call's `function_call_output.output` and the image parts shown after the outputs,
        numbered from `first` there."""
        text = text_of(record.content)
        pictures = [part for part in record.content if isinstance(part, ImagePart)]
        if not pictures:
            return text, []
        if not images:
            labels = ", ".join(dict.fromkeys(part.label or "image" for part in pictures))
            stub = (
                f"[{len(pictures)} images left out to save context ({labels}); "
                f"{record.name}({_brief_arguments(record)}) shows them again]"
            )
            return f"{text}\n{stub}", []
        if delivery == "output":
            return [_text(text), *(item for part in pictures for item in self.image(part))], []
        last = first + len(pictures) - 1
        numbers = f"image {first}" if last == first else f"images {first}-{last}"
        shown = [item for k, part in enumerate(pictures) for item in self.image(part, first + k)]
        return f"{text}\n[{numbers} attached below]", shown


def _replay(item: dict[str, Any], current: bool) -> dict[str, Any] | None:
    """An output item as input: as it came in the current turn; in earlier turns calls and text
    without their ids, and nothing else."""
    if current:
        return item
    kind = item.get("type")
    if kind == "function_call":
        return {key: item[key] for key in ("type", "call_id", "name", "arguments") if key in item}
    if kind == "message":
        text = "".join(
            part.get("text", "")
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        return {"role": "assistant", "content": text} if text else None
    return None


def _brief_arguments(record: CallRecord) -> str:
    if record.arguments is None:
        return _cut(str(record.item.get("arguments", "")), _SUMMARY_ARGS)
    shown = {key: value for key, value in record.arguments.items() if key != "rationale"}
    return _cut(json.dumps(shown, ensure_ascii=False, separators=(",", ":")), _SUMMARY_ARGS)


def _cut(text: str, limit: int) -> str:
    text = " / ".join(line.strip() for line in text.splitlines() if line.strip())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _text(text: str) -> dict[str, Any]:
    return {"type": "input_text", "text": text}


def _user(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "user", "content": content}
