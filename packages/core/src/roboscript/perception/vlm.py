"""VLM grounding through the model gateway: Set-of-Mark and boxes
(docs/concepts/perception.md).

Both backends take an OpenAI client of the gateway, injected by the caller (layers below the
agent never import the gateway), and ask one model (the alias `grounder`, by
default) through the Responses API: the images as JPEG, the answer bound to a strict JSON
schema and validated (finite numbers, the marks shown). Each call comes back as an `Asked` (the
prompt, the images as blobs when a store is given, the raw answer, latency, token usage, the
deployment that answered) for the caller's log.

- `SomGrounder`: Set-of-Mark. The plain image, the same image with the frame's proposals outlined
  and numbered, and one attribute card line per mark (color, footprint, height, position, from
  pixels and depth); one call answers every role. For the one-time identification the roles are
  abstract (what each does in the task, what kind of object it is: the Memory's OBJECTS) and the
  prompt carries the episode's live instruction, whose words decide which object plays each
  role: the model gives each role's mark (or null) and the instruction's phrase for it (the
  cross-check's query). Without an instruction each role comes with a description to find it by.
- `BoxGrounder`: a box (and a point) in pixels, or not found: for one description,
  or for several roles in one call (`locate_roles`, with the instruction as above). Boxes are
  lifted with depth; a point only ever selects a proposal (a lifted point lands away from
  the top center, and on thin parts it falls through to the table).
- `instruction_roles`: the objects an instruction is about, as roles with descriptions to find
  them by (text only), for identifying objects before any Memory names them.
- `role_words`: for abstract roles and the live instruction (text only), whether it names the
  role's object at all, each role's words, its
  noun, and the color, size or place words that single its object out, which geometry then
  checks (`roboscript.perception.reorder`).
- `compare`: look-alike candidates cut out and enlarged side by side (`overlay.crop_sheet`),
  with the whole view; which one is each role's object.
- `part_mark`: an object's part candidates (`roboscript.perception.parts`) outlined and
  numbered; which one a description of a part names.
- `choose_axis`: an object's candidate x arrows drawn labeled (A, B, ...: where its shape leaves
  its frame's x open, `perception.direction`), on the whole image and on a close-up of the
  object; which one points toward the role's `x toward:` words (the Memory's OBJECTS), with
  what common sense says is the front of its kind of object (a category prior's words, when the
  caller has them) as context. The perception toolset asks twice with the labels in opposite
  orders and turns the frame only when both answers name the same arrow.
- `goal_match`: the current views beside a demonstration's final-state views, camera by camera;
  whether the task's end state looks reached, and what differs (`check_done` with goal frames).

A Set-of-Mark prompt says that a mark's number sits on its object or is joined to it by a line
(`overlay.place_labels`), and, for several roles, that different roles are different objects.

Every call carries a short system instruction (`INSTRUCTIONS`): a deployment that gets none may
put a long one of its own in its place. A call the gateway could not serve (a 5xx such as
`queue_timeout` or `no_deployment_available`, a 408, 409 or 429, or no connection) is asked again
after each of `VlmConfig.retry_waits_s`, within the calling tool's own timeout; a model that still
does not answer raises `InfraError`, one that refuses the request (any other 4xx, never
asked again) `ToolError`; an answer that
breaks its schema counts as no answer for the roles it breaks (`Asked.error` says why). Usage the
API leaves out counts as zero.
"""

import base64
import io
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal, get_args

import anyio
import numpy as np
import openai
from numpy.typing import NDArray
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    NonNegativeFloat,
    StrictBool,
    StrictInt,
    ValidationError,
)

from roboscript.blobs import BlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import InfraError, ToolError
from roboscript.perception.config import DEFAULTS, PerceptionConfig
from roboscript.perception.grounding import (
    Candidate,
    Grounding,
    Mode,
    ProposalSource,
    Resources,
    mask_box,
    proposal_source,
    snap,
)
from roboscript.perception.overlay import attribute_card, box_overlay, som_overlay
from roboscript.perception.proposals import Proposal
from roboscript.perception.views import FrameGeometry

JPEG: Final = "image/jpeg"
INSTRUCTIONS: Final = (
    "You identify objects in robot camera images and answer in the requested JSON."
)
_NUMBERS: Final = {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}]}
BOX_SCHEMA: Final = {
    "type": "object",
    "properties": {"found": {"type": "boolean"}, "box": _NUMBERS, "point": _NUMBERS},
    "required": ["found", "box", "point"],
    "additionalProperties": False,
}
_ROLE: Final = {
    "type": "object",
    "properties": {"role": {"type": "string"}, "description": {"type": "string"}},
    "required": ["role", "description"],
    "additionalProperties": False,
}
ROLES_SCHEMA: Final = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": _ROLE}},
    "required": ["objects"],
    "additionalProperties": False,
}
ROLE_CHARS: Final = re.compile(r"[^a-z0-9_]+")
Size = Literal["largest", "middle", "smallest"]
Place = Literal["leftmost", "rightmost", "nearest", "farthest", "middle"]
_MARKS: Final = (
    "candidate objects outlined and numbered (the marks); a mark's number sits on its object or "
    "is joined to it by a line."
)
_ROLES_NOTE: Final = (
    "The task's objects, by role: what each does in the task and what kind of object it is. "
    "Which object on the table plays a role is decided by the instruction's words for it (such "
    "as a color, a size, a shape or where it is). A role's description is only a hint at its "
    "shape and use, written for another episode of the task: where it names a kind of object "
    "the instruction does not, the instruction's object plays the role, never one named only by "
    "the description."
)


class VlmConfig(BaseModel):
    """How the VLM backends ask: the gateway `model` (or alias), its reasoning effort, the JPEG
    quality of the images and the detail the model looks at them with.

    `retry_waits_s`: the waits before asking again after a failure the gateway could not serve
    (85 s in all by default; `identify`'s timeout leaves room for them on each of its calls in a
    row). The calling tool's timeout bounds the whole: a gateway `queue_timeout` alone takes its
    `queue_timeout_s` (300 s)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = "grounder"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    detail: Literal["low", "high", "auto"] = "high"
    retry_waits_s: list[NonNegativeFloat] = Field(default=[5.0, 20.0, 60.0], max_length=10)


class Asked(BaseModel):
    """One model call: what was asked (the prompt, the images as blobs if they were stored), the
    raw answer and the parsed one (None when it is not JSON; `error` says why), seconds (the
    retries' waits included), token usage, the deployment the gateway picked and the failures
    asked again (`retried`)."""

    model_config = ConfigDict(frozen=True)

    model: str
    prompt: str
    images: list[BlobRef]
    answer: str
    parsed: Any
    error: str | None = None
    latency_s: float
    usage: dict[str, int]
    deployment: str | None = None
    retried: list[str] = []


@dataclass(frozen=True, eq=False)
class Marks:
    """A Set-of-Mark answer: the mark chosen per role (None: no mark shows it), the words of the
    instruction naming each role's object (its description without an instruction; None: the
    answer gave none), the marks shown, the overlay (H, W, 3) and the call (None when no mark was
    shown, so nothing was asked)."""

    chosen: dict[str, int | None]
    phrases: dict[str, str | None]
    shown: list[int]
    overlay: NDArray[np.uint8]
    asked: Asked | None


@dataclass(frozen=True)
class BoxAnswer:
    """A box answer in continuous pixel coordinates (`roboscript.contracts.vision`), None when
    the model found nothing or broke the format, and the point it gave with the box."""

    box: tuple[float, float, float, float] | None
    point: tuple[float, float] | None
    asked: Asked


@dataclass(frozen=True)
class Boxes:
    """One call's boxes, per role (None for a role the model did not find or answered out of
    format), and the call."""

    boxes: dict[str, tuple[float, float, float, float] | None]
    points: dict[str, tuple[float, float] | None]
    asked: Asked


class _MarkJson(BaseModel):
    phrase: str = Field(max_length=200)
    mark: StrictInt | None


class _BoxJson(BaseModel):
    found: StrictBool
    box: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None
    point: tuple[FiniteFloat, FiniteFloat] | None


async def ask_json(
    client: openai.AsyncOpenAI,
    config: VlmConfig,
    prompt: str,
    images: Sequence[NDArray[np.uint8]],
    name: str,
    schema: dict[str, Any],
    blobs: BlobStore | None = None,
) -> Asked:
    """Ask for a JSON answer bound to a strict `schema`: one user message with the prompt, then
    the images (JPEG), and nothing kept upstream (`store=False`)."""
    encoded = [await anyio.to_thread.run_sync(jpeg, image, config.jpeg_quality) for image in images]
    refs = [] if blobs is None else [await blobs.aput_bytes(data, JPEG) for data in encoded]
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for data in encoded:
        url = f"data:{JPEG};base64,{base64.b64encode(data).decode()}"
        content.append({"type": "input_image", "image_url": url, "detail": config.detail})
    text_format = {"type": "json_schema", "name": name, "schema": schema, "strict": True}
    request: dict[str, Any] = {
        "model": config.model,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": config.reasoning_effort},
        "text": {"format": text_format},
        "store": False,
    }
    started = time.monotonic()
    retried: list[str] = []
    waits = iter(config.retry_waits_s)
    while True:
        try:
            raw = await client.responses.with_raw_response.create(**request)
            break
        except openai.APIStatusError as error:
            if 400 <= error.status_code < 500 and error.status_code not in (408, 409, 429):
                message = f"the grounding model refused the request: {error.message}"
                raise ToolError(message) from None
            failure = f"{error.status_code} {_code(error)}"
        except openai.APIError as error:
            failure = type(error).__name__
        wait = next(waits, None)
        if wait is None:
            tried = "; ".join([*retried, failure])
            raise InfraError("not_ready", f"the grounding model did not answer ({tried})") from None
        retried.append(failure)
        await anyio.sleep(wait)
    response = raw.parse()
    answer = response.output_text
    parsed, problem = None, None
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError as error:
        problem = f"not JSON: {error}"
    return Asked(
        model=config.model,
        prompt=prompt,
        images=refs,
        answer=answer,
        parsed=parsed,
        error=problem,
        latency_s=round(time.monotonic() - started, 3),
        usage=usage_of(response.usage),
        deployment=raw.headers.get("x-rs-deployment"),
        retried=retried,
    )


def _code(error: openai.APIStatusError) -> str:
    """The error's code (the gateway's `queue_timeout`, `no_deployment_available`, ...) or
    its type."""
    body = error.body if isinstance(error.body, dict) else {}
    found = body.get("code") or body.get("type")
    return str(found) if found else "error"


def jpeg(rgb: NDArray[np.uint8], quality: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3])).save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def camera_words(
    camera: str, pipeline: PerceptionConfig = DEFAULTS, anchor: str | None = None
) -> str:
    """How a prompt names the camera that took an image: the anchor one (`anchor`, else the
    pipeline's) in the pipeline's `anchor_words`, another by its `camera_words`, else by its
    name."""
    if camera == (pipeline.anchor if anchor is None else anchor):
        return pipeline.anchor_words
    return pipeline.camera_words.get(camera, camera)


def som_prompt(
    card: str,
    queries: Mapping[str, str],
    camera: str = DEFAULTS.anchor_words,
    scene: str = DEFAULTS.scene_words,
) -> str:
    """The Set-of-Mark prompt; `queries` maps answer keys (r1, r2, ...) to descriptions."""
    listing = "\n".join(f'- "{key}": {text}' for key, text in queries.items())
    answer = ", ".join(f'"{key}": <mark number or null>' for key in queries)
    return (
        f"Two images of the same {scene} from {camera}: the plain image, then the "
        f"same image with {_MARKS} Measured attributes "
        "of each mark (color is the median pixel color; sizes and positions are from depth):\n"
        f"{card}\n\nFor each object description below, give the number of the mark that shows "
        f"that object, or null if no mark shows it.\n{listing}\n\nAnswer with JSON: {{{answer}}}"
    )


def som_roles_prompt(
    card: str,
    roles: Mapping[str, tuple[str, str]],
    instruction: str,
    camera: str = DEFAULTS.anchor_words,
    scene: str = DEFAULTS.scene_words,
) -> str:
    """The Set-of-Mark prompt for abstract roles and the live instruction; `roles` maps answer
    keys to (role, what it does and what kind of object it is)."""
    listing = "\n".join(f'- "{key}" {role}: {text}' for key, (role, text) in roles.items())
    answer = ", ".join(f'"{key}": {{"phrase": <text>, "mark": <number or null>}}' for key in roles)
    distinct = " Different roles are different objects." if len(roles) > 1 else ""
    return (
        f"Two images of the same {scene} from {camera}: the plain image, then the "
        f"same image with {_MARKS} Measured attributes "
        "of each mark (color is the median pixel color; sizes and positions are from depth):\n"
        f'{card}\n\nThe robot\'s instruction: "{instruction}"\n{_ROLES_NOTE}\n{listing}\n\n'
        'For each role, give "phrase", the words of the instruction that name its object, '
        "copied from the instruction (the role's name if the instruction does not name it; "
        'never words of the description), and "mark", the number of the mark '
        f"that shows that object, or null if no mark shows it.{distinct}\n"
        f"Answer with JSON: {{{answer}}}"
    )


def som_schema(keys: Sequence[str], marks: Sequence[int], phrases: bool = False) -> dict[str, Any]:
    """Each key answered by one of the marks, or null; with `phrases`, as `{phrase, mark}`."""
    mark = {"anyOf": [{"type": "integer", "enum": list(marks)}, {"type": "null"}]}
    value: dict[str, Any] = mark
    if phrases:
        value = {
            "type": "object",
            "properties": {"phrase": {"type": "string"}, "mark": mark},
            "required": ["phrase", "mark"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": dict.fromkeys(keys, value),
        "required": list(keys),
        "additionalProperties": False,
    }


def box_prompt(
    query: str, width: int, height: int, camera: str, scene: str = DEFAULTS.scene_words
) -> str:
    """The grounding prompt; `camera` names what took the image (`camera_words`)."""
    return (
        f"The image is {width}x{height} pixels (width x height), a {scene} seen by {camera}. "
        f'Locate: "{query}". Answer with JSON: '
        '{"found": true|false, "box": [x_min, y_min, x_max, y_max], "point": [x, y]} '
        "in pixel coordinates of this image (origin top-left, x to the right, y down). "
        "The box must tightly enclose the whole object; the point must lie on the object itself. "
        "If the object is not visible, answer found=false."
    )


def boxes_prompt(
    roles: Mapping[str, tuple[str, str]],
    instruction: str | None,
    width: int,
    height: int,
    camera: str,
    scene: str = DEFAULTS.scene_words,
) -> str:
    """Several boxes in one call: `roles` maps answer keys to (role, text): with the live
    `instruction`, abstract roles whose object its words decide; without, descriptions."""
    if instruction is None:
        what = "Locate each object described below.\n"
        listing = "\n".join(f'- "{key}": {text}' for key, (_, text) in roles.items())
    else:
        what = f'The robot\'s instruction: "{instruction}"\n{_ROLES_NOTE}\n'
        listing = "\n".join(f'- "{key}" {role}: {text}' for key, (role, text) in roles.items())
    shape = '{"found": true|false, "box": [x_min, y_min, x_max, y_max], "point": [x, y]}'
    answer = ", ".join(f'"{key}": {shape}' for key in roles)
    return (
        f"The image is {width}x{height} pixels (width x height), a {scene} seen by {camera}. "
        f"{what}{listing}\n\nFor each, locate the object"
        + ("" if instruction is None else " the instruction means")
        + ": the box must tightly enclose the whole object and the point must lie on the object "
        "itself, in pixel coordinates of this image (origin top-left, x to the right, y down). "
        "If the object is not visible, answer found=false.\n"
        f"Answer with JSON: {{{answer}}}"
    )


def boxes_schema(keys: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict.fromkeys(keys, BOX_SCHEMA),
        "required": list(keys),
        "additionalProperties": False,
    }


def roles_prompt(instruction: str, robot: str = "a robot") -> str:
    """`robot`: the robot in the prompt's words (`EmbodimentProfile.description`)."""
    return (
        f'{robot[:1].upper()}{robot[1:]} is told: "{instruction}"\n'
        "List the objects on the table that this instruction is about (not the robot, its arms "
        "or grippers, and not the table). For each give a role, a short lowercase name of one or "
        "two words joined by _ (such as <noun> or <noun>_<qualifier>), and a description to find "
        "it by in an image: the instruction's own words for it, with the color, shape or "
        'material the instruction gives. Answer with JSON: {"objects": [{"role": ..., '
        '"description": ...}]}'
    )


async def instruction_roles(
    client: openai.AsyncOpenAI,
    instruction: str,
    config: VlmConfig | None = None,
    blobs: BlobStore | None = None,
    robot: str = "a robot",
) -> tuple[dict[str, str], Asked]:
    """The objects `instruction` is about, role -> description (roles lowercase words joined by
    `_`, at most 40 characters; descriptions at most 200), in the order named; and the call."""
    prompt = roles_prompt(instruction, robot)
    asked = await ask_json(client, config or VlmConfig(), prompt, [], "roles", ROLES_SCHEMA, blobs)
    listed = asked.parsed.get("objects") if isinstance(asked.parsed, dict) else None
    roles: dict[str, str] = {}
    for item in listed if isinstance(listed, list) else []:
        if not isinstance(item, dict):
            continue
        role = ROLE_CHARS.sub("_", str(item.get("role", "")).lower()).strip("_")[:40]
        description = " ".join(str(item.get("description", "")).split())[:200]
        if role and role[0].isalpha() and description and role not in roles:
            roles[role] = description
    return roles, asked


class GoalMatch(BaseModel):
    """`goal_match`'s answer: whether the views show the goal's end state, and what differs."""

    model_config = ConfigDict(frozen=True)

    reached: bool | None
    differs: str


def goal_prompt(
    cameras: Sequence[str], instruction: str, pipeline: PerceptionConfig = DEFAULTS
) -> str:
    """The prompt comparing current views with goal views, one pair per camera in `cameras`."""
    pairs = "; ".join(f"pair {i}: {camera_words(c, pipeline)}" for i, c in enumerate(cameras, 1))
    return (
        f"Image pairs from a robot's cameras ({pairs}): in each pair the first image is the "
        "scene now, the second the end state of a demonstration of the same task on another "
        f'scene. The task: "{instruction}". Objects may sit elsewhere on the table and look '
        "different in detail; compare only how the task's objects sit relative to each other "
        "and to the robot's grippers. Does the scene now show the task's end state? Answer with "
        'JSON: {"reached": <true, false, or null if the views cannot tell>, "differs": '
        '"<what still differs from the end state, empty if nothing>"}'
    )


async def goal_match(
    client: openai.AsyncOpenAI,
    now: Mapping[str, NDArray[np.uint8]],
    goal: Mapping[str, NDArray[np.uint8]],
    instruction: str,
    config: VlmConfig | None = None,
    blobs: BlobStore | None = None,
    pipeline: PerceptionConfig = DEFAULTS,
) -> tuple[GoalMatch, Asked]:
    """Whether the views `now` show the end state the `goal` views (same cameras) show; an answer
    out of format counts as undecided."""
    cameras = [c for c in goal if c in now]
    if not cameras:
        raise ToolError("no camera of the goal views is in the current observation")
    schema = {
        "type": "object",
        "properties": {
            "reached": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "differs": {"type": "string"},
        },
        "required": ["reached", "differs"],
        "additionalProperties": False,
    }
    images = [image for c in cameras for image in (now[c], goal[c])]
    prompt = goal_prompt(cameras, instruction, pipeline)
    asked = await ask_json(client, config or VlmConfig(), prompt, images, "goal", schema, blobs)
    try:
        return GoalMatch.model_validate(asked.parsed), asked
    except ValidationError:
        return GoalMatch(reached=None, differs="the answer was out of format"), asked


class RoleWords(BaseModel):
    """What the instruction says of a role's object: its words, its kind in one word (`noun`),
    and the words that single it out among objects of that kind: a color, its size, or where it
    is now as the robot sees it (None: the words say none); `named` false where the instruction
    asks for no object in that role (it names fewer objects of a kind than the roles list);
    `arm` the robot arm the instruction has act on that object (None: it says none for it)."""

    model_config = ConfigDict(frozen=True)

    phrase: str = Field(max_length=200)
    noun: str = Field(max_length=40)
    color: str | None = Field(default=None, max_length=40)
    size: Size | None = None
    place: Place | None = None
    named: bool = True
    arm: Literal["left", "right"] | None = None


def words_prompt(roles: Mapping[str, tuple[str, str]], instruction: str) -> str:
    """`roles` maps answer keys to (role, what it does and what kind of object it is)."""
    listing = "\n".join(f'- "{key}" {role}: {text}' for key, (role, text) in roles.items())
    answer = ", ".join(f'"{key}": {{...}}' for key in roles)
    return (
        f'The robot\'s instruction: "{instruction}"\n{_ROLES_NOTE}\n{listing}\n\n'
        'For each role give: "phrase", the instruction\'s words for its object, copied from the '
        "instruction (the role's name if the instruction does not name it; never words of the "
        'description); "noun", the kind of object the instruction names in one lowercase word; '
        "and only from words that describe which object it is (never from where it goes): "
        '"color", a color word the instruction gives it, else null; "size", "largest", "middle" '
        'or "smallest" when the words single it out by size among objects of its kind, else '
        'null; "place", "leftmost", "rightmost", "nearest", "farthest" or "middle" when the '
        'words single it out by where it is now, as the robot sees it, else null; "named", '
        "false only when the instruction asks for no object in this role (it names fewer "
        "objects of this kind than there are roles of it: the roles it leaves out), else true; "
        "words that refer to several objects at once (a number such as two, both, all, each, "
        "the rest, the others, a plural) name every role of the kind they cover; "
        '"arm", "left" or "right" when the instruction says which robot arm (hand, gripper) acts '
        "on this role's object (read lists in parallel: the k-th arm it lists goes with the k-th "
        "object it lists), else null (left of or right of an object is where something goes, "
        "not an arm).\n"
        f"Answer with JSON: {{{answer}}}"
    )


def words_schema(keys: Sequence[str]) -> dict[str, Any]:
    text = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    role = {
        "type": "object",
        "properties": {
            "phrase": {"type": "string"},
            "noun": {"type": "string"},
            "color": text,
            "size": {"anyOf": [{"type": "string", "enum": list(get_args(Size))}, {"type": "null"}]},
            "place": {
                "anyOf": [{"type": "string", "enum": list(get_args(Place))}, {"type": "null"}]
            },
            "named": {"type": "boolean"},
            "arm": {"anyOf": [{"type": "string", "enum": ["left", "right"]}, {"type": "null"}]},
        },
        "required": ["phrase", "noun", "color", "size", "place", "named", "arm"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": dict.fromkeys(keys, role),
        "required": list(keys),
        "additionalProperties": False,
    }


async def role_words(
    client: openai.AsyncOpenAI,
    roles: Mapping[str, str],
    instruction: str,
    config: VlmConfig | None = None,
    blobs: BlobStore | None = None,
) -> tuple[dict[str, RoleWords], Asked]:
    """What `instruction` says of each abstract role's object (role -> what it does and is), in
    one text-only call; a role the answer breaks is left out."""
    keys = {f"r{i}": role for i, role in enumerate(roles, 1)}
    listed = {key: (role, roles[role]) for key, role in keys.items()}
    prompt = words_prompt(listed, instruction)
    schema = words_schema(list(keys))
    asked = await ask_json(client, config or VlmConfig(), prompt, [], "role_words", schema, blobs)
    answer = asked.parsed if isinstance(asked.parsed, dict) else {}
    found: dict[str, RoleWords] = {}
    for key, role in keys.items():
        try:
            found[role] = RoleWords.model_validate(answer.get(key))
        except ValidationError:
            continue
    if len(found) < len(keys) and asked.error is None:
        broken = ", ".join(k for k, r in keys.items() if r not in found)
        asked = asked.model_copy(update={"error": f"out of format for {broken}"})
    return found, asked


def compare_prompt(
    roles: Mapping[str, tuple[str, str, str, list[str]]],
    instruction: str,
    wrists: bool,
    anchor_words: str = DEFAULTS.anchor_words,
) -> str:
    """`roles` maps answer keys to (role, what it does and is, the instruction's words for it,
    the letters of its candidates)."""
    listing = "\n".join(
        f'- "{key}" {role}: {text}; the instruction\'s words for it: "{words}"; candidates '
        + ", ".join(letters)
        for key, (role, text, words, letters) in roles.items()
    )
    answer = ", ".join(f'"{key}": <letter or null>' for key in roles)
    closer = "; below a candidate, the same object seen closer by a wrist camera" if wrists else ""
    return (
        f"The first image shows candidate objects from {anchor_words}, cut out and "
        f"enlarged, each under its letter{closer}; the second image is the whole view with the "
        "same candidates outlined and numbered in letter order (1 = A, 2 = B, ...).\n"
        f'The robot\'s instruction: "{instruction}"\n{_ROLES_NOTE}\n{listing}\n\n'
        "For each role, give the letter of the candidate that is the object the instruction "
        "means for it, or null if none of its candidates is. Different roles are different "
        f"objects.\nAnswer with JSON: {{{answer}}}"
    )


async def compare(
    client: openai.AsyncOpenAI,
    sheet: NDArray[np.uint8],
    overview: NDArray[np.uint8],
    roles: Mapping[str, tuple[str, str, list[str]]],
    instruction: str,
    wrists: bool,
    config: VlmConfig | None = None,
    blobs: BlobStore | None = None,
    anchor_words: str = DEFAULTS.anchor_words,
) -> tuple[dict[str, str | None], Asked]:
    """Which candidate each role's object is: `roles` maps a role to (what it does and is, the
    instruction's words for it, the letters of its candidates on `sheet`, which `overview`
    shows numbered in letter order); `wrists`: the sheet has closer wrist views. A role the
    answer leaves out or breaks gets None."""
    keys = {f"r{i}": role for i, role in enumerate(roles, 1)}
    listed = {key: (role, *roles[role]) for key, role in keys.items()}
    prompt = compare_prompt(listed, instruction, wrists, anchor_words)
    letter: dict[str, Any] = {
        key: {"anyOf": [{"type": "string", "enum": roles[role][2]}, {"type": "null"}]}
        for key, role in keys.items()
    }
    schema = {
        "type": "object",
        "properties": letter,
        "required": list(keys),
        "additionalProperties": False,
    }
    asked = await ask_json(
        client, config or VlmConfig(), prompt, [sheet, overview], "compare", schema, blobs
    )
    answer = asked.parsed if isinstance(asked.parsed, dict) else {}
    picked: dict[str, str | None] = {}
    for key, role in keys.items():
        value = answer.get(key)
        picked[role] = value if isinstance(value, str) and value in roles[role][2] else None
    return picked, asked


def part_prompt(query: str, camera: str) -> str:
    """The part prompt; `camera` names what took the image (`camera_words`)."""
    return (
        f"Two images from {camera}: the plain image, then the same image with regions of one "
        "object outlined and numbered (the marks); a mark's number sits on its region or is "
        f'joined to it by a line. Which mark shows "{query}"? Left and right are as the robot '
        'sees them. Answer with JSON: {"mark": <number, or null if no mark shows it>}'
    )


async def part_mark(
    client: openai.AsyncOpenAI,
    rgb: NDArray[np.uint8],
    overlay: NDArray[np.uint8],
    query: str,
    marks: Sequence[int],
    camera: str,
    config: VlmConfig | None = None,
    blobs: BlobStore | None = None,
) -> tuple[int | None, Asked]:
    """Which of the `marks` on `overlay` (numbered regions of one object over `rgb`) shows the
    part `query` describes (None: none, or an answer out of format)."""
    mark_or_null = {"anyOf": [{"type": "integer", "enum": list(marks)}, {"type": "null"}]}
    schema = {
        "type": "object",
        "properties": {"mark": mark_or_null},
        "required": ["mark"],
        "additionalProperties": False,
    }
    prompt = part_prompt(query, camera)
    asked = await ask_json(
        client, config or VlmConfig(), prompt, [rgb, overlay], "part_mark", schema, blobs
    )
    value = asked.parsed.get("mark") if isinstance(asked.parsed, dict) else None
    if value is None or (type(value) is int and value in marks):
        return value, asked
    return None, asked.model_copy(update={"error": f"out of format: mark {value!r}"})


def axis_prompt(
    role: str, toward: str, labels: Sequence[str], camera: str, front: str | None = None
) -> str:
    """The prompt that settles which way an object's frame points: the object's candidate x
    arrows are drawn labeled, on the whole image and on a close-up of the object; which points
    toward `toward`? `front`: what common sense says is the front of its kind (a category
    prior's words), said as context."""
    listed = ", ".join(labels)
    known = "" if front is None else f' Common sense: this kind of object\'s front is "{front}".'
    return (
        f"Three images from {camera}: the plain image, the same image with arrows drawn from "
        f"one object, {role}: its candidate x axes in red, labeled {listed}, and its z (up) in "
        f"blue, then a close-up of that object with the same arrows.{known} Which labeled arrow "
        f'points toward "{toward}" of that object? Answer with JSON: '
        '{"arrow": <label, or null if no arrow does or the object cannot be seen>}'
    )


async def choose_axis(
    client: openai.AsyncOpenAI,
    rgb: NDArray[np.uint8],
    overlay: NDArray[np.uint8],
    close_up: NDArray[np.uint8],
    role: str,
    toward: str,
    labels: Sequence[str],
    camera: str,
    config: VlmConfig | None = None,
    blobs: BlobStore | None = None,
    front: str | None = None,
) -> tuple[str | None, Asked]:
    """Which of the labeled candidate x arrows on `overlay` (and on `close_up`, the object cut
    out of it) points toward `toward` (the role's `x toward:` words; `front`, its kind's front
    as common sense has it, is said as context); None: none, or an answer out of format."""
    choice = {"anyOf": [{"type": "string", "enum": list(labels)}, {"type": "null"}]}
    schema = {
        "type": "object",
        "properties": {"arrow": choice},
        "required": ["arrow"],
        "additionalProperties": False,
    }
    prompt = axis_prompt(role, toward, labels, camera, front)
    asked = await ask_json(
        client, config or VlmConfig(), prompt, [rgb, overlay, close_up], "axis", schema, blobs
    )
    value = asked.parsed.get("arrow") if isinstance(asked.parsed, dict) else None
    if value is None or value in labels:
        return value, asked
    return None, asked.model_copy(update={"error": f"out of format: arrow {value!r}"})


class SomGrounder:
    """Set-of-Mark over the frame's proposals seen in one camera (the anchor by default)."""

    name: ClassVar[str] = "som"

    def __init__(
        self,
        client: openai.AsyncOpenAI,
        config: VlmConfig | None = None,
        blobs: BlobStore | None = None,
        proposals: ProposalSource | None = None,
        pipeline: PerceptionConfig = DEFAULTS,
    ) -> None:
        self.client = client
        self.config = VlmConfig() if config is None else config
        self.blobs = blobs
        self.pipeline = pipeline
        self.proposals = proposal_source(pipeline) if proposals is None else proposals

    @classmethod
    def build(cls, resources: Resources) -> "SomGrounder":
        return cls(
            resources.client, resources.vlm, resources.blobs, resources.source(), resources.config
        )

    async def choose(
        self,
        geometry: FrameGeometry,
        proposals: Sequence[Proposal],
        roles: Mapping[str, str],
        camera: str | None = None,
        instruction: str | None = None,
    ) -> Marks:
        """The mark of each role among the proposals `camera` sees: roles map to descriptions,
        or, with the live `instruction`, to what each role does and is (abstract roles)."""
        camera = camera or geometry.anchor
        image = geometry.view(camera).image
        shown = [p for p in proposals if p.masks[camera].any()]
        overlay, _ = await anyio.to_thread.run_sync(
            som_overlay, image.rgb, {p.id: p.masks[camera] for p in shown}
        )
        marks = [p.id for p in shown]
        given: dict[str, str | None] = dict(roles) if instruction is None else dict.fromkeys(roles)
        if not marks:
            return Marks(dict.fromkeys(roles), given, [], overlay, None)
        card = await anyio.to_thread.run_sync(
            attribute_card, image.rgb, shown, geometry.table, camera, geometry.base
        )
        keys = {f"r{i}": role for i, role in enumerate(roles, 1)}
        words = camera_words(camera, self.pipeline, geometry.anchor)
        scene = self.pipeline.scene_words
        if instruction is None:
            queries = {key: roles[role] for key, role in keys.items()}
            prompt = som_prompt(card, queries, words, scene)
        else:
            listed = {key: (role, roles[role]) for key, role in keys.items()}
            prompt = som_roles_prompt(card, listed, instruction, words, scene)
        schema = som_schema(list(keys), marks, phrases=instruction is not None)
        asked = await ask_json(
            self.client,
            self.config,
            prompt,
            [image.rgb, overlay],
            "set_of_mark",
            schema,
            self.blobs,
        )
        answer = asked.parsed if isinstance(asked.parsed, dict) else {}
        chosen: dict[str, int | None] = {}
        broken = []
        for key, role in keys.items():
            if instruction is None:
                chosen[role] = _mark(answer.get(key), marks)
                continue
            try:
                item = _MarkJson.model_validate(answer.get(key))
            except ValidationError:
                broken.append(key)
                chosen[role] = None
                continue
            chosen[role] = item.mark if item.mark in marks else None
            given[role] = " ".join(item.phrase.split()) or None
        if broken and asked.error is None:
            asked = asked.model_copy(update={"error": f"out of format for {', '.join(broken)}"})
        return Marks(chosen, given, marks, overlay, asked)

    async def ground(
        self, geometry: FrameGeometry, camera: str, query: str, mode: Mode = "box"
    ) -> Grounding:
        """The mark the model picks for `query` as a region candidate, in either mode (none if
        it picks none)."""
        started = time.monotonic()
        proposals = await self.proposals(geometry)
        marks = await self.choose(geometry, proposals, {"query": query}, camera)
        candidates = [
            Candidate(region=p.id, box=mask_box(p.masks[camera]), label=f"mark {p.id}")
            for p in proposals
            if p.id == marks.chosen["query"]
        ]
        elapsed = time.monotonic() - started
        return Grounding(
            geometry.frame,
            camera,
            query,
            candidates,
            marks.overlay,
            self.name,
            elapsed,
            marks.asked,
        )


class BoxGrounder:
    """Boxes (and points) from the VLM: for one description, or for several roles at once."""

    name: ClassVar[str] = "vlm"

    def __init__(
        self,
        client: openai.AsyncOpenAI,
        config: VlmConfig | None = None,
        blobs: BlobStore | None = None,
        proposals: ProposalSource | None = None,
        pipeline: PerceptionConfig = DEFAULTS,
    ) -> None:
        self.client = client
        self.config = VlmConfig() if config is None else config
        self.blobs = blobs
        self.pipeline = pipeline
        self.proposals = proposal_source(pipeline) if proposals is None else proposals

    @classmethod
    def build(cls, resources: Resources) -> "BoxGrounder":
        return cls(
            resources.client, resources.vlm, resources.blobs, resources.source(), resources.config
        )

    async def locate(
        self,
        rgb: NDArray[np.uint8],
        query: str,
        camera: str | None = None,
        anchor: str | None = None,
    ) -> BoxAnswer:
        """Where `query` is in the image `rgb` that `camera` took (`anchor`: the frame's anchor
        camera, which the prompt names in the pipeline's `anchor_words`; None for either: the
        pipeline's `anchor`)."""
        anchor = self.pipeline.anchor_camera if anchor is None else anchor
        camera = anchor if camera is None else camera
        height, width = rgb.shape[:2]
        words = camera_words(camera, self.pipeline, anchor)
        prompt = box_prompt(query, width, height, words, self.pipeline.scene_words)
        asked = await ask_json(
            self.client, self.config, prompt, [rgb], "box", BOX_SCHEMA, self.blobs
        )
        box, point, problem = _box_answer(asked.parsed, width, height)
        if problem is not None and asked.error is None:
            asked = asked.model_copy(update={"error": problem})
        return BoxAnswer(box, point, asked)

    async def locate_roles(
        self,
        rgb: NDArray[np.uint8],
        roles: Mapping[str, str],
        camera: str | None = None,
        anchor: str | None = None,
        instruction: str | None = None,
    ) -> Boxes:
        """Where each role's object is in the image `rgb` that `camera` took, in one call: roles
        map to descriptions, or, with the live `instruction`, to what each role does and is
        (`camera`, `anchor` as for `locate`)."""
        anchor = self.pipeline.anchor_camera if anchor is None else anchor
        camera = anchor if camera is None else camera
        height, width = rgb.shape[:2]
        keys = {f"r{i}": role for i, role in enumerate(roles, 1)}
        listed = {key: (role, roles[role]) for key, role in keys.items()}
        words = camera_words(camera, self.pipeline, anchor)
        prompt = boxes_prompt(listed, instruction, width, height, words, self.pipeline.scene_words)
        asked = await ask_json(
            self.client, self.config, prompt, [rgb], "boxes", boxes_schema(list(keys)), self.blobs
        )
        answer = asked.parsed if isinstance(asked.parsed, dict) else {}
        boxes: dict[str, tuple[float, float, float, float] | None] = {}
        points: dict[str, tuple[float, float] | None] = {}
        broken = []
        for key, role in keys.items():
            boxes[role], points[role], problem = _box_answer(answer.get(key), width, height)
            if problem is not None:
                broken.append(key)
        if broken and asked.error is None:
            asked = asked.model_copy(update={"error": f"out of format for {', '.join(broken)}"})
        return Boxes(boxes, points, asked)

    async def ground(
        self, geometry: FrameGeometry, camera: str, query: str, mode: Mode = "box"
    ) -> Grounding:
        """The model's box (mode `box`) or point (mode `point`, with the proposal it falls on,
        if any)."""
        started = time.monotonic()
        image = geometry.view(camera).image
        answer = await self.locate(image.rgb, query, camera, geometry.anchor)
        candidates = []
        if mode == "box" and answer.box is not None:
            candidates.append(Candidate(box=answer.box, label=query))
        elif mode == "point" and answer.point is not None:
            proposals = await self.proposals(geometry)
            under = snap(proposals, camera, point=answer.point)
            region = None if under is None else under.id
            candidates.append(Candidate(point=answer.point, region=region, label=query))
        boxes = {i: c.box for i, c in enumerate(candidates, 1) if c.box is not None}
        points = {i: c.point for i, c in enumerate(candidates, 1) if c.point is not None}
        overlay = await anyio.to_thread.run_sync(box_overlay, image.rgb, boxes, points)
        elapsed = time.monotonic() - started
        return Grounding(
            geometry.frame, camera, query, candidates, overlay, self.name, elapsed, answer.asked
        )


def usage_of(usage: Any) -> dict[str, int]:
    """Token usage as the API reports it; what it leaves out counts as zero."""
    inputs = getattr(usage, "input_tokens_details", None)
    outputs = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None) or 0,
        "cached_tokens": getattr(inputs, "cached_tokens", None) or 0,
        "output_tokens": getattr(usage, "output_tokens", None) or 0,
        "reasoning_tokens": getattr(outputs, "reasoning_tokens", None) or 0,
    }


def _mark(value: Any, marks: Sequence[int]) -> int | None:
    valid = isinstance(value, int) and not isinstance(value, bool) and value in marks
    return value if valid else None


def _box_answer(
    value: Any, width: int, height: int
) -> tuple[tuple[float, float, float, float] | None, tuple[float, float] | None, str | None]:
    """A box answer's box (clipped to the image; None unless found with area) and point, and
    what was wrong with it (None: in format)."""
    try:
        answer = _BoxJson.model_validate(value)
    except ValidationError as error:
        return None, None, f"out of format: {error.error_count()} problems"
    if not answer.found or answer.box is None:
        return None, None, None
    u0, v0, u1, v1 = answer.box
    u0, u1 = max(0.0, min(u0, u1)), min(float(width), max(u0, u1))
    v0, v1 = max(0.0, min(v0, v1)), min(float(height), max(v0, v1))
    if not (u0 < u1 and v0 < v1):
        return None, None, None
    return (u0, v0, u1, v1), answer.point, None
