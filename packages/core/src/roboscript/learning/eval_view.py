"""A read-only, tier-labelled view of evaluation records (docs/concepts/learning.md,
docs/architecture.md).

What a reflection (P4's `p4_reflect`) reads of an evaluation: per episode the instruction, tool
calls and receipts, keyframes, the Agent's scratchpad and finish, and the benchmark's success. Each
part's tier is fixed by its kind, so a record cannot relabel ground truth as D. The default view
holds D parts and the benchmark's success; a view with P (the truth-diagnosis variant, reported
separately as an upper bound) has to be asked for by name (`privileged`).

`read_eval_records` builds the episodes from the Agent host's records (`agent.host`:
`<records>/<key>/ep<NNN>/`): the instruction and the benchmark's success from `summary.json`, the
frames the episode took as keyframes (its `env.observe` events, each at its step and the turn
it was taken in; an episode without its own `events.jsonl` has them read from the host's log by
its episode id, `episode_events`, and says so in `EvalEpisode.events_from`), and per turn of
`turns.jsonl` its observation as text (images by label), the model's own text, and per model
call the tool call and its receipt, the Agent's `scratchpad` and
`finish` (a refused one followed by its receipt), each at the step of the latest frame of its
turn and with the turn's number; a `diagnosis.json` beside them is the P part (the post-hoc
attribution's, written after the evaluation). `EvalEpisode.outcome` holds how the episode ended
(`summary.json`: the end's reason, turns, model calls, steps, the benchmark's step count and
limit, `infra`, the voided end; the model-call limit from `config.json`), and `episode_memory`
the Task Memory its Actor read (`memory_before.md`), both tier D. Episodes infrastructure lost
(`agent.host.infra_lost`) are left out, never counted as the agent's failures, and so is an
episode directory without a `summary.json` (its host died mid-episode, or the summary could not
be written), which counts as lost to the infrastructure too. The host's records
do not hold the env seed of an episode (the benchmark's client picks it), so the caller names it
(`SeedOf`: the benchmark's mapping of a slot's episodes to seeds). Each episode keeps its records
directory (`EvalEpisode.directory`).

A keyframe part holds the frame's alias and its observation by reference; `frame_images` reads the
camera images behind an episode's keyframes from the run's blob store (`blobs/<id[:2]>/<id>`): per
frame (by alias or frame id, default all, in the order taken) and camera, the RGB image's
`BlobRef` (a PNG), after checking that the blob is there. These are the Actor's own observations,
so they share the keyframe part's tier (D). A blob the store lacks (records copied without it) is
listed in `FrameImages.missing`, never skipped silently, and so is an episode without its
event log (`events.jsonl`).
"""

import json
import re
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, NonNegativeInt, model_validator
from typing_extensions import Self

from roboscript.agent.host import MEMORY_USED, infra_lost
from roboscript.agent.recording import read_lines
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.events import Event, EventKind
from roboscript.contracts.ids import FrameId, TaskName
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.contracts.tools import Tier
from roboscript.records import read_events

PartKind = Literal[
    "instruction",
    "observation",
    "assistant",
    "rationale",
    "tool_call",
    "receipt",
    "keyframe",
    "scratchpad",
    "finish",
    "diagnosis",
]
PART_TIERS: Final[dict[PartKind, Tier]] = {
    "instruction": "D",
    "observation": "D",  # a turn's observation as the Actor read it (images by label)
    "assistant": "D",  # the model's own text beside its calls
    "rationale": "D",  # the rationale the model gave with its turn's action, if any
    "tool_call": "D",
    "receipt": "D",
    "keyframe": "D",
    "scratchpad": "D",
    "finish": "D",
    "diagnosis": "P",  # ground truth from replaying the episode's recorded waypoints
}


class RecordPart(BaseModel):
    """One part of an episode's record; `step`: the env step it belongs to, if any; `turn`: the
    Agent's turn it was taken or made in (a keyframe: the turn its frame was taken in), if
    any."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: PartKind
    content: JsonValue
    step: NonNegativeInt | None = None
    turn: NonNegativeInt | None = None

    @property
    def tier(self) -> Tier:
        return PART_TIERS[self.kind]


class EpisodeOutcome(BaseModel):
    """How an episode ended, from its `summary.json` and `config.json` (tier D, beside the
    official success): the end's reason, the Agent's turns and model calls and the model-call
    limit, the env steps the Agent's actions used, the benchmark's step count and limit (the
    client's: steps after a `finish` count too), whether infrastructure failed it (`infra`) and
    the end the host voided, if any."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    end: str | None = None
    turns: NonNegativeInt | None = None
    model_calls: NonNegativeInt | None = None
    max_model_calls: NonNegativeInt | None = None
    steps: NonNegativeInt | None = None
    official_steps: NonNegativeInt | None = None
    step_limit: NonNegativeInt | None = None
    infra: bool = False
    voided: str | None = None


EventSource = Literal["episode", "host", "none"]
"""Where an episode's events were read: its own `events.jsonl`; the host's log, filtered by the
episode's id (an episode cut by the step limit can lack its own copy); none found."""


class EvalEpisode(BaseModel):
    """One evaluated episode: its task and seed, the benchmark's success, its parts in order, its
    records directory (`frame_images` reads its keyframes' images from there) and how it ended
    (`outcome`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskName
    seed: NonNegativeInt
    success: bool
    parts: tuple[RecordPart, ...]
    directory: Path | None = None
    outcome: EpisodeOutcome | None = None
    events_from: EventSource = "episode"


class EvalRecordView(BaseModel):
    """Episodes holding only parts of the view's `tiers`; build one with `of`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tiers: frozenset[Tier] = frozenset({"D"})
    episodes: tuple[EvalEpisode, ...]

    @model_validator(mode="after")
    def _within_tiers(self) -> Self:
        for episode in self.episodes:
            outside = {part.kind for part in episode.parts if part.tier not in self.tiers}
            if outside:
                where = f"{episode.task} seed {episode.seed}"
                raise ValueError(f"{where}: {sorted(outside)} outside {set(self.tiers)}")
        return self

    @classmethod
    def of(cls, episodes: list[EvalEpisode], *, privileged: bool = False) -> "EvalRecordView":
        """The view of `episodes`: D parts and success, plus P parts when `privileged`."""
        tiers: frozenset[Tier] = frozenset({"D", "P"}) if privileged else frozenset({"D"})
        kept = tuple(
            episode.model_copy(update={"parts": tuple(p for p in episode.parts if p.tier in tiers)})
            for episode in episodes
        )
        return cls(tiers=tiers, episodes=kept)


SeedOf = Callable[[str, int, Mapping[str, Any]], int]
"""The env seed of a host episode: from its slot key, its index in the slot and its
`summary.json`."""

SUMMARY: Final = "summary.json"
EVENTS: Final = "events.jsonl"
EPISODE_DIR: Final = re.compile(r"ep[0-9]+")
"""An episode's directory under its slot (`agent.host`: `ep{index:03d}`, so `ep1000` on)."""

DIAGNOSIS: Final = "diagnosis.json"
"""An episode's ground-truth diagnosis, beside its records (tier P)."""

OWN_KINDS: Final[dict[str, PartKind]] = {"finish": "finish", "scratchpad": "scratchpad"}
"""The Hub's own tools whose calls are parts of their own kind, not tool calls."""


def read_eval_records(records: Path, seed_of: SeedOf) -> list[EvalEpisode]:
    """The episodes of an Agent host's records directory (`HostConfig.records`), by slot and
    index, each with all its parts (`EvalRecordView.of` keeps those of the view's tiers);
    episodes infrastructure lost are left out."""
    episodes = []
    for directory in _episode_dirs(records):
        if not (path := directory / SUMMARY).is_file():
            continue  # no summary: lost to the infrastructure, as `infra_lost`
        summary = json.loads(path.read_text(encoding="utf-8"))
        if infra_lost(summary):
            continue
        seed = seed_of(directory.parent.name, int(directory.name[2:]), summary)
        episodes.append(_episode(directory, summary, seed))
    return episodes


def read_eval_episode(directory: Path, seed: int) -> EvalEpisode | None:
    """One host episode (`<records>/<key>/ep<NNN>`) that ran `seed` (a benchmark that knows its
    episodes' seeds picks them itself); None when infrastructure lost
    it."""
    if not (path := directory / SUMMARY).is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return None if infra_lost(summary) else _episode(directory, summary, seed)


def _episode(directory: Path, summary: Mapping[str, Any], seed: int) -> EvalEpisode:
    official = summary.get("official") or {}
    events, source = episode_events(directory)
    return EvalEpisode(
        task=TaskName(summary["task"]),
        seed=seed,
        success=bool(official.get("success")),
        parts=tuple(_parts(directory, summary, events)),
        directory=directory,
        events_from=source,
        outcome=_outcome(directory, summary),
    )


def _outcome(directory: Path, summary: Mapping[str, Any]) -> EpisodeOutcome:
    official = summary.get("official") or {}
    counters = summary.get("counters") or {}
    limits: Mapping[str, Any] = {}
    if (config := directory / "config.json").is_file():
        limits = json.loads(config.read_text(encoding="utf-8")).get("limits") or {}
    return EpisodeOutcome(
        end=(summary.get("end") or {}).get("reason"),
        turns=summary.get("turns"),
        model_calls=summary.get("model_calls"),
        max_model_calls=limits.get("max_model_calls"),
        steps=counters.get("steps"),
        official_steps=official.get("steps"),
        step_limit=official.get("step_limit"),
        infra=bool(summary.get("infra")),
        voided=summary.get("voided_end"),
    )


MEMORY: Final = "memory_before.md"
"""The Task Memory as an episode's Actor read it (tier D: part of its prompt)."""


def episode_memory(directory: Path) -> str | None:
    """The Task Memory text the episode's Actor read (`memory_before.md`), else, in library
    mode, the Memories it had in use at the end (`memory_used.md`), if recorded."""
    for name in (MEMORY, MEMORY_USED):
        path = directory / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def _episode_dirs(records: Path) -> list[Path]:
    """`<records>/<key>/ep<N>` directories (`N` any number of digits) by slot, then by index."""
    if not records.is_dir():
        return []
    slots = sorted(path for path in records.iterdir() if path.is_dir())
    found = [
        (slot.name, int(path.name[2:]), path)
        for slot in slots
        for path in slot.iterdir()
        if path.is_dir() and EPISODE_DIR.fullmatch(path.name)
    ]
    return [path for _, _, path in sorted(found)]


def _parts(directory: Path, summary: Mapping[str, Any], events: list[Event]) -> list[RecordPart]:
    parts = [RecordPart(kind="instruction", content=summary.get("instruction"))]
    frames: list[tuple[int, int]] = []  # (turn, step) of each frame the episode took
    for event in events:
        if event.kind != EventKind.ENV_OBSERVE:
            continue
        data = event.data
        turn = data.get("turn", 0)
        frames.append((turn, data["step"]))
        frame = {"frame": data["alias"], "observation": data["observation"]}
        parts.append(RecordPart(kind="keyframe", content=frame, step=data["step"], turn=turn))
    if (turns := directory / "turns.jsonl").is_file():
        for turn in read_lines(turns):
            step = None  # the latest frame's at or before this turn, in the order taken
            for taken, at in frames:
                if taken <= turn["turn"]:
                    step = at
            number = turn["turn"]
            if seen := _seen(turn.get("observation")):
                parts.append(RecordPart(kind="observation", content=seen, step=step, turn=number))
            for exchange in turn["exchanges"]:
                if text := exchange.get("text"):
                    parts.append(RecordPart(kind="assistant", content=text, step=step, turn=number))
                for made in exchange["calls"]:
                    parts += _call_parts(made, step, number)
            if rationale := turn.get("rationale"):
                parts.append(
                    RecordPart(kind="rationale", content=rationale, step=step, turn=number)
                )
    if (diagnosis := directory / DIAGNOSIS).is_file():
        content = json.loads(diagnosis.read_text(encoding="utf-8"))
        parts.append(RecordPart(kind="diagnosis", content=content))
    return parts


def _seen(observation: Any) -> str:
    """A turn's observation as text, each image as its label (`[image: <label>]`)."""
    if not observation:
        return ""
    texts = []
    for part in observation:
        if part.get("type") == "text":
            texts.append(part["text"])
        else:
            texts.append(f"[image: {part.get('label') or 'image'}]")
    return "\n".join(texts)


def _call_parts(made: Mapping[str, Any], step: int | None, turn: int) -> list[RecordPart]:
    name, arguments = made["name"], made["arguments"]
    receipt = {key: made[key] for key in ("status", "error", "content")}
    if name in OWN_KINDS:
        own = [RecordPart(kind=OWN_KINDS[name], content=arguments, step=step, turn=turn)]
        if made["error"] is None:
            return own
        # a refused finish (for whichever cause its text gives) is what a reflection needs to see
        return [*own, RecordPart(kind="receipt", content=receipt, step=step, turn=turn)]
    call = {"name": name, "arguments": arguments}
    return [
        RecordPart(kind="tool_call", content=call, step=step, turn=turn),
        RecordPart(kind="receipt", content=receipt, step=step, turn=turn),
    ]


class FrameImage(BaseModel):
    """A camera's RGB image of a keyframe: the frame's alias and id, its step, the camera."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame: str
    frame_id: FrameId
    step: NonNegativeInt
    camera: str
    image: BlobRef

    @property
    def tier(self) -> Tier:
        return PART_TIERS["keyframe"]


class FrameImages(BaseModel):
    """The images `frame_images` found, and what is missing: the blob ids the store lacks (an
    observation's, then its images are unknown too; or an image's), or `events.jsonl` when the
    episode has no event log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    images: tuple[FrameImage, ...]
    missing: tuple[str, ...] = ()


def episode_events(directory: Path) -> tuple[list[Event], EventSource]:
    """An episode's events and where they were read (`EventSource`): its own `events.jsonl`, else
    the host's log (`<records>/events.jsonl`, two levels up) filtered by the episode's id from
    its `summary.json` or `config.json`, else none."""
    if (own := directory / EVENTS).is_file():
        return list(read_events(own)), "episode"
    host = directory.parent.parent / EVENTS
    episode_id = _episode_id(directory)
    if episode_id is None or not host.is_file():
        return [], "none"
    tag = json.dumps(episode_id)
    found = []
    with host.open("rb") as lines:
        for line in lines:
            # the id's text first: the host's log holds every episode of the run
            if tag.encode() in line and line.endswith(b"\n"):
                event = Event.model_validate_json(line)
                if event.episode_id == episode_id:
                    found.append(event)
    return found, ("host" if found else "none")


def _episode_id(directory: Path) -> str | None:
    for name in (SUMMARY, "config.json"):
        if (path := directory / name).is_file():
            found = json.loads(path.read_text(encoding="utf-8")).get("episode_id")
            if found:
                return str(found)
    return None


def episode_frames(directory: Path) -> list[FrameEntry]:
    """The frames an episode took (its `env.observe` events, `episode_events`), in order."""
    return [
        FrameEntry.model_validate(event.data)
        for event in episode_events(directory)[0]
        if event.kind == EventKind.ENV_OBSERVE
    ]


def frame_images(
    directory: Path,
    blobs: LocalBlobStore,
    frames: Collection[str] = (),
    cameras: Collection[str] = (),
) -> FrameImages:
    """The RGB images of the keyframes of the episode at `directory` (`frames`: aliases or frame
    ids, default every frame; `cameras`: default every camera), from `blobs`, the store its run
    wrote. LookupError for a frame the episode did not take."""
    events, source = episode_events(directory)
    if source == "none":
        return FrameImages(images=(), missing=(EVENTS,))
    taken = [FrameEntry.model_validate(e.data) for e in events if e.kind == EventKind.ENV_OBSERVE]
    known = {entry.alias for entry in taken} | {entry.frame_id for entry in taken}
    if unknown := sorted(set(frames) - known):
        raise LookupError(f"{directory} took no frames {unknown}")
    images: list[FrameImage] = []
    missing: list[str] = []
    for entry in taken:
        if frames and entry.alias not in frames and entry.frame_id not in frames:
            continue
        if not blobs.exists(entry.observation):
            missing.append(entry.observation.id)
            continue
        observation = Observation.model_validate_json(blobs.get_bytes(entry.observation))
        for camera, frame in sorted(observation.cameras.items()):
            if cameras and camera not in cameras:
                continue
            if not blobs.exists(frame.rgb):
                missing.append(frame.rgb.id)
                continue
            images.append(
                FrameImage(
                    frame=entry.alias,
                    frame_id=entry.frame_id,
                    step=entry.step,
                    camera=camera,
                    image=frame.rgb,
                )
            )
    return FrameImages(images=tuple(images), missing=tuple(missing))
