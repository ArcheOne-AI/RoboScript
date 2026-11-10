"""Evaluated episodes' records written as the Agent host writes them (`summary.json`,
`turns.jsonl`, `events.jsonl` with each frame's observation and images in a blob store,
`config.json`, `memory_before.md`), for the reflection's tests."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.camera import CameraFrame, CameraModel
from roboscript.contracts.events import EventKind
from roboscript.contracts.ids import FrameId, TaskName
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.learning.eval_view import EvalEpisode, read_eval_episode
from roboscript.learning.seeds import SeedRange, SeedSplit
from roboscript.records import EventWriter
from roboscript.testing.scenes import robot_state
from roboscript_methods.rsi.doc import EpisodeKey

BLOCK = "block1"
START = 200000
SPLIT = SeedSplit(
    blocks={BLOCK: SeedRange(start=START, count=100), "block2": SeedRange(start=300000, count=100)},
    reflects={"block2": BLOCK},
)
MEMORY = "# Memory\nworkflow: reach, grasp, lift, place.\n"
CAMERAS = ("head", "wrist")


def made(name: str, text: str = "ok", error: str | None = None, **arguments: Any) -> dict[str, Any]:
    """One call of a turn as `turns.jsonl` keeps it; its result `text`, an error's code `error`."""
    return {
        "name": name,
        "arguments": arguments,
        "status": "error" if error else "ok",
        "error": error,
        "content": [{"type": "text", "text": text}],
    }


@dataclass
class Turn:
    calls: list[dict[str, Any]]
    text: str = ""
    frames: int = 1
    """How many frames the turn took (each at its start)."""
    rationale: dict[str, str] | None = None
    """The rationale the model gave with the turn's action (`turns.jsonl`), if any."""


@dataclass
class Episode:
    task: str
    index: int
    success: bool
    turns: Sequence[Turn]
    end: str = "finish"
    infra: bool = False
    voided: str | None = None
    instruction: str = "move the part"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> EpisodeKey:
        return EpisodeKey(block=BLOCK, task=TaskName(self.task), episode=self.index)

    @property
    def seed(self) -> int:
        return START + self.index


def _camera(name: str) -> CameraModel:
    k = ((1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
    pose = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0, 0, 0, 1.0))
    return CameraModel(name=name, width=2, height=2, K=k, cam2world=pose)


def write(root: Path, blobs: LocalBlobStore, spec: Episode) -> Path:
    """`spec`'s records under `root/records/<task>.p3.seed0/ep<index>`; its directory."""
    directory = root / "records" / f"{spec.task}.p3.seed0" / f"ep{spec.index:03d}"
    directory.mkdir(parents=True)
    step = 0
    lines = []
    with EventWriter(directory / "events.jsonl", "agent_host") as events:
        for number, turn in enumerate(spec.turns):
            for _ in range(turn.frames):
                cameras = {}
                for camera in CAMERAS:
                    rgb = np.full((2, 2, 3), step % 250, dtype=np.uint8)
                    rgb[0, 0] = (spec.index, number, CAMERAS.index(camera))  # blobs of their own
                    cameras[camera] = CameraFrame(camera=_camera(camera), rgb=blobs.put_image(rgb))
                frame_id = FrameId(f"{spec.task}-{spec.index}-{step}")
                observation = Observation(
                    frame_id=frame_id, step=step, cameras=cameras, robot=robot_state()
                )
                ref = blobs.put_bytes(observation.model_dump_json().encode(), "application/json")
                alias = f"frame_{step // 10}"
                entry = FrameEntry(frame_id=frame_id, alias=alias, step=step, observation=ref)
                data = {**entry.model_dump(mode="json"), "turn": number}
                events.emit(EventKind.ENV_OBSERVE, data)
                step += 10
            seen = [{"type": "text", "text": f"turn {number}: the scene"}]
            seen.append({"type": "image", "label": f"frame_{step // 10 - 1} head"})
            lines.append(
                {
                    "turn": number,
                    "observation": seen,
                    "exchanges": [{"text": turn.text, "calls": turn.calls}],
                    **({"rationale": turn.rationale} if turn.rationale else {}),
                }
            )
    (directory / "turns.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    summary = {
        "task": spec.task,
        "instruction": spec.instruction,
        "official": {"success": spec.success, "steps": step, "step_limit": 400},
        "end": {"reason": spec.end},
        "turns": len(spec.turns),
        "model_calls": len(spec.turns),
        "counters": {"steps": step},
        "infra": spec.infra,
        **({"voided_end": spec.voided} if spec.voided else {}),
        **spec.extra,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    config = {"limits": {"max_model_calls": 60}}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "memory_before.md").write_text(MEMORY, encoding="utf-8")
    return directory


def episodes(
    root: Path, specs: Sequence[Episode]
) -> tuple[list[EvalEpisode], list[EpisodeKey], LocalBlobStore]:
    """The view's episodes of `specs`, their keys, and the records' blob store."""
    blobs = LocalBlobStore(root / "records-blobs")
    found = []
    for spec in specs:
        episode = read_eval_episode(write(root, blobs, spec), spec.seed)
        assert episode is not None
        found.append(episode)
    return found, [spec.key for spec in specs], blobs


# Turns the tests reuse

EMPTY = (
    "Receipt: empty_grasp (hard failure)\ngripper: left opening 0.02 (closed on nothing)\n"
    "Note: the gripper closed on nothing"
)
DONE = "done: yes (the end state matches the demonstration's last keyframe)"


def failed_episode(task: str, index: int) -> Episode:
    """A false finish: an empty grasp, a grasp and a place, the done check passing, a finish."""
    return Episode(
        task=task,
        index=index,
        success=False,
        turns=[
            Turn([made("look")], text="I see the part."),
            Turn([made("grasp_object", EMPTY, actor="part")]),
            Turn([made("grasp_object", actor="part"), made("place_object", target="pad")]),
            Turn([made("check_done", DONE)]),
            Turn([made("scratchpad", note="following G-01 and T-x-02"), made("finish", note="ok")]),
        ],
    )
