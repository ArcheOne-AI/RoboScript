"""What P3's `eval()` and the relay send each other over RoboTwin's socket (JSON, numpy arrays as
base64, both done by RoboTwin): the observation `eval()` packs from the shim's readings, the
Observation the relay makes of it (by `roboscript_robotwin.frames`, as the env host makes them, in
the blob store shared with the Agent host), and the relay's answers: a `Poll`, `ok`, or `failed`
when the host cannot be reached or refused, each with `hold`, how many actions `eval()` holds still
for once the agent is done."""

import functools
from collections.abc import Mapping
from functools import cache
from typing import Any, Final

import anyio
import numpy as np

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.action import ActionBatch
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import RobotState
from roboscript.embodiments import EmbodimentProfile
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.frames import Frame
from roboscript_robotwin.frames import observation as build

Packed = dict[str, Any]
INFRA_LOST: Final = "RoboScript: infrastructure lost this episode"
"""What `eval()` prints as an episode infrastructure lost ends, into the client's log before the
official client's own lines of that episode: `launch.finished` reads the official client's log
alone, which counts such an episode as a failure, and marks it lost from this line."""
Answer = dict[str, Any]


@cache
def profile() -> EmbodimentProfile:
    return ROBOTWIN.profile()


def pack(frames: Mapping[str, Frame], robot: RobotState, instruction: str | None) -> Packed:
    """An observation for the socket; the arrays stay numpy, which RoboTwin's encoder handles."""
    cameras = {
        name: {"rgb": frame.rgb, "depth": frame.depth, "K": frame.K, "cam2world": frame.cam2world}
        for name, frame in frames.items()
    }
    return {"cameras": cameras, "robot": robot.model_dump(mode="json"), "instruction": instruction}


async def observation(packed: Packed, blobs: LocalBlobStore) -> Observation:
    """A packed observation as an `Observation`, its frames put into `blobs`
    (`roboscript_robotwin.frames`, as the env host does)."""
    frames = {
        name: Frame(
            K=np.asarray(frame["K"], dtype=float),
            cam2world=np.asarray(frame["cam2world"], dtype=float),
            rgb=np.asarray(frame["rgb"]),
            depth=np.asarray(frame["depth"]),
        )
        for name, frame in packed["cameras"].items()
    }
    robot = RobotState.model_validate(packed["robot"])
    encode = functools.partial(build, frames, robot, packed["instruction"], blobs)
    return await anyio.to_thread.run_sync(encode)


def batch(answer: Answer) -> ActionBatch:
    """The batch of a `batch` answer."""
    return ActionBatch.model_validate(answer["batch"])


def answer(status: str, hold: int, **fields: Any) -> Answer:
    """A relay answer: `status` (`pending`, `batch`, `done`, `ok` or `failed`), `hold` and the
    rest (`batch`, `reason`, `infra`)."""
    return {"status": status, "batch": None, "reason": None, "hold": hold, "infra": False, **fields}
