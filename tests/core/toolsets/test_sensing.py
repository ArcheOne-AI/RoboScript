"""The sensing toolset over the Hub and MockEnv: observations of every camera with the TCPs marked,
the Actor's frames seen by a Reviser under the same aliases, and who sees which tool."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from roboscript.contracts.action import ActionBatch, ActionStep
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.geometry import Pose
from roboscript.contracts.hub import CallOutcome, OpenEpisode
from roboscript.contracts.ids import BatchId, EpisodeId, new_id
from roboscript.contracts.observation import Observation
from roboscript.contracts.tools import ImagePart, TextPart, ToolCall
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.records import read_events
from roboscript.spatial.camera import project
from roboscript.tools import TABLE, ToolContext, Toolset, seed_from_log, tool
from roboscript.toolsets.sensing import Sensing

Model = Callable[..., ToolCall]
HEAD = "overhead"
EXTENT = [-0.6, 0.6, -0.35, 0.35]


class Done(BaseModel):
    ok: bool = True


class Tables(Toolset):
    """Stands in for perception: sets the table estimate in the episode's store."""

    name = "tables"

    @tool(kind="pure", tier="D")
    async def set_table(self, ctx: ToolContext, z: float) -> Done:
        """Set the table estimate."""
        ctx.frames.set_fact(TABLE, {"z": z, "extent": EXTENT})
        return Done()


async def move_left(rig: Any, env_episode: EpisodeId, tcp: tuple[float, float, float]) -> None:
    """Put the left TCP at `tcp`, the gripper pointing down (mock joints are the ee pose)."""
    qpos = list((await rig.env.robot(env_episode)).qpos_cmd)
    qpos[:3] = [tcp[0], tcp[1], tcp[2] + 0.1]  # the TCP is 0.1 m below the ee
    batch = ActionBatch(batch_id=BatchId(new_id("b")), steps=[ActionStep(qpos=qpos)])
    await rig.env.execute(env_episode, batch)


def pixel(camera: dict[str, Any], point: tuple[float, float, float]) -> tuple[int, int]:
    """Where `point` appears in the image of a `get_camera` result (as JSON)."""
    K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
    uv, _, in_front = project(point, K, Pose.model_validate(camera["pose"]).matrix())
    assert in_front
    u, v = (int(x) for x in np.round(uv))
    assert 0 <= u < camera["width"] and 0 <= v < camera["height"]
    return u, v


async def image_of(rig: Any, outcome: CallOutcome, index: int = 0) -> np.ndarray:
    images = [part for part in outcome.content if isinstance(part, ImagePart)]
    return np.asarray(await rig.blobs.aget_image(images[index].blob))


def text_of(outcome: CallOutcome) -> str:
    return "\n".join(part.text for part in outcome.content if isinstance(part, TextPart))


def cyan(rgb: np.ndarray) -> bool:
    return bool(rgb[0] < 60 and rgb[1] > 180 and rgb[2] > 200)


async def test_observe_shows_every_camera_with_the_tcps_marked(rig: Any, model: Model) -> None:
    async with rig.hub(Sensing()) as hub:
        ep, env_ep = await rig.episode(hub, role="demo_reader")  # get_camera
        await move_left(rig, env_ep, (-0.1, -0.1, 0.85))
        observed = await hub.call(ep, model("observe"))
        plain = await hub.call(ep, model("observe", overlay=False))
        head = await hub.call(ep, model("get_camera", camera=HEAD, frame="frame_1"))
    found = observed.structured
    assert found is not None and head.structured is not None and plain.structured is not None
    assert (found["frame"], found["step"], found["step_limit"]) == ("frame_1", 1, 40)
    left = found["arms"]["left"]
    assert left["tcp"]["p"] == pytest.approx([-0.1, -0.1, 0.85])
    assert (left["gripper_cmd"], left["holding"]) == (1.0, False)
    assert found["overlays"] and plain.structured["overlays"] == []
    assert text_of(observed).startswith("Observation frame_1")
    images = [part for part in observed.content if isinstance(part, ImagePart)]
    assert [(part.label, part.blob.media_type) for part in images] == [
        (HEAD, "image/png"),
        ("wrist_left", "image/png"),
        ("wrist_right", "image/png"),
    ]
    marked, raw = await image_of(rig, observed), await image_of(rig, plain)
    assert marked.shape == raw.shape == (384, 512, 3)
    u, v = pixel(head.structured, (-0.1, -0.1, 0.85))
    assert cyan(marked[v, u]) and not cyan(raw[v, u])
    # The left wrist camera looks along its gripper, so its TCP is at the image center.
    wrist, wrist_raw = await image_of(rig, observed, 1), await image_of(rig, plain, 1)
    assert wrist.shape == (192, 256, 3)
    assert cyan(wrist[96, 128]) and not cyan(wrist_raw[96, 128])
    puts = [e.data for e in read_events(rig.events.path) if e.kind == "env.observe"]
    assert [put["alias"] for put in puts] == ["frame_1", "frame_2"]  # each one in the log
    stored = BlobRef.model_validate(puts[1]["observation"])
    frame = Observation.model_validate_json(await rig.blobs.aget_bytes(stored))
    shown = [part.blob for part in plain.content if isinstance(part, ImagePart)]
    assert shown == [frame.cameras[part.label].rgb for part in images]  # the PNGs as captured


async def test_a_reviser_sees_the_actors_frames_under_the_same_aliases(
    rig: Any, model: Model
) -> None:
    async with rig.hub(Sensing()) as hub:
        actor, env_ep = await rig.episode(hub, role="demo_reader")  # get_camera
        await hub.call(actor, model("observe", views=[HEAD]))
        await move_left(rig, env_ep, (0.1, -0.05, 0.9))
        await hub.call(actor, model("observe"))
        before = await hub.call(actor, model("get_camera", camera="wrist_left", frame="frame_2"))
        seed = seed_from_log(read_events(rig.events.path), actor)
        spec = OpenEpisode(
            stage="s2", role="reviser", seed=seed, embodiment=str(MOCK_BIMANUAL), structured=True
        )
        reviser = await hub.open_episode(spec)
        after = await hub.call(
            reviser.episode_id, model("get_camera", camera="wrist_left", frame="frame_2")
        )
        shown = await hub.call(reviser.episode_id, model("get_image", camera=HEAD, frame="frame_1"))
    assert [entry.alias for entry in seed.frames] == ["frame_1", "frame_2"]
    names = [tool.name for tool in reviser.tools]
    assert names == ["get_image", "show_frames", "get_camera", "finish"]
    assert after.structured == before.structured  # the same frame, the same calibration
    assert shown.structured is not None and shown.structured["frame"] == "frame_1"


async def test_the_catalog_shows_sensing_by_role(rig: Any) -> None:
    visible = {}
    async with rig.hub(Sensing()) as hub:
        for role in ("actor", "demo_reader", "reviser"):
            ep, _ = await rig.episode(hub, stage="s2", role=role)
            visible[role] = [tool.name for tool in await hub.tools(ep)]
    live = ["observe", "get_image", "show_frames", "get_camera", "get_robot_state", "finish"]
    assert visible == {
        "actor": ["observe", "get_image", "show_frames", "get_robot_state", "finish", "scratchpad"],
        "demo_reader": live,
        "reviser": ["get_image", "show_frames", "get_camera", "finish"],
    }
