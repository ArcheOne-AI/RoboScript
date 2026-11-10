"""The per-episode frame store: frames by id and alias from a JSON blob, scene facts, the scene
registry, the log it leaves, and seeding another episode from that log under the same aliases."""

from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
from pydantic import BaseModel

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraFrame, CameraModel
from roboscript.contracts.errors import ToolError
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.ids import FrameId
from roboscript.contracts.observation import Observation
from roboscript.contracts.robot import RobotState
from roboscript.perception.registry import SceneRegistry
from roboscript.tools import (
    REGISTRY,
    REGISTRY_CHANGED,
    TABLE,
    FrameRef,
    FrameStore,
    ToolContext,
    Toolset,
    tool,
)

EYE = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def observation(frame: str, step: int, rgb: BlobRef) -> Observation:
    K = ((1.0, 0.0, 2.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0))
    camera = CameraModel(name="head", width=4, height=2, K=K, cam2world=EYE)
    robot = RobotState(step=step, step_limit=10, arms={}, qpos_cmd=[])
    cameras = {"head": CameraFrame(camera=camera, rgb=rgb)}
    return Observation(frame_id=FrameId(frame), step=step, cameras=cameras, robot=robot)


@pytest.fixture
async def rgb(blobs: LocalBlobStore) -> BlobRef:
    return await blobs.aput_array(np.zeros((2, 4, 3), np.uint8))


@pytest.fixture
def blobs(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


async def test_frames_by_id_and_alias_facts_and_their_log(
    blobs: LocalBlobStore, rgb: BlobRef
) -> None:
    logged: list[str] = []
    aliases = {"frame-a": "frame_1", "frame-b": "frame_2"}
    store = FrameStore(blobs, alias=aliases.__getitem__, emit=lambda kind, _: logged.append(kind))
    first = await store.put(observation("frame-a", 0, rgb))
    await store.put(observation("frame-b", 3, rgb))
    assert (first.alias, first.step) == ("frame_1", 0)
    assert first.observation.media_type == "application/json"
    assert (await store.get("frame_2")).frame_id == "frame-b"  # by alias
    assert (await store.get("frame-a")).step == 0  # by id
    with pytest.raises(LookupError, match="frame_9"):
        await store.get("frame_9")
    store.set_fact(TABLE, {"z": 0.7, "extent": [0, 1, 0, 1]})
    assert store.fact(TABLE) == {"z": 0.7, "extent": [0, 1, 0, 1]} and store.fact("x") is None
    assert logged == ["env.observe", "env.observe", "scene.fact"]
    # Another store seeded with the entries reads the observations back from their blobs.
    seed = EpisodeSeed(frames=store.entries(), facts={TABLE: {"z": 0.7, "extent": [0, 1, 0, 1]}})
    seeded = FrameStore(blobs, alias=str, emit=lambda kind, data: None, seed=seed)
    assert await seeded.get("frame_2") == await store.get("frame_2")
    assert seeded.fact(TABLE) == store.fact(TABLE)


class Out(BaseModel):
    frame: FrameRef


class Frames(Toolset):
    """Puts frames into the store and reads them back."""

    name = "frames"

    def __init__(self, rgb: BlobRef) -> None:
        self.rgb = rgb
        self.taken = 0

    @tool(kind="perception", tier="D")
    async def snap(self, ctx: ToolContext) -> Out:
        """Take a frame."""
        self.taken += 1
        entry = await ctx.frames.put(observation(f"frame-{self.taken}", self.taken, self.rgb))
        return Out(frame=entry.frame_id)

    @tool(kind="pure", tier="D")
    async def read(self, ctx: ToolContext, frame: FrameRef) -> Out:
        """Read a frame."""
        return Out(frame=(await ctx.frames.get(frame)).frame_id)


CUBE = np.array([[x, y, z] for x in (0, 0.05) for y in (0, 0.05) for z in (0.7, 0.75)])


def register(registry: SceneRegistry, name: str, dx: float = 0.0) -> None:
    registry.register(name, CUBE + np.array([dx, 0.0, 0.0]), frame=FrameId("frame-a"), step=0)


async def test_the_registry_is_kept_as_a_versioned_fact_and_seeds_the_next_episode(
    blobs: LocalBlobStore,
) -> None:
    logged: list[tuple[str, dict[str, Any]]] = []
    store = FrameStore(blobs, alias=str, emit=lambda kind, data: logged.append((kind, data)))
    first = await store.registry()
    assert first is not await store.registry() and not first.objects  # readers get copies
    async with store.edit_registry() as registry:
        register(registry, "cube")
    async with store.edit_registry() as registry:
        assert list(registry.objects) == ["o1"]  # unchanged: no new version
    [(kind, fact)] = logged
    assert (kind, fact["name"]) == ("scene.fact", REGISTRY)
    assert (fact["value"]["objects"], fact["value"]["version"]) == (["o1"], 1)
    assert not first.objects  # the copy taken before never changes
    seeded = FrameStore(
        blobs,
        alias=str,
        emit=lambda kind, data: None,
        seed=EpisodeSeed(facts={REGISTRY: fact["value"]}),
    )
    again = await seeded.registry()
    assert again.to_json() == (await store.registry()).to_json()


class Copy:
    """A remote toolset's copy of `store`, made from its snapshot (`base`), and the facts set in
    it, the last of each, as its call's result brings them back."""

    def __init__(self, blobs: LocalBlobStore, store: FrameStore) -> None:
        self.base = store.snapshot()
        self.facts: dict[str, dict[str, Any]] = {}
        self.store = FrameStore(blobs, alias=str, emit=self._emit, seed=self.base)

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "scene.fact":
            self.facts[data["name"]] = data["value"]


def names(registry: SceneRegistry) -> list[str]:
    return [o.name for o in registry.objects.values()]


async def test_a_copy_s_facts_are_kept_all_or_none(blobs: LocalBlobStore) -> None:
    """What a remote copy set is kept whole, unless it changed the registry and the registry
    here changed since the copy was made: then none of it is kept, the lost race is logged and
    the call fails with a tool error that asks to call again. An edit in progress finishes
    first; within an edit, a copy's registry never is kept. Only edits change the registry."""
    logged: list[tuple[str, dict[str, Any]]] = []
    store = FrameStore(blobs, alias=str, emit=lambda kind, data: logged.append((kind, data)))
    with pytest.raises(ValueError, match="edit_registry"):
        store.set_fact(REGISTRY, {"version": 9})
    first = Copy(blobs, store)
    async with first.store.edit_registry() as registry:
        register(registry, "cube")
    first.store.set_fact(TABLE, {"z": 0.7})
    await store.merge(first.facts, first.base)
    assert store.fact(TABLE) == {"z": 0.7} and names(await store.registry()) == ["cube"]

    late = Copy(blobs, store)  # from version 1, like the edit below
    async with late.store.edit_registry() as registry:
        register(registry, "mat", 0.2)
    late.store.set_fact(TABLE, {"z": 0.75})
    editing, release, failed = anyio.Event(), anyio.Event(), []

    async def edit() -> None:
        async with store.edit_registry() as registry:
            editing.set()
            await release.wait()
            register(registry, "cup", 0.6)

    async def merge() -> None:
        with pytest.raises(ToolError) as error:
            await store.merge(late.facts, late.base)
        failed.append(error.value.detail)

    async with anyio.create_task_group() as group:
        group.start_soon(edit)
        await editing.wait()
        group.start_soon(merge)
        await anyio.wait_all_tasks_blocked()  # the merge waits for the edit's lock
        release.set()
    assert failed == [REGISTRY_CHANGED]
    assert names(await store.registry()) == ["cube", "cup"] and store.fact(TABLE) == {"z": 0.7}
    assert logged[-1] == ("scene.stale", {"name": REGISTRY, "version": 2, "base": 1, "kept": 2})

    current = Copy(blobs, store)  # from the version kept now
    async with current.store.edit_registry() as registry:
        register(registry, "bowl", 0.4)
    with pytest.raises(ToolError, match="call it again"):
        async with store.edit_registry():
            await store.merge(current.facts, current.base)
    await store.merge(current.facts, current.base)  # outside an edit, it is kept
    assert names(await store.registry()) == ["cube", "cup", "bowl"]
