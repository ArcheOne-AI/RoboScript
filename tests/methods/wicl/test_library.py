"""P3's Memory library (`memory.library`, `toolsets.library`, the Agent host's library mode): the
library's opaque, stable ids and cards, text served in parts and pages that fit the Actor's cap,
only a library host seeing the library, and leave-one-out hiding the task's own Memory and
demonstration."""

import contextlib
import dataclasses
import importlib.util
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

import anyio
from pydantic import BaseModel

from roboscript.agent import ScriptedBrain, call, read_lines
from roboscript.agent.host import AgentHostService, DemoView, HostConfig, HostLibrary
from roboscript.agent.overlay import Overlay
from roboscript.agent.scripted import Script
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.artifact import ArtifactLayer, ArtifactRef, ArtifactStack
from roboscript.contracts.errors import ToolError
from roboscript.contracts.ids import TaskName, ToolName
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.tools import TextPart
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs import BridgeConfig, Bridges
from roboscript.envs.mock import MOCK, MockConfig, MockEnv
from roboscript.hub import Hub
from roboscript.learning.agent_host import (
    AgentHostConfig,
    actor_prompt,
    frozen_memory,
    host_library,
    identify_hook,
    opening_hook,
    with_library,
)
from roboscript.memory import (
    DoneSpecParam,
    FileMemoryStore,
    MemoryDoc,
    VersionSource,
    build_manifest,
)
from roboscript.memory.artifact import MemoryArtifacts, manifest_stack
from roboscript.memory.library import (
    PART_CHARS,
    card_pages,
    entry_id,
    library_entries,
    memory_text,
    split_parts,
)
from roboscript.perception.criteria import DoneSpec
from roboscript.records import EventWriter
from roboscript.testing.hold import Hold
from roboscript.testing.host import MockClient
from roboscript.testing.memory import sample_memory
from roboscript.testing.mock import Mock
from roboscript.testing.phases import TEST_PHASES
from roboscript.testing.scenes import PERCEPTION
from roboscript.tools import LocalProvider, ToolContext, Toolset, tool
from roboscript.tools.frames import IDENTIFIED
from roboscript.tools.skillsets import guides
from roboscript.toolsets.library import LibraryTools
from roboscript.toolsets.perception import RoleQuery
from roboscript.toolsets.scene import SceneConfig
from roboscript.toolsets.sensing import Sensing
from roboscript.toolsets.worldstate import WorldState
from roboscript_methods.wicl.method import P3, Wicl
from roboscript_methods.wicl.toolsets.done import Done

REPO = Path(__file__).resolve().parents[3]
WICL = Wicl.runtime(REPO / "recipes" / "wicl")
PROFILE = load_embodiment(MOCK_BIMANUAL)
RATIONALE = {"stage": "wait", "scene": "three boxes", "progress": "at rest", "plan": "stay"}
OWN = "lift_red"
OTHERS = ("carry_tool", "stack_two")


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("core_test_host_library", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def memory(task: str, *, statement: str = "", criterion: bool = False) -> MemoryDoc:
    doc = sample_memory(task)
    if statement:
        doc = doc.replace(statement=statement)
    if criterion:
        gap = DoneSpec(metric="gap_z", a="tool", b="table", value=0.0, tol=0.01)
        doc = doc.replace(params={**doc.params, "tool_down": DoneSpecParam(value=gap)})
    return doc


async def frozen(root: Path, docs: list[MemoryDoc]) -> FileMemoryStore:
    store = FileMemoryStore(root)
    for doc in docs:
        await store.commit(doc, status="s1_draft", source=VersionSource(kind="s1"))
        await store.freeze(doc.task, 0, reason="for the test")
    return store


def library_docs() -> list[MemoryDoc]:
    return [
        memory(OWN, statement="Raise the red block."),
        memory(OTHERS[0], criterion=True),
        memory(OTHERS[1], statement="Stack the green block on the blue block."),
    ]


# The library itself


async def test_ids_are_opaque_and_stable_and_cards_are_extracts(tmp_path: Path) -> None:
    store = await frozen(tmp_path / "m", library_docs())
    versions = [(t, await store.load(t, (await store.frozen(t)).hash)) for t in (OWN, *OTHERS)]  # type: ignore[union-attr]
    entries = library_entries((t, v, v) for t, v in versions)
    again = library_entries((t, v, v) for t, v in versions)
    assert [e.id for e in entries] == [e.id for e in again]  # same Memory, same id and card
    assert [e.card for e in entries] == [e.card for e in again]
    for entry, (task, version) in zip(entries, versions, strict=True):
        assert entry.id == entry_id(version.meta.hash) and entry.id.startswith("mem-")
        assert task not in entry.id  # the id is no task name
        assert entry.card.task == task  # the card says where the Memory came from
    card = entries[1].card
    assert card.tools == ["grasp_object", "place_object"]  # the workflow's calls, each once
    assert [r.split(":")[0] for r in card.roles] == ["tool", "base"]
    assert card.stages == ["grasp_tool", "set_down"]
    assert "gap_z(tool, table)" in card.done  # its done criterion
    assert card.done[0].startswith("near(A, base")  # the final stage's done line
    assert entries[1].card.brief().startswith(f"[{entries[1].id}] {OTHERS[0]}: Put the tool")
    assert "Stack the green block" in entries[2].card.render()


async def test_text_goes_out_in_parts_and_pages_that_fit_the_actor_s_cap(tmp_path: Path) -> None:
    store = await frozen(tmp_path / "m", library_docs())
    version = await store.load(OWN, (await store.frozen(OWN)).hash)  # type: ignore[union-attr]
    [entry] = library_entries([(OWN, version, version)])
    text = memory_text(entry)
    assert text.startswith(f"# TASK MEMORY {entry.id} (from task {OWN})")
    assert "Raise the red block." in text and "## WORKFLOW" in text
    long = text * 6
    parts = split_parts(long)
    assert len(parts) > 1 and all(len(p) <= PART_CHARS + 1 for p in parts)
    assert "".join(p.rstrip() for p in parts).replace("\n", "") == long.replace("\n", "")
    pages = card_pages([entry] * 30)
    assert len(pages) > 1 and all(len(p) <= PART_CHARS for p in pages)


def host_config(
    tmp_path: Path, library: dict[str, Any], access: str = "library"
) -> AgentHostConfig:
    return AgentHostConfig.model_validate(
        {
            "manifest": str(tmp_path / "manifest.json"),
            "method": "wicl",
            "recipe": "recipes/wicl",
            "episodes": {"records": str(tmp_path / "host"), "embodiment": str(MOCK_BIMANUAL)},
            "toolsets": [{"name": "sensing"}],
            "memory_access": access,
            "memory_library": library,
        }
    )


def test_only_a_library_host_sees_the_library() -> None:
    """The method's tables have no `library.*` (bound, S1 and S2 runs, their catalog locks and
    tokens); a host in library mode adds it last to its phase's Actor list (`with_library`)."""
    assert AgentHostConfig.model_fields["memory_access"].default == "library"
    assert all(
        "library.*" not in entries for phase in Wicl.phases for entries in phase.visibility.values()
    )
    added = with_library(Wicl.phases, "p3")
    for before, after in zip(Wicl.phases, added, strict=True):
        if before.name != "p3":
            assert after == before
            continue
        assert after.visibility["actor"] == [*P3.visibility["actor"], "library.*"]
        assert {r: v for r, v in after.visibility.items() if r != "actor"} == {
            r: v for r, v in before.visibility.items() if r != "actor"
        }


# A host episode in library mode

KEY = "lift_red.test.seed0"
PHASES = [
    phase.model_copy(
        update={
            "visibility": {
                **phase.visibility,
                "actor": ["library.*", *phase.visibility["actor"], "eyes.identify"],
            }
        }
    )
    if phase.name == "p3"
    else phase
    for phase in TEST_PHASES
]


@contextlib.asynccontextmanager
async def library_host(
    tmp_path: Path,
    events: EventWriter,
    script: Script,
    *,
    exclude_own: bool = False,
    phases: list[PhaseSpec] = PHASES,
    perception: "FakePerception | None" = None,
    use_timeout_s: float | None = None,
    demos: dict[str, DemoView] | None = None,
    own_memory: bool = True,
    docs: list[MemoryDoc] | None = None,
    overlaid: bool = False,
) -> AsyncIterator[tuple[MockClient, HostLibrary]]:
    """A started host in library mode over lift_red's manifest (`own_memory`; else another
    task's: lift_red has no Memory of its own), the library of three tasks; with `perception`,
    a fake `identify` mounted (`use_timeout_s`: `library_use`'s timeout made short)."""
    blobs = LocalBlobStore(tmp_path / "blobs")
    store = await frozen(tmp_path / "memory", docs or library_docs())
    manifest = await build_manifest(store, [OWN if own_memory else OTHERS[0]])
    bridges = Bridges()
    tools = LibraryTools()
    providers = [
        LocalProvider(Sensing()),
        LocalProvider(Hold()),
        LocalProvider(Mock(), {"act_seconds": 0.0}),
        LocalProvider(WorldState(), SceneConfig(pipeline=PERCEPTION)),
        LocalProvider(Done(), SceneConfig(pipeline=PERCEPTION)),
        LocalProvider(tools),
        *([] if perception is None else [LocalProvider(perception)]),
    ]
    hub = Hub(providers, blobs=blobs, phases=phases, events=events, env_client=bridges.client)
    if use_timeout_s is not None:
        use = hub.mounted[ToolName("library_use")]
        meta = use.meta.model_copy(update={"timeout_s": use_timeout_s})
        hub._tools[ToolName("library_use")] = dataclasses.replace(use, meta=meta)
    config = HostConfig(
        records=tmp_path / "host",
        embodiment=str(MOCK_BIMANUAL),
        bridge=BridgeConfig(observe_timeout_s=5, report_timeout_s=5),
        latch_finish=False,
    )
    prompts = WICL.prompts({}, guides(MOCK.skillsets))
    prompt, digest = actor_prompt(prompts, PROFILE, "p3")
    host = host_config(tmp_path, {"exclude_own": exclude_own})
    full = await build_manifest(store, [OWN, *OTHERS])  # the library: every task of the store
    library = await host_library(
        host, full, MemoryArtifacts(memory=store), WICL, prompts, PROFILE, "p3", tools
    )
    service = AgentHostService(
        config,
        hub=hub,
        bridges=bridges,
        blobs=blobs,
        events=events,
        load=frozen_memory(MemoryArtifacts(memory=store), WICL.frozen),
        manifest=manifest,
        brain=lambda: ScriptedBrain(script),
        prompt=prompt,
        phase="p3",
        prompt_digest=digest,
        opening=opening_hook(hub, WICL.opening),
        finish=WICL.finish.host,
        demos=demos,
        library=library,
        identify=identify_hook(hub, WICL.opening),
        **(overlay_of(manifest) if overlaid else {}),
    )
    env = MockEnv(MockConfig(task=OWN, blob_dir=tmp_path / "blobs"), blobs)
    async with hub, service:
        await service.start()
        yield MockClient(env, service, KEY, OWN, manifest.id), library


def overlay_of(manifest: Any) -> dict[str, Any]:
    """A stack with an RSI-like document beside the Memory, a part for every task and one for
    lift_red, and a loader that renders each part as `PART <task>`."""
    doc = ArtifactRef(kind="rsi", name="G", version="v000", hash="2" * 64)
    layers = [
        *manifest_stack(manifest).layers,
        ArtifactLayer(ref=doc),
        ArtifactLayer(ref=doc, task=TaskName(OWN)),
    ]

    async def load(task: TaskName, given: Any) -> Overlay:
        parts = tuple(f"PART {layer.task}" for layer in given if layer.ref.kind == "rsi")
        rsi = tuple(layer for layer in given if layer.ref.kind == "rsi")
        return Overlay(after=parts, layers=rsi)

    return {"stack": ArtifactStack(layers=layers), "overlay": load}


def ids(library: HostLibrary) -> dict[str, str]:
    return {e.task: e.id for e in library.entries}


async def test_leave_one_out_hides_the_own_memory(tmp_path: Path, events: EventWriter) -> None:
    store = await frozen(tmp_path / "probe", library_docs())
    own = entry_id((await store.frozen(OWN)).hash)  # type: ignore[union-attr]
    script: Script = [
        [call("library_read", id=own)],
        [call("finish", note="done")],
    ]
    async with library_host(tmp_path, events, script, exclude_own=True) as (bench, _):
        with anyio.fail_after(30):
            await bench.episode()
    episode = tmp_path / "host" / KEY / "ep000"
    record = json.loads((episode / "summary.json").read_text(encoding="utf-8"))["library"]
    assert record["excluded"] == [OWN] and own not in record["seen"] and len(record["seen"]) == 2
    turns = read_lines(episode / "turns.jsonl")
    [refused] = turns[0]["exchanges"][0]["calls"]
    assert "no entry" in refused["content"][0]["text"] and record["read"] == []


# Bound mode: no library


# Identification as the Actor adds Memories (a perception toolset mounted)


class Row(BaseModel):
    role: str
    obj: str
    reason: str


class Found(BaseModel):
    roles: list[Row]


class FakePerception(Toolset):
    """A perception toolset with the system-only `identify` alone (named apart from the real one,
    whose other tools the test phases list): each call takes the next of
    `modes` (`ok`: every role found and the `IDENTIFIED` fact set; `refuse`: the tool's own
    error, a configuration failure; `hang`: it never answers), then `ok`."""

    name = "eyes"

    def __init__(self, modes: list[str] | None = None) -> None:
        self.modes = list(modes or [])
        self.asked: list[list[str]] = []

    @tool(kind="perception", tier="D", timeout_s=60.0, system_only=True)
    async def identify(
        self,
        ctx: ToolContext,
        instruction: str,
        roles: list[RoleQuery] | None = None,
    ) -> Found:
        """Identify the roles' objects (a fake)."""
        mode = self.modes.pop(0) if self.modes else "ok"
        names = [r.role for r in roles] if roles else ["target"]
        self.asked.append(names)
        if mode == "refuse":
            raise ToolError("the model refused to name the objects")
        if mode == "hang":
            await anyio.sleep(60)
        found = {n: {"object": f"obj_{n}", "reason": "ok"} for n in names}
        ctx.frames.set_fact(IDENTIFIED, {"frame": "f0", "roles": found})
        return Found(roles=[Row(role=n, obj=f"obj_{n}", reason="ok") for n in names])


def episode_dir(tmp_path: Path) -> Path:
    return tmp_path / "host" / KEY / "ep000"


def results(tmp_path: Path, name: str) -> list[str]:
    """The text the Actor got back from each `name` call."""
    out = []
    for turn in read_lines(episode_dir(tmp_path) / "turns.jsonl"):
        for exchange in turn["exchanges"]:
            for made in exchange["calls"]:
                if made["name"] == name:
                    out.append(" ".join(p.get("text", "") for p in made["content"]))
    return out


async def test_leave_one_out_withholds_the_demonstration_too(
    tmp_path: Path, events: EventWriter
) -> None:
    demo = DemoView(
        brief=[TextPart(text="DEMONSTRATION OF LIFT_RED")], seed=None, label={"demo_context": "x"}
    )
    for exclude_own in (False, True):
        directory = tmp_path / str(exclude_own)
        directory.mkdir()
        script: Script = [[call("finish", note="done")]]
        async with library_host(
            directory, events, script, exclude_own=exclude_own, demos={OWN: demo}
        ) as (bench, _):
            with anyio.fail_after(30):
                await bench.episode()
        episode = episode_dir(directory)
        calls = (episode / "llm_calls.jsonl").read_text(encoding="utf-8")
        config = json.loads((episode / "config.json").read_text(encoding="utf-8"))
        assert ("DEMONSTRATION OF LIFT_RED" in calls) is not exclude_own
        assert config["demo_context"] == ({"demo_context": "none"} if exclude_own else demo.label)
