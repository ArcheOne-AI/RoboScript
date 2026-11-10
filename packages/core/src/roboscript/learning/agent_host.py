"""The Agent host process (docs/concepts/learning.md, docs/concepts/operations.md):

    python -m roboscript.learning.agent_host --config <file> [--token-file <path>]
        [--ready-file <path>]

composes what `roboscript.agent.host.AgentHostService` runs on (the Hub over the toolsets named
in `roboscript.toolsets`, in this process or served by their own servers (`ToolsetSpec.url`,
checked as it starts: `roboscript.learning.mounts`), the gateway's model clients, the Memory
store and manifest, the blob store it shares with the benchmark's policy module), with the
Actor's brain (`brain_maker`) and prompt (the method's, `MethodRuntime.prompts`, under the rules
of the phase the method runs the host in, `MethodRuntime.host_phase`: wicl's P3) made as in
S1 and S2, and the one-time identification as each episode opens (`opening_hook`: the hook the
episode driver runs), the Actor's view of the demonstration at `demo_context` (`demo_views`, as
the method shows it), the artifacts the Actor reads beside its Task Memory when the config names an
overlay (`host_overlay`: P4's RSI document), in library mode (`memory_access`, the default) the
Memory library and its `library` toolset (`host_library`), and serves the `agent.host` plane
until SIGINT or SIGTERM. Its bearer token
is read from `--token-file`, a file only its owner can read (a launcher writes it),
else `RS_HOST_TOKEN` (the environment, then `.env`). Relative paths in the config are relative to
the repository root.
"""

import argparse
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import anyio
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from roboscript.agent.host import (
    NO_DEMO,
    ActorPrompt,
    AgentHostService,
    DemoView,
    HostConfig,
    HostLibrary,
    IdentifyHook,
    MemoryLoader,
    Opened,
    OpeningHook,
)
from roboscript.agent.overlay import OverlayLoader, overlay_loader
from roboscript.blobs import BlobStore, LocalBlobStore
from roboscript.contracts.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactLayer,
    ArtifactRef,
    ArtifactStack,
    ArtifactStore,
)
from roboscript.contracts.host import AgentHost
from roboscript.contracts.hub import EpisodeSession
from roboscript.contracts.ids import PhaseName, TaskName
from roboscript.contracts.phase import PhaseSpec
from roboscript.demos.manifest import content_sha256
from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.envs.bridge import Bridges
from roboscript.envs.cards import load_cards
from roboscript.gateway.client import openai_client
from roboscript.hub import Hub, HubConfig
from roboscript.learning.artifact import ArtifactStoreSpec, load_frozen_ref, open_artifact_store
from roboscript.learning.config import BrainSpec
from roboscript.learning.demo import PACKAGE_FILE, import_blobs, load_package
from roboscript.learning.method import Declaration, load_method
from roboscript.learning.mounts import provider, remote_toolsets
from roboscript.learning.opening import Opener, Retries
from roboscript.learning.runner import brain_maker
from roboscript.learning.runtime import (
    MethodRuntime,
    PromptSet,
    Rewrite,
    check_level,
    check_mounted,
)
from roboscript.memory import MemoryDoc, MemoryManifest, StoreSpec, Version, read_manifest
from roboscript.memory.artifact import KIND, TaskMemory, memory_ref
from roboscript.memory.library import (
    LibraryConfig,
    LibraryEntry,
    MemoryAccess,
    library_entries,
    read_index,
)
from roboscript.perception.criteria import EndTarget
from roboscript.records import EventWriter, log_in_background
from roboscript.rpc import make_app, routes, serve_while
from roboscript.settings import EnvSettings, load_config_file, read_secret, repo_root
from roboscript.tools import LocalProvider, ToolProvider
from roboscript.tools.done import load_end_targets
from roboscript.tools.provider import provider_guides
from roboscript.toolsets.library import LibraryTools

log = logging.getLogger(__name__)


class ToolsetSpec(BaseModel):
    """A toolset registered in `roboscript.toolsets`, its config and the skillsets mounted on it
    (`tools.skillsets`: the benchmark's, `Benchmark.skillsets`); `url`: the MCP endpoint of a
    server that serves it already (`python -m roboscript.integrations.mcp.serve`, which holds
    its config and names the same skillsets; `learning.mounts`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    config: dict[str, Any] = {}
    skillsets: list[str] = []
    url: str | None = Field(default=None, exclude_if=lambda url: url is None)

    @model_validator(mode="after")
    def _remote(self) -> Self:
        if self.url is not None and self.config:
            raise ValueError(f"toolset {self.name}: a remote toolset's config is its server's")
        return self


class OverlayConfig(BaseModel):
    """What the Actor reads beside its Task Memory (`agent.overlay`): `stack`, a JSON file of the
    artifact stack the method composed (`contracts.artifact.ArtifactStack`, in render order; its
    Task Memory layers are the manifest's), and by artifact kind, the store each other layer is
    loaded from (frozen, by reference)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stack: Path
    stores: dict[ArtifactKind, ArtifactStoreSpec]


class AgentHostConfig(BaseModel):
    """The Agent host's config: the manifest of frozen Memory versions and the store that
    holds them, the blob store shared with the policy module (None: `RS_BLOB_DIR`, which a stack
    sets), the port of the plane on 127.0.0.1 (0: a free one), the method whose evaluation phase
    the episodes run (`roboscript.methods` name or `package.module:Class`), the recipe whose
    prompts the Actor reads, the episodes (`HostConfig`), the toolsets mounted, the Hub, the
    brain, and how often the one-time identification is tried in place when the infrastructure
    fails it (nothing re-runs a P3 episode). `demo_context`: what the Actor reads of the
    demonstration, a level the method declares (`runtime.DemoBriefs.levels`, checked as the host
    starts) and shows its own way; none (`agent.host.NO_DEMO`), the default, is the plain
    one-shot; from the demo package `demo` (of a task of the manifest; needed unless none), as in
    S2. `done_goal`: a demo package whose final state `check_done` compares the scene with,
    whatever the Actor reads (the method's finish gate; None: `demo`'s, shown or not, else
    none). `end_targets`: a file of the demonstration's end as targets
    (`read_end_targets`' report, read on its seed under the frozen Memory's roles), which
    `check_done` also measures (`tools.done.end_for` keeps those that carry over); needs the
    package. `overlay`: the artifacts the Actor reads beside its Task Memory (None, the default:
    none; left out of a dump then). `memory_access`: `library` (the default),
    the Actor gets no Memory of its task id but the library of `memory_library` (by default every
    task of the manifest; an index file for per-task stores; exclusions) through the `library`
    toolset, and uses the Memories it chooses (`memory.library`); `bound`, the Memory of its
    task id is in its prompt. Every library episode records its access mode (`config.json`,
    `summary.json`); a record without the field was bound (`access_of`). `cards`: the benchmark's
    directory of success
    condition cards (`Benchmark.cards`, `envs.cards`), which the Actor reads under the instruction
    in either access mode (None, the default: none; left out of a dump then)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: Path
    memory: StoreSpec = StoreSpec()
    blob_dir: Path | None = None
    port: int = Field(default=0, ge=0, le=65535)
    method: str
    recipe: Path
    episodes: HostConfig
    toolsets: list[ToolsetSpec]
    hub: HubConfig = HubConfig()
    brain: BrainSpec = BrainSpec()
    identify: Retries = Retries(attempts=3, unresolved=1)
    demo_context: str = NO_DEMO
    demo: Path | None = None
    done_goal: Path | None = None
    end_targets: Path | None = None
    overlay: OverlayConfig | None = Field(default=None, exclude_if=lambda overlay: overlay is None)
    memory_access: MemoryAccess = "library"
    memory_library: LibraryConfig = LibraryConfig()
    cards: Path | None = Field(default=None, exclude_if=lambda cards: cards is None)

    @model_validator(mode="after")
    def _demo(self) -> Self:
        if (self.demo_context == NO_DEMO) != (self.demo is None):
            raise ValueError("demo names the package demo_context shows, and only then")
        if self.end_targets is not None and self.demo is None and self.done_goal is None:
            raise ValueError("end_targets are a demonstration's: name its package (done_goal)")
        if None not in (self.demo, self.done_goal) and self.demo != self.done_goal:
            raise ValueError("done_goal and demo name different packages: one task, one demo")
        return self

    def shared_blob_dir(self) -> Path:
        directory = self.blob_dir or EnvSettings().blob_dir
        if directory is None:
            raise ValueError("set blob_dir in the Agent host's config, or RS_BLOB_DIR")
        return _absolute(directory)


def host_token(token_file: Path | None = None) -> SecretStr:
    """The plane's bearer token: `token_file`'s, else `RS_HOST_TOKEN` (the environment or
    `.env`)."""
    token = read_secret(token_file) if token_file is not None else EnvSettings().host_token
    if token is None or len(token.get_secret_value()) < 16:
        where = token_file or "RS_HOST_TOKEN in .env"
        raise ValueError(f"set {where} to a random string of 16 or more characters")
    return token


def actor_prompt(
    prompts: PromptSet, profile: EmbodimentProfile, rules: str
) -> tuple[ActorPrompt, str]:
    """The host's Actor prompt from the method's prompts (`MethodRuntime.prompts`, made with no
    tools and the guides of the skillsets its toolsets mount) under `rules` (the host's phase,
    `MethodRuntime.host_phase`), as S1 and S2 render theirs, and their digest."""

    def prompt(instruction: str, memory: MemoryDoc) -> str:
        return prompts.actor(instruction=instruction, memory=memory, rules=rules, profile=profile)

    return prompt, prompts.digest


def host_phase(method: type[Declaration], runtime: MethodRuntime) -> PhaseName:
    """The phase the method runs the host's episodes in (`MethodRuntime.host_phase`), one it
    declares: else the host does not start."""
    phase = runtime.host_phase
    if phase is None:
        raise ValueError(f"method {method.name!r} names no phase for the Agent host (host_phase)")
    if phase not in {declared.name for declared in method.phases}:
        raise ValueError(f"method {method.name!r} runs the host in {phase!r}, a phase it lacks")
    return phase


def opening_hook(hub: Hub, opener: Opener, retries: Retries | None = None) -> OpeningHook:
    """The method's open-time hook (`opener`, its `MethodRuntime.opening`) as a P3 episode opens,
    as the episode driver runs it (`roboscript.learning.opening`), tried again in place after an
    infrastructure failure as `retries` allows; wicl's has nothing to do without the
    perception toolset."""

    async def hook(session: EpisodeSession, instruction: str, memory: MemoryDoc) -> Opened:
        opened = await opener.open(
            hub,
            session,
            instruction=instruction,
            memory=memory,
            source=session.episode_id,
            retries=retries,
        )
        if opened.failed:
            return Opened(end=opened.end_reason, note=opened.note)
        return Opened(brief=opened.brief())

    return hook


def identify_hook(hub: Hub, opener: Opener, retries: Retries | None = None) -> IdentifyHook:
    """The method's identification in an episode that runs (`Opener.identify`), for the roles of
    a Memory a library Actor adds, tried again in place as `retries` allows; a failure is
    reported, never ends the episode."""

    async def hook(session: EpisodeSession, instruction: str, memory: MemoryDoc) -> Opened:
        found = await opener.identify(
            hub, session, instruction=instruction, memory=memory, retries=retries
        )
        if found.failed:
            return Opened(end=found.end_reason, note=found.note)
        return Opened(brief=found.brief())

    return hook


LIBRARY_ENTRY: Final = "library.*"


def with_library(phases: Sequence[PhaseSpec], phase: PhaseName) -> list[PhaseSpec]:
    """The method's phases with `library.*` last in the Actor's list of the host's `phase`: only a
    host in library mode mounts the `library` toolset and sees it, so the method's tables, the
    catalog lock and every bound, S1 and S2 run do not depend on it."""
    out = []
    for spec in phases:
        if spec.name == phase:
            actor = [*spec.visibility.get("actor", []), LIBRARY_ENTRY]
            seen = {**spec.visibility, "actor": actor}
            spec = spec.model_copy(update={"visibility": seen})
        out.append(spec)
    return out


def access_of(config: Mapping[str, Any]) -> MemoryAccess:
    """How a recorded host config (`agent_host.yaml`, an episode's `config.json`) gave the Actor
    its Memory: a record without `memory_access` was bound."""
    return "library" if config.get("memory_access") == "library" else "bound"


def toolset_provider(spec: ToolsetSpec, profile: EmbodimentProfile | None = None) -> ToolProvider:
    """`spec`'s toolset, mounted for `profile`'s robot (None: no fit check)."""
    return provider(spec.name, spec.config, spec.skillsets, spec.url, profile)


async def serve_host(config: AgentHostConfig, token: SecretStr, ready_file: Path | None) -> None:
    """Serve the Agent host until SIGINT or SIGTERM (or cancellation): the ready file as soon as
    it listens, `/readyz` once the service has warmed up."""
    episodes = config.episodes.model_copy(
        update={
            "records": _absolute(config.episodes.records),
            "embodiment": _embodiment(config.episodes.embodiment),
        }
    )
    profile = load_embodiment(episodes.embodiment)
    method = load_method(config.method)
    runtime = method.runtime(_absolute(config.recipe))
    phase = host_phase(method, runtime)
    providers = [toolset_provider(spec, profile) for spec in config.toolsets]
    prompts = runtime.prompts({}, provider_guides(providers))
    prompt, digest = actor_prompt(prompts, profile, phase)
    manifest = read_manifest(_absolute(config.manifest))
    cards = {} if config.cards is None else load_cards(_absolute(config.cards))
    artifacts = open_artifact_store(KIND, config.memory.model_dump())
    brain = brain_maker(config.brain)
    blobs = LocalBlobStore(config.shared_blob_dir())
    demos = await demo_views(config, manifest.tasks, blobs, profile, runtime)
    stack, overlay = host_overlay(config)
    library, phases = None, list(method.phases)
    if config.memory_access == "library":
        tools = LibraryTools()
        providers = [*providers, LocalProvider(tools)]
        phases = with_library(phases, phase)
        library = await host_library(
            config, manifest, artifacts, runtime, prompts, profile, phase, tools
        )
        write_library(episodes.records / "library.json", library)
    bridges = Bridges()
    log_file = episodes.records / "events.jsonl"  # appended off the loop: the disk may stall
    with EventWriter(log_file, "agent_host", background=True) as events:
        hub = Hub(
            providers,
            blobs=blobs,
            phases=phases,
            config=config.hub,
            events=events,
            env_client=bridges.client,
            models=openai_client,
        )
        service = AgentHostService(
            episodes,
            hub=hub,
            bridges=bridges,
            blobs=blobs,
            events=events,
            load=frozen_memory(artifacts, runtime.frozen),
            manifest=manifest,
            brain=brain,
            prompt=prompt,
            phase=phase,
            prompt_digest=digest,
            models=openai_client,
            opening=opening_hook(hub, runtime.opening, config.identify),
            finish=runtime.finish.host,
            demos=demos,
            stack=stack,
            overlay=overlay,
            library=library,
            identify=identify_hook(hub, runtime.opening, config.identify),
            cards=cards,
        )
        async with hub, remote_toolsets(hub, providers), service:
            check_mounted(runtime, [p for p in phases if p.kind == "eval"], hub.mounted)
            app = make_app(routes(AgentHost, service, token=token), ready=lambda: service.ready)
            log.info("Agent host for manifest %s, records in %s", manifest.id, episodes.records)
            name = "the Agent host"
            await serve_while(
                app, service.start, name=name, port=config.port, ready_file=ready_file
            )


async def host_library(
    config: AgentHostConfig,
    manifest: MemoryManifest,
    artifacts: ArtifactStore,
    runtime: MethodRuntime,
    prompts: PromptSet,
    profile: EmbodimentProfile,
    phase: PhaseName,
    tools: LibraryTools,
) -> HostLibrary:
    """Library mode's library (`config.memory_library`): the frozen versions its index names,
    each from its own store, else the manifest's from the host's; each still frozen
    (`load_frozen_ref`) and read as the method's Actor reads a frozen Memory
    (`MethodRuntime.frozen`); the Actor's prompt is the method's (`prompts.library_actor`) under
    the host's `phase`. An excluded task the library lacks is an error."""
    spec = config.memory_library
    stored: list[tuple[str, Version]] = []
    if spec.index is None:
        for task, entry in manifest.tasks.items():
            loaded = await load_frozen_ref(artifacts, memory_ref(task, entry))
            assert isinstance(loaded, TaskMemory)
            stored.append((task, loaded.version))
    else:
        stores: dict[str, ArtifactStore] = {}
        for item in read_index(_absolute(spec.index)).entries:
            key = item.store.model_dump_json()
            if key not in stores:
                stores[key] = open_artifact_store(KIND, item.store.model_dump())
            ref = memory_ref(item.task, item.manifest_entry)
            loaded = await load_frozen_ref(stores[key], ref)
            assert isinstance(loaded, TaskMemory)
            stored.append((item.task, loaded.version))
    entries = library_entries(
        (task, version, runtime.frozen(version)[0]) for task, version in stored
    )
    if unknown := sorted(set(spec.exclude) - {e.task for e in entries}):
        raise ValueError(f"memory_library.exclude names tasks the library lacks: {unknown}")
    check_size(entries, spec, manifest.tasks)

    def prompt(instruction: str) -> str:
        return prompts.library_actor(instruction=instruction, rules=phase, profile=profile)

    return HostLibrary(
        entries=entries,
        mount=tools,
        prompt=prompt,
        exclude=frozenset(spec.exclude),
        exclude_own=spec.exclude_own,
        digest=prompts.library_digest,
    )


MIN_ENTRIES: Final = 2
"""A library of fewer entries is the bound mode in effect: a warning."""


def check_size(entries: Sequence[LibraryEntry], spec: LibraryConfig, tasks: Iterable[str]) -> None:
    """Refuse a library an episode of one of `tasks` would see empty (`exclude`, and with
    `exclude_own` its own task's left out); warn when one would see fewer than `MIN_ENTRIES`, as a
    library built from one task's manifest does (build it across the benchmark's tasks:
    `python -m roboscript.memory.library`)."""
    kept = [e for e in entries if e.task not in spec.exclude]
    for task in tasks:
        seen = [e for e in kept if not (spec.exclude_own and e.task == task)]
        if not seen:
            raise ValueError(f"{task}: its episodes would see an empty Memory library")
        if len(seen) < MIN_ENTRIES:
            log.warning(
                "%s: its episodes see %d library entries (%s): build the library across the "
                "benchmark's tasks",
                task,
                len(seen),
                ", ".join(e.task for e in seen),
            )


def write_library(path: Path, library: HostLibrary) -> None:
    """The library as the host loaded it, cards included (`<records>/library.json`): which
    entry is which task's version, for the records' readers."""
    data = {
        "exclude": sorted(library.exclude),
        "exclude_own": library.exclude_own,
        "entries": [_entry(e) for e in library.entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _entry(entry: LibraryEntry) -> dict[str, Any]:
    return {**entry.record(), "card": entry.card.model_dump(mode="json")}


def frozen_memory(
    artifacts: ArtifactStore, rewrite: Callable[[Version], tuple[Version, Rewrite]]
) -> MemoryLoader:
    """P3's loader: a task's one Task Memory layer, still its frozen version, as the method's
    Actor reads it (`rewrite`, its `MethodRuntime.frozen`: wicl's rewrites its arm
    bindings by role, `unbind_arms`; records name the rewrite's hash)."""

    async def load(task: TaskName, layers: Sequence[ArtifactLayer]) -> Version:
        if len(layers) != 1 or layers[0].ref.kind != KIND:
            shown = ", ".join(f"{layer.ref.kind} {layer.ref.name}" for layer in layers) or "none"
            raise ValueError(f"{task}: P3 reads exactly one Task Memory, not {shown}")
        memory = await load_frozen_ref(artifacts, layers[0].ref)
        assert isinstance(memory, TaskMemory)  # a task_memory store loads Task Memories
        version, changed = rewrite(memory.version)
        if changed.changes:
            log.info("%s: unbind_arms rewrote %d fields", task, len(changed.changes))
        return version

    return load


def host_overlay(config: AgentHostConfig) -> tuple[ArtifactStack | None, OverlayLoader | None]:
    """The stack the config's overlay names and the loader of its layers beside the Task Memory,
    each from the store of its kind, still its frozen version (`load_frozen_ref`); (None, None)
    without an overlay. A kind the stack uses without a store, or a store of another kind than it
    is named for, is an error here, before any episode."""
    if config.overlay is None:
        return None, None
    path = _absolute(config.overlay.stack)
    stack = ArtifactStack.model_validate_json(path.read_text(encoding="utf-8"))
    stores = {
        kind: open_artifact_store(spec.name, spec.options)
        for kind, spec in config.overlay.stores.items()
    }
    if wrong := sorted(kind for kind, store in stores.items() if store.kind != kind):
        raise ValueError(f"overlay stores named for {wrong} hold other kinds")
    kinds = {layer.ref.kind for layer in stack.layers} - {KIND}
    if missing := sorted(kinds - set(stores)):
        raise ValueError(f"{path}: no overlay store for its {missing} layers")

    async def load(ref: ArtifactRef) -> Artifact:
        return await load_frozen_ref(stores[ref.kind], ref)

    return stack, overlay_loader(load, KIND)


async def demo_views(
    config: AgentHostConfig,
    tasks: Sequence[TaskName],
    blobs: BlobStore,
    profile: EmbodimentProfile,
    runtime: MethodRuntime,
) -> dict[TaskName, DemoView]:
    """The Actor's view of its task's demonstration at `config.demo_context` (as the method
    shows it: its `MethodRuntime.demo`), the package's blobs put into `blobs` (where its brief,
    frames and final-state views resolve), with what `check_done` gates on, the package's
    (`done_goal`'s, else `demo`'s, shown or not), as the method reads its end
    (`MethodRuntime.goal`, read with `profile` and the end targets of `config.end_targets`);
    none without a package. A level the method does not declare is an error (`check_level`)."""
    check_level(runtime.demo, config.demo_context)
    named = config.demo or config.done_goal
    if named is None:
        return {}
    path = _absolute(named)
    package = load_package(path)
    if package.task not in tasks:
        raise ValueError(f"{path}: its package is of {package.task}, not a task of the manifest")
    await import_blobs(path, blobs)
    level = config.demo_context
    digest = f"{path} (sha256 {content_sha256(path / PACKAGE_FILE)})"
    label = {"demo_context": level}
    if config.demo is not None:
        label["demo_package"] = digest
    if config.done_goal is not None:
        label["done_goal"] = digest
    brief = runtime.demo.brief(package, level)
    end: list[EndTarget] = []
    if config.end_targets is not None:
        where = _absolute(config.end_targets)
        end = load_end_targets(where)
        label["end_targets"] = f"{where} (sha256 {content_sha256(where)})"
    goal = await runtime.goal(package, blobs, profile, end)
    view = DemoView(brief=brief, seed=runtime.demo.seed(package, level), label=label, goal=goal)
    return {package.task: view}


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def _embodiment(ref: str) -> str:
    """A profile's YAML path made absolute; a registered name as it is."""
    return str(_absolute(Path(ref))) if ref.endswith((".yaml", ".yml")) else ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Agent host (P3 official evaluation).")
    parser.add_argument("--config", type=Path, required=True, help="an AgentHostConfig YAML file")
    parser.add_argument("--token-file", type=Path, help="the bearer token (else RS_HOST_TOKEN)")
    parser.add_argument("--ready-file", type=Path, help="where to write the plane's URL")
    args = parser.parse_args()
    log_in_background(logging.INFO, "%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # a line per poll otherwise
    config = load_config_file(args.config, AgentHostConfig)
    anyio.run(serve_host, config, host_token(args.token_file), args.ready_file)


if __name__ == "__main__":
    main()
