"""An episode's frame store (docs/concepts/perception.md, docs/concepts/agent-runtime.md). The Hub
keeps one per episode and hands it to every toolset call as `ToolContext.frames`.

Frames are immutable observations: a toolset `put`s each one it takes (stored as a JSON blob,
logged as one `env.observe` event, registered under the alias models will see, `frame_3`), and
any toolset `get`s one by frame id or alias. The store also keeps the episode's scene facts by
name, such as perception's table estimate (`TABLE`) and the open-time identification
(`IDENTIFIED`), each one logged as a `scene.fact` event, and the episode's scene registry.

The toolsets of an episode (perception, geometry, worldstate, manipulation) share its registry.
`registry()` hands out a copy of it as it is now, which never changes under its reader; a change
goes through `edit_registry()`, which holds the episode's one lock while it loads the registry
(a newer `REGISTRY` fact first, e.g. one a remote toolset saved), applies the change and saves the
result as the next version of the `REGISTRY` fact (its JSON in the blob store), for the log, for
a seeded episode and for remote toolsets. A remote toolset works on a copy of the store
(`snapshot()`), and what it set comes back through `merge`: all of it, or, when it changed the
registry and the registry here changed since the copy was made, none, logged (`scene.stale`),
and the call fails with a tool error that asks the model to call again, so that no result claims
a change that was not kept. `derived` holds what the toolsets of this process derive per episode
and share (analyses, proposals); it is not logged.

An episode may start with another's frames, facts and aliases (`EpisodeSeed`; see
`seed_from_log`), so that a reviewing role can look at the Actor's `frame_3` and objects, or with
several earlier episodes' (the tries of a seed, `seed_from_logs`) under try-qualified frame
aliases after the frame-source naming `log:<try>:<turn>` (`log:2:3.2`: the second frame of try
2's turn 3), with each try's registry at its end as the fact `registry@<try>`, which
`registry(frame)` reads for that try's frames. Each `env.observe` event records the turn its
frame was taken in.
"""

import re
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Final, NoReturn

import anyio
from pydantic import BaseModel

from roboscript.blobs import BlobStore
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import ToolError
from roboscript.contracts.events import Event, EventKind
from roboscript.contracts.hub import EpisodeSeed, SeedAlias
from roboscript.contracts.ids import EpisodeId, FrameId
from roboscript.contracts.observation import FrameEntry, Observation
from roboscript.perception.registry import SceneRegistry

LOG_ALIAS: Final = re.compile(r"log:[1-9][0-9]*:[0-9]+(?:\.(?:[2-9]|[1-9][0-9]+))?")
"""A try-qualified frame alias, `log:<try>:<turn>[.<n>]` (`seed_from_logs`)."""
DEMO_ALIAS: Final = re.compile(r"demo:[1-9][0-9]*")
"""A demonstration's frame shown to an Actor, `demo:<n>` (`demo_seed`): apart from the
episode's own `frame_<n>` and from the tries' `log:` frames it may also be seeded with."""

TABLE: Final = "table"
"""The table estimate: `{"z": <the plane's height at the world origin>, "extent": [x0, x1, y0,
y1]}`, meters in the world frame (perception stores its whole `Table`, which has both)."""
REGISTRY: Final = "registry"
"""The scene registry: `{"blob": <BlobRef of its JSON>, "objects": [<object ids>], "version":
<the save it is, from 1>}`; `registry@<try>` is a try's, in a reviewing role's store."""
DEMO: Final = "demo"
"""The demonstration a demo-reading role and an imitating Actor study: its keypoints
(`KeypointSet` JSON, TCP poses included), which `demo_relation` reads and no model is shown; the
runner seeds it with the recording's frames."""
DEMO_ARMS: Final = "demo_arms"
"""Which arm the demonstration grasped with, where: `{"grasps": [{"arm": "left", "keypoint":
"left K2", "tcp": [x, y, z]}, ...]}` in frame order (its closes; recorded proprioception, no
object ground truth). A method seeds it into every episode on the demonstration's scene, so that
`plan_arms` gives the demonstration's arms there (`toolsets.manipulation.arms.demo_arm`)."""
DEMO_TRACKS: Final = "demo_tracks"
"""The demonstration's recorded proprioception at every frame: `{"blob": <BlobRef of its
`learning.demo.DemoTracks` JSON>}`, which fitting a written skill reads (`tools.skill_fit`); a
method seeds it where the package has it."""
DEMO_FITS: Final = "demo_fits"
"""The specs `demo_relation` fitted in this episode: `{"<arm> K<n>": [<spec JSON>, ...]}`, which
the memory toolset checks a fitted spec's definition against."""
DEMO_RELEASES: Final = "demo_releases"
"""Where this episode measured the objects the demonstration let go of, again on its frames
(wicl's `releases.demo_releases`): `{"objects": {<id>: <first pose>}, "tried":
[[<open frame>, <id>], ...], "poses": {"<open frame>": {<id>: <pose>}}}`, for the replays of its
grips (wicl's `relation.replay_grips`); `objects` is the registry it was made for, and a pose
holds for an object while the registry has it with that first pose (`relation.released_poses`)."""
ARM_SIDES: Final = "arm_sides"
"""The Memory's `arm side:` lines (`memory.objects.ARM_SIDES`): `{"roles": {<role>: "toward its
x" | "away from its x" | "either"}}`; the runner seeds it with `MEMORY_SPECS` (`spec_seed`), so
that `plan_arms` gives the arm on the side the role's frame x points to (or away from it)."""
MEMORY_SPECS: Final = "memory_specs"
"""The specs of the Memory an Actor reads, by parameter name: `{"specs": {<name>: {"kind":
"grasp_spec" | "place_spec" | "hinge_spec", "value": <spec JSON>}}, "stage_arms": {<role>:
[<arm>, ...]}, "chains": <memory.chains.Chains JSON>, "numbers": {<name>: <value>}}`
(`stage_arms`: the roles its workflow grasps with both arms in turn; `chains`: its holds, which
`plan_arms` assigns arms to; `numbers`: its number parameters, which `move_by` takes by
name); the runner seeds it in every Actor episode with a Memory (`spec_seed`), so that a call
naming a spec (`{name}`) runs it with all its stored fields (`toolsets.manipulation`)."""
ARM_PLAN: Final = "arm_plan"
"""The episode's arm plan (`plan_arms`): `{"stages": {<stage>: [<arm>, ...]}, "roles": {<role>:
[<arm of each of its holds, in order>]}, "holds": [{"role", "arm", "stages", "why"}, ...],
"override": <the Agent's reason or null>}`; the last plan made wins (P3 may plan again after a
hard failure), and a tool resolving `<arm of <role>>` reads the role's newest hold."""
IDENTIFIED: Final = "identified"
"""The open-time identification: `{"frame": <frame id>, "roles": {<role>: {"object": <object
id or null>, "reason": ..., "candidates": [<region ids>]}}}` (`roboscript.toolsets.perception`)."""

DONE: Final = "done"
"""The Memory's done criteria that `check_done` measures: `{"criteria": [<DoneSpec JSON>, ...]}`
(wicl's `perception.done`); the runner seeds it where the method gates `finish` on it."""

REGISTRY_CHANGED: Final = (
    "Another call changed the scene registry while this one ran, so nothing this call changed "
    "was kept; call it again."
)
"""The tool error of a remote call whose registry change lost the race (`FrameStore.merge`)."""

_JSON: Final = "application/json"

Emit = Callable[[str, dict[str, Any]], None]


class FrameStore:
    """One episode's frames and scene facts. `alias(frame_id)` registers a frame id in the
    episode's alias table and returns its alias; `emit(kind, data)` logs an event; `turn()`, if
    given, is the episode's current turn, logged with each observation."""

    def __init__(
        self,
        blobs: BlobStore,
        alias: Callable[[FrameId], str],
        emit: Emit,
        seed: EpisodeSeed | None = None,
        turn: Callable[[], int] | None = None,
    ) -> None:
        self._blobs = blobs
        self._alias = alias
        self._emit = emit
        self._turn = turn
        self._entries: dict[FrameId, FrameEntry] = {}
        self._ids: dict[str, FrameId] = {}  # alias -> frame id
        self._observations: dict[FrameId, Observation] = {}
        self._facts: dict[str, dict[str, Any]] = {}
        self._live: FrameId | None = None  # the newest frame this episode took
        self._registry: SceneRegistry | None = None
        self._version = 0  # the REGISTRY version `_registry` holds (0: none saved yet)
        self._lock = anyio.Lock()
        self._editor: int | None = None  # the task in `edit_registry`
        self._tried: dict[str, SceneRegistry] = {}  # a `registry@<try>` blob id -> its registry
        self.derived: dict[Any, Any] = {}
        if seed is not None:
            for entry in seed.frames:
                self._add(entry)
            self._facts.update(seed.facts)
            self._live = seed.live

    async def put(self, observation: Observation) -> FrameEntry:
        """Keep a new observation."""
        data = observation.model_dump_json().encode()
        blob = await self._blobs.aput_bytes(data, _JSON, meta={"kind": "observation"})
        frame_id = observation.frame_id
        entry = FrameEntry(
            frame_id=frame_id, alias=self._alias(frame_id), step=observation.step, observation=blob
        )
        self._add(entry)
        self._observations[frame_id] = observation
        self._live = frame_id
        turn = {} if self._turn is None else {"turn": self._turn()}
        self._emit(EventKind.ENV_OBSERVE, {**entry.model_dump(mode="json"), **turn})
        return entry

    async def get(self, frame: str) -> Observation:
        """The observation of a frame id or alias; LookupError if the episode has no such frame."""
        entry = self._entries.get(self._ids.get(frame, FrameId(frame)))
        if entry is None:
            raise LookupError(f"no frame {frame!r} in this episode")
        if entry.frame_id not in self._observations:
            data = await self._blobs.aget_bytes(entry.observation)
            self._observations[entry.frame_id] = Observation.model_validate_json(data)
        return self._observations[entry.frame_id]

    def entries(self) -> list[FrameEntry]:
        """The frames in the order they came: seeded ones, then those put."""
        return list(self._entries.values())

    def live(self) -> FrameEntry | None:
        """The newest frame the episode took itself; never a seeded one (a demonstration's)."""
        return None if self._live is None else self._entries[self._live]

    def alias_of(self, frame: str) -> str | None:
        """The alias of a frame of the store, given by id or alias (None: no such frame)."""
        entry = self._entries.get(self._ids.get(frame, FrameId(frame)))
        return None if entry is None else entry.alias

    def snapshot(self) -> EpisodeSeed:
        """The frames and facts now, for a store in another process (a remote toolset's)."""
        return EpisodeSeed(frames=self.entries(), facts=dict(self._facts), live=self._live)

    def fact(self, name: str) -> dict[str, Any] | None:
        return self._facts.get(name)

    def set_fact(self, name: str, value: BaseModel | Mapping[str, Any]) -> None:
        """Set a scene fact (plain JSON, e.g. `TABLE`); the last one set wins. The `REGISTRY`
        fact changes with the registry only (`edit_registry`, `merge`)."""
        if name == REGISTRY:
            raise ValueError("the scene registry changes through edit_registry")
        self._set(name, value.model_dump(mode="json") if isinstance(value, BaseModel) else value)

    def drop_fact(self, name: str) -> None:
        """Remove a scene fact, if set (a `scene.fact` event with value null says so)."""
        if name == REGISTRY:
            raise ValueError("the scene registry changes through edit_registry")
        if self._facts.pop(name, None) is not None:
            self._emit(EventKind.SCENE_FACT, {"name": name, "value": None})

    async def merge(self, facts: Mapping[str, Mapping[str, Any]], base: EpisodeSeed) -> None:
        """Keep the facts that a copy of this store made from `base` (`snapshot()`) set in
        another process (a remote toolset's call), all or none. None when they include the
        registry and the registry here changed since `base`: the copy changed an outdated
        registry, so keeping it would undo that change, and dropping it silently would leave the
        call's result claiming a change that was not kept. That is logged (`scene.stale`) and
        raises `ToolError(REGISTRY_CHANGED)`, which asks the model to call again. An edit in
        progress in another task finishes first; one in this task counts as a change."""
        registry = facts.get(REGISTRY)
        if registry is None:
            for name, value in facts.items():
                self._set(name, value)
            return
        if self._editing():
            self._stale(registry, base)
        async with self._lock:
            if _version(self._facts.get(REGISTRY)) != _version(base.facts.get(REGISTRY)):
                self._stale(registry, base)
            for name, value in facts.items():
                self._set(name, value)

    async def registry(self, frame: str | None = None) -> SceneRegistry:
        """A copy of the episode's scene registry as it is now (empty at the start, else the
        state the `REGISTRY` fact holds); for a frame of an earlier try whose registry the store
        was seeded with (`registry@<try>`), that try's. The copy is the reader's: nothing
        changes it, and changing it changes nothing (use `edit_registry`)."""
        if frame is not None and (fact := self._try_fact(frame)) is not None:
            return await self._tried_registry(fact)
        if self._editing():
            assert self._registry is not None
            return self._registry.clone()
        async with self._lock:
            return (await self._load()).clone()

    @asynccontextmanager
    async def edit_registry(self) -> AsyncIterator[SceneRegistry]:
        """The episode's registry, to change: under the episode's one lock it is loaded (a newer
        `REGISTRY` fact first), changed by the caller and, if changed, saved as the fact's next
        version. Within the task editing it, `registry()` and a nested `edit_registry` see it
        as edited. An edit that raises is not saved: the next load starts from the fact."""
        if self._editing():
            assert self._registry is not None
            yield self._registry
            return
        async with self._lock:
            self._editor = _task()
            try:
                registry = await self._load()
                before, counter = dict(registry.objects), registry.counter
                try:
                    yield registry
                except BaseException:
                    self._registry = None  # maybe half-changed: reload from the fact
                    raise
                changed = registry.counter != counter or registry.objects.keys() != before.keys()
                if changed or any(registry.objects[k] is not before[k] for k in before):
                    await self._save(registry)
            finally:
                self._editor = None

    async def _load(self) -> SceneRegistry:
        fact = self._facts.get(REGISTRY)
        version = _version(fact)
        if self._registry is None or version > self._version:
            if fact is None:
                self._registry = SceneRegistry()
            else:
                data = await self._blobs.aget_bytes(BlobRef.model_validate(fact["blob"]))
                self._registry = SceneRegistry.model_validate_json(data)
            self._version = version
        return self._registry

    async def _save(self, registry: SceneRegistry) -> None:
        data = registry.to_json().encode()
        blob = await self._blobs.aput_bytes(data, _JSON, meta={"kind": "scene_registry"})
        version = max(self._version, _version(self._facts.get(REGISTRY))) + 1
        self._version = version
        fact = {
            "blob": blob.model_dump(mode="json"),
            "objects": sorted(registry.objects),
            "version": version,
        }
        self._set(REGISTRY, fact)

    def _set(self, name: str, value: Mapping[str, Any]) -> None:
        data = dict(value)
        self._facts[name] = data
        self._emit(EventKind.SCENE_FACT, {"name": name, "value": data})

    def _stale(self, registry: Mapping[str, Any], base: EpisodeSeed) -> NoReturn:
        stale = {
            "name": REGISTRY,
            "version": _version(registry),
            "base": _version(base.facts.get(REGISTRY)),
            "kept": _version(self._facts.get(REGISTRY)),
        }
        self._emit(EventKind.SCENE_STALE, stale)
        raise ToolError(REGISTRY_CHANGED)

    def _editing(self) -> bool:
        return self._editor is not None and self._editor == _task()

    def _try_fact(self, frame: str) -> dict[str, Any] | None:
        alias = self.alias_of(frame)
        if alias is None or LOG_ALIAS.fullmatch(alias) is None:
            return None
        return self._facts.get(f"{REGISTRY}@{alias.split(':')[1]}")

    async def _tried_registry(self, fact: dict[str, Any]) -> SceneRegistry:
        blob = BlobRef.model_validate(fact["blob"])
        if blob.id not in self._tried:
            data = await self._blobs.aget_bytes(blob)
            self._tried[blob.id] = SceneRegistry.model_validate_json(data)
        return self._tried[blob.id].clone()

    def _add(self, entry: FrameEntry) -> None:
        self._entries[entry.frame_id] = entry
        self._ids[entry.alias] = entry.frame_id


def seed_from_log(events: Iterable[Event], episode_id: EpisodeId) -> EpisodeSeed:
    """The frames an episode kept (its seed's too), its last scene facts and the aliases it
    showed, from its log (e.g. the merged `events.jsonl`), to open another episode with."""
    frames: list[FrameEntry] = []
    facts: dict[str, dict[str, Any]] = {}
    aliases: dict[str, SeedAlias] = {}
    for event in events:
        if event.episode_id != episode_id:
            continue
        if event.kind == EventKind.EPISODE_OPEN and event.data.get("seed"):
            seeded = EpisodeSeed.model_validate(event.data["seed"])
            frames += seeded.frames
            facts.update(seeded.facts)
            aliases.update({given.alias: given for given in seeded.aliases})
        elif event.kind == EventKind.ENV_OBSERVE:
            frames.append(FrameEntry.model_validate(event.data))
        elif event.kind == EventKind.SCENE_FACT:
            facts[event.data["name"]] = event.data["value"]
        elif event.kind == EventKind.HUB_ALIAS and event.data["kind"] != "frame":
            made = SeedAlias.model_validate(event.data)
            aliases[made.alias] = made
    return EpisodeSeed(frames=frames, facts=facts, aliases=list(aliases.values()))


def demo_alias(alias: str) -> str:
    """The `demo:<n>` alias of a demonstration's frame `frame_<n>`."""
    return "demo:" + alias.removeprefix("frame_")


def log_alias(attempt: int, turn: int, n: int = 1) -> str:
    """The alias of the `n`-th frame taken in turn `turn` of try `attempt`."""
    return f"log:{attempt}:{turn}" + (f".{n}" if n > 1 else "")


def seed_from_logs(
    logs: Iterable[tuple[int, Iterable[Event], EpisodeId]],
) -> tuple[EpisodeSeed, dict[int, dict[str, str]]]:
    """The frames of several earlier episodes (`(try, events, episode id)`: the tries of a seed),
    aliased `log:<try>:<turn>` by the turn each was taken in (turn 0 when its event does not
    say; a frame a try was seeded with keeps the alias of the try that took it), the last one's
    scene facts, each try's registry at its end (`registry@<try>`), and the aliases of objects
    and regions the tries showed (one that named different things in two tries is left out);
    with each try's own frame aliases mapped to these."""
    frames: list[FrameEntry] = []
    facts: dict[str, dict[str, Any]] = {}
    tried: dict[str, dict[str, Any]] = {}
    renamed: dict[int, dict[str, str]] = {}
    named: dict[FrameId, str] = {}  # frame id -> its log alias
    shown: dict[str, SeedAlias | None] = {}  # alias -> what it names (None: named differently)
    for attempt, events, episode_id in logs:
        taken: Counter[int] = Counter()
        renamed[attempt], facts = {}, {}
        for event in events:
            if event.episode_id != episode_id:
                continue
            if event.kind == EventKind.EPISODE_OPEN and event.data.get("seed"):
                seeded = EpisodeSeed.model_validate(event.data["seed"])
                for entry in seeded.frames:
                    if entry.frame_id in named:
                        renamed[attempt][entry.alias] = named[entry.frame_id]
                facts.update(seeded.facts)
                for given in seeded.aliases:
                    _show(shown, given)
            elif event.kind == EventKind.ENV_OBSERVE:
                entry = FrameEntry.model_validate(event.data)
                turn = int(event.data.get("turn", 0))
                taken[turn] += 1
                alias = renamed[attempt][entry.alias] = log_alias(attempt, turn, taken[turn])
                named[entry.frame_id] = alias
                frames.append(entry.model_copy(update={"alias": alias}))
            elif event.kind == EventKind.SCENE_FACT:
                facts[event.data["name"]] = event.data["value"]
            elif event.kind == EventKind.HUB_ALIAS and event.data["kind"] != "frame":
                _show(shown, SeedAlias.model_validate(event.data))
        if REGISTRY in facts:
            tried[f"{REGISTRY}@{attempt}"] = facts[REGISTRY]
    aliases = [given for given in shown.values() if given is not None]
    return EpisodeSeed(frames=frames, facts={**facts, **tried}, aliases=aliases), renamed


def _show(shown: dict[str, SeedAlias | None], given: SeedAlias) -> None:
    if given.alias not in shown:
        shown[given.alias] = given
    elif shown[given.alias] != given:
        shown[given.alias] = None


def _version(fact: Mapping[str, Any] | None) -> int:
    return 0 if fact is None else int(fact.get("version", 0))


def _task() -> int:
    return anyio.get_current_task().id
