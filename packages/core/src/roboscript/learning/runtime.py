"""What a method hands the core's episode driver and the Agent host to run its episodes with
(docs/how-to/add-a-method.md).

A method builds one `MethodRuntime` (`Declaration.runtime`), held by the `ProtocolRunner` (and
its `EpisodeDriver`) and by the Agent host, constructor-injected like any other dependency: the
open-time hook (`opening`), the finish gate (`finish`), the roles' prompts (`prompts`), what
an Actor reads of its demonstration (`demo`), the reading of a try's ground truth (`explain`),
the hook on a judged try (`judged`), what an evaluated Actor reads of a frozen Memory
(`frozen`), the demonstration's end the done check compares with (`goal`) and the checks of a
model's Memory edits (`edit_checks`). The core runs them at fixed points; what they do is the
method's. An episode's own settings for these hooks travel in `Episode.method`, which the core
carries without reading.
"""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from roboscript.agent.host import NO_DEMO, DemoView, OpaqueGoal
from roboscript.blobs import BlobStore
from roboscript.contracts.env import Diagnosis, EnvControl
from roboscript.contracts.hub import EpisodeSeed
from roboscript.contracts.ids import EpisodeId, PhaseName, ToolName
from roboscript.contracts.phase import PhaseRules, PhaseSpec
from roboscript.contracts.robot import RobotState
from roboscript.contracts.tools import ContentPart
from roboscript.demos import ObjectShape
from roboscript.embodiments import EmbodimentProfile
from roboscript.hub import FinishGuard, Hub
from roboscript.learning.demo import DemoPackage
from roboscript.learning.opening import NoOpening, Opener
from roboscript.memory import MemoryDoc, ToolSig, Version
from roboscript.perception.criteria import EndTarget
from roboscript.perception.registry import SceneRegistry
from roboscript.toolsets.memory import EditCheck

if TYPE_CHECKING:
    from roboscript.learning.episode import Episode


class FinishGate(Protocol):
    """What gates a role's `finish`, as its episode opens: `episode` gives a learning episode's
    rules and seed (with the facts its done check reads) and whether its finish is gated, from
    the episode's own (`rules`, `seed`) and the tools the Hub has (`mounted`); `guard`, the
    finish guard set on an episode with an env that brings none of its own
    (`Episode.finish_guard`; None: none), with the env's control plane; `unconfirmed`, once it
    ended, the end reason a finish the env's success check did not confirm gets, a failure
    (`fail_fast`; None: the finish stands); `host`, an evaluation episode's rules and seed from
    the frozen Memory and the task's demonstration."""

    def episode(
        self,
        episode: "Episode",
        rules: PhaseRules | None,
        seed: EpisodeSeed | None,
        *,
        instruction: str | None,
        actor: bool,
        mounted: Collection[ToolName],
    ) -> tuple[PhaseRules | None, EpisodeSeed | None, bool]: ...

    def guard(
        self, episode: "Episode", control: EnvControl, env_episode: EpisodeId
    ) -> FinishGuard | None: ...

    def unconfirmed(self, episode: "Episode", gated: bool) -> str | None: ...

    def host(
        self,
        doc: MemoryDoc,
        rules: PhaseRules | None,
        demo: DemoView | None,
        instruction: str,
        positional: bool,
    ) -> tuple[PhaseRules | None, EpisodeSeed | None]: ...


class NoFinishGate:
    """The finish gate of a method without one: rules and seed as given, no finish refused."""

    def episode(
        self,
        episode: "Episode",
        rules: PhaseRules | None,
        seed: EpisodeSeed | None,
        *,
        instruction: str | None,
        actor: bool,
        mounted: Collection[ToolName],
    ) -> tuple[PhaseRules | None, EpisodeSeed | None, bool]:
        return rules, seed, False

    def guard(
        self, episode: "Episode", control: EnvControl, env_episode: EpisodeId
    ) -> FinishGuard | None:
        return None

    def unconfirmed(self, episode: "Episode", gated: bool) -> str | None:
        return None

    def host(
        self,
        doc: MemoryDoc,
        rules: PhaseRules | None,
        demo: DemoView | None,
        instruction: str,
        positional: bool,
    ) -> tuple[PhaseRules | None, EpisodeSeed | None]:
        return rules, None if demo is None else demo.seed


class PromptSet(Protocol):
    """A method's role prompts, as the core renders an Actor's: `digest`, the prompts' version
    (in a run's `config.json` and each episode's records); `actor`, the Actor's system prompt
    under `rules` (the method's name for the stage's rules: the core passes its phase's name,
    `MethodRuntime.evaluate_phase` or `host_phase`; wicl's own protocols pass theirs).
    The method's other roles read theirs through its own protocols."""

    @property
    def digest(self) -> str: ...

    def actor(
        self, *, instruction: str, memory: MemoryDoc, rules: str, profile: EmbodimentProfile
    ) -> str: ...

    def library_actor(self, *, instruction: str, rules: str, profile: EmbodimentProfile) -> str:
        """The Actor's system prompt in the host's library mode (`memory.library`): no Task Memory,
        a library of them to read and use through the `library` toolset."""
        ...

    @property
    def library_digest(self) -> str:
        """The version of what `library_actor` adds (its template), for `config.json`."""
        ...


class PromptFactory(Protocol):
    """The prompts with the tools a Memory's calls may name (`tools`: the runner's; the Agent
    host's are none) and the guides of the skillsets the stack mounts (`guides`: Markdown
    templates, `SkillSet.guide`, in mount order), which the method renders into the prompts of
    the roles it shows them to (`skills.render_guide` fills in the robot's values)."""

    def __call__(self, tools: Mapping[str, ToolSig], guides: Sequence[str] = ()) -> PromptSet: ...


class NoPrompts:
    """The prompts of a method without any: no episode of it can have an Actor."""

    digest = ""

    def actor(
        self, *, instruction: str, memory: MemoryDoc, rules: str, profile: EmbodimentProfile
    ) -> str:
        raise NotImplementedError("the method declares no prompts (MethodRuntime.prompts)")

    library_digest = ""

    def library_actor(self, *, instruction: str, rules: str, profile: EmbodimentProfile) -> str:
        raise NotImplementedError("the method declares no prompts (MethodRuntime.prompts)")


def no_prompts(tools: Mapping[str, ToolSig], guides: Sequence[str] = ()) -> PromptSet:
    """`NoPrompts`, whatever the tools and guides."""
    return NoPrompts()


class DemoBriefs(Protocol):
    """What a practising or evaluated Actor reads of a task's demonstration at `level`, a name the
    core does not read: `levels`, the names the method accepts besides `agent.host.NO_DEMO`
    (`check_level`); `brief`, its brief's parts; `seed`, the frames its episode starts with (None:
    none)."""

    @property
    def levels(self) -> Collection[str]: ...

    def brief(self, package: DemoPackage, level: str) -> list[ContentPart]: ...

    def seed(self, package: DemoPackage, level: str) -> EpisodeSeed | None: ...


class NoDemoBriefs:
    """The demonstration views of a method that shows none: no level but `NO_DEMO`, no brief, no
    frames."""

    levels: Collection[str] = ()

    def brief(self, package: DemoPackage, level: str) -> list[ContentPart]:
        return []

    def seed(self, package: DemoPackage, level: str) -> EpisodeSeed | None:
        return None


def check_level(briefs: DemoBriefs, level: str) -> None:
    """Raise unless `level` is `NO_DEMO` or a level the method's views declare (`briefs.levels`)."""
    if level != NO_DEMO and level not in briefs.levels:
        known = ", ".join(dict.fromkeys((NO_DEMO, *briefs.levels)))
        raise ValueError(f"demo_context {level!r} is not a level the method shows ({known})")


class Explainer(Protocol):
    """A failed try's ground truth (`diagnosis`) in the try's own terms, as a reviewing role reads
    it (`referee/explained.json`): with the try's `registry` at the end, `alias` (the model's name
    of a registered object's id), the robot at the end, the referee's object shapes by the
    simulator's names (None: not given) and the episode's own settings (`Episode.method`)."""

    def __call__(
        self,
        diagnosis: Diagnosis,
        registry: SceneRegistry,
        alias: Callable[[str], str],
        robot: RobotState | None,
        shapes: Mapping[str, ObjectShape] | None,
        method: object | None,
    ) -> BaseModel: ...


class JudgedHook(Protocol):
    """Run on a judged Actor episode before its Hub episode closes, with the Hub, the episode, its
    Hub episode and whether the env's success check passed: what the method records of the
    try's end (into the episode's directory). It must not change how the episode ended."""

    async def __call__(
        self, hub: Hub, episode: "Episode", episode_id: EpisodeId, success: bool
    ) -> None: ...


class Rewrite(Protocol):
    """What a method changed in a frozen Memory for the Actor that reads it (`changes`; none:
    nothing) and its record (`record()`, `unbind_arms.json` in an evaluation)."""

    @property
    def changes(self) -> Sequence[object]: ...

    def record(self) -> dict[str, object]: ...


class Unchanged:
    """The rewrite of a method that leaves a frozen Memory as stored."""

    changes: Sequence[object] = ()

    def record(self) -> dict[str, object]:
        return {}


def as_stored(version: Version) -> tuple[Version, Rewrite]:
    """A frozen Memory as the Actor reads it in a method without a rewrite: as stored."""
    return version, Unchanged()


class GoalReader(Protocol):
    """The demonstration's end as the method reads it (`agent.host.OpaqueGoal`, the method's own
    shape: wicl's `goal.DemoGoal`) from its package, whose blobs are in `blobs`, read with
    `profile`, with `end` (its end as targets, from a file: none, when not read); None: the method
    compares with none. The core hands it to the method's finish gate (`Episode.goal`,
    `DemoView.goal`) and reads only its `hint`."""

    async def __call__(
        self,
        package: DemoPackage,
        blobs: BlobStore,
        profile: EmbodimentProfile,
        end: Sequence[EndTarget],
    ) -> OpaqueGoal | None: ...


async def no_goal(
    package: DemoPackage, blobs: BlobStore, profile: EmbodimentProfile, end: Sequence[EndTarget]
) -> OpaqueGoal | None:
    """The demonstration's end in a method that compares with none."""
    return None


def check_mounted(
    runtime: "MethodRuntime", phases: Iterable[PhaseSpec], mounted: Collection[ToolName]
) -> None:
    """Raise unless every tool the method's runtime calls in `phases` (`MethodRuntime.mounts`) is
    among the `mounted` ones: as a runner or the Agent host starts, not at its first episode."""
    missing = {
        phase.name: sorted(set(runtime.mounts.get(phase.name, ())) - set(mounted))
        for phase in phases
    }
    if wanted := {name: tools for name, tools in missing.items() if tools}:
        shown = "; ".join(f"{name}: {', '.join(tools)}" for name, tools in sorted(wanted.items()))
        raise ValueError(f"the method's phases call tools no mounted toolset has ({shown})")


@dataclass(frozen=True)
class MethodRuntime:
    """`opening`: the hook run as each episode opens; `finish`: the gate on its finish (defaults:
    none); `prompts`: the roles' prompts (default: none, so no Actor); `demo`: what an Actor
    reads of its demonstration (default: nothing); `explain`: what a failed try's ground truth
    is written as for a reviewing role (default: None, nothing written); `judged`: the hook on
    a judged Actor episode (default: None, none); `frozen`: a frozen Memory as
    an evaluated Actor reads it, with what that changed (default: as stored); `goal`: the
    demonstration's end the done check compares with (default: none); `mounts`: by phase, the
    tools its runtime calls there, which a stack must mount (`check_mounted`; default: none);
    `edit_checks`: checks of a model's Memory edits beyond the editor's own, run in order on every
    `memory_edit` (`toolsets.memory.EditCheck`; default: none).
    `evaluate_phase`: the phase whose rules the core's `evaluate` runs its tries under (its Hub
    phase, its env episodes' stage and the Actor's prompt rules; wicl's `s2`); `host_phase`:
    the phase the Agent host runs its episodes in (wicl's eval phase `p3`). None (the
    defaults): the method has no such evaluation, and `evaluate` or the host refuses to start."""

    opening: Opener = field(default_factory=NoOpening)
    finish: FinishGate = field(default_factory=NoFinishGate)
    prompts: PromptFactory = no_prompts
    demo: DemoBriefs = field(default_factory=NoDemoBriefs)
    explain: Explainer | None = None
    judged: JudgedHook | None = None
    frozen: Callable[[Version], tuple[Version, Rewrite]] = as_stored
    goal: GoalReader = no_goal
    mounts: Mapping[PhaseName, tuple[ToolName, ...]] = field(default_factory=dict)
    evaluate_phase: PhaseName | None = None
    host_phase: PhaseName | None = None
    edit_checks: Sequence[EditCheck] = ()
