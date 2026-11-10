"""Frozen artifacts a method learns, references to them, the stores that keep them, and the
stacks an episode loads (docs/concepts/agent-runtime.md, docs/how-to/add-a-method.md,
docs/concepts/learning.md).

Here, not in `learning`, because the skillset manifest (`contracts.skills`) is an artifact too,
and the Agent host (another process, in the agent layer) loads artifacts by reference: it gets
`ArtifactRef`s as data (an evaluation's manifest), opens the store its config names
(`roboscript.artifact_stores`) and loads them there; no Python object crosses.

A store is read and frozen through `ArtifactStore`. Committing a version is the kind's own
call, because what a version records beside its content is the kind's (a Task Memory version's
parent, status, source and merge record, `memory.store.MemoryStore.commit`); a store that
dropped it to fit one signature would lose what resumes and audits read.
"""

from typing import Annotated, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from roboscript.contracts.ids import TaskName

ArtifactKind = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
"""What an artifact is, e.g. `task_memory`, `skillset`, `rsi`."""


class Artifact(BaseModel):
    """Base of every frozen artifact; a subclass names its `kind`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ClassVar[str]


class ArtifactRef(BaseModel):
    """One version of one artifact: `name` is its key within the kind (a task for Task Memory,
    the skillset's name), `version` the store's label for it, `hash` the sha256 of its content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ArtifactKind
    name: str
    version: str
    hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ArtifactStore(Protocol):
    """Versions of one artifact kind, by name; a frozen version is what evaluation reads.
    Registered in `roboscript.artifact_stores`; `roboscript.testing.artifacts` checks one."""

    kind: ClassVar[str]

    async def load(self, ref: ArtifactRef) -> Artifact:
        """The version `ref` names, checked against its hash; LookupError if there is none,
        ValueError if `ref` is of another kind."""
        ...

    async def versions(self, name: str) -> list[ArtifactRef]:
        """`name`'s versions, oldest first."""
        ...

    async def freeze(self, ref: ArtifactRef, *, reason: str) -> ArtifactRef:
        """Freeze a version; freezing the frozen one again returns it, another is refused."""
        ...

    async def frozen(self, name: str) -> ArtifactRef | None: ...


class ArtifactLayer(BaseModel):
    """One artifact in a stack, for every task (`task` None) or for one task only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: ArtifactRef
    task: TaskName | None = None


class ArtifactStack(BaseModel):
    """The artifacts an episode loads, in render order. The same artifact may appear once for
    every task and once per task (an RSI document's GLOBAL and TASK parts). With one layer per
    task (a Task Memory) what the Actor reads is that artifact alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layers: tuple[ArtifactLayer, ...]

    @field_validator("layers")
    @classmethod
    def _unique(cls, layers: tuple[ArtifactLayer, ...]) -> tuple[ArtifactLayer, ...]:
        seen = set()
        for layer in layers:
            key = (layer.ref, layer.task)
            if key in seen:
                ref = layer.ref
                raise ValueError(f"{ref.kind} {ref.name} stacked twice for {layer.task}")
            seen.add(key)
        return layers

    def for_task(self, task: TaskName) -> list[ArtifactLayer]:
        """The layers an episode of `task` loads, in order."""
        return [layer for layer in self.layers if layer.task in (None, task)]
