"""Artifact overlays (docs/concepts/learning.md): what an evaluated Actor reads beside the method's
own artifact, its Task Memory.

An episode loads an artifact stack (`contracts.artifact.ArtifactStack`): its task's layers, in
the order the method composed them (P4's `roboscript_methods.rsi.context_stack`: the Task
Memory, then the RSI document's GLOBAL part, then the task's own part). The Task Memory layer is
read by the method's Actor prompt, as it always is; every other layer is an artifact that renders
its own part for the layer's scope (`ContextArtifact.context`: every task, or one) and goes
around that prompt in stack order, those before the Task Memory layer before it, the rest after
(`Overlay.wrap`). With the Task Memory alone (no overlay, every method without one) the prompt
is the method's, byte for byte.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from roboscript.contracts.artifact import Artifact, ArtifactLayer, ArtifactRef
from roboscript.contracts.ids import TaskName

SEPARATOR: Final = "\n\n"
"""Between the overlay's parts and the method's prompt."""


@runtime_checkable
class ContextArtifact(Protocol):
    """An artifact an Actor can read beside its Task Memory: its part for a layer's scope (`task`
    None: the part for every task), as text; "" when it has nothing for that scope."""

    def context(self, task: TaskName | None) -> str: ...


@dataclass(frozen=True)
class Overlay:
    """The rendered layers of one task's stack beside its Task Memory: `before` and `after` the
    method's prompt, in stack order; `layers`, the overlay's layers (what `config.json`
    records)."""

    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    layers: tuple[ArtifactLayer, ...] = ()

    def wrap(self, prompt: str) -> str:
        """`prompt` with the overlay around it; `prompt` itself without one."""
        return SEPARATOR.join((*self.before, prompt, *self.after))


NO_OVERLAY: Final = Overlay()

ArtifactLoader = Callable[[ArtifactRef], Awaitable[Artifact]]
"""Loads a layer's artifact by its ref (its store's, checked frozen)."""

OverlayLoader = Callable[[TaskName, Sequence[ArtifactLayer]], Awaitable[Overlay]]
"""A task's overlay from its stack's layers (`ArtifactStack.for_task`), the Task Memory's
included: where it stands splits the rest into before and after."""


def overlay_loader(load: ArtifactLoader, primary: str) -> OverlayLoader:
    """The overlay of every layer that is not of kind `primary` (the Task Memory's), each artifact
    loaded once by `load` and rendered for its layer's scope; a layer whose artifact renders
    nothing for it is left out, and one that cannot be rendered is an error."""

    async def overlay(task: TaskName, layers: Sequence[ArtifactLayer]) -> Overlay:
        at = [i for i, layer in enumerate(layers) if layer.ref.kind == primary]
        if len(at) != 1:
            raise ValueError(f"{task}: an overlay goes around one {primary} layer, not {len(at)}")
        loaded: dict[ArtifactRef, Artifact] = {}
        before: list[str] = []
        after: list[str] = []
        kept = []
        for i, layer in enumerate(layers):
            if i == at[0]:
                continue
            if layer.ref not in loaded:
                loaded[layer.ref] = await load(layer.ref)
            artifact = loaded[layer.ref]
            if not isinstance(artifact, ContextArtifact):
                ref = layer.ref
                raise ValueError(f"{ref.kind} {ref.name} has no part an Actor reads (`context`)")
            text = artifact.context(layer.task)
            if text:
                (before if i < at[0] else after).append(text)
                kept.append(layer)
        return Overlay(before=tuple(before), after=tuple(after), layers=tuple(kept))

    return overlay
