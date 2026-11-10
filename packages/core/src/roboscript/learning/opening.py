"""The open-time hook's record and interface (docs/concepts/perception.md,
docs/concepts/learning.md).

A method may run a hook as each episode opens, before the brain (`Opener`, the method's
`MethodRuntime.opening`; `NoOpening`, the default, does nothing), e.g. one that identifies the
Memory's OBJECTS roles once (wicl's, `roboscript_methods.wicl.opening`). What it leaves is an
`Opening` (identified, reused, skipped, or a failure that ends the episode as infrastructure or
configuration), and a later episode of the same scene may start from it (`Reuse`, saved as the
episode's `identify.json`) when the scene has not drifted (`Drift`, at most `DRIFT`).
"""

from pathlib import Path
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveFloat, PositiveInt

from roboscript.blobs import BlobStore
from roboscript.contracts.hub import EpisodeSeed, EpisodeSession
from roboscript.contracts.observation import Observation
from roboscript.contracts.tools import ContentPart, TextPart
from roboscript.hub import Hub
from roboscript.memory import (
    MemoryDoc,
)
from roboscript.records import write_atomic

IDENTIFY: Final = "identify"
"""The perception toolset's system-only identification."""


SAVED: Final = "identify.json"


INFRA_CODES: Final = frozenset({"infra_error", "timeout"})


BRIEF: Final = (
    "The objects were identified once, as this episode opened, and follow the grippers from now on:"
)


DRIFT: Final = 0.5
"""A try's first frame may differ from the frame a reused identification saw by this much (the
mean absolute difference of their images, 0-255, in the camera that differs most, at 1/8 scale).
In a deterministic simulator the resets of a seed are identical to the bit, and the next seed
differs by more than this."""


class Reuse(BaseModel):
    """An identification to start another episode of the same scene from: the episode's frames,
    facts and aliases after it (`seed`), its rendered outcome, the episode it was made in
    (`source`), and what it identified (`key`: the roles and the instruction)."""

    model_config = ConfigDict(frozen=True)

    source: str
    key: str
    seed: EpisodeSeed
    content: list[ContentPart]


class Retries(BaseModel):
    """How often the identification is tried when it fails as infrastructure (the model or the
    vision service did not answer), and the wait before the second try, doubling after it;
    `unresolved`: how often the roles an identification left without an object (not found, or
    ambiguous with no object registered) are identified again on a new observation (the roles
    it found keep their objects)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempts: PositiveInt = 1
    backoff_s: PositiveFloat = 2.0
    unresolved: NonNegativeInt = 0


class Drift(BaseModel):
    """How far a try's first frame is from the frame the identification of `source` saw (see
    `DRIFT`; None: they share no camera), and so whether it may be reused."""

    model_config = ConfigDict(frozen=True)

    source: str
    difference: float | None

    @property
    def too_far(self) -> bool:
        return self.difference is None or self.difference > DRIFT


class Opening(BaseModel):
    """What the open-time hook left an episode: `identified`, `reused` (from `reuse.source`),
    `skipped` (no `identify`, no instruction), or a failure, `infra` or `config` (`note` says
    what); the content for the role's brief and what a later episode may reuse."""

    model_config = ConfigDict(frozen=True)

    status: Literal["identified", "reused", "skipped", "infra", "config"]
    note: str = ""
    content: list[ContentPart] = []
    reuse: Reuse | None = None

    @property
    def failed(self) -> bool:
        return self.status in ("infra", "config")

    @property
    def end_reason(self) -> str:
        """How a failed opening ends the episode (an infrastructure or a configuration end)."""
        return "infra_error" if self.status == "infra" else "config_error:identify"

    def brief(self) -> list[ContentPart]:
        return [TextPart(text=BRIEF), *self.content] if self.content else []


def save(directory: Path, reuse: Reuse) -> None:
    """Keep an identification in an episode's directory (`identify.json`)."""
    write_atomic(directory / SAVED, (reuse.model_dump_json(indent=2) + "\n").encode())


def load(directory: Path) -> Reuse | None:
    """The identification an episode's directory keeps, if it has one."""
    path = directory / SAVED
    return Reuse.model_validate_json(path.read_bytes()) if path.is_file() else None


class Opener(Protocol):
    """A method's open-time hook (`learning.runtime.MethodRuntime.opening`): whether an episode
    reading `memory` under `instruction` may start from `reuse`, how far a try's first frame is
    from the one `reuse` saw, and the hook itself, run once the Hub episode is open (it may end
    the episode, `Opening.failed`)."""

    def reusable(
        self, reuse: Reuse | None, memory: MemoryDoc | None, instruction: str | None
    ) -> bool: ...

    async def drift(self, reuse: Reuse, first: Observation, blobs: BlobStore) -> Drift: ...

    async def open(
        self,
        hub: Hub,
        session: EpisodeSession,
        *,
        instruction: str | None,
        memory: MemoryDoc | None,
        source: str,
        reuse: Reuse | None = None,
        drift: Drift | None = None,
        retries: Retries | None = None,
    ) -> Opening: ...

    async def identify(
        self,
        hub: Hub,
        session: EpisodeSession,
        *,
        instruction: str | None,
        memory: MemoryDoc,
        retries: Retries | None = None,
    ) -> Opening:
        """Identify `memory`'s roles once more in an episode that runs (a P3 Actor adding a
        library Memory, `agent.host`): as `open` identifies, tried again in place as `retries`
        allows, but a failure is only reported (`Opening.failed`): it never ends the episode,
        and the episode's clock is left to the caller."""
        ...


class NoOpening:
    """The open-time hook of a method without one: nothing is identified or reused."""

    def reusable(
        self, reuse: Reuse | None, memory: MemoryDoc | None, instruction: str | None
    ) -> bool:
        return False

    async def drift(self, reuse: Reuse, first: Observation, blobs: BlobStore) -> Drift:
        return Drift(source=reuse.source, difference=None)

    async def open(
        self,
        hub: Hub,
        session: EpisodeSession,
        *,
        instruction: str | None,
        memory: MemoryDoc | None,
        source: str,
        reuse: Reuse | None = None,
        drift: Drift | None = None,
        retries: Retries | None = None,
    ) -> Opening:
        return Opening(status="skipped")

    async def identify(
        self,
        hub: Hub,
        session: EpisodeSession,
        *,
        instruction: str | None,
        memory: MemoryDoc,
        retries: Retries | None = None,
    ) -> Opening:
        return Opening(status="skipped")
