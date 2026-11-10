"""What a protocol runs one task with, and the protocol extension point (`roboscript.protocols`).

Only the runner and its protocols hold the control plane: seeds, env episodes, budgets, referees
and Memory versions. A protocol is a class with a `name`, a pydantic `Config` (its section of the
recipe's `protocols.yaml` is validated against it) and `async run(ctx, config)`, which returns a
pydantic result and leaves its records under `ctx.directory`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel

from roboscript.contracts.capabilities import EpisodeReset
from roboscript.contracts.events import EventKind
from roboscript.contracts.ids import TaskName
from roboscript.contracts.motion import Planner
from roboscript.embodiments import EmbodimentProfile
from roboscript.envs.cards import TaskCard
from roboscript.learning.episode import EpisodeDriver
from roboscript.learning.hooks import Hooks
from roboscript.learning.runtime import PromptSet
from roboscript.memory import MemoryRecipe, MemoryStore
from roboscript.records import EventWriter

GROUP = "roboscript.protocols"


@dataclass(frozen=True)
class TaskContext:
    """One task: its directory (`<run_dir>/<task>`), the episode driver, the Memory store and
    recipe, the method's role prompts, the robot, the debug hooks, the mounted toolsets, the log,
    the planner whose FK S1's referee uses (None where no protocol needs one), and how the env
    resets the episodes a protocol opens (`EpisodeSpec.reset_by`: the run's env's reset kind,
    `contracts.capabilities.episode_reset`). `card` is the task's success condition card, if it
    has one (the driver's, `envs.cards`): every role reads it under the instruction."""

    task: TaskName
    directory: Path
    driver: EpisodeDriver
    store: MemoryStore
    memory: MemoryRecipe
    prompts: PromptSet
    profile: EmbodimentProfile
    hooks: Hooks
    toolsets: tuple[str, ...]
    events: EventWriter
    planner: Planner | None = None
    reset_by: EpisodeReset = "seed"

    @property
    def card(self) -> TaskCard | None:
        return self.driver.cards.get(self.task)

    def emit(self, step: str, **data: Any) -> None:
        """A `protocol.step` event of this task."""
        self.events.emit(EventKind.PROTOCOL_STEP, {"task": self.task, "step": step, **data})


class LearningProtocol(Protocol):
    """A learning or evaluation protocol."""

    name: ClassVar[str]
    Config: ClassVar[type[BaseModel]]

    async def run(self, ctx: TaskContext, config: Any) -> BaseModel: ...
