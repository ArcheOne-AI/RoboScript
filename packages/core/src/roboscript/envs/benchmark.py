"""The benchmark extension point (docs/concepts/embodied-core.md): what the core needs to
run an env.

In-repo integrations are referenced by import path (`my_package.benchmark:BENCHMARK`); packaged
ones register in the `roboscript.benchmarks`
entry-point group. Either way the object is a `Benchmark`.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from roboscript.contracts.capabilities import EnvCapabilities, SensingCapabilities
from roboscript.contracts.ids import TaskName
from roboscript.embodiments import EmbodimentProfile, load_embodiment
from roboscript.envs.cards import TaskCard, load_cards
from roboscript.kernel import entrypoints

GROUP = "roboscript.benchmarks"


class TaskInfo(BaseModel):
    """A task's official step limit and its category. Agent-side: what a task's success check
    reads belongs to the benchmark's referee, never here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_limit: PositiveInt
    category: str | None = None


class EvalSeeds(BaseModel):
    """The benchmark's own evaluation seed blocks: block `s` holds the seeds from
    `first + stride * s` up to the next block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first: NonNegativeInt
    stride: PositiveInt

    def block(self, index: int) -> tuple[int, int]:
        """Block `index` as (first seed, number of seeds)."""
        return self.first + self.stride * index, self.stride


class Benchmark(BaseModel):
    """A benchmark integration: its tasks; the env server, run as `python -m <env_server> --config
    <file> --ready-file <path>` by `interpreter` with the extra `pythonpath` entries; the
    embodiment profile (a YAML path or a registered name); the env server's default config;
    `capabilities`: the optional faces its env servers offer (docs/how-to/add-a-sim-benchmark.md),
    which the startup check compares with a method's `requires` before any server runs (default:
    none). Its servers' `discover()` says the same, with their sensing (`discovered`).
    `env_config_model`: the env server config's model by import path (`config_model`);
    `eval_seeds`: the benchmark's evaluation blocks, which no learning seeds may overlap;
    `skillsets`: the skillsets its stacks mount on their toolsets (`roboscript.skillsets` names or
    import paths, `tools.skillsets`): its bindings for the core's entries and the skills it adds.
    `cards`: the directory of its tasks' success condition cards (`envs.cards`, `<task>.md`; a
    task without a file has none; None, the default: no task has one)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    tasks: dict[TaskName, TaskInfo]
    env_server: str
    interpreter: Literal["sim", "agent"]
    pythonpath: list[Path] = []
    embodiment: Path | str
    env_config: dict[str, Any] = {}
    capabilities: EnvCapabilities = EnvCapabilities()
    env_config_model: str | None = None
    eval_seeds: EvalSeeds | None = None
    skillsets: list[str] = []
    cards: Path | None = Field(default=None, exclude_if=lambda cards: cards is None)

    def task_cards(self) -> dict[TaskName, TaskCard]:
        """Its tasks' success condition cards, checked (`envs.cards.load_cards`); none without
        `cards`."""
        return {} if self.cards is None else load_cards(self.cards, self.tasks)

    def discovered(self, sensing: SensingCapabilities) -> EnvCapabilities:
        """What an env server of this benchmark declares in `discover()`: the faces declared
        here, with its `sensing`."""
        return self.capabilities.model_copy(update={"sensing": sensing})

    def config_model(self) -> type[BaseModel]:
        """The env server config's model (`env_config_model`)."""
        if self.env_config_model is None:
            raise ValueError(f"benchmark {self.name} declares no env_config_model")
        found = entrypoints.load_ref(self.env_config_model)
        if not (isinstance(found, type) and issubclass(found, BaseModel)):
            raise TypeError(f"{self.env_config_model} is not a pydantic model")
        return found

    def profile(self) -> EmbodimentProfile:
        return load_embodiment(self.embodiment)


def load_benchmark(ref: str) -> Benchmark:
    """A benchmark by import path `package.module:attr` or by registered name."""
    found = entrypoints.load_ref(ref) if ":" in ref else entrypoints.load(GROUP, ref)
    if not isinstance(found, Benchmark):
        raise TypeError(f"{ref} is a {type(found).__name__}, not a Benchmark")
    return found
