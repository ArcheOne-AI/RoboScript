"""The precheck extension point (`roboscript.prechecks`): checks that run before a run starts,
each returning what is wrong (one line per problem; empty: fine), so that a run that cannot work
stops before any job, server or model call.

A registered `Precheck` names the check, its config model and `check(facts, config)` (sync or
async), which returns the problems, or a `PrecheckResult` with notes beside them: what the run
goes without and why, which does not refuse it (`run_prechecks` logs them; `precheck_result`
returns both). Built-ins register through the same group: `env_capabilities` (the env offers what
the method `requires`, docs/concepts/embodied-core.md, but the faces a run waives with its
reason, `CapabilitiesConfig`, and is no real robot, which the runner does not drive:
`REAL_ROBOT`), `seed_split` (the run's seed blocks and the
benchmark's evaluation blocks do not overlap, and the run's reflection chain holds:
`learning.seeds`; a block both name must be the same range) and `skillsets` (every skillset the
env and the run name fits the run's embodiment: its arms, by count and name, the kinds of action it
takes, and the bindings its skills ask for; and every toolset the run mounts mounts some of its
tools; as notes, each entry of those toolsets that is not mounted, not fitting the robot or
lacking a binding, and why: `skills.check_mounts`), which every run checks
(`DEFAULT_PRECHECKS`);
and the example `gpu_headroom` (each listed GPU has `min_free_gb` free, by `nvidia-smi`; off by
default).
"""

import inspect
import logging
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import anyio
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    ValidationError,
    field_validator,
)

from roboscript.contracts.capabilities import EnvCapabilities, missing
from roboscript.embodiments import load_embodiment
from roboscript.envs import load_benchmark
from roboscript.kernel import entrypoints
from roboscript.learning.runspec import RunSpec
from roboscript.learning.seeds import SeedRange, SeedSplit
from roboscript.skills import check_mounts

GROUP: Final = "roboscript.prechecks"
log = logging.getLogger(__name__)
TOOLSETS: Final = "roboscript.toolsets"


class RunFacts(BaseModel):
    """What a precheck sees: the run, what its env offers and its method requires, and the
    benchmark's evaluation seed blocks the run uses (by name)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: RunSpec
    offered: EnvCapabilities
    required: EnvCapabilities
    eval_blocks: dict[str, SeedRange] = {}
    first_seeds: dict[str, list[NonNegativeInt]] = {}
    """By the run's seed block, the first seeds the protocols drawing from it start at (each
    variant's, say): `seed_split` checks that each lies inside its block."""


class PrecheckResult(BaseModel):
    """What prechecks found: `problems` refuse the run (one line each); `notes` say what it goes
    without and why (an entry not mounted on its robot, say), and do not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    problems: list[str] = []
    notes: list[str] = []


Check = Callable[
    [RunFacts, Any], "list[str] | PrecheckResult | Awaitable[list[str] | PrecheckResult]"
]


class NoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class Precheck:
    """A check as a plugin: `check(facts, config)`, `config` an instance of `Config`."""

    name: str
    check: Check
    Config: type[BaseModel] = NoConfig


class PrecheckFailed(Exception):
    """Prechecks found problems; `problems` lists them as `<precheck>: <problem>`."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("the run cannot start:\n  " + "\n  ".join(problems))
        self.problems = problems


def load_precheck(ref: str) -> Precheck:
    """The precheck registered as `ref` in `roboscript.prechecks`, or by import path
    `package.module:attribute`."""
    found = entrypoints.load_ref(ref) if ":" in ref else entrypoints.load(GROUP, ref)
    if not isinstance(found, Precheck):
        where = ref if ":" in ref else f"{ref} in {GROUP}"
        raise TypeError(f"{where} is a {type(found).__name__}, not a Precheck")
    return found


async def precheck_result(
    checks: Sequence[tuple[Precheck, dict[str, Any]]], facts: RunFacts
) -> PrecheckResult:
    """Every check's problems and notes, in order, as `<name>: <line>`."""
    problems, notes = [], []
    for precheck, values in checks:
        try:
            config = precheck.Config.model_validate(values)
        except ValidationError as error:
            raise ValueError(f"invalid config for precheck {precheck.name!r}: {error}") from error
        found = precheck.check(facts, config)
        result = await found if inspect.isawaitable(found) else found
        if not isinstance(result, PrecheckResult):
            result = PrecheckResult(problems=result)
        problems += [f"{precheck.name}: {line}" for line in result.problems]
        notes += [f"{precheck.name}: {line}" for line in result.notes]
    return PrecheckResult(problems=problems, notes=notes)


async def run_prechecks(
    checks: Sequence[tuple[Precheck, dict[str, Any]]], facts: RunFacts
) -> list[str]:
    """Every check's problems, in order, as `<name>: <problem>`; their notes are logged (INFO)."""
    result = await precheck_result(checks, facts)
    for note in result.notes:
        log.info("precheck note: %s", note)
    return result.problems


REAL_ROBOT: Final = (
    "the env is a real robot: the learning runner does not drive real robots (it opens episodes "
    "with env.control open(EpisodeSpec)); a robot server serves robot.control (open with a manual "
    "reset, a person's judgment, heartbeats while an episode is open)"
)
"""Why `env_capabilities` refuses a real robot (`EnvCapabilities.real_robot`): real robots are
driven through their own server (docs/how-to/add-a-real-robot.md)."""


class CapabilitiesConfig(BaseModel):
    """`waive`: required faces a run goes without, each with the run's reason (a note, not a
    problem). Only `expert`: a method needs the expert pre-run to tell a solvable seed from an
    unsolvable one, which a run whose seeds the benchmark itself screened does not (a benchmark's
    official evaluation layouts, an unstable one skipped as unsolvable by the env)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    waive: dict[Literal["expert"], str] = {}

    @field_validator("waive")
    @classmethod
    def _reasons(cls, waive: dict[str, str]) -> dict[str, str]:
        if blank := sorted(face for face, why in waive.items() if not why.strip()):
            raise ValueError(f"waiving {blank} needs a reason")
        return waive


def _capabilities(facts: RunFacts, config: CapabilitiesConfig) -> PrecheckResult:
    required = facts.required
    notes = []
    for face, why in config.waive.items():
        if getattr(required, face) and not getattr(facts.offered, face):
            required = required.model_copy(update={face: False})
            notes.append(f"the env lacks {face}, waived: {why}")
    lines = [f"the env lacks {line}" for line in missing(required, facts.offered)]
    if facts.offered.real_robot:
        lines.append(REAL_ROBOT)
    return PrecheckResult(problems=lines, notes=notes)


def _skillsets(facts: RunFacts, config: NoConfig) -> PrecheckResult:
    benchmark = load_benchmark(facts.spec.env.name)
    names = [*benchmark.skillsets]
    names += [item.name for item in facts.spec.skillsets if item.name not in names]
    toolsets = {item.name: entrypoints.load(TOOLSETS, item.name) for item in facts.spec.toolsets}
    found = check_mounts(names, toolsets, load_embodiment(facts.spec.embodiment.name))
    return PrecheckResult(problems=found.problems, notes=found.unmounted)


def _seed_split(facts: RunFacts, config: NoConfig) -> list[str]:
    split = facts.spec.seeds
    # a run naming a benchmark block (a P4 chain's, say) names the same range
    clash = sorted(
        name
        for name in set(split.blocks) & set(facts.eval_blocks)
        if split.blocks[name] != facts.eval_blocks[name]
    )
    if clash:
        return [f"the run and the benchmark both name seed blocks {clash}"]
    try:
        SeedSplit(
            blocks={**split.blocks, **facts.eval_blocks},
            reflects=split.reflects,
            final=split.final,
            transductive=split.transductive,
        )
    except ValidationError as error:
        return [problem["msg"] for problem in error.errors(include_url=False)]
    problems = []
    for name, seeds in facts.first_seeds.items():
        block = split.blocks.get(name)
        if block is None:
            problems.append(f"protocols start at seeds {seeds} of a block {name!r} the run lacks")
            continue
        if outside := sorted({s for s in seeds if not block.start <= s < block.stop}):
            problems.append(
                f"first seeds {outside} lie outside seed block {name!r} "
                f"({block.start}..{block.stop - 1})"
            )
    return problems


class GpuHeadroomConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gpus: list[NonNegativeInt] = []
    min_free_gb: NonNegativeFloat = 8.0


async def _gpu_headroom(facts: RunFacts, config: GpuHeadroomConfig) -> list[str]:
    if not config.gpus:
        return []
    query = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    done = await anyio.to_thread.run_sync(
        lambda: subprocess.run(query, capture_output=True, text=True, check=False)
    )
    if done.returncode != 0:
        return [f"nvidia-smi failed: {done.stderr.strip() or done.returncode}"]
    free = {}
    for line in done.stdout.strip().splitlines():
        index, total, used = (int(v) for v in line.split(","))
        free[index] = (total - used) / 1000  # MiB to the GB nvidia-smi users count in
    lines = [f"no GPU {gpu}" for gpu in config.gpus if gpu not in free]
    lines += [
        f"GPU {gpu} has {free[gpu]:.1f} GB free, under {config.min_free_gb:g}"
        for gpu in config.gpus
        if gpu in free and free[gpu] < config.min_free_gb
    ]
    return lines


ENV_CAPABILITIES: Final = Precheck(
    name="env_capabilities", check=_capabilities, Config=CapabilitiesConfig
)
SEED_SPLIT: Final = Precheck(name="seed_split", check=_seed_split)
SKILLSETS: Final = Precheck(name="skillsets", check=_skillsets)
GPU_HEADROOM: Final = Precheck(name="gpu_headroom", check=_gpu_headroom, Config=GpuHeadroomConfig)
DEFAULT_PRECHECKS: Final[tuple[Precheck, ...]] = (ENV_CAPABILITIES, SEED_SPLIT, SKILLSETS)
"""What every run checks."""
