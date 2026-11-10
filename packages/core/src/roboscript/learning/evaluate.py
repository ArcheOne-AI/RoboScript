"""`evaluate` (docs/concepts/learning.md): a Memory version, by default the frozen one, on a list
of seeds, one try each under the rules of the phase the method names for it
(`MethodRuntime.evaluate_phase`: wicl's S2), writing nothing to Memory; unsolvable seeds are
left out. It runs the optional validation block (seeds from 20000) and re-runs a seed with any
version (`version`, `seeds`, `name`). With `done_goal`, a demo package, `check_done` also compares
the scene with the demonstration's final state, as in practice (the method's finish gate).

    <task>/eval/<name>/<seed>/try1/   results.json (the success rate with its Wilson interval)
"""

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
)

from roboscript.agent.host import OpaqueGoal
from roboscript.contracts.env import EpisodeOpened, EpisodeSpec
from roboscript.contracts.ids import TaskName
from roboscript.envs.cards import task_text
from roboscript.learning.context import TaskContext
from roboscript.learning.demo import import_blobs, load_package
from roboscript.learning.episode import Episode, InfraExhausted
from roboscript.learning.ledger import TryRecord, wilson
from roboscript.records import write_atomic
from roboscript.settings import repo_root


class EvaluateConfig(BaseModel):
    """`seeds`, or `count` seeds from `first_seed`; `version`: a number, `v003` or a hash (None:
    the frozen version); `name`: the folder under `<task>/eval/` (default `v<NNN>`);
    `unbind_arms`: the Actor reads the version as the method rewrites a frozen Memory
    (`MethodRuntime.frozen`: wicl's rewrites its arm bindings by role; false reads it as
    stored); `done_goal`: the task's demo package whose final state
    (last keyframe, every camera, as the method reads it: `MethodRuntime.goal`) gates `finish`
    through `check_done` (None: only done criteria gate it); `reset_options`: the benchmark's own
    reset options for every episode (`EpisodeSpec.options`; empty: its defaults)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seeds: list[NonNegativeInt] | None = None
    first_seed: NonNegativeInt = 20000
    count: PositiveInt = 10
    version: int | str | None = None
    name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")] | None = None
    prerun_attempts: PositiveInt = 3
    reset_options: dict[str, JsonValue] = {}
    unbind_arms: bool = True
    done_goal: Path | None = None

    @field_validator("done_goal")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        return path if path is None or path.is_absolute() else repo_root() / path

    def seed_list(self) -> list[int]:
        if self.seeds is not None:
            return self.seeds
        return list(range(self.first_seed, self.first_seed + self.count))


class EvaluateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: TaskName
    version: int
    hash: str
    n: int
    successes: int
    rate: float | None
    wilson95: tuple[float, float]
    unsolvable: list[int]
    tries: list[TryRecord]


class Evaluate:
    name = "evaluate"
    Config = EvaluateConfig

    async def run(self, ctx: TaskContext, config: EvaluateConfig) -> EvaluateResult:
        return await evaluate(ctx, config)


async def evaluate(ctx: TaskContext, config: EvaluateConfig) -> EvaluateResult:
    phase = ctx.driver.runtime.evaluate_phase
    if phase is None:
        raise ValueError("the method names no phase to evaluate in (MethodRuntime.evaluate_phase)")
    if config.version is None:
        frozen = await ctx.store.frozen(ctx.task)
        if frozen is None:
            raise ValueError(f"{ctx.task} has no frozen Memory: give a version to evaluate")
        version = await ctx.store.load(ctx.task, frozen.version)
    else:
        version = await ctx.store.load(ctx.task, config.version)
    folder = ctx.directory / "eval" / (config.name or f"v{version.meta.version:03d}")
    if folder.exists():
        raise ValueError(f"{folder} exists: name this evaluation (name) or use a new run")
    if config.unbind_arms:
        version, rewrite = ctx.driver.runtime.frozen(version)
        if rewrite.changes:
            data = json.dumps(rewrite.record(), indent=2, ensure_ascii=False) + "\n"
            folder.mkdir(parents=True)
            write_atomic(folder / "unbind_arms.json", data.encode())
            ctx.emit("unbind_arms", version=version.meta.version, changes=len(rewrite.changes))
    meta, doc = version.meta, version.doc
    label = f"v{meta.version:03d}"
    goal: OpaqueGoal | None = None  # the method's own, for its finish gate
    if (path := config.done_goal) is not None:
        package = load_package(path)
        if package.task != ctx.task:
            raise ValueError(f"{path}: its package is of {package.task}, not {ctx.task}")
        await import_blobs(path, ctx.driver.blobs)
        goal = await ctx.driver.runtime.goal(package, ctx.driver.blobs, ctx.profile, [])
    tries: list[TryRecord] = []
    unsolvable: list[int] = []
    for seed in config.seed_list():
        spec = EpisodeSpec(
            task=ctx.task,
            seed=seed,
            stage=phase,
            prerun_attempts=config.prerun_attempts,
            options=config.reset_options,
            reset_by=ctx.reset_by,
        )

        def prompt(opened: EpisodeOpened | None) -> tuple[str, None]:
            assert opened is not None and opened.instruction is not None
            text = ctx.prompts.actor(
                instruction=task_text(opened.instruction, ctx.card),
                memory=doc,
                rules=phase,
                profile=ctx.profile,
            )
            return text, None

        episode = Episode(
            task=ctx.task,
            stage=phase,
            role="actor",
            directory=folder / str(seed) / "try1",
            prompt=prompt,
            spec=spec,
            diagnose=True,
            memory=doc,
            memory_label=label,
            memory_hash=meta.hash,
            goal=goal,
            extra={"protocol": "evaluate", "seed": seed, "attempt": 1},
        )
        try:
            outcome, reruns = await ctx.driver.run_counted(episode)
        except InfraExhausted as error:
            if any(o.end.reason != "open_failed:sim_fault" for o in error.outcomes):
                raise
            unsolvable.append(seed)
            continue
        if not outcome.solvable:
            unsolvable.append(seed)
            continue
        await ctx.hooks.on_try_end(ctx.task, seed, 1, outcome)
        tries.append(TryRecord.of(1, label, outcome, reruns, ctx.directory))
    successes = sum(record.success for record in tries)
    result = EvaluateResult(
        task=ctx.task,
        version=meta.version,
        hash=meta.hash,
        n=len(tries),
        successes=successes,
        rate=successes / len(tries) if tries else None,
        wilson95=wilson(successes, len(tries)),
        unsolvable=unsolvable,
        tries=tries,
    )
    data: dict[str, Any] = result.model_dump(mode="json")
    write_atomic(folder / "results.json", (json.dumps(data, indent=2) + "\n").encode())
    ctx.emit("evaluate", version=meta.version, n=result.n, successes=successes)
    return result
