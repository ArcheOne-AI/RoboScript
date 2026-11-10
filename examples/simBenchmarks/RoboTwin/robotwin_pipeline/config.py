"""The pipeline's config: the tasks and variants to run, what each run is made of (the learning
settings, the protocols' parameters, the demonstrations, the env servers), where it runs (the
interpreters, the GPUs and their shared services), how it recovers (respawns, service restarts),
P3's settings, the seed split, the run's plugins and the disk guard. Relative paths are relative to
the repository root.

It is layered (docs/concepts/operations.md, `compose_config`): the generic recipe
(`recipes/wicl/pipeline.yaml`), the env binding (`examples/simBenchmarks/RoboTwin/run/
wicl/pipeline.yaml`), the machine's file (`configs/pipeline.yaml`, copy
configs/pipeline.example.yaml; it names the binding, by default RoboTwin's), then `--set`; the
protocol recipe the same way (`protocols.yaml` of the recipe and the binding) into
`protocol_recipe`. A pipeline directory keeps the composed config it runs (`pipeline.yaml`,
`load_directory`), and a resume reads only that."""

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from roboscript.agent.context import ContextBudget
from roboscript.contracts.ids import TaskName
from roboscript.envs import load_benchmark
from roboscript.kernel import entrypoints
from roboscript.learning import LearningConfig, ProtocolRecipe, load_protocol_recipe
from roboscript.learning.compose import compose_run
from roboscript.learning.config import PROTOCOLS_FILE, BrainSpec
from roboscript.learning.method import load_method
from roboscript.learning.prechecks import PrecheckFailed, RunFacts, precheck_result
from roboscript.learning.runspec import (
    PluginItem,
    RunItem,
    RunSpec,
    binding_layers,
    recipe_of,
    resolve,
)
from roboscript.learning.seeds import SeedRange, SeedSplit
from roboscript.orchestrate.slots import MIB_PER_GB
from roboscript.settings import config_ref_path, load_config_file, read_mapping, repo_root
from roboscript.stack import Interpreters, ServiceSpec
from roboscript_robotwin.eval.blocks import P3_PLACES, P3Settings

log = logging.getLogger(__name__)

CONFIG: Final = "pipeline.yaml"
BINDING: Final = Path("examples/simBenchmarks/RoboTwin/run/wicl")
"""The env binding a machine's config layers over unless it names another (`binding`)."""
PROTOCOLS: Final = ("imitate", "practice")
"""The stages whose parameters the config sets; a variant sets only `practice`'s (S1 is shared)."""
RUN_SETS: Final = {
    "imitate": ("demo",),
    "practice": ("demo", "done_goal", "resume", "start", "pause_after"),
}
"""Protocol parameters that each job sets itself (the demo from `demo`, which S2 and P3 show
only at a `demo_context` level and compare the scene with under `done_goal`; resuming from its
records) or that would stop it short of P3 (`start` would skip the S1 -> S2 gate, `pause_after`
the freeze)."""
MOTION: Final = ServiceSpec(
    name="motion",
    module="roboscript.motion",
    interpreter="sim",
    args=["--config", "examples/simBenchmarks/RoboTwin/configs/motion.example.yaml"],
    ready_timeout_s=900,
)
"""The cuRobo planner service, one per GPU, with the config the demo recording uses (from
the repository root, where the pipeline runs)."""
_NAME: Final = re.compile(r"[a-z0-9][a-z0-9_-]*")
REPLACED: Final = ("variants", "learning.brains.*")
"""Keys the highest layer that names them sets whole (`compose_layers`), instead of merging into
the lower layers': a machine's file that lists only `default` runs only that variant, and a role's
brain it names has only the settings it gives (none of the recipe's brain settings for it)."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


class ReplaySettings(BaseModel):
    """The replay tape of every job (`roboscript.records.replay`; None: off, the default): with
    `record`, each job records its model, motion and vision answers and its comparison stream into
    `<directory>/<job dir>/<phase>/`; with `replay`, it is answered from `source` (a recorded
    pipeline's `directory`) instead and writes its comparison stream
    (examples/simBenchmarks/RoboTwin/README.md, "Replay equivalence")."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["record", "replay"]
    directory: Path
    source: Path | None = None

    @field_validator("directory", "source")
    @classmethod
    def _paths(cls, path: Path | None) -> Path | None:
        return None if path is None else _absolute(path)

    @model_validator(mode="after")
    def _source(self) -> Self:
        if (self.mode == "replay") != (self.source is not None):
            raise ValueError("a replay needs the recorded tape as source; a recording takes none")
        return self


Unconverged = Literal["best_memory", "last", "stop"]
"""What a variant whose S2 stopped at its seed cap without converging does (`PipelineP3`)."""


class PipelineP3(P3Settings):
    """P3's settings (`roboscript_robotwin.eval.blocks.P3Settings`) with what the pipeline does
    with a variant whose S2 stopped at its seed cap without converging (`not converged`):
    `unconverged`. `best_memory` (the default): P3 evaluates the S2 Memory version with the best
    first-try record (`wicl.ledger.best_version`: the Wilson 95 % lower bound of its counted
    first tries, then more successes, then the later version), frozen in place of the cap's
    (`run.pick_s2`); with no first-try success at all the variant stops (`not_converged`) and
    counts 0 in P3. `stop`: the variant ends `not_converged` and its P3 blocks are not counted
    (those done before are kept; `run.p3_gated`). `last`: P3 evaluates the Memory frozen at the
    cap. A started job keeps it unless it is the default (`run.definition`)."""

    unconverged: Unconverged = "best_memory"


class PipelineConfig(BaseModel):
    """Each task's S1 runs once, in `<root>/<stamp>/<task>/S1/`; each variant's S2 starts from it
    and P3 evaluates the Memory S2 froze, in `<root>/<stamp>/<task>/<variant>/`.

    `benchmark`: RoboTwin, or MockEnv for tests; `tasks`: the tasks to run (empty: every task of
    the benchmark). `protocols` holds protocol parameters by protocol
    name (`imitate`, `practice`) over the recipe's, a variant `practice`'s over those; in a string
    value `{task}` is the task. `demo` is each task's demo package, whose final state S2's and P3's
    `check_done` compare the scene with while `done_goal` is on; `learning` the `LearningConfig`
    of every job but its `run_dir`; `env` the env server config's fields but `task` and `blob_dir`.
    `per_gpu` are the services each GPU runs for the jobs placed there, each in a stack of its own
    (named `<name>-gpu<n>`; one that dies restarts after each of `service_restart_waits_s`, then its
    GPU takes no more jobs): `planner` names the one S1's referee and the toolsets plan with, and
    one named `vision` (`roboscript_vision`) is the cross-check of S2's scene toolsets and P3's;
    all share the pipeline's blob store. A job holds simulator places on its GPU while it runs, one
    for S1 or S2 (its env server) and `P3_PLACES` for a P3 block: at most `max_per_gpu` of ours on a
    GPU (a sanity bound: memory decides), and `max_runs` jobs at once. `gpus` are filled in their
    order: a job goes on the first GPU
    whose services are up, with room by our count and by memory: what `nvidia-smi` shows in use,
    plus what we reserved and others claimed that it does not show yet, plus `free_gb_per_place` per
    place the job holds, stays within `gpu_cap_gb` less `gpu_margin_gb` (GB: `MIB_PER_GB`; the
    margin takes what grows between readings). A placement reserves the larger of its
    `free_gb_per_place` per place and the peak the job's own processes showed, for as long as the
    job runs; a GPU's services start when a job first needs the GPU (no GPU that runs them has room
    for it) and the GPU has room for their `service_gb` (a service it does not name:
    `free_gb_per_place`) and the job's, reserved the same way; they stop once no job of ours was on
    the GPU for `service_idle_s`, and start again when a job needs it. Once a job ran for
    `PEAK_LEARN_S`, the most memory its processes showed, per task and places
    (`<stamp>/gpu_peaks.json`, across launches), plus `peak_headroom_gb` is what that task's next
    jobs reserve instead of `free_gb_per_place`, so that tasks whose simulators take less pack
    closer (a task not seen yet: the most any task showed for as many places); a placement whose
    job ran `PEAK_LEARN_S` holds its peak plus `peak_headroom_gb`, and a GPU whose use went over
    the cap less the margin takes no job for `OVER_BACKOFF_S`. `gpu_claims` is the claims
    directory other tenants of the cards write (None: no other
    tenant claims): every placement and service start claims there too. A job that infrastructure
    failed is respawned after each of `respawn_waits_s`, and so is one whose worker gave no
    heartbeat for `watchdog_s` (killed); one that ran out of GPU memory (its logs or its GPU's
    services' say so) waits for room and runs again without using up its respawns, for
    `pressure_retry_s` in all, as a GPU's service that ran out of memory restarts without using up
    its restarts; while the disk has less than `min_free_gb` free, no job starts. A GPU service
    restarts on its port, so that the URLs its jobs hold stay valid; one that came up and then was
    not ready (`/readyz`) for `service_unready_s` straight is stopped and restarts as one that died
    does, and one ready for `service_healthy_s` straight has its restarts back
    (`roboscript.orchestrate.launch.watch_ready`).

    `s1_fallback`: an S1 that runs out of attempts yields its best Memory all the same
    (`ImitateConfig.fallback`), and its variants start from it (`ops.FALLBACK`), picked under
    `s1_fallback_rule` (`ImitateConfig.fallback_rule`): `success`,
    only if an attempt had an official success, else the task stops at S1 and counts 0 in P3;
    `progress`, the version with the most referee passes, success or not.
    `model` (by default every model call of a run is the run's model): the
    run's model, which every brain of `learning.brains` that names a `model` takes, and P3's Agent
    host brain (`run.run_models`), and the VLM of the scene toolsets of every stage (S1 and S2's,
    `configs/scene.yaml`'s `vlm.model` otherwise; P3's host's, their config's otherwise) unless
    `perception_model` names that one (`vlm_model`); None: each as its own config names it (the
    brains' models, the scene configs' gateway alias). `context_windows`: context windows in
    tokens by model, over each brain's `context.windows` (`agent.context.ContextBudget`; its
    defaults otherwise), P3's host brain's too.
    `spare_preempt_after_s`: a job that is not a spare unit (`p3.randomized: spare`) and waited
    this long for room that stopping running spare units on one GPU would give it has the newest
    of them stopped (`preempted`; they resume from their finished episodes once there is spare
    room again).
    `placement_order`: which waiting jobs place first (`roboscript.orchestrate.slots.Turn`):
    `finish_tasks`, the later stages (S2, P3) of started tasks before S1s; `by_stage`, S1s before
    S2s before P3s. Among each, the jobs that ran before first.

    `shared_stacks` (a registry directory, `roboscript.stack.shared`; None: the pipeline's own
    services, in `<directory>/services/`) shares each GPU's services with every tenant of the
    machine that runs the same service code: the pipeline attaches to the stack that runs them
    on the GPU (then only the job needs room there) and starts one, as above, only when there is
    none; its keeper, not the pipeline, runs and restarts the services (`service_restart_waits_s`,
    `pressure_retry_s`; their restarts back after `service_healthy_s`) and claims their memory
    (`service_gb`), and stops them `shared_linger_s` after the last tenant left. Its keeper
    restarts a service that stays up not ready on its own rule (`/readyz` failing
    `stack.keeper.UNHEALTHY_CHECKS` times in a row): `service_unready_s` applies to the
    pipeline's own services only. The pipeline holds the stack while a job of its own was on the
    GPU within `service_idle_s`; the services read the jobs' images from the pipeline's blob
    store.

    `binding`: the env binding the config was composed with (`compose_config`); `protocol_recipe`:
    the protocol recipe composed with it (the recipe's `protocols.yaml` under the binding's), which
    every job runs with (None: the recipe's file, `recipe`); `seeds`: the run's learning seed
    blocks, which the `seed_split` precheck keeps
    apart from each other and from the benchmark's evaluation blocks; `plugins`: the run's
    guards, event sinks and prechecks beside the built-in ones (`roboscript.learning.compose`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    root: Path = Path("runs/pipeline")
    benchmark: Literal["robotwin", "mock"] = "robotwin"
    tasks: list[TaskName] = []
    variants: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=lambda: {"default": {}}, min_length=1
    )
    protocols: dict[str, dict[str, Any]] = {}
    demo: str = "data/demos/recorded/{task}/package"
    done_goal: bool = True
    learning: dict[str, Any] = {}
    env: dict[str, Any] = {}
    interpreters: Interpreters
    sim_pythonpath: list[Path] = [
        Path("packages/core/src"),
        Path("packages/vision/src"),
        Path("packages/benchmarks/robotwin/src"),
    ]
    vision_pythonpath: list[Path] = [Path("packages/core/src"), Path("packages/vision/src")]
    gpus: list[NonNegativeInt] = Field(default=[0], min_length=1)
    gpu_cap_gb: PositiveFloat = 46.0
    gpu_margin_gb: NonNegativeFloat = 1.0
    gpu_claims: Path | None = None
    shared_stacks: Path | None = None
    shared_linger_s: PositiveFloat = 60.0
    free_gb_per_place: NonNegativeFloat = 4.0
    peak_headroom_gb: NonNegativeFloat = 0.5
    service_gb: dict[str, NonNegativeFloat] = {}
    max_per_gpu: PositiveInt = 10
    max_runs: PositiveInt = 15
    per_gpu: list[ServiceSpec] = Field(default_factory=lambda: [MOTION])
    planner: str = MOTION.name
    env_ready_timeout_s: PositiveFloat = 600.0
    p3: PipelineP3 = PipelineP3()
    respawn_waits_s: list[PositiveFloat] = [60.0, 300.0, 900.0]
    pressure_retry_s: PositiveFloat = 3600.0
    service_restart_waits_s: list[PositiveFloat] = [10.0, 60.0, 300.0]
    service_idle_s: PositiveFloat = 600.0
    service_unready_s: PositiveFloat = 60.0
    service_healthy_s: PositiveFloat = 1800.0
    watchdog_s: PositiveFloat = 1800.0
    min_free_gb: NonNegativeFloat = 100.0
    status_interval_s: PositiveFloat = 30.0
    s1_fallback: bool = True
    s1_fallback_rule: Literal["success", "progress"] = "success"
    model: str | None = None
    perception_model: str | None = None
    context_windows: dict[str, PositiveInt] = {}
    spare_preempt_after_s: PositiveFloat = 300.0
    placement_order: Literal["finish_tasks", "by_stage"] = "finish_tasks"
    replay: ReplaySettings | None = None
    binding: Path | None = None
    protocol_recipe: ProtocolRecipe | None = None
    seeds: SeedSplit | None = None
    plugins: list[PluginItem] = []

    @model_validator(mode="before")
    @classmethod
    def _retired(cls, values: Any) -> Any:
        # `reservation_s` in a stored directory's config is ignored
        if isinstance(values, dict):
            values = {k: v for k, v in values.items() if k != "reservation_s"}
        return values

    @model_validator(mode="before")
    @classmethod
    def _run_model(cls, values: Any) -> Any:
        """The brains of `learning` take the run's `model` and `context_windows` (idempotent:
        a pipeline directory's saved config has them already)."""
        if not isinstance(values, dict):
            return values
        learning = values.get("learning") or {}
        brains = learning.get("brains") or {}
        if not brains:
            return values
        model, windows = values.get("model"), values.get("context_windows") or {}
        if model is None and not windows:
            return values
        found = {role: brain_with(entry, model, windows) for role, entry in brains.items()}
        return {**values, "learning": {**learning, "brains": found}}

    @property
    def vlm_model(self) -> str | None:
        """The model the scene toolsets' VLM asks (`perception_model`, else `model`); None: as
        their configs name it."""
        return self.perception_model or self.model

    @field_validator("tasks")
    @classmethod
    def _every_task(cls, tasks: list[TaskName], info: ValidationInfo) -> list[TaskName]:
        if tasks or "benchmark" not in info.data:
            return tasks
        return [TaskName(task) for task in load_benchmark(info.data["benchmark"]).tasks]

    @field_validator("root")
    @classmethod
    def _root(cls, path: Path) -> Path:
        return _absolute(path)

    @field_validator("gpu_claims", "shared_stacks", "binding")
    @classmethod
    def _claims(cls, path: Path | None) -> Path | None:
        return None if path is None else _absolute(path)

    @field_validator("sim_pythonpath", "vision_pythonpath")
    @classmethod
    def _paths(cls, paths: list[Path]) -> list[Path]:
        return [_absolute(path) for path in paths]

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name, values in (("tasks", self.tasks), ("gpus", self.gpus)):
            if len(set(values)) < len(values):
                raise ValueError(f"{name} lists an entry twice: {values}")
        if bad := [v for v in self.variants if not _NAME.fullmatch(v)]:
            raise ValueError(f"variant names are directory names ([a-z0-9_-]): {bad}")
        if unknown := sorted(set(self.protocols) - set(PROTOCOLS)):
            raise ValueError(f"protocols {unknown}: a run sets only {list(PROTOCOLS)}")
        if bad := sorted({p for v in self.variants.values() for p in v} - {"practice"}):
            raise ValueError(f"variants set {bad}: only practice (every variant shares S1)")
        for stage in [self.protocols, *self.variants.values()]:
            for name, keys in RUN_SETS.items():
                if clash := sorted(set(stage.get(name, {})) & set(keys)):
                    raise ValueError(f"{name} sets {clash}, which each job sets itself")
        if "{task}" not in self.demo:
            raise ValueError("demo must name the task's package with {task}")
        if "run_dir" in self.learning:
            raise ValueError("learning.run_dir is each job's own directory: leave it out")
        learning = LearningConfig.model_validate({**self.learning, "run_dir": self.root})
        if self.binding is not None and (
            _absolute(learning.recipe).resolve() != recipe_of(self.binding).resolve()
        ):
            raise ValueError(
                f"learning.recipe {learning.recipe} is not the recipe the binding {self.binding} "
                f"binds ({recipe_of(self.binding)}), whose protocols the run composed"
            )
        if clash := {"task", "blob_dir"} & set(self.env):
            raise ValueError(f"env sets {sorted(clash)}, which the pipeline sets per job")
        names = [service.name for service in self.per_gpu]
        if len(set(names)) < len(names) or any(s.gpu is not None for s in self.per_gpu):
            raise ValueError("per_gpu services need distinct names and no gpu (set per GPU)")
        if with_port := [s.name for s in self.per_gpu if "--port" in s.args]:
            raise ValueError(
                f"per_gpu {with_port}: leave --port out (the pipeline keeps each service's port "
                "across its restarts)"
            )
        if any(service.depends_on for service in self.per_gpu):
            raise ValueError(
                "per_gpu services start apart, each in a stack of its own: no depends_on"
            )
        if unknown := sorted(set(self.service_gb) - set(names)):
            raise ValueError(f"service_gb names {unknown}, none of the per_gpu services {names}")
        if self.planner not in names:
            raise ValueError(f"planner {self.planner!r} is none of the per_gpu services {names}")
        if self.gpu_margin_gb >= self.gpu_cap_gb:
            raise ValueError("gpu_margin_gb leaves no room under gpu_cap_gb")
        if self.p3.blocks and self.max_per_gpu < P3_PLACES * self.p3.shards:
            places = P3_PLACES * self.p3.shards
            raise ValueError(f"a P3 block holds {places} places: max_per_gpu is too small")
        return self

    def recipe(self) -> ProtocolRecipe:
        """The protocol recipe the jobs run with: `protocol_recipe`, else the recipe's file."""
        if self.protocol_recipe is not None:
            return self.protocol_recipe
        learning = LearningConfig.model_validate({**self.learning, "run_dir": self.root})
        return load_protocol_recipe(learning.recipe)

    def service_mib(self, name: str) -> float:
        """The memory (MiB) a GPU's service `name` reserves as it starts."""
        return self.service_gb.get(name, self.free_gb_per_place) * MIB_PER_GB

    def demo_of(self, task: str) -> Path:
        return _absolute(Path(self.demo.replace("{task}", task)))

    def protocol(self, name: str, task: str, variant: str | None = None) -> dict[str, Any]:
        """Protocol `name`'s parameters over the recipe's (the runner adds those): the config's,
        a variant's over them, `{task}` replaced."""
        own = self.variants[variant].get(name, {}) if variant is not None else {}
        values = {**self.protocols.get(name, {}), **own}
        return {
            k: v.replace("{task}", task) if isinstance(v, str) else v for k, v in values.items()
        }


def compose_layers(ref: str, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """The config the machine's file `ref` (a name under configs/ or a YAML file) makes, layered
    over its binding and that binding's recipe (see the module docstring), as plain values. The
    layers merge, but for `REPLACED`."""
    site = read_mapping(config_ref_path(ref))
    binding = _absolute(Path(site.get("binding", BINDING)))
    if not binding.is_dir():
        raise FileNotFoundError(f"the env binding {binding} does not exist")
    layers: dict[Any, dict[str, Any]] = binding_layers(binding, CONFIG)
    for name, values in binding_layers(binding, PROTOCOLS_FILE).items():
        layers[name] = {**layers.get(name, {}), "protocol_recipe": values}
    layers["site"] = {**site, "binding": str(binding)}
    for name, values in layers.items():
        _explicit_brains(values, f"{name} layer")
    composed = resolve(layers, overrides, replace=REPLACED)
    _explicit_brains(composed, "the composed config (--set)")
    return composed


BRAIN_KEYS: Final = ("model", "reasoning_effort")
"""The settings a brain entry must name itself, where its brain has them: a layer's entry
replaces the lower layers' whole (`REPLACED`), so one that left them out would fall back to the
brain class's defaults unseen."""


def _explicit_brains(values: Mapping[str, Any], where: str) -> None:
    for role, entry in ((values.get("learning") or {}).get("brains") or {}).items():
        spec = BrainSpec.model_validate(entry)
        fields = entrypoints.load("roboscript.brains", spec.name).Config.model_fields
        if missing := [k for k in BRAIN_KEYS if k in fields and k not in spec.config]:
            raise ValueError(
                f"{where}: learning.brains.{role} names no {missing}: a brain entry replaces the "
                "lower layers' whole, so name each of them"
            )


def brain_with(entry: Any, model: str | None, windows: Mapping[str, int]) -> Any:
    """A brain entry (`BrainSpec` values) with the run's `model` where it names one, and the
    context `windows` over its own where its brain's config has a context budget (`context`)."""
    if not isinstance(entry, dict):
        return entry
    config = dict(entry.get("config") or {})
    if model is not None and "model" in config:
        config["model"] = model
    if windows:
        name = entry.get("name", BrainSpec.model_fields["name"].default)
        fields = entrypoints.load("roboscript.brains", name).Config.model_fields
        if "context" in fields:
            context = dict(config.get("context") or {})
            own = context.get("windows") or ContextBudget().windows  # the defaults, else
            context["windows"] = {**own, **windows}
            config["context"] = context
    return {**entry, "config": config}


def compose_config(ref: str, overrides: Sequence[str] = ()) -> PipelineConfig:
    """The composed config of the machine's file `ref` (`compose_layers`), validated."""
    values = compose_layers(ref, overrides)
    try:
        return PipelineConfig.model_validate(values)
    except ValidationError as error:
        raise ValueError(f"{ref} over its binding {values['binding']}: {error}") from error


def run_spec(config: PipelineConfig) -> RunSpec:
    """The run the config declares (`roboscript.learning.runspec`): the benchmark, its
    embodiment, the method its binding binds, the tasks, the seed split and the plugins."""
    benchmark = load_benchmark(config.benchmark)
    method = (config.binding or repo_root() / BINDING).name
    return RunSpec(
        env=RunItem(name=config.benchmark),
        embodiment=RunItem(name=str(benchmark.embodiment)),
        method=RunItem(name=method),
        tasks=config.tasks,
        seeds=config.seeds or SeedSplit(blocks={}),
        plugins=config.plugins,
    )


def run_facts(config: PipelineConfig) -> RunFacts:
    """What the prechecks see: the run, what its benchmark offers, what its method requires (the
    method's own `requires`), and the official P3 blocks it evaluates (`p3_<block>`)."""
    benchmark = load_benchmark(config.benchmark)
    blocks = {}
    if benchmark.eval_seeds is not None:
        for block in config.p3.blocks:
            start, count = benchmark.eval_seeds.block(block)
            blocks[f"p3_{block}"] = SeedRange(start=start, count=count)
    spec = run_spec(config)
    return RunFacts(
        spec=spec,
        offered=benchmark.capabilities,
        required=load_method(spec.method.name).requires,
        eval_blocks=blocks,
        first_seeds={} if config.seeds is None else first_seeds(config),
    )


FIRST_SEEDS: Final = {"practice": "practice", "evaluate": "evaluate"}
"""The protocols whose `first_seed` starts in a seed block of the run, by block name."""


def first_seeds(config: PipelineConfig) -> dict[str, list[int]]:
    """By seed block, the `first_seed`s of the protocols drawing from it: the recipe's, the
    config's over it, each variant's over that (`PipelineConfig.protocol`; variants set only
    `practice`)."""
    recipe = config.recipe().protocols
    found: dict[str, list[int]] = {}
    for protocol, block in FIRST_SEEDS.items():
        variants: list[str | None] = [*config.variants] if protocol == "practice" else [None]
        for variant in variants:
            values = {**recipe.get(protocol, {}), **config.protocol(protocol, "", variant)}
            if (seed := values.get("first_seed")) is not None and seed not in found.get(block, []):
                found.setdefault(block, []).append(int(seed))
    return found


async def check_run(config: PipelineConfig) -> None:
    """The run's prechecks (the built-in ones and those its plugins add) before any job starts:
    their notes (what the run goes without, which does not refuse it) are printed and logged, one
    `precheck note: <precheck>: <note>` line each; then `PrecheckFailed` names every problem."""
    async with compose_run(config.plugins) as composed:
        found = await precheck_result(composed.prechecks, run_facts(config))
    for note in found.notes:
        print(f"precheck note: {note}", flush=True)
        log.info("precheck note: %s", note)
    if found.problems:
        raise PrecheckFailed(found.problems)


def load_directory(directory: Path, overrides: Sequence[str] = ()) -> PipelineConfig:
    """The config a pipeline directory runs, with dotted `overrides`."""
    return load_config_file(directory / CONFIG, PipelineConfig, overrides=overrides)
