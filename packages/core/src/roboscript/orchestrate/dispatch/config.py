"""The dispatch config (`DispatchConfig`; a benchmark's example ships one): the workers, the
tasks and their costs, the pipeline config every worker runs and how a unit is launched and read.

The pipeline config is composed once, on the coordinator, as plain YAML data (relative paths stay
relative, so each worker resolves them in its own code copy). A worker's copy differs only in the
task subset, its GPUs and the `local_keys` it sets (`WorkerSpec.set`: interpreters, run root, ...);
`provenance` hashes the rest, which every worker checks before it starts."""

import copy
import hashlib
import re
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.ids import TaskName
from roboscript.kernel import entrypoints
from roboscript.kernel.compose import compose
from roboscript.settings import repo_root

_NAME: Final = re.compile(r"[a-z0-9][a-z0-9_-]*")
_SECRET: Final = re.compile(r"TOKEN|SECRET|KEY|PASSWORD", re.IGNORECASE)


class WorkerSpec(BaseModel):
    """A machine (or a share of one) that runs units.

    `host`: the ssh alias (`transport: ssh`) or anything for `local`; `code_dir`: its code copy,
    built from the dispatch's frozen `sha` (`.roboscript/source.sha` or a clean git checkout);
    `python`: the agent venv's interpreter there (relative to `code_dir`); `gpus`: the cards its
    pipeline may use; `run_root`: where its units live (every unit directory is inside it, and
    the agent writes and deletes nowhere else); `gateway_url`: the model gateway its jobs call
    (`RS_GATEWAY_URL` in the unit's environment; None: that machine's own resolution,
    `roboscript.gateway.client.gateway_url`); `set`: its
    values of the unit's `local_keys` (dotted); `env`: environment for its units (no secrets: a
    worker reads its tokens from its own `.env`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    transport: str = "ssh"
    host: str
    code_dir: str
    python: str = ".venv/bin/python"
    gpus: list[NonNegativeInt] = Field(min_length=1)
    run_root: str
    gateway_url: str | None = None
    set: dict[str, Any] = {}
    env: dict[str, str] = {}

    @field_validator("name")
    @classmethod
    def _name(cls, name: str) -> str:
        if not _NAME.fullmatch(name):
            raise ValueError(f"worker name {name!r}: use [a-z0-9_-]")
        return name

    @field_validator("code_dir", "run_root")
    @classmethod
    def _absolute(cls, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError(f"{path}: a worker's paths are absolute")
        return path.rstrip("/")

    @field_validator("gpus")
    @classmethod
    def _gpus(cls, gpus: list[int]) -> list[int]:
        if len(set(gpus)) < len(gpus):
            raise ValueError(f"gpus lists a card twice: {gpus}")
        return gpus

    @field_validator("env")
    @classmethod
    def _no_secrets(cls, env: dict[str, str]) -> dict[str, str]:
        if named := sorted(k for k in env if _SECRET.search(k)):
            raise ValueError(f"env {named}: tokens stay in the worker's own .env")
        return env


class PinnedPath(BaseModel):
    """A path relative to the code directory that must be byte-identical on every worker (git
    keeps the code; these are the git-ignored inputs a run reads: configs, demo packages).
    `{task}` expands to each of a unit's tasks; `ship`: the coordinator copies its own to a
    worker where it is missing or differs, instead of refusing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    ship: bool = False

    @field_validator("path")
    @classmethod
    def _relative(cls, path: str) -> str:
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"{path}: pinned paths are relative to the code directory")
        return path


class UnitSpec(BaseModel):
    """How a unit (one pipeline launch on a worker) is started and read; nothing here is
    specific to a benchmark.

    `argv` runs in the worker's `code_dir` and both starts and resumes a unit (`{dir}` is the
    unit directory, `{python}` the worker's interpreter), as the pipeline's `--resume` does for a
    directory holding only its config; `config_file` is that config, written by the agent;
    `tasks_key` and `gpus_key` the config's fields that the dispatch sets; `local_keys` the
    dotted fields a worker may set (`WorkerSpec.set`), left out of the provenance; `forced`
    dotted values every worker's config gets and none may set (the GPU cap). `status_files`:
    the unit's status, newest first by time (the pipeline's `rows`, each `run` starting with
    its task and a `/`); `terminal`: the statuses a job ends in; a task whose jobs all ended is
    moved. `tombstones`: globs in a moved task's directory left behind so that a resumed launch
    does not run it again; `blobs`: the unit's content-addressed blob store;
    `blob_min_bytes`: a moved task's blobs at least this large that no other task names are
    deleted at the move (None: all wait for the unit to drain); `strip`: globs of a
    checkpoint's files left out when it seeds a unit on another worker (the lost machine's pid
    records, which a pid there could match by chance); `summary_argv`: run on the
    coordinator over its run root at the end (`{dir}`), for the pipeline's own summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: list[str] = Field(min_length=1)
    env: dict[str, str] = {}
    config_file: str = "pipeline.yaml"
    tasks_key: str = "tasks"
    gpus_key: str = "gpus"
    local_keys: list[str] = []
    forced: dict[str, Any] = {}
    status_files: list[str] = ["summary.json", "status.json"]
    terminal: list[str] = Field(min_length=1)
    tombstones: list[str] = []
    blobs: str | None = "blobs"
    blob_min_bytes: NonNegativeInt | None = 65536
    strip: list[str] = []
    pinned: list[PinnedPath] = []
    summary_argv: list[str] | None = None
    stop_grace_s: PositiveFloat = 120.0

    @model_validator(mode="after")
    def _check(self) -> Self:
        if "{dir}" not in " ".join(self.argv):
            raise ValueError("unit.argv must name the unit directory with {dir}")
        if clash := sorted(set(self.forced) & set(self.local_keys)):
            raise ValueError(f"{clash} are both forced and local: pick one")
        if named := sorted(k for k in self.env if _SECRET.search(k)):
            raise ValueError(f"unit.env {named}: tokens stay in each worker's own .env")
        return self


class DispatchConfig(BaseModel):
    """`name`: the dispatch (its directory `<root>/<name>/` on the coordinator, its units'
    `<run_root>/<name>/<unit>/` on the workers); `sha`: the frozen code revision every worker
    and the coordinator must run; `pipeline`: the pipeline config (a YAML file, relative paths
    relative to the repository root) and `pipeline_set` dotted overrides, composed by
    `composer` when the pipeline layers its file over others (`package.module:function` taking
    the file and the overrides, returning plain YAML data with relative paths; None: the file
    alone); `costs`: a task's
    expected cost (1 when not given) for the assignment; `policy`: the assignment policy
    (entry-point group `roboscript.dispatch_policies`); `poll_s` between the coordinator's
    rounds; a worker that cannot be reached for `lost_after_s` is lost and its unfinished tasks
    go to the others (from their last checkpoint, else afresh); a unit whose process ended with
    tasks unfinished is resumed on its worker after each of `relaunch_waits_s`, then its tasks
    are stuck (left on the worker for a human); `checkpoint`: copy an unfinished task's
    directory to the coordinator whenever one of its jobs ends (a second copy, while it runs,
    in exchange for not restarting from scratch on a lost worker); `seeds`: a directory here
    holding a prepared directory per task (`<seeds>/<task>/`), pushed into its unit's directory
    before the unit's first launch, so that the unit resumes work prepared on the coordinator (P4's
    evaluation blocks of a frozen Memory: `roboscript_methods.rsi`); None: units start empty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    root: Path
    sha: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    workers: list[WorkerSpec] = Field(min_length=1)
    tasks: list[TaskName] = Field(min_length=1)
    costs: dict[TaskName, PositiveFloat] = {}
    policy: str = "balanced"
    pipeline: Path
    pipeline_set: list[str] = []
    composer: str | None = None
    unit: UnitSpec
    poll_s: PositiveFloat = 60.0
    command_timeout_s: PositiveFloat = 600.0
    lost_after_s: PositiveFloat = 1800.0
    relaunch_waits_s: list[PositiveFloat] = [60.0, 300.0, 900.0]
    checkpoint: bool = False
    max_parallel_moves: PositiveInt = 2
    seeds: Path | None = Field(default=None, exclude_if=lambda seeds: seeds is None)

    @field_validator("name")
    @classmethod
    def _name(cls, name: str) -> str:
        if not _NAME.fullmatch(name):
            raise ValueError(f"dispatch name {name!r}: use [a-z0-9_-]")
        return name

    @field_validator("root", "pipeline", "seeds")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else repo_root() / path

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [w.name for w in self.workers]
        if len(set(names)) < len(names):
            raise ValueError(f"worker names repeat: {names}")
        if len(set(self.tasks)) < len(self.tasks):
            raise ValueError(f"tasks lists a task twice: {self.tasks}")
        if unknown := sorted(set(self.costs) - set(self.tasks)):
            raise ValueError(f"costs names tasks not dispatched: {unknown}")
        hosts: dict[tuple[str, str], WorkerSpec] = {}
        for worker in self.workers:
            if bad := sorted(set(worker.set) - set(self.unit.local_keys)):
                raise ValueError(
                    f"worker {worker.name} sets {bad}: only unit.local_keys "
                    f"{self.unit.local_keys} differ between workers"
                )
            other = hosts.setdefault((worker.transport, worker.host), worker)
            if other is not worker:  # two shares of one machine
                if shared := sorted(set(worker.gpus) & set(other.gpus)):
                    raise ValueError(f"workers {other.name} and {worker.name} share GPUs {shared}")
                if worker.run_root == other.run_root or worker.code_dir == other.code_dir:
                    raise ValueError(
                        f"workers {other.name} and {worker.name} on one host need their own "
                        "code_dir and run_root"
                    )
        return self

    def cost(self, task: TaskName) -> float:
        return self.costs.get(task, 1.0)

    @property
    def directory(self) -> Path:
        return self.root / self.name


def load_pipeline(config: DispatchConfig) -> dict[str, Any]:
    """The pipeline config every unit runs, as YAML data: the file with `pipeline_set` (or what
    the `composer` makes of them), then the unit's `forced` values, with the dispatch's tasks."""
    if config.composer is not None:
        data = entrypoints.load_ref(config.composer)(config.pipeline, list(config.pipeline_set))
    else:
        with config.pipeline.open(encoding="utf-8") as file:
            data = compose(yaml.safe_load(file) or {}, overrides=config.pipeline_set)
    if not isinstance(data, dict):
        raise ValueError(f"{config.pipeline}: the top level must be a mapping")
    forced = [f"{key}={canonical_json(value)}" for key, value in config.unit.forced.items()]
    composed = compose(data, overrides=forced)
    composed[config.unit.tasks_key] = list(config.tasks)
    return composed


def worker_config(
    pipeline: dict[str, Any], unit: UnitSpec, worker: WorkerSpec, tasks: list[TaskName]
) -> dict[str, Any]:
    """A worker's copy of the pipeline config for a unit of `tasks`."""
    sets = [f"{key}={canonical_json(value)}" for key, value in worker.set.items()]
    composed = compose(pipeline, overrides=sets)
    composed[unit.tasks_key] = list(tasks)
    composed[unit.gpus_key] = list(worker.gpus)
    return composed


def provenance(config: dict[str, Any], local_keys: list[str]) -> str:
    """The sha256 of the config without its local keys (dotted): what every worker shares."""
    shared = _without(config, local_keys)
    return hashlib.sha256(canonical_json(shared).encode("utf-8")).hexdigest()


def local_keys(unit: UnitSpec) -> list[str]:
    """The keys a provenance leaves out: the worker's own and the dispatch's per-unit ones."""
    return [*unit.local_keys, unit.tasks_key, unit.gpus_key]


def _without(config: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for key in keys:
        *parents, last = key.split(".")
        node: Any = result
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(last, None)
    return result


def load_dispatch(path: Path, overrides: list[str] | None = None) -> DispatchConfig:
    """A dispatch config from YAML, with dotted overrides."""
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return DispatchConfig.model_validate(compose(data, overrides=overrides or []))
