"""Job graphs as data (docs/how-to/add-a-method.md, `Method.jobs()`): a launch's nodes and edges,
and the interfaces of the parts that add them (a method's jobs, `TaskJobs`; a benchmark's
evaluation, `Evaluation`; any owner of nodes, `Part`). `roboscript.orchestrate.graph` composes
and runs them; these are what a method or a benchmark declares, importable from any layer.

A node is a job (a worker process on a GPU's simulator places: `JobSpec`) or an action (a short
step of its part on the run's records). Its edges name the nodes it follows and the outcomes of
each it accepts (`Edge`); an outcome is a status the node's records show once it ran, an
action's own, `SKIPPED`, `ERROR` or `REJECTED`.

A job's worker runs its body by the job's phase name (`JobKind`, which the method and the
evaluation give by phase: `Method.kinds`, `EvaluationFactory.kinds`), with what the worker's
command line says (`JobRequest`), and records how it ended (`JobEnd`)."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Self

SKIPPED: Final = "skipped"
REJECTED: Final = "rejected"
ERROR: Final = "error"
ANY: Final = "*"
"""In `Edge.accept`: every outcome of a node that ran (an error too; a rejection never)."""
EVALUATIONS: Final = "roboscript.evaluations"
"""The entry-point group of benchmarks' evaluations (`EvaluationFactory`), by benchmark name."""


class Edge(BaseModel):
    """A node follows `node`, accepting the outcomes in `accept` (`ANY`: all). A skipped `node`
    (`skip`): `pass` lets the node follow it as if accepted, `status` judges it by the status
    its records show now (`JobGraph.status`, against `accept`), `reject` rejects the node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node: str
    accept: frozenset[str] = frozenset()
    skip: Literal["pass", "status", "reject"] = "reject"


class JobSpec(BaseModel):
    """What a job node runs: its records' key (`unit`: the task's unit, e.g. a variant, and its
    `lane`, a job of its own beside the unit's), the phase its worker runs, the simulator places
    it holds (`weight`, of `shards` alike), its turn (`stage`, an index into `JobGraph.stages`;
    `first`: a task's first job; `spare`: a unit that waits behind every other job), what its
    records say while it waits as a spare unit (`note`), and what the worker gets besides
    (`modes`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str
    unit: str
    lane: str | None = None
    phase: str
    stage: NonNegativeInt
    first: bool = False
    weight: PositiveInt = 1
    shards: PositiveInt = 1
    spare: bool = False
    note: str | None = None
    modes: list[str] | None = None


class JobNode(BaseModel):
    """A node of a job graph: a job (`job` set) or an action (`action`, a name the graph acts
    on), after its edges."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    after: list[Edge] = []
    job: JobSpec | None = None
    action: str | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.job is None) == (self.action is None):
            raise ValueError(f"node {self.id!r}: a job or an action, exactly one")
        return self


class JobRequest(BaseModel):
    """What a job's worker is asked to run (its command line, `orchestrate.worker.command`): the
    run's directory, the job's phase, its key (task, `unit`, `lane`), its GPU, the URLs of that
    GPU's services by name and the modes it runs (None: its kind's default)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: Path
    phase: str
    task: str
    unit: str
    lane: str | None = None
    gpu: NonNegativeInt
    services: dict[str, str]
    modes: list[str] | None = None


class JobEnd(BaseModel):
    """How a job's body ended: the status its records show (`orchestrate.jobs.Status`), the stage
    they show from then on (None: the stage it ran at) and a note beside the status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    stage: str | None = None
    note: str | None = None


class JobKind(Protocol):
    """A job's body, by the phase it runs: the stage its records show while it runs (`stage`), and
    `run`, which runs it on the pipeline's `host` (the interface of what the kind needs of the
    pipeline is the kind's package's own) under `request`, `made` being what the job is made of
    (its records' `definition`)."""

    stage: str

    async def run(self, host: Any, request: JobRequest, made: dict[str, Any] | None) -> JobEnd: ...


Outcomes = Mapping[str, str]
"""The outcome of each node ended so far, by id."""


class UnitView(Protocol):
    """What a part sees of a node as it becomes ready: its key (a job key: task, unit, lane) and
    step, whether the launch took its unit up as the unit's chain began (`began`), the outcomes so
    far, the run's records (`records.retry_failed`) and a job's state and directory there."""

    key: Any
    step: str
    began: bool
    outcomes: Outcomes
    records: Any

    def state(self, key: Any = None) -> Any: ...

    def directory(self, key: Any = None) -> Any: ...


class GraphBuilder(Protocol):
    """A graph parts add nodes to, in order (`add`: a node of `part` for job key `key`, a job's
    `JobSpec` or an action's name; its id), with the phases of its jobs in the order `by_stage`
    places them (`stages`)."""

    stages: tuple[str, ...]

    def add(
        self, part: Any, key: Any, step: str, after: Sequence[Edge] = (), **what: Any
    ) -> str: ...


class Part(Protocol):
    """One owner's nodes in a `ComposedGraph` (a method's learning jobs, a benchmark's evaluation
    jobs): whether one is due, and an action's outcome."""

    async def due(self, unit: UnitView) -> bool: ...

    async def act(self, unit: UnitView) -> str: ...


class Evaluation(Part, Protocol):
    """A benchmark's evaluation jobs of a unit's frozen artifact (docs/how-to/add-a-method.md: the
    env package provides them): the phases its jobs run (`phases`, after the method's in the
    graph's stages), and `add`, which adds a unit's nodes to the graph after `learned` (the unit's
    artifact froze, or was frozen before) and `begun` (the unit's chain may begin: its task's
    first phase is done)."""

    phases: Sequence[str]

    def add(self, graph: GraphBuilder, key: Any, learned: Edge, begun: Edge) -> None: ...


class EvaluationFactory(Protocol):
    """A benchmark's evaluation as registered (`EVALUATIONS`): the model of its settings
    (`settings`, from the launch's config), its jobs' bodies (`kinds`), and the evaluation under
    them, where `gated(directory, task)` holds a unit from it and `modes` (None: all) limits a
    launch to reopening those."""

    settings: type[BaseModel]
    kinds: Mapping[str, JobKind]
    """The bodies of its jobs, by phase (`Evaluation.phases`)."""

    def __call__(
        self,
        settings: Any,
        gated: Callable[[Path, Any], bool],
        modes: Sequence[Any] | None = None,
    ) -> Evaluation: ...


class TaskJobs(Protocol):
    """A method's jobs of a launch (`Method.jobs`): the graph's `stages` (the method's phases,
    then the evaluation's), and `add`, which adds a task's nodes (the evaluation's of each unit
    among them) to the graph."""

    stages: Sequence[str]

    def add(self, graph: GraphBuilder, task: str) -> None: ...
