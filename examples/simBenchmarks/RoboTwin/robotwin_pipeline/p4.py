"""P4 on RoboTwin (docs/concepts/learning.md): the run config (`run/rsi/p4.yaml`, `P4Settings`), the
plan a new run freezes from a base pipeline run (`plan_from_base`), each evaluated stage prepared
and dispatched (`RoboTwinEvaluator`) and read back (`stage_episodes`), and the reflection of a round
(`reflector`). `scripts/run_p4.py` drives it with `roboscript_methods.rsi.rounds.Driver`.

A stage (an arm's block) is a P3-only pipeline run: each task's directory is prepared from the
base run (`seed_task`): its S1 recorded done, its variant queued at P3 with the base run's frozen
Task Memory (the version its P3 read, best-Memory fallback included), S2's stop and exit records
and end targets, the base run's P3 manifest, and the stage's block (`P3/p4.json`,
`roboscript_robotwin.eval.p4.P4Block`); the RSI arm's units also hold the artifact stack their
Actor reads and the round's frozen document store. `roboscript.orchestrate.dispatch` runs the
stage on the workers (`DispatchConfig.seeds`), the official client and server unchanged, and moves
each finished task back to the stage's directory.
"""

import json
import shutil
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from typing_extensions import Self

from roboscript.agent import ScriptedBrain, call
from roboscript.agent.session import Brain
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.ids import TaskName
from roboscript.gateway import openai_client
from roboscript.hub import Hub
from roboscript.learning.config import BrainSpec
from roboscript.learning.eval_view import EvalRecordView, read_eval_episode
from roboscript.learning.runner import brain_maker
from roboscript.learning.seeds import SeedRange
from roboscript.memory import FileMemoryStore
from roboscript.memory.artifact import manifest_stack, memory_ref
from roboscript.memory.manifest import MemoryManifest, read_manifest
from roboscript.orchestrate.dispatch import Coordinator, DispatchConfig, load_dispatch
from roboscript.orchestrate.jobs import RunState
from roboscript.records import EventWriter
from roboscript.tools import LocalProvider
from roboscript_methods.rsi.doc import EpisodeKey
from roboscript_methods.rsi.lint import LintContext
from roboscript_methods.rsi.method import CURATOR, REVIEWER, Rsi, RsiConfig, context_stack
from roboscript_methods.rsi.recipe import RsiRecipe
from roboscript_methods.rsi.reflect import Reflected, Reflection, reflect
from roboscript_methods.rsi.rounds import (
    STORE,
    DocSource,
    Reflector,
    RoundInput,
    Stage,
    StageEpisode,
)
from roboscript_methods.rsi.store import FileRsiStore
from roboscript_methods.rsi.toolset import RsiTools
from roboscript_methods.wicl.ops import S1
from roboscript_robotwin.benchmark import ROBOTWIN
from roboscript_robotwin.config import LINT_OVERLAYS
from roboscript_robotwin.eval.blocks import MANIFEST
from roboscript_robotwin.eval.p4 import STACK_FILE, STORE_DIR, P4Block, write_block
from roboscript_robotwin.eval.seed_of import block_episodes

S2_FILES: Final = ("s2_exit.json", "stop.json", "end_targets.json", "ledger.jsonl")
"""What a P3 job reads of S2 besides the Memory (`run.pick_s2`, `p3_gated`, `p3_demo`)."""
PREPARED: Final = ".p4"
"""Under a stage's directory: its prepared task directories (`seeds/`)."""


class ReflectSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    context_window: PositiveInt | None = None
    """`model`'s context window in tokens (None: the context budget's default)."""
    effort: str = "high"
    concurrency: int = Field(default=4, ge=1, le=8)
    frames_shown: PositiveInt = 12
    demos: Path | None = None
    """Each task's demonstration package `<demos>/<task>/package` (its final keyframe's RGB)."""
    scripted: bool = False
    """Scripted reflection roles (tests and plumbing checks)."""


class P4Settings(BaseModel):
    """`run/rsi/p4.yaml`: `root` holds the runs (`<root>/<name>/`); `base` the base pipeline run
    (read only) and its `variant`; `tasks` (empty: every task whose base P3 froze a Memory);
    `rounds`, `episodes`, `shards`, `gates`, the modes, `method` and `recipe` as the plan
    (`rounds.P4Plan`); `lint_overlay` RoboTwin's lint vocabulary over the recipe's; `dispatch` the
    dispatch config each stage runs with (its `name`, `root`,
    `tasks`, `seeds`, `sha` are set per stage; each worker's `run_root` gets
    `<name>/<stage>` appended) and `stage_set` the dotted overrides of the pipeline config every
    stage adds (P3's block, episodes, shards and modes are the stage's); `reflect` the
    reflection's models; `poll_s` how often a lane at a gate looks for its approval. `only`: the
    stages the run evaluates (`rounds.P4Plan.only`; empty: all), e.g. `[rounds/r1/eval/block2]`
    for one RSI arm paired afterwards against another run's baseline; `pins`: the directory, on
    the evaluating machine, of the expert's verdicts the run's units share (None: the run's own,
    `shared_pins`), e.g. a copy of another run's pins so both evaluate the same seeds at the
    same indices. `first_read`: the base run's blocks round 1 reflects on, block 1 on in order
    (default `[1]`); round 1 is evaluated on the block after them, e.g. `[1, 2]` with an imported
    document reflected on blocks 1 and 2 evaluates round 1 on block 3."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path
    base: Path
    variant: str = "default"
    tasks: list[TaskName] = []
    rounds: PositiveInt = 3
    episodes: int = Field(default=10, ge=1, le=100)
    shards: PositiveInt = 4
    gates: list[str] = []
    eval_modes: list[str] = ["clean"]
    final_modes: list[str] = ["clean", "randomized"]
    method: str = "wicl+rsi"
    recipe: str = "recipes/rsi"
    lint_overlay: Path = Path(LINT_OVERLAYS["rsi"])
    dispatch: Path
    stage_set: list[str] = []
    reflect: ReflectSettings
    poll_s: float = Field(default=60.0, gt=0)
    only: list[str] = []
    pins: Path | None = None
    first_read: list[PositiveInt] = Field(default=[1], min_length=1)

    @model_validator(mode="after")
    def _pins(self) -> Self:
        if self.pins is not None and not self.pins.is_absolute():
            raise ValueError(f"pins {self.pins}: an absolute directory on the evaluating machine")
        if self.first_read != list(range(1, len(self.first_read) + 1)):
            raise ValueError(f"first_read {self.first_read}: blocks 1 to k in order, e.g. [1, 2]")
        return self


def eval_block(index: int) -> SeedRange:
    """RoboTwin's official block `index` (100000 x (1 + index) on)."""
    start, count = ROBOTWIN.eval_seeds.block(index)
    return SeedRange(start=start, count=count)


def base_memories(
    base: Path, variant: str, tasks: Sequence[TaskName]
) -> tuple[dict[TaskName, ArtifactRef], dict[TaskName, str]]:
    """Each task's Task Memory as the base run's P3 read it (its P3 manifest, which must name
    the store's frozen version), and the tasks left out with why."""
    found, excluded = {}, {}
    names = tasks or sorted(TaskName(p.name) for p in base.iterdir() if (p / variant).is_dir())
    for task in names:
        unit = base / task / variant
        if not (unit / MANIFEST).is_file():
            excluded[task] = "the base run's P3 has no manifest (no frozen Memory evaluated)"
            continue
        entry = read_manifest(unit / MANIFEST).tasks.get(task)
        if entry is None:
            excluded[task] = f"the base run's P3 manifest names no Memory of {task}"
            continue
        frozen = json.loads((FileMemoryStore(unit).folder(task) / "frozen.json").read_bytes())
        if frozen["hash"] != entry.hash:
            raise ValueError(
                f"{unit}: the P3 manifest names {entry.hash}, the store froze {frozen['hash']}"
            )
        found[task] = memory_ref(task, entry)
    return found, excluded


def seed_task(
    base: Path,
    variant: str,
    target: Path,
    block: P4Block,
    store: Path | None,
) -> None:
    """Prepare `target` (`<seeds>/<task>`), new, from the base run's task directory: see the
    module. `store`: the round's document store (the RSI arm), copied in."""
    task = block.memory.name
    if target.exists():
        raise ValueError(f"{target} exists: a stage's task is prepared once")
    unit_base, unit = base / task / variant, target / variant
    shutil.copytree(FileMemoryStore(unit_base).folder(task), FileMemoryStore(unit).folder(task))
    for name in S2_FILES:
        if (found := unit_base / task / "S2" / name).is_file():
            (unit / task / "S2").mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, unit / task / "S2" / name)
    (unit / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(unit_base / MANIFEST, unit / MANIFEST)
    manifest = read_manifest(unit / MANIFEST)
    entry = manifest.tasks[task]
    if memory_ref(task, entry) != block.memory:
        raise ValueError(f"{unit_base}: its P3 manifest no longer names {block.memory}")
    write_block(unit, block)
    if block.doc is not None:
        if store is None:
            raise ValueError(f"{block.stage} reads {block.doc.name}: no store to copy")
        shutil.copytree(store / block.doc.name, unit / STORE_DIR / block.doc.name)
        stack = context_stack(manifest_stack(_one(manifest, task)), block.doc)
        (unit / STACK_FILE).write_text(stack.model_dump_json(indent=2) + "\n", encoding="utf-8")
    s1 = RunState(task=task, variant=S1, status="done", stage="s1")
    (target / S1).mkdir(parents=True)
    s1.save(target / S1)
    RunState(task=task, variant=variant, status="queued", stage="p3").save(unit)
    _owner_writable(target)


def _owner_writable(root: Path) -> None:
    """Every file and directory under `root` writable by its owner: the copies keep the base
    run's modes (a read-only base copy stays read-only), and the dispatch removes a moved task
    from its worker."""
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _one(manifest: MemoryManifest, task: str) -> MemoryManifest:
    return manifest.model_copy(update={"tasks": {task: manifest.tasks[task]}})


def stage_set(settings: P4Settings, stage: Stage, episodes: int, shards: int) -> list[str]:
    """The stage's pipeline overrides: one block, its episodes and shards, randomized in the
    same job when the stage evaluates it."""
    values: dict[str, Any] = {
        "p3.blocks": [stage.block],
        "p3.episodes": episodes,
        "p3.episodes_randomized": None,
        "p3.randomized": "randomized" in stage.modes,
        "p3.spare_parallel": False,
        "p3.shards": shards,
        "p3.steal_shards": shards > 1,
    }
    return [*settings.stage_set, *(f"{k}={canonical_json(v)}" for k, v in values.items())]


def stage_dispatch(
    settings: P4Settings,
    run_name: str,
    code: str,
    stage: Stage,
    directory: Path,
    tasks: Sequence[TaskName],
    episodes: int,
    shards: int,
) -> DispatchConfig:
    """The dispatch config of `stage` (its directory `directory`, `<root>/<name>`)."""
    template = load_dispatch(settings.dispatch)
    workers = [
        w.model_copy(update={"run_root": f"{w.run_root.rstrip('/')}/{run_name}/{stage.name}"})
        for w in template.workers
    ]
    return template.model_copy(
        update={
            "name": directory.name,
            "root": directory.parent,
            "sha": code,
            "tasks": list(tasks),
            "workers": workers,
            "pipeline_set": [*template.pipeline_set, *stage_set(settings, stage, episodes, shards)],
            "seeds": directory / PREPARED / "seeds",
        }
    )


def stage_episodes(
    directory: Path, stage: Stage, tasks: Sequence[TaskName], episodes: int
) -> list[StageEpisode]:
    """The first `episodes` official episodes of each task's block in each of the stage's modes,
    as `seed_of` reads them, with their tokens (the host's summary)."""
    found = []
    for task in tasks:
        for mode in stage.modes:
            for e in block_episodes(directory, task, mode, stage.block, episodes):
                summary = json.loads((e.directory / "summary.json").read_text(encoding="utf-8"))
                usage = summary.get("usage") or {}
                tokens = (
                    usage.get("input_tokens", 0) + usage.get("output_tokens", 0) if usage else None
                )
                lost = e.infra or e.voided is not None
                found.append(
                    StageEpisode(
                        task=task,
                        index=e.index,
                        seed=e.seed,
                        success=None if lost else e.success,
                        voided=e.voided is not None,
                        infra=e.infra,
                        tokens=tokens,
                        mode=mode,
                        directory=e.directory,
                    )
                )
    return found


WAITING: Final = ("queued", "stopped", "blocked", "preempted")
"""Job statuses (`roboscript.orchestrate.jobs.Status`) that have yet to (re)start."""


def shared_pins(template: DispatchConfig, run_name: str) -> Path:
    """Where the arms' units of a block share the expert's verdicts on its seeds (`P4Block.pins`):
    `<run_root>/<run name>/pins` on the machine that evaluates the task. The verdicts file is
    shared under a file lock on one machine, so every worker names the same `run_root` (one path,
    on each machine its own directory) and a task's units of a block run on one worker
    (`colocated`)."""
    roots = {w.run_root.rstrip("/") for w in template.workers}
    if len(roots) != 1:
        raise ValueError(
            f"the arms share the expert's verdicts under one path: the dispatch's workers must "
            f"name one run_root, not {sorted(roots)}"
        )
    (run_root,) = roots
    return Path(run_root) / run_name / "pins"


def colocated(run: Path, stage: Stage, workers: dict[TaskName, str]) -> None:
    """Refuse `stage` when one of its tasks goes to another worker than the block's other stages
    gave it (`workers`: task -> worker), since its verdicts would then be pinned on two machines.
    The built-in assignment is deterministic, so the stages of a block agree unless the dispatch
    template changed between them; a worker lost mid-stage moves its tasks, which this cannot
    catch."""
    block = f"block{stage.block}"
    rounds = sorted(run.glob(f"rounds/*/eval/{block}"))
    siblings = [run / "baseline" / block, *rounds, run / "final" / block]
    for other in siblings:
        state = other / "state.json"
        if other == run / stage.name or not state.is_file():
            continue
        tasks = json.loads(state.read_text(encoding="utf-8"))["tasks"]
        moved = {
            t: (tasks[t]["worker"], w)
            for t, w in workers.items()
            if t in tasks and tasks[t]["worker"] != w
        }
        if moved:
            raise RuntimeError(
                f"{stage.name}: tasks on other workers than in {other.relative_to(run)} "
                f"(there, here) {moved}: a block's arms must share each task's machine"
            )


class RoboTwinEvaluator:
    """`rounds.Evaluator` on RoboTwin: each stage prepared from the base run and dispatched."""

    def __init__(
        self,
        settings: P4Settings,
        run: Path,
        plan_name: str,
        code: str,
        method: str,
        memories: dict[TaskName, ArtifactRef],
        episodes: int,
        shards: int,
    ) -> None:
        self.settings, self.run, self.name, self.code = settings, run, plan_name, code
        self.method, self.memories, self.episodes_per, self.shards = (
            method,
            memories,
            episodes,
            shards,
        )
        self.pins = settings.pins or shared_pins(load_dispatch(settings.dispatch), plan_name)

    def tasks(self) -> list[TaskName]:
        return sorted(self.memories)

    def block_of(self, stage: Stage, task: TaskName, doc: DocSource | None) -> P4Block:
        return P4Block(
            run=self.name,
            stage=stage.name,
            arm=stage.arm,
            round=stage.round if doc is not None else None,
            block=stage.block,
            modes=stage.modes,
            episodes=self.episodes_per,
            shards=self.shards,
            memory=self.memories[task],
            method=self.method if doc is not None else None,
            doc=None if doc is None else doc.ref,
            pins=self.pins,
        )

    def prepare(self, stage: Stage, doc: DocSource | None) -> Path:
        """The stage's prepared task directories (made once: an interrupted preparation is made
        again from scratch)."""
        directory = self.run / stage.name
        seeds = directory / PREPARED / "seeds"
        if seeds.is_dir():
            return seeds
        making = directory / PREPARED / "seeds.making"
        if making.exists():
            shutil.rmtree(making)
        making.mkdir(parents=True)
        for task in self.tasks():
            block = self.block_of(stage, task, doc)
            seed_task(
                self.settings.base,
                self.settings.variant,
                making / task,
                block,
                None if doc is None else doc.store,
            )
        making.rename(seeds)
        return seeds

    async def evaluate(self, stage: Stage, doc: DocSource | None) -> None:
        self.prepare(stage, doc)
        directory = self.run / stage.name
        config = stage_dispatch(
            self.settings,
            self.name,
            self.code,
            stage,
            directory,
            self.tasks(),
            self.episodes_per,
            self.shards,
        )
        coordinator = Coordinator(config)
        workers = {t: s.worker for t, s in coordinator.state.tasks.items()}
        colocated(self.run, stage, workers)
        await coordinator.run()
        unfinished = {
            t: s.status for t, s in coordinator.state.tasks.items() if s.status != "moved"
        }
        if unfinished:
            raise RuntimeError(
                f"{stage.name}: tasks did not finish {unfinished} (dispatch status in {directory})"
            )

    def episodes(self, stage: Stage) -> list[StageEpisode]:
        return stage_episodes(self.run / stage.name, stage, self.tasks(), self.episodes_per)

    def waiting(self, stage: Stage) -> bool:
        """Whether a task of the stage has a job that has not started, or waits to start again
        (its dispatch's `status.json`; none yet: nothing started)."""
        status = self.run / stage.name / "status.json"
        if not status.is_file():
            return True
        found = json.loads(status.read_text(encoding="utf-8"))
        seen = {row["run"].split("/")[0] for row in found["rows"]}
        if set(self.tasks()) - seen:  # a unit that has not reported yet
            return True
        return any(row.get("status") in WAITING for row in found["rows"])


def combined_blobs(sources: Sequence[Path], target: Path) -> LocalBlobStore:
    """One blob store holding the stores of `sources` (each `<stage>/blobs`), hard-linked into
    `target` (blobs are named by content, so the union is safe)."""
    for source in sources:
        blobs = source / "blobs"
        if not blobs.is_dir():
            continue
        for path in blobs.rglob("*"):
            if path.is_file():
                out = target / path.relative_to(blobs)
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        out.hardlink_to(path)
                    except OSError:
                        shutil.copy2(path, out)
    return LocalBlobStore(target)


def _scripted(role: str, label: str, keys: Sequence[EpisodeKey]) -> Brain:
    """A role that lists its task's episodes (a reflector), reads one and finishes; the curator
    and the reviewer read the document and finish (plumbing checks, as p4_reflect --scripted)."""
    if role in (CURATOR, REVIEWER):
        return ScriptedBrain([[call("rsi_doc")], [call("finish", note=f"{role}: nothing to do")]])
    first = next(key for key in keys if key.task == label)
    return ScriptedBrain(
        [
            [call("rsi_episodes", task=label)],
            [call("rsi_episode", key=first.label(), turns="1-2")],
            [call("finish", note="scripted: nothing proposed. Gaps: none named")],
        ]
    )


def reflector(settings: ReflectSettings, recipe: RsiRecipe, tasks: Sequence[TaskName]) -> Reflector:
    """`rounds.Reflector`: a round's reflection (`rsi.reflect`) on the D parts of its episodes,
    keyed with their arms, the images from the evaluated stages' blob stores (`blobs/`
    hard-linked into the round's `source-blobs/`), each task's demonstration package under
    `settings.demos`, the brains through the model gateway (or scripted)."""

    async def run(round_input: RoundInput) -> Reflected:
        out = round_input.directory
        episodes, keys = [], []
        for entry, key in round_input.episodes:
            assert entry.directory is not None  # the driver passes recorded episodes only
            read = read_eval_episode(entry.directory, entry.seed)
            if read is None:
                continue
            episodes.append(read)
            keys.append(key)
        view = EvalRecordView.of(episodes)
        source = combined_blobs(round_input.sources, out / "source-blobs")
        demos = {}
        if settings.demos is not None:
            demos = {t: settings.demos / t / "package" for t in sorted({k.task for k in keys})}
        if settings.scripted:

            def brain(role: str, label: str) -> Brain:
                return _scripted(role, label, keys)
        else:
            config = {
                "model": settings.model,
                "reasoning_effort": settings.effort,
                "parallel_tool_calls": True,
                "history": {"tool_images": settings.frames_shown},
            }
            if settings.context_window is not None:
                config["context"] = {"windows": {settings.model: settings.context_window}}
            make = brain_maker(BrainSpec(name="responses", config=config))

            def brain(role: str, label: str) -> Brain:
                return make()

        tools = RsiTools()
        own = LocalBlobStore(out / "blobs")
        with EventWriter(out / "events.jsonl", "p4_reflect") as events:
            hub = Hub([LocalProvider(tools)], blobs=own, phases=Rsi.phases, events=events)
            reflection = Reflection(
                hub=hub,
                tools=tools,
                brain=brain,
                blobs=own,
                store=FileRsiStore(out / STORE),
                directory=out,
                recipe=recipe,
                context=LintContext(vocabulary=recipe.lint, tasks=sorted(tasks)),
                source=source,
                demos=demos,
                models=None if settings.scripted else openai_client,
                events=events,
                concurrency=settings.concurrency,
            )
            async with hub:
                return await reflect(
                    reflection,
                    RsiConfig(evaluation=round_input.evaluation),
                    view,
                    round_input.split,
                    round=round_input.round,
                    keys=keys,
                    start=round_input.start,
                )

    return run
