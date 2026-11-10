"""The P4 round driver (`rsi.rounds`) end to end with a fake benchmark: three rounds, a final block
and an added round, and the seed guard stopping an RSI arm evaluated on a block it read."""

import json
from pathlib import Path

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.ids import TaskName
from roboscript.learning.seeds import SeedRange
from roboscript.memory.artifact import KIND as TASK_MEMORY
from roboscript_methods.rsi.doc import Evidence, RsiDoc, RsiEntry
from roboscript_methods.rsi.recipe import load_rsi_recipe
from roboscript_methods.rsi.reflect import Reflected
from roboscript_methods.rsi.rounds import (
    DONE,
    DocSource,
    Driver,
    RoundInput,
    Stage,
    StageEpisode,
    add_rounds,
    import_round,
    load_plan,
    make_plan,
    write_plan,
)
from roboscript_methods.rsi.store import FileRsiStore, round_name
from tests.methods.rsi.records import Episode, Turn, made, write

RECIPE = load_rsi_recipe(Path("recipes/rsi"))
TASKS = [TaskName("lift_red"), TaskName("lift_blue")]
EPISODES = 2


def block(index: int) -> SeedRange:
    return SeedRange(start=100000 * (1 + index), count=100000)


def entry(entry_id: str, round: int) -> RsiEntry:
    evidence = Evidence(block=f"block{round}", task=TASKS[0], episode=0)
    return RsiEntry(
        id=entry_id,
        when="the box rests on the table",
        do="grasp it from above",
        check="it rises",
        evidence=(evidence,),
        added=round,
    )


async def round_dir(source: Path, round: int, doc: RsiDoc) -> Path:
    """A round's directory as scripts/p4_reflect.py writes it: G<r>.json and the store."""
    await FileRsiStore(source / "rsi").commit_round(round, doc, reason="test")
    (source / f"{round_name(round)}.json").write_text(doc.model_dump_json(), encoding="utf-8")
    (source / "gaps.md").write_text(f"round {round}: no gap\n", encoding="utf-8")
    return source


class FakeBenchmark:
    """Each stage's episodes: seeds from its block's start, the RSI arm succeeding on the first
    task where the baseline fails (its Actor citing G-01), and recorded once evaluated."""

    def __init__(self, root: Path, fail_on: str | None = None, seed_shift: int = 0) -> None:
        self.root, self.fail_on, self.shift = root, fail_on, seed_shift
        self.calls: list[tuple[str, str | None]] = []
        self.done: set[str] = set()
        self.unstarted: set[str] = set()  # stages with a task not started yet (`waiting`)

    async def evaluate(self, stage: Stage, doc: DocSource | None) -> None:
        self.calls.append((stage.name, None if doc is None else doc.ref.name))
        if stage.name == self.fail_on:
            self.fail_on = None
            raise RuntimeError(f"killed while evaluating {stage.name}")
        if stage.arm == "rsi":
            assert doc is not None and doc.ref.name == round_name(stage.round or 0)
        else:
            assert doc is None
        self.done.add(stage.name)

    def waiting(self, stage: Stage) -> bool:
        return stage.name in self.unstarted

    def episodes(self, stage: Stage) -> list[StageEpisode]:
        assert stage.name in self.done or (self.root / stage.name / DONE).is_file()
        found = []
        start = block(stage.block).start + self.shift
        for mode in stage.modes:
            for task in TASKS:
                for index in range(EPISODES):
                    rsi = stage.arm == "rsi"
                    success = (task == TASKS[0] and rsi) or index == 0
                    note = "applying G-01" if rsi else "lifting"
                    turn = Turn([made("scratchpad", note=note), made("finish", note="done")])
                    spec = Episode(task=task, index=index, success=success, turns=[turn])
                    where = self.root / "records" / stage.name / mode
                    directory = where / "records" / f"{task}.p3.seed0" / f"ep{index:03d}"
                    if not directory.exists():
                        write(where, LocalBlobStore(where / "blobs"), spec)
                    found.append(
                        StageEpisode(
                            task=task,
                            index=index,
                            seed=start + index,
                            success=success,
                            mode=mode,
                            directory=directory,
                            tokens=100,
                        )
                    )
        return found


class FakeReflector:
    """Round r's document: the previous one (after its counts) and one more GLOBAL entry."""

    def __init__(self) -> None:
        self.inputs: list[RoundInput] = []

    async def __call__(self, given: RoundInput) -> Reflected:
        self.inputs.append(given)
        doc = given.start.model_copy(
            update={
                "global_entries": (
                    *given.start.global_entries,
                    entry(f"G-{given.round:02d}", given.round),
                )
            }
        )
        ref = await FileRsiStore(given.directory / "rsi").commit_round(given.round, doc, reason="t")
        (given.directory / "gaps.md").write_text(f"round {given.round}: none\n", encoding="utf-8")
        return Reflected(ref=ref, doc=doc)


async def new_run(tmp: Path, gates: tuple[str, ...] = (), rounds: int = 3) -> Path:
    run = tmp / "run"
    run.mkdir()
    source = await round_dir(tmp / "d4", 1, RsiDoc(global_entries=(entry("G-01", 1),)))
    imported = import_round(run, source)
    memories = {
        t: ArtifactRef(kind=TASK_MEMORY, name=t, version="v003", hash=str(n) * 64)
        for n, t in enumerate(TASKS, 1)
    }
    plan = make_plan(
        eval_modes=("clean",),
        final_modes=("clean", "randomized"),
        count_mode="clean",
        name="p4-test",
        code="abcdef0",
        base="/base",
        tasks=memories,
        excluded={},
        rounds=rounds,
        block=block,
        episodes=EPISODES,
        shards=1,
        method="wicl+rsi",
        recipe="recipes/rsi",
        models={},
        gates=gates,
        imported=imported,
        taken={"demo": SeedRange(start=0, count=68)},
    )
    write_plan(run, plan)
    return run


async def failure(run: Driver) -> str:
    """The messages of what stopped `run` (inside the task groups' exception groups)."""
    try:
        await run.run()
    except BaseException as error:  # an exception group, or the error itself
        found, todo = [], [error]
        while todo:
            e = todo.pop()
            todo += list(getattr(e, "exceptions", ()))
            found.append(str(e))
        return " | ".join(found)
    raise AssertionError("the run did not stop")


def driver(run: Path, bench: FakeBenchmark, reflector: FakeReflector) -> Driver:
    return Driver(directory=run, evaluator=bench, reflector=reflector, recipe=RECIPE, poll_s=0.01)


async def test_three_rounds_a_final_block_and_an_added_round(tmp_path: Path) -> None:
    run = await new_run(tmp_path)
    plan = load_plan(run)
    assert [plan.eval_block(r) for r in (1, 2, 3)] == [2, 3, 4]
    assert plan.split.final == ("block5",) and plan.read_blocks(2) == (2,)
    bench, reflector = FakeBenchmark(run), FakeReflector()
    status = await driver(run, bench, reflector).run()
    assert status["finished"] and status["rounds"] == dict.fromkeys(("r1", "r2", "r3"), "counted")
    evaluated = dict(bench.calls)
    assert evaluated == {
        "baseline/block2": None,
        "baseline/block3": None,
        "baseline/block4": None,
        "baseline/block5": None,
        "rounds/r1/eval/block2": "G1",
        "rounds/r2/eval/block3": "G2",
        "rounds/r3/eval/block4": "G3",
        "final/block5": "G3",
    }
    # within the RSI lane the round's own block goes first: its count waits on it, the final not
    rsi = [name for name, _ in bench.calls if not name.startswith("baseline/")]
    assert rsi.index("final/block5") > rsi.index("rounds/r3/eval/block4")
    assert [n for n, _ in bench.calls if n.startswith("baseline/")] == [
        f"baseline/block{b}" for b in (2, 3, 4, 5)
    ]
    # round 1 imported, rounds 2 and 3 reflected on both arms of the previous round's block
    assert [i.round for i in reflector.inputs] == [2, 3]
    second = reflector.inputs[0]
    assert {(k.block, k.arm) for _, k in second.episodes} == {
        ("block2", "baseline"),
        ("block2", "rsi"),
    }
    assert second.evaluation == "block3" and [g.id for _, g in second.start.entries()] == ["G-01"]
    assert second.sources == [run / "baseline/block2", run / "rounds/r1/eval/block2"]
    # counts from paired outcomes and the cited id: the RSI arm won lift_red's seed 1, citing G-01
    counted = RsiDoc.model_validate_json((run / "rounds/r1/G1.counted.json").read_bytes())
    assert counted.find("G-01")[1].helpful == 1
    report = json.loads((run / "rounds/r1/report.json").read_bytes())
    assert (report["pairs"], report["rsi_won"], report["rsi_lost"]) == (4, 1, 0)
    assert report["entries"][0]["cited"] == 4 and report["tasks"]["lift_red"]["rsi_won"] == 1
    assert "## Tasks where RSI lost 2 or more pairs" in (run / "rounds/r1/report.md").read_text(
        encoding="utf-8"
    )
    for mode in ("clean", "randomized"):
        assert (run / f"final/block5/report-{mode}.json").is_file()
    assert "| 3 | block5 | randomized |" in (run / "report.md").read_text(encoding="utf-8")
    assert "round 2: none" in (run / "gaps.md").read_text(encoding="utf-8")

    bench.calls.clear()
    grown = add_rounds(run, 1, block)
    assert grown.split.final == ("block5", "block7") and grown.eval_block(4) == 6
    assert grown.read_blocks(4) == (4,)
    status = await driver(run, bench, reflector).run()
    assert status["finished"]
    assert sorted(bench.calls) == sorted(
        [
            ("baseline/block6", None),
            ("baseline/block7", None),
            ("rounds/r4/eval/block6", "G4"),
            ("final/block7", "G4"),
        ]
    )
    assert {(k.block, k.arm) for _, k in reflector.inputs[-1].episodes} == {
        ("block4", "baseline"),
        ("block4", "rsi"),
    }
    changes = (run / "plan-changes.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(changes[0])["final"] == 7


async def test_the_seed_guard_stops_an_rsi_arm_evaluated_on_a_block_it_read(tmp_path: Path) -> None:
    run = await new_run(tmp_path)
    plan = load_plan(run)
    stage = next(s for s in plan.stages() if s.name == "rounds/r2/eval/block3")
    episodes = [
        StageEpisode(task=TASKS[0], index=0, seed=block(2).start, success=True, mode="clean")
    ]
    with pytest.raises(ValueError, match="outside"):
        driver(run, FakeBenchmark(run), FakeReflector())._check_stage(plan, stage, episodes)
