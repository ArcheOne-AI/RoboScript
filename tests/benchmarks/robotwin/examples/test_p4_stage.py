"""A P4 stage on MockEnv (`robotwin_pipeline.p4`): task directories prepared from a finished base
run evaluate one block of its frozen Task Memory, P3 only; the baseline arm's Actor reads the base
run's P3 prompt byte for byte, the RSI arm's the same prompt with the round's document after it."""

import json
from pathlib import Path

import anyio
import pytest
import yaml
from pipeline_fakes import config as pipeline_config
from pipeline_fakes import record_demo, wiring
from robotwin_pipeline import load_directory, new_directory, run_pipeline
from robotwin_pipeline.p4 import (
    P4Settings,
    ReflectSettings,
    RoboTwinEvaluator,
    base_memories,
    stage_set,
)
from test_pipeline import RED, agent_venv
from test_pipeline import root as root  # the fixture

from roboscript.contracts.ids import TaskName
from roboscript.kernel.compose import compose
from roboscript_methods.rsi.doc import Evidence, RsiDoc, RsiEntry
from roboscript_methods.rsi.rounds import DocSource, Stage
from roboscript_methods.rsi.store import FileRsiStore
from roboscript_robotwin.eval.seed_of import block_episodes

DOC = RsiDoc(
    global_entries=(
        RsiEntry(
            id="G-01",
            when="the box rests on the table",
            do="grasp it from above",
            check="it rises",
            evidence=(Evidence(block="block1", task=RED, episode=0),),
            added=1,
        ),
    )
)


def episode_configs(run: Path, block: int) -> list[dict]:
    found = sorted((run / RED / "default" / "P3" / "clean" / f"block{block}").rglob("config.json"))
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in found
        if (p.parent / "summary.json").is_file()
    ]


@agent_venv
async def test_a_stage_evaluates_the_base_runs_frozen_memory(root: Path) -> None:
    await record_demo(RED, root / "demos" / RED / "package")
    base_config = pipeline_config(root, [RED], {"default": {}})
    base = new_directory(base_config)
    with anyio.fail_after(240):
        await run_pipeline(base_config, wiring(), base)
    memories, excluded = base_memories(base, "default", [TaskName(RED)])
    assert list(memories) == [RED] and excluded == {}
    for path in [base, *base.rglob("*")]:  # a read-only base copy (P4 reads it, never writes)
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    store = root / "r1" / "rsi"
    ref = await FileRsiStore(store).commit_round(1, DOC, reason="test")
    episodes = base_config.p3.episodes_of("clean")
    workers = root / "workers"  # the stages run here in-process; the template names the pins
    template = {
        "name": "template",
        "root": str(root / "coord"),
        "sha": "abcdef0",
        "workers": [
            {
                "name": "w",
                "transport": "local",
                "host": "localhost",
                "code_dir": str(root),
                "gpus": [0],
                "run_root": str(workers),
            }
        ],
        "tasks": [RED],
        "pipeline": str(root / "pipeline.yaml"),
        "unit": {"argv": ["true", "{dir}"], "terminal": ["done"]},
    }
    (root / "dispatch.yaml").write_text(yaml.safe_dump(template), encoding="utf-8")
    reflect = ReflectSettings(model="scripted", scripted=True)
    settings = P4Settings(
        root=root / "p4", base=base, dispatch=root / "dispatch.yaml", reflect=reflect
    )
    evaluator = RoboTwinEvaluator(
        settings, root / "p4", "p4-test", "abcdef0", "p4_fakes:TestMethodRsi", memories, episodes, 1
    )
    baseline = Stage(name="baseline/block2", arm="baseline", block=2, modes=("clean",))
    rsi = Stage(name="rounds/r1/eval/block2", arm="rsi", round=1, block=2, modes=("clean",))
    for stage, doc in ((baseline, None), (rsi, DocSource(ref=ref, store=store))):
        seeds = evaluator.prepare(stage, doc)
        directory = root / "p4" / stage.name
        for task in seeds.iterdir():
            (directory / task.name).parent.mkdir(parents=True, exist_ok=True)
            task.rename(directory / task.name)
        data = compose(
            base_config.model_dump(mode="json"), overrides=stage_set(settings, stage, episodes, 1)
        )
        (directory / "pipeline.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        with anyio.fail_after(240):
            found = await run_pipeline(load_directory(directory), wiring(), directory)
        assert found["runs"] == {"done": 2}, found
        assert len(evaluator.episodes(stage)) == len(block_episodes(base, RED, "clean", 1)) > 0
    prepared = root / "p4" / baseline.name / RED / "default" / RED / "memory" / "frozen.json"
    assert prepared.stat().st_mode & 0o200  # the prepared copy is the stage's own, removable
    # the arms share the expert's verdicts on the block: the same seeds at the same indices
    pins = workers / "p4-test" / "pins"
    assert evaluator.block_of(baseline, RED, None).pins == pins
    assert (pins / RED / "clean-block2.jsonl").stat().st_size > 0
    indexed = [
        [(e.index, e.seed) for e in block_episodes(root / "p4" / s.name, RED, "clean", 2)]
        for s in (baseline, rsi)
    ]
    assert indexed[0] == indexed[1] and indexed[0]
    for stage in (baseline, rsi):
        block = root / "p4" / stage.name / RED / "default" / "P3" / "clean" / "block2"
        assert not (block / "shards" / "verdicts.jsonl").exists()
    [base_one, *_] = episode_configs(base, 1)
    [plain, *_] = episode_configs(root / "p4" / "baseline/block2", 2)
    [rsi_one, *_] = episode_configs(root / "p4" / "rounds/r1/eval/block2", 2)
    assert plain["instruction"] == base_one["instruction"]
    assert plain["prompt"] == base_one["prompt"] and plain["tools"] == base_one["tools"]
    assert plain["memory"] == base_one["memory"] == rsi_one["memory"]
    assert rsi_one["prompt"]["sha256"] != plain["prompt"]["sha256"]
    assert "overlay" not in plain and "overlay" not in base_one
    assert {layer["ref"]["hash"] for layer in rsi_one["overlay"]} == {ref.hash}
    assert plain["host"]["provenance"]["config"] == "p4:baseline"
    assert rsi_one["host"]["provenance"]["config"] == "p4:rsi"
    with pytest.raises(ValueError, match="prepared once"):
        from robotwin_pipeline.p4 import seed_task

        seed_task(
            base,
            "default",
            root / "p4" / baseline.name / RED,
            evaluator.block_of(baseline, RED, None),
            None,
        )
