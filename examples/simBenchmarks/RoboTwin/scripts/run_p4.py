"""Run P4 (self-improvement over rounds, docs/concepts/learning.md) on RoboTwin from a base pipeline
run's frozen Task Memory: `roboscript_methods.rsi.rounds.Driver` with RoboTwin's evaluator and
reflection (`robotwin_pipeline.p4`).

    PYTHONPATH=examples/simBenchmarks/RoboTwin .venv/bin/python \\
        examples/simBenchmarks/RoboTwin/scripts/run_p4.py new run/rsi/p4.yaml --name p4-run \\
        --import-g1 /path/to/g1-run [--set rounds=3]
    ... resume <run dir> [--approve r1]   go on (an approval reaches a running driver too)
    ... add-round <run dir> [--rounds 1]  more rounds on the next unused blocks, a new final block
    ... status <run dir>

`new` makes `<root>/<name>/`: the settings it runs with (`p4.yaml`), the frozen plan (`p4.json`:
the code, the base run, each task's Task Memory as the base run's P3 read it, the seed chain), round
1's document imported from a reflection made elsewhere (`--import-g1`, scripts/p4_reflect.py's
directory), then runs. A run stops at a gate (`gates: [r1]`) once nothing else runs; `resume
--approve r1` goes on. The coordinator's code must be a clean checkout (the dispatch's frozen
revision).
"""

import argparse
import json
import sys
from pathlib import Path

import anyio
import yaml
from robotwin_pipeline.p4 import (
    P4Settings,
    RoboTwinEvaluator,
    base_memories,
    eval_block,
    reflector,
)

from roboscript.kernel.compose import compose
from roboscript.learning.seeds import SeedRange
from roboscript.orchestrate.dispatch.agent import code_sha
from roboscript.settings import repo_root
from roboscript_methods.rsi.recipe import load_rsi_recipe
from roboscript_methods.rsi.rounds import (
    Driver,
    Imported,
    P4Plan,
    add_rounds,
    approve,
    import_round,
    load_plan,
    make_plan,
    run_status,
    running,
    write_plan,
)

SETTINGS = "p4.yaml"


def load_settings(path: Path, overrides: list[str]) -> P4Settings:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return P4Settings.model_validate(compose(data, overrides=overrides))


def driver(directory: Path) -> Driver:
    settings = P4Settings.model_validate(yaml.safe_load((directory / SETTINGS).read_text("utf-8")))
    plan = load_plan(directory)
    recipe = load_rsi_recipe(Path(plan.recipe), settings.lint_overlay)
    evaluator = RoboTwinEvaluator(
        settings,
        directory,
        plan.name,
        plan.code,
        plan.method,
        dict(plan.tasks),
        plan.episodes,
        plan.shards,
    )
    return Driver(
        directory=directory,
        evaluator=evaluator,
        reflector=reflector(settings.reflect, recipe, sorted(plan.tasks)),
        recipe=recipe,
        poll_s=settings.poll_s,
    )


def new(args: argparse.Namespace) -> int:
    settings = load_settings(args.config, args.set)
    sha, dirty = code_sha(repo_root())
    if dirty:
        print(f"{repo_root()} has uncommitted changes: P4 runs a frozen revision", file=sys.stderr)
        return 2
    directory = settings.root / args.name
    if directory.exists():
        print(f"{directory} exists: resume it, or name a new run", file=sys.stderr)
        return 2
    base = settings.base.resolve()
    memories, excluded = base_memories(base, settings.variant, settings.tasks)
    pipeline = yaml.safe_load((base / "pipeline.yaml").read_text(encoding="utf-8"))
    taken = {
        name: SeedRange.model_validate(block)
        for name, block in ((pipeline.get("seeds") or {}).get("blocks") or {}).items()
    }
    brains = (pipeline.get("learning") or {}).get("brains") or {}
    load_rsi_recipe(
        Path(settings.recipe), settings.lint_overlay
    )  # a recipe that does not load stops the run here

    def planned(imported: Imported | None) -> P4Plan:
        return make_plan(
            name=args.name,
            code=sha,
            base=str(base),
            tasks=memories,
            excluded=excluded,
            rounds=settings.rounds,
            block=eval_block,
            episodes=settings.episodes,
            shards=settings.shards,
            method=settings.method,
            recipe=settings.recipe,
            models={
                "reflect": settings.reflect.model_dump(mode="json"),
                "actor": brains.get("actor"),
            },
            gates=settings.gates,
            eval_modes=settings.eval_modes,
            final_modes=settings.final_modes,
            count_mode="clean",
            imported=imported,
            taken=taken,
            only=settings.only,
            first_read=len(settings.first_read),
        )

    plain = planned(None)  # the chain and its seed guard, before anything is written
    directory.mkdir(parents=True)
    (directory / SETTINGS).write_text(
        yaml.safe_dump(settings.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    plan = planned(import_round(directory, args.import_g1.resolve(), plain))
    write_plan(directory, plan)
    print(f"{directory}: {len(memories)} tasks, {len(excluded)} left out {sorted(excluded)}")
    return run(directory)


def run(directory: Path) -> int:
    status = anyio.run(driver(directory).run)
    print(json.dumps(status, indent=2))
    return 0


def resume(args: argparse.Namespace) -> int:
    directory = args.directory.resolve()
    for gate in args.approve:
        approve(directory, gate)
    if (pid := running(directory)) is not None:
        print(f"driver pid {pid} runs {directory}: it reads the approvals as it waits")
        return 0
    return run(directory)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    made = sub.add_parser("new")
    made.add_argument("config", type=Path)
    made.add_argument("--name", required=True)
    made.add_argument("--import-g1", type=Path, required=True)
    made.add_argument("--set", action="append", default=[])
    again = sub.add_parser("resume")
    again.add_argument("directory", type=Path)
    again.add_argument("--approve", action="append", default=[])
    more = sub.add_parser("add-round")
    more.add_argument("directory", type=Path)
    more.add_argument("--rounds", type=int, default=1)
    shown = sub.add_parser("status")
    shown.add_argument("directory", type=Path)
    args = ap.parse_args()
    if args.command == "new":
        return new(args)
    if args.command == "resume":
        return resume(args)
    if args.command == "add-round":
        plan = add_rounds(args.directory.resolve(), args.rounds, eval_block)
        print(f"{plan.rounds} rounds; final blocks {list(plan.split.final)}")
        return 0
    print(json.dumps(run_status(args.directory.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
