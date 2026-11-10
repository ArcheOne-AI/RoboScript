# RoboTwin 2.0 integration

50 bimanual tasks on the `aloha_agilex` robot, clean and randomized. The importable package is
`roboscript_robotwin` in `packages/benchmarks/robotwin/`. This directory holds configs, scripts,
the `robotwin_pipeline` scheduler, the wicl binding in `run/wicl/`, reflection config in `run/rsi/` and
the skill builder glue in `p1/`.

## Installation

Install [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) at the commit
`roboscript_robotwin/eval/install.py` expects and set `ROBOTWIN_ROOT` in `.env`. Two interpreters:

**Agent venv** (runs the orchestrator, Hub, toolsets, brains):
```bash
uv sync --all-packages
```

**Simulator env** (runs the env server, nothing of ours installed):
```bash
conda create -n roboscript-sim --clone <robotwin-env>
conda activate roboscript-sim
export PYTHONNOUSERSITE=1
python -m pip install -r examples/simBenchmarks/RoboTwin/requirements-sim.txt
export PYTHONPATH=$PWD/packages/core/src:$PWD/packages/vision/src:$PWD/packages/benchmarks/robotwin/src
```

The vision service has its own environment (`packages/vision/README.md`). Copy `.env`, `.env.api`
and `configs/*.yaml` from their `*.example` files.

## Recording demonstrations

```bash
python examples/simBenchmarks/RoboTwin/scripts/select_demos.py --out data/demos
cp examples/simBenchmarks/RoboTwin/configs/demos.example.yaml configs/demos.yaml
python examples/simBenchmarks/RoboTwin/scripts/record_demos.py --demos data/demos --gpus 0,1 --per-gpu 4
```

Results go to `data/demos/recorded/` with `summary.json` and `index.html`.

## Running the pipeline

`scripts/run_pipeline.py` runs S1, S2 and the official evaluation for every task. Copy configs first: `pipeline.yaml`,
`manipulation.yaml`, `vision.yaml` and `p3.yaml` from their examples. Start the gateway, then:

```bash
mkdir -p <root>
PYTHONPATH=examples/simBenchmarks/RoboTwin nohup .venv/bin/python \
    examples/simBenchmarks/RoboTwin/scripts/run_pipeline.py \
    > <root>/launch.log 2>&1 & echo $! > <root>/pipeline.pid

# stop (each job can resume)
kill -TERM $(cat <root>/pipeline.pid)

# resume
... run_pipeline.py --resume <root>/<stamp>
```

Flags: `--retry-failed`, `--dry-run`, `--s1-only`, `--redo-p3`, `--p3-mode`, `--set`,
`--allow-dirty`. See `--help`.

## Official evaluation

RoboTwin's own client/server drives evaluation. `roboscript_robotwin.eval.install` links the
policy package into the checkout. The pipeline runs the evaluation itself; `roboscript_robotwin.eval.launch`
shows how to evaluate one block by hand.

## Reading results

A launch writes `<root>/<stamp>/`:

- `pipeline.yaml` -- composed config a resume reads
- `status.md` -- refreshes while running
- `summary.md`, `summary.json` -- S1 attempts, S2 first-try curve, evaluation success rate with Wilson interval
- `<task>/P3/<mode>/block<n>/result.json` -- per-block outcomes

`RS_RUNS_ROOT` puts `runs/` on another disk.

## Tests

```bash
uv run pytest tests/benchmarks/robotwin        # no simulator needed
uv run pytest tests/benchmarks/robotwin -m sim  # needs ROBOTWIN_ROOT
```
