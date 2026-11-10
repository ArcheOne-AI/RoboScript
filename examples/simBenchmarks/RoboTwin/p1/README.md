# Building the RoboTwin skillset

RoboTwin's glue for the skill builder (`roboscript_methods.skill_builder`). A coding agent
writes skills for each task's demonstrations, a driver gates them, and accepted skills are
frozen as a skillset. RoboTwin's current skillset is the builder's output.

## Setup

Copy `configs/builder.example.yaml` to `configs/builder.yaml`. Set `backend` (`codex` or `fake`)
and `model`. The gateway reads `configs/gateway.yaml`; authenticate with `RS_GATEWAY_TOKEN` from
`.env`.

## Steps

1. Record demonstrations with transfer demos:
   ```bash
   python examples/simBenchmarks/RoboTwin/scripts/record_demos.py --demos DIR --transfer 3
   ```

2. Export for the builder:
   ```bash
   python demo_export.py --recorded DIR/recorded --out <recordings> --transfer
   ```

3. Run entry checks and measure criteria:
   ```bash
   uv run python entry_check.py --task place_empty_cup --python <sim>/bin/python \
       --recordings <recordings> --planner http://127.0.0.1:<port> --out entry/
   uv run python entry_check.py --measure-criteria entry/run1 entry/run2 --out entry/
   ```

4. Build (`--dry-run` first, drop it to run):
   ```bash
   uv run python build.py --run <name> --runs <runs> --python <sim>/bin/python \
       --recordings <recordings> --criteria entry/ --gpus 0,1 --dry-run
   ```

5. Freeze:
   ```bash
   uv run python freeze.py --run <runs>/<name> --name robotwin --version v1 --store <store>
   ```

Every job importing agent-written skills runs in a bubblewrap sandbox. `--no-sandbox` is
debug-only (accepted only with `--smoke`, and `freeze.py` refuses it).
