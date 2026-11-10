# Operations

How runs are launched, resumed and spread over machines, and what they leave on disk.

## The pipeline

A pipeline takes each task from imitate through practice to evaluation. The core provides `roboscript.orchestrate`; each benchmark supplies a concrete pipeline. Jobs form a graph of nodes and edges. The launcher places workers by GPU memory, starts each GPU's services, and respawns workers on infrastructure failures. Each worker runs in its own process, so a crash ends only that job.

Before the first job, prechecks validate the run spec, confirm the brain names a model, and verify the code has a known clean version. Each task runs imitate once; variants then run practice and evaluation from the same Memory.

A variant sets only S2's `practice` parameters (`variants` in the pipeline config). The ablations:

- `success_only`: `{practice: {diagnosis: false}}`, the S2 Reviser gets no ground-truth diagnosis.
- `budget2x`: `{practice: {budget_factor: 2}}`, the S2 Reviser's edit budget doubled.
- `demo_none`, `demo_goal`, `demo_ops`, `demo_full`: `{practice: {demo_context: <level>}}`, what the S2 Actor and P3's Agent host read of the demonstration: nothing (a clean one-shot), the success frame, every operation's before/after views and the keypoint table, or those and the recording's frames.

## Resuming

`--resume <dir>` continues every unfinished job. `--retry-failed` also retakes failed ones. A resume refuses code other than the version the directory started with. Other flags: `--redo-p3 <task>:<mode>` discards and reruns a task's evaluation blocks in one mode, `--dry-run` shows what would happen, `--s1-only` stops after imitate, `--allow-dirty` permits uncommitted code.

## Dispatch

`roboscript.orchestrate.dispatch` spreads a run's tasks over machines with no shared filesystem. The coordinator assigns tasks once (`dispatch_policies`, built-in `balanced`), starts one unit per worker as a pipeline launch over that worker's tasks and GPUs. Worker commands run over a transport (`ssh` or `local`). Finished tasks copy back to the coordinator with a manifest check. If a worker is lost, its tasks go to others. If the coordinator dies, units keep running and `resume` picks up from `state.json`.

## Records

```
runs/<run_id>/
  run.json                 manifest: code version, configs, models, method, tiers
  <task>/<stage>/<seed>/<attempt>/
    config.json            resolved config, Memory hash, tool-schema versions
    events.jsonl           the episode's merged event stream
    llm_calls.jsonl        model request/response pairs
    turns.jsonl            per turn: observation, calls, action, end
    tool_calls.jsonl       per call: args, targets, planner result, reason
    physics.npz            per step: TCP poses, gripper command, measured opening
    summary.json           outcome, steps, turns, tokens, wall clock
  <task>/memory/           Memory versions; frozen.json names the one evaluation reads
  <task>/S2/
    ledger.jsonl  stop.json
    <seed>/first.json  try1/ ... try3/  cache/  merge.json
```

## Tests

Tests live in `tests/` by area. Markers: `unit` (default, no simulator/GPU/network), `sim`, `gpu`, `model`. Only LLMs, network and clock are mocked. A pytest plugin gives each run a directory and keeps the last ten under `test-results/`, with `summary.json` listing counts and failures.

## Runtime

The supervisor (`roboscript.stack`) starts services in dependency order, each in its own process group. Services bind port 0 on 127.0.0.1 and write a ready file. Three interpreters: the agent venv (uv), the simulator env (may pin older Python), and the vision env (hashed requirements, torch kept out of `uv.lock`). All cross-process communication uses pydantic contracts over JSON with bearer tokens. Bulk data goes through `roboscript.blobs` as `BlobRef`s.

Configuration is layered: core defaults, the method's, the generic recipe, the env binding, a variant, the machine's file, then `--set`. Copy from tracked `*.example.yaml` files; real configs are git-ignored.
