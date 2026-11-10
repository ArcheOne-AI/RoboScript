# RoboTwin binding of `wicl`

RoboTwin-specific overrides layered over the generic recipe in `recipes/wicl/`:

| File | Holds |
|---|---|
| `pipeline.yaml` | Benchmark, tasks, demonstrations, env servers, GPU services, evaluation settings, seed split |
| `memory_lint.yaml` | RoboTwin object words, asset annotation names, simulator names |
| `protocols.yaml` | First seeds and expert pre-run tries |

The machine's `configs/pipeline.yaml` goes on top and names this directory as its `binding`.
