# wicl

Content of the `wicl` method: learns each task from one demonstration (S1), practices on new seeds
to distill a per-task Memory (S2), then evaluates the frozen Memory. These files hold no
paths, ports, GPUs or secrets.

| File | What it holds |
|---|---|
| `protocols.yaml` | Role episode bounds, stage parameters for S1, S2 and evaluate |
| `memory.yaml` | Edit rules per stage, the learning-rate schedule, budget formula |
| `memory_lint.yaml` | Linter vocabularies (asset names, simulator words, coordinates, etc.) |
| `pipeline.yaml` | Ablation variants, protocol parameters, role brains |
| `prompts/*.md` | Each role's method text |

A benchmark overlays its own words in `memory_lint.yaml` via `LearningConfig.lint_overlay`.
