# rsi

Content of the reflection stage. `roboscript_methods.rsi.recipe.load_rsi_recipe` reads these
files. They hold no paths, ports, GPUs or secrets.

| File | What it holds |
|---|---|
| `rsi.yaml` | Edit budgets per round, entry field caps, token budgets, dedup threshold, retire rules |
| `lint.yaml` | Vocabulary rules for what an Actor reads (seeds, coordinates, arms, simulator names) |
| `reflect.yaml` | Reflection rules: tool names for keyframes, digest caps, review passes, model-call limits |
| `prompts/*.md` | Reflector, curator and reviewer prompts (plus consolidation variants) |

A round reads its own seed blocks' records and is evaluated on a block it did not read.
