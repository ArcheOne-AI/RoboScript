"""P4 self-improvement (RSI, docs/concepts/learning.md): after an evaluation, reflection roles
read its records and write an RSI document apart from the Task Memory; an evaluation on another
seed block then reads both. Attaches after any method's evaluation (`method.attach`).

- `doc`: the RSI document (`RsiDoc`, artifact kind `rsi`: GLOBAL and `TASK <name>` parts, entries
  (rules or traps) with a stable id, when, do, check, evidence, helpful and harmful counts,
  status and provenance), `render`, its `context` for an Actor
- `edits`: typed edits (add, modify, merge, retire, vote) under a round's decaying budgets, field
  caps, token budgets, dedup and GLOBAL's evidence rule (`RsiEditor`)
- `lint`: GLOBAL names no task, object instance or number; no seeds, coordinates or evaluator
  internals anywhere (the recipe's vocabulary); names the Task Memory sets are flagged
- `counters`: helpful and harmful counts from paired outcomes, probation, promotion, retirement
- `recipe`: `recipes/rsi/` loaded and validated (`RsiRules`, `RsiLintConfig`, `ReflectRules`,
  the roles' prompts)
- `store`: `FileRsiStore` (`roboscript.artifact_stores`: `rsi`), one frozen `G<r>` per round
- `digest`: the deterministic per-episode digest a reflection reads first (D tier only)
- `staging`: proposals, what each seat may propose, verdicts, one revision, the order applied
- `toolset`: `RsiTools` (`roboscript.toolsets`: `rsi`; `rsi_episodes`, `rsi_episode`,
  `rsi_frames`, `rsi_doc`, `rsi_memory`, `rsi_edit`, `rsi_drop`, `rsi_verdict`)
- `method`: `p4_reflect` and `p4_eval` (`P4_REFLECT`, `P4_EVAL`), the reflector, curator and
  reviewer, `Rsi` (`roboscript.methods`: `rsi`), `attach`, `RsiConfig`, `context_stack`
- `reflect`: one reflection round (`reflect`, `Reflection`)
- `rounds`: the round driver (`P4Plan`, `Driver`): evaluate -> count -> reflect -> freeze -> next
  round over a seed chain, the baseline and RSI arms paired by seed, resumable, with approval gates
- `report`: a round's report (`round_report`): the two arms paired, McNemar's test, per task
"""
