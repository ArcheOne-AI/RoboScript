# Learning

The core's learning harness (`roboscript.learning`) runs staged protocols over tasks. It knows phases, roles and tiers as data and names no stage. Stage logic lives in `roboscript_methods.wicl`; parameters live in `recipes/wicl/`.

## Roles

Three roles work on a task. The **Demo Reader** writes Memory v0 from the demonstration. The **Actor** acts. The **Reviser** edits Memory after a failure.

## The turn loop

Each Actor turn starts from a fresh observation, allows any number of information calls, and ends with one physical action or `finish`. Context holds the stage prompt, instruction, Memory, a brief and history. Images are sent for the last two exchanges (up to `tool_images`, default 8). Context is budgeted at 60% of the model's window; exceeding it shrinks history, and a second too-long refusal ends the episode as `context_overflow`.

## How an episode ends

| End | Causes | Effect |
|---|---|---|
| Success | official latch fires | A success, whatever the agent was doing |
| Finish | agent calls `finish` | Official check decides |
| Hard failure | empty grasp, plan failed, etc. | Ends the try in S1/S2; in evaluation, the agent recovers |
| Policy failure | max turns, wall clock, format errors, context overflow | Counts as a failure |
| Infrastructure | gateway error, env crash, timeout | Voided and rerun automatically |
| Configuration | bad model config, identify failure | Stops the entire run |

Budgets scale with the task's step limit (`RoleConfig.per_step`). By default an Actor gets 0.09 turns per step (range 45-150) and 0.4 model calls per step (floor 200).

## Imitate (S1)

The Demo Reader studies one demonstration and writes Memory v0. Then the Actor attempts the task on the demonstration's own scene. Hinted attempts see keypoint progress; unhinted attempts see only Memory. Exit requires `exit_passes` (3) unhinted passes in the last `exit_window` (5), within `max_attempts` (60). After each failure the Reviser edits Memory directly. If imitate runs out of attempts, the best Memory with an official success goes forward with `s1_converged: false`.

## Practice (S2)

The Actor replays the task on new scenes at tier D. Each seed gets up to 3 tries; only try 1 counts. On failure, the Reviser edits a cache copy within a budget. A success on try 2 or 3 merges the cache back into Memory.

```
for each seed in the practice block:
    stop if 3 of the last 5 firsts passed, or the cap (60) is reached
    try 1 with M  ->  success: record pass, next seed
                  ->  failure: record fail, C = copy(M), Reviser edits C
    try 2 with C  ->  success: merge C into M, next seed
                  ->  failure: Reviser edits C again
    try 3 with C  ->  success: merge C into M
                  ->  failure: discard C
```

Merging interpolates numbers: `alpha_k = max(0.2, 1/(1 + (k-1)/10))`, `B_k = max(1, round(4 * alpha_k))`. Each typed edit costs budget from B_k. A task that never converges uses `p3.unconverged` to pick its best Memory version by Wilson lower bound.

## Evaluate

The frozen Memory runs under the benchmark's official evaluation client. The benchmark owns the episode loop. The Agent host bridges the official client to the Hub. Hard failures trigger recovery (move to safe pose, try again) instead of ending the episode. An episode lost to infrastructure is rerun on the same seed, up to 3 times (`MAX_LOSSES`).

Memory access defaults to `library`: the Actor browses every task's frozen Memory and chooses what to use, rather than being handed its own task's Memory directly.

## Memory

A Task Memory is Markdown (TASK, OBJECTS, WORKFLOW, LESSONS, NOTES, SCRATCH) plus typed parameters in `params.json`. TASK states the task by role, with no arm and no appearance words; the demonstration's instruction is kept as metadata that no view shows, and each episode's own instruction is what the agent reads. Parameters are numbers with trust radii, categories, specs and done criteria. Every change is a typed `memory_edit`. The linter (`recipes/wicl/memory_lint.yaml`) catches asset annotations, simulator names, seeds and coordinates. Memory never binds a specific arm: a stage's arm is `<arm>`, which `plan_arms` assigns once per episode.

## Reflect

`rsi` attaches to any method's evaluation. A reflector reads evaluation records and writes an RSI document (rules and traps). The next evaluation reads that document beside the Memory. Rounds are capped in edits (8/5/3 for GLOBAL, 4/3/2 per task) and compared with McNemar's test on paired episodes.

## Demonstrations

A demonstration package holds the instruction, keyframes, keypoint tables, frames and an optional ground-truth layer. `demos.selection` ranks expert episodes offline; one is chosen per task before learning. The keypoint referee guards against skipped steps: a pass needs official success plus required keypoints reached in order.
