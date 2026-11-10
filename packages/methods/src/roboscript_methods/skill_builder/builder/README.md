# Builder: written skills, gated by the driver

The step where a coding agent writes the skills the demonstrations need
(`docs/how-to/add-a-skillset.md` §8). The coverage pipeline (`../core_entry`, `../demo_coverage`
and a benchmark's glue) is fixed. What changes from round to round is the set of skills the agent
wrote and the driver accepted.

## Pieces

| File | What |
|---|---|
| `plugins.py` | `WrittenSkill`, `load_skills(dirs)`: a written skill as coverage and the gate run it |
| `static_check.py` | the gate's static checks on a skill directory |
| `codex.py`, `prompts/builder.md`, `brief.py` | one coding session on one gap cluster, in a sandbox workspace, through the gateway |
| `prompts/task.md`, `plans.py` | one coding session on one task, and its task plan (data) |
| `clusters.py`, `driver.py` | rounds: coverage and reuse, gap clusters, sessions, gates, acceptance, rerun, per-task closure |

A benchmark's glue provides the jobs the driver runs: a coverage job that takes written skills
(they join reuse and the ceiling), a chain job (the task's plan rolled out from the reset), a case
job (one segment, JSON out) and a transfer case job (the gate's cases on exported demonstrations).
Each benchmark's skill-builder directory documents its commands, e.g. `examples/simBenchmarks/RoboTwin/p1/`.

## A written skill

A directory named after the skill (snake_case, never a task name), holding:

- `skill.py`, run at tier D, as the Agent at S2 and in evaluation would run it:
  - `NAME: str` and `KINDS: tuple[str, ...]`, the segment kinds it may explain (`grasp`,
    `place`, `carry`, `touch`, `release`, `move`, `home`; see `demo_coverage.segments`).
  - `SPEC`: a `roboscript.contracts.skills.SkillSpec`.
  - `async def act(api, arm, obj, target, params) -> list[Built]`, which builds the motions. The
    caller runs them in order and stops at the first receipt that is not ok. It sees only the
    action API, the object proxies the caller gives and `params`. It imports only
    `roboscript.contracts`, `roboscript.spatial`, `roboscript.manipulation`, numpy and the
    standard library.
- `fit.py`, run at build time only (tier P): `fit(segment, recording, scene, base, poses)`
  returns the parameters of `act` that reproduce one segment of a demonstration, or None where
  the skill does not apply. It may read the recording and the truth scene. Its parameters hold no
  absolute position (`demo_coverage.fitting.RELATIVE_RULE`): each fit runs again on the
  demonstration moved across the table, and a number that moves with it by more than 1 cm
  refuses the fit. A pose the robot returns to in every scene (its rest pose) goes in joint
  space.
- `cases.yaml`: `gaps: [<task>/<segment>, ...]`, the gaps it was written for.

## Gaps

Each task is judged two ways. From each segment's own start (a replay of the demonstration up to
there), every segment at once. And in one rollout of the task's plan from the reset (`--chain`),
each call starting where the calls before it left the scene. Where the chain breaks, the
segment's alternatives are tried there and the first that explains it replaces the call. A
chained break is its own gap (`<task>/<segment>#chained`, or `#stages` when every segment is
explained but the benchmark's stages fall behind the demonstration's). Each round's
`summary.md` opens with five totals, before and after: segments explained in the chain, tasks
whose chain reaches the end, the ceiling's official successes, the stages done, and segments
explained from their own start.

## The gate

The driver runs the gate itself and never takes the agent's report. It judges a snapshot of the
skill whose content hash is checked again before acceptance. Every process that imports the skill
runs in the sandbox (`sandbox.py`) and writes only a fresh directory of its own.

1. **Static check and contract**, without network: allowed imports only, no reflection, no task
   name or object label, no position literal in `skill.py`; then the import and the contract
   (`plugins.load_one`).
2. **Cases**: the skill's `cases.yaml` gaps that are open this round. A case judged and not
   explained runs once more. It passes when one is explained. When every run broke on the
   infrastructure, the skill is held for the next gate instead of rejected.
3. **Transfer**, the check that a skill is relative and not a replay. Its fitted params run on
   the matching segment of up to 3 other demonstrations of the task whose object lies at least
   5 cm away. Only layouts where the core's own entry for the segment transfers count. It fails
   when at least 2 are judged and none is explained. These scenes are seen by the driver only;
   the agent is told the verdict.
4. **Regression**: the round's sweeps rerun with every accepted skill and the candidates, paired
   against the round's coverage. A loss is swept again before it blames a candidate. A candidate
   is dropped only where its confirmed losses outweigh its gains.

A written skill's `act` is bounded (900 s, 32 motions). A timeout or error inside it is the
skill's failure, never the infrastructure's.

## Per-task closure

A task is at its target when at least 2 of its ceiling's first 4 runs latch the official success
and at least 4 of 8 do after 4 more runs. Every task below its target gets competing sessions of
its own (`prompts/task.md`). They work the whole task: the transitions between segments, the
benchmark's success checks with their margins, and the stages. Their output is `plan.json`, a
task plan that is data, never code: per segment a skill and its fitted params, small deltas
toward the middle of the success region, and generic transition calls. New skills go through the
gate like any other.

The driver accepts a plan when fresh trials, paired against the task's current plan, latch the
official success more often and do no fewer stages, or chain deeper and lose nothing. A small
rise is confirmed by a second trial. The plan is the builder's evidence only. It stays in the run
directory and is never in the skillset, a Memory, or any input to S1, S2 or evaluation.

Tasks whose ceiling runs all stop on a cause no skill can fix (the executor's table-top refusal,
the episode's step budget, a segment kind no core entry serves) get no session. This is judged
each round from that round's reports, never by task name.

## Running and stopping

`build.py --run NAME` resumes a run where it stopped. A `STOP` file in the run directory halts it
before its next round. `--rounds N` caps the rounds (0: no cap). `--until-target` runs until
every task is at its target; a round in which nothing grew is logged, not a stop. Without it, a
run stops after a round in which no total grew and no plan was accepted. `--regate RUN --run NEW`
gates a finished run's skills again with the current gate, and `--seed-candidates DIR...` makes
skills from another run candidates of the next round.

## Budgets

These are pipeline constants, the same for every gap cluster.

| | |
|---|---|
| case runs per session | 12, at most 4 at once |
| wall clock per session | 1.5 h |
| tokens per session | 5 M |
| relays per cluster and round | 4; a relay needs progress beyond the sessions before it |
| gaps per session | 4 (`clusters.CAP`); a kind and class split into at most 2 clusters |
| competing sessions per cluster | `--variants`, default 2, each with its own approach |
| sessions at once | `--sessions-at-once`, default 24 |
| ceiling repeats per task | 4, and 4 more when 2 of those latch |
| per-task sessions | `--task-variants`, default 2 |

A case run the infrastructure broke does not count against the session's budget. A session that
ends at its budget with progress (a gap explained, or a measure past the best the round reached)
is relayed to a new session in the same workspace, told what the last one did. Without progress
the cluster is reported open.
