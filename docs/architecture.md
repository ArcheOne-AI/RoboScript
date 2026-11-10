# Architecture

RoboScript is a Python framework for agentic robot manipulation. An LLM agent solves manipulation tasks by calling tools over RGB-D cameras, joint readings and a motion planner. The framework enforces what the agent can see, gives it reliable ways to act, and lets it learn a task through staged rounds of trying and revising a text artifact, with no weights trained.

## Vocabulary

| Word | Meaning |
|---|---|
| build skills | A coding agent writes skills from demonstrations. Optional. |
| imitate (`S1`) | The agent learns a task's workflow from one demonstration. |
| practice (`S2`) | The agent repeats the task on new scenes and revises its Memory. |
| evaluate | The frozen Memory runs under the benchmark's official evaluation. |
| reflect | A separate agent reads evaluation records and writes a document for the next evaluation. |
| episode | One run of a task, from reset to end. Called a *try* in practice, an *attempt* in imitate. |
| Memory | The per-task text artifact: Markdown sections plus typed parameters. |
| spec | A structured grasp or placement description (`GraspSpec`, `PlaceSpec`), stored in Memory. |
| receipt | The result of a physical action (`ActionReceipt`). |
| tier | What a tool may read. `D` = deployed robot. `P` = simulator privilege. |
| toolset | A group of tools sharing one resource profile. |
| skillset | The operation API of one environment: extra tools plus data binding core entries to it. |
| Hub | The service every tool call passes through. Enforces visibility, budgets and one action at a time. |
| brain | What drives a model through an episode. The reference is `responses`. |
| gateway | Holds model API keys and routes calls over endpoints. |
| method | A learning approach declaration: phases, roles, tool visibility, frozen artifact. `wicl`, `rsi` and `skill_builder` ship. |
| recipe | A method's content in `recipes/<method>/`: prompts, lint vocabularies, parameters. |
| roles | The agents a method defines. In `wicl`: Actor (acts), Demo Reader (writes first Memory), Reviser (edits Memory after failure). |

## Packages and layers

| Distribution | Location | Holds |
|---|---|---|
| `roboscript` | `packages/core` | Framework mechanisms. No simulator or GPU needed. |
| `roboscript-vision` | `packages/vision` | GroundingDINO, SAM 3 and the vision service. |
| `roboscript-methods` | `packages/methods` | Methods `wicl`, `rsi` and `skill_builder`. |
| `roboscript-robotwin` | `packages/benchmarks/robotwin` | RoboTwin benchmark adapter. |
| `roboscript-behavior` | `packages/benchmarks/behavior` | BEHAVIOR benchmark adapter (in development). |
| `roboscript-robots` | `packages/robots` | Robot server, safety rules, config schemas. No hardware backend ships. |

The Mock benchmark lives in the core (`roboscript.envs.mock`). Outside the packages: `recipes/` holds method content, `examples/simBenchmarks/<Name>/` holds benchmark configs and scripts, `examples/realRobots/_template/` is the starting point for a real robot, and `examples/tutorials/` holds walkthroughs.

Units are SI, quaternions wxyz, every pose names its frame. World x right, y forward, z up. Python 3.10+.

## One episode

A practice seed in `wicl`, simplified:

1. The runner opens the env with a task, seed and stage. The env checks solvability; unsolvable seeds are skipped.
2. The Hub opens an episode. `identify` finds the objects Memory names.
3. The brain runs the turn loop. Each turn: observe, any number of information calls, one physical action or `finish`.
4. A physical action (e.g. `grasp_object`) goes through the Hub (visibility, budget, action lock), the manipulation library (synthesize candidates, plan with cuRobo), then `env.agent.execute` (joint-position batch).
5. A receipt reports what happened: target reached, gripper holding, steps used.
6. The episode ends on success (official latch), `finish`, hard failure, budget or infrastructure error.
7. The runner calls `judge` for the official verdict. After a failure, the Reviser edits Memory.

## Stages

| | Imitate (S1) | Practice (S2) | Evaluate |
|---|---|---|---|
| Seeds | demonstration's | practice block | official blocks |
| Actor reads | Memory + demo package | Memory only | frozen Memory |
| Judged by | latch + keypoint referee | latch | official client |
| Hard failure | ends attempt | ends try | agent recovers |
| Memory writes | Reviser edits directly | cache, merged on success | none |

## Information boundary

### Tiers

**Tier D** (deployed robot): RGB-D with intrinsics/extrinsics, joint readings, finger opening, TCP poses, URDF, planner feedback, step count/limit, instruction.

**Tier P** (simulator privilege): ground-truth object poses, asset annotations, meshes, URDFs, contacts, segmentation, `check_success`, expert traces.

### What each role reads

| Stage and role | Tier | Official outcome | Ground truth | Demonstration |
|---|---|---|---|---|
| Build skills: coding agent | D (written skills), P (`fit.py`) | success condition + margins | `fit.py` reads truth | all demonstrations |
| Imitate: Demo Reader | P | no | truth layer | whole package |
| Imitate: Actor (hinted) | P | latch refusal + scene diff | truth layer, `true_objects` | package + keypoint progress |
| Imitate: Actor (unhinted) | P | latch refusal | `true_objects` where mounted | none |
| Imitate: Reviser | P | yes | diagnosis + expert diff | package, keypoint deviations |
| Practice: Actor | D | latch refusal | none | `demo_context` (default: none) |
| Practice: Reviser | D | yes | truth diagnosis (unless `diagnosis: false`) | off by default |
| Evaluate: Actor | D | latch refusal | none | `demo_context` (default: none) |
| Reflect: all roles | D | each episode's success | only with `RsiConfig.diagnosis` | no |

### What this means, plainly

An Actor never reads simulator data during practice or evaluation, directly or inside a tool. Memory is different: it is written during imitate and revised during practice by roles that read ground truth, so parameter values in Memory can reflect privileged learning. Execution is free of privilege; learning is not.

Success is always the simulator's verdict. Only the official check, latched after each action, makes an episode a success. `finish` never does.
