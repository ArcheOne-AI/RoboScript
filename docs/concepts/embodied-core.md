# The embodied core

Everything below `tools` in the layer diagram runs without a model: environments, robot descriptions, motion planning and the manipulation API. `spatial` sits at the bottom as pure numpy for poses, quaternions, transforms, projection and point clouds.

## Envs and benchmarks

An env server hosts one task instance, with separate ports for its planes.

| Plane | Caller | Operations |
|---|---|---|
| public | anyone | `discover`, `healthz`, `readyz` |
| `env.control` | orchestrator | `open(EpisodeSpec)`, `judge`, `diagnose`, `gt_snapshot`, `close` |
| `env.agent` | toolsets | `robot`, `observe`, `execute` |
| `robot.control` | real robots | `describe`, `open`, `judge`, `close`, `estop`, `home`, `heartbeat` |

Only discovery, open/close and the agent plane are required. Everything else is an optional capability declared with `EnvCapabilities`: how it judges (`latch`, `terminal_score`, `external`), how it resets (`seed`, `initial_state`, `manual`), whether one reset may open twice (`reruns`), and optional faces (`truth`, `expert`, `demos`, `real_robot`, `sensing`). A method states what it needs in `requires`, and the `env_capabilities` precheck compares before any job starts.

A `Benchmark` descriptor ties it together: name, tasks, env server module, interpreter, embodiment, capabilities, eval seed blocks and skillsets. The core registers the `mock` benchmark in `roboscript.envs.mock`. Other benchmarks are downstream packages. A task can have a success condition card (`envs.cards`), at most 1200 characters, reaching the agent as part of the instruction.

When a benchmark has an official evaluation client, evaluation runs behind a bridge that serves `env.agent`, so toolsets cannot tell practice from evaluation.

A real robot is an env server of its own (`roboscript_robots.base.server`). It serves the required planes plus `robot.control`, never declares ground truth or an expert, and the learning runner refuses it (`REAL_ROBOT`).

## Embodiments

An `EmbodimentProfile` is data: arms by name, the joint vector, grippers, cameras, planner config, asset paths, pose conventions and world axes. `Gripper.holding_threshold` is the one rule for holding: a gripper commanded closed holds something if its measured opening is at or above it. Register a profile in `roboscript.embodiments` as a loaded object or a YAML path.

## Motion

`roboscript.motion` is a planner interface: plan to a pose, orientation-locked plan, reachability, batches, IK and FK. `curobo` and `mock` are registered providers. The service runs as `python -m roboscript.motion --config <file>`, one per GPU, serving any number of episodes. Its collision world is the table cuboid plus up to `MAX_OBSTACLES` oriented boxes. Plan before executing, so a failure costs no steps.

## Manipulation

`roboscript.manipulation` is the action API every caller goes through. `ActionApi` wraps an `Executor`; the `manipulation` toolset exposes it to the agent, and a scripted planner calls the same methods.

| Action | What it does |
|---|---|
| `grasp_object` | Synthesize candidates from a `GraspSpec`, approach and close |
| `place_object` | Place by a `PlaceSpec`, searching regions for a plannable pre-place |
| `move_by` | Straight, bounded move from the measured pose |
| `move_to_pose` | Move to an end-effector pose |
| `open_gripper`, `close_gripper`, `return_home` | Grippers, and the home pose |
| `follow_path` | Carry along a `PathSpec` as one rigid motion |
| `nudge` | Small correction of a held object's center and yaw |
| `together` | One action per arm (`moves=[{arm, action}, ...]`), sent as one batch |
| `check_reach` | Plans nothing, moves nothing |

A skillset can add more (e.g. `rotate_about` for hinge manipulation).

### Specs

`GraspSpec` and `PlaceSpec` describe where and how to grasp or place, stored as Memory parameters. Every number is relative to an object, part, landmark or the robot base. Absolute object positions never enter a spec. Which arm does which stage is planned once per episode by `plan_arms`, over the holds the workflow makes (`memory.chains`): holds open at the same time take different arms, a hand-over's receiver is the other arm, and an arm the instruction names keeps its stage; a spec holds no arm.

### Receipts

An action returns one `ActionReceipt`: its `reason` (`ok`, `empty_grasp`, `plan_failed`, `unreachable`, `lost_hold`, `stopped_short`, `deviated`, `budget_exhausted`, etc.), commanded and achieved poses, deviation, steps used, gripper state and whether it holds. Hard-failure reasons (`unreachable`, `plan_failed`, `empty_grasp`, `lost_hold`) end a practice try.
