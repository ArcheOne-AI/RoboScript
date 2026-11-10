# Extension points

Every extension is a lazily loaded entry-point group with an interface and a conformance suite in
`roboscript.testing`. Third-party packages are named `roboscript-<name>` / `roboscript_<name>`.
Register in your `pyproject.toml` and run `uv sync --all-packages`.

## Groups

| Group | Registered | Conformance suite |
|---|---|---|
| `roboscript.benchmarks` | `Benchmark` | `benchmarks.BenchmarkConformance`, `env.EnvConformance` |
| `roboscript.embodiments` | `EmbodimentProfile` or YAML path | `embodiments.EmbodimentConformance` |
| `roboscript.toolsets` | `Toolset` class | `toolsets.ToolsetConformance` |
| `roboscript.skillsets` | `SkillSetPlugin` | `skills.SkillSetConformance` |
| `roboscript.perception` | grounder or segmenter class | `perception.GrounderConformance` |
| `roboscript.planners` | planner class | `planner.PlannerConformance` |
| `roboscript.brains` | brain class | `brain.BrainConformance` |
| `roboscript.methods` | method class | `methods.MethodConformance` |
| `roboscript.protocols` | protocol class | `protocols.ProtocolConformance` |
| `roboscript.memory_stores` | store class | `memory.MemoryStoreConformance` |
| `roboscript.artifact_stores` | store class | `artifacts.ArtifactStoreConformance` |
| `roboscript.gateway_policies` | routing policy class | `routing.RoutingPolicyConformance` |
| `roboscript.guards` | `Guard` | `guards.GuardConformance` |
| `roboscript.event_sinks` | `Sink` | `plugins.SinkConformance` |
| `roboscript.prechecks` | `Precheck` | `plugins.PrecheckConformance` |
| `roboscript.dispatch_transports` | transport class | `transport.TransportConformance` |
| `roboscript.dispatch_policies` | assignment policy class | `assignment.AssignmentConformance` |
| `roboscript.evaluations` | `EvaluationFactory` | `jobs.check_evaluation` (function) |
| `roboscript.robots` | `RobotSpec` (in `roboscript_robots`) | `roboscript_robots.testing.RobotConformance` |

---

## benchmarks

Interface: `roboscript.envs.Benchmark` -- `name`, `tasks`, `env_server`, `interpreter`,
`embodiment`, `capabilities`, `eval_seeds`, `skillsets`. The env server speaks
`roboscript.contracts.env`.

```toml
[project.entry-points."roboscript.benchmarks"]
mybench = "roboscript_mybench.benchmark:MYBENCH"
```

```python
from pathlib import Path
from roboscript.envs import Benchmark, TaskInfo

MYBENCH = Benchmark(
    name="mybench",
    tasks={"lift_cube": TaskInfo(step_limit=400)},
    env_server="roboscript_mybench.env",
    interpreter="sim",
    embodiment=Path(__file__).parent / "embodiments" / "myrobot.yaml",
)
```

## robots

Interface: `roboscript_robots.base.RobotSpec` -- `name`, `config`, `hardware`, `fake`. Register
under `roboscript.robots`, `roboscript.embodiments` and `roboscript.benchmarks` at once.

```toml
[project.entry-points."roboscript.robots"]
my_arm = "my_arm:MY_ARM"
[project.entry-points."roboscript.embodiments"]
my_arm = "my_arm:EMBODIMENT"
[project.entry-points."roboscript.benchmarks"]
my_arm = "my_arm:BENCHMARK"
```

## embodiments

Interface: `roboscript.embodiments.EmbodimentProfile` -- arms, joints, grippers, cameras, planner
config, pose conventions. Register an `EmbodimentProfile` or a `Path` to its YAML.

```toml
[project.entry-points."roboscript.embodiments"]
myrobot = "roboscript_myrobot:PROFILE"
```

## toolsets

Interface: subclass `roboscript.tools.Toolset` with a `name` and `@tool`-decorated async methods.

```toml
[project.entry-points."roboscript.toolsets"]
mytools = "roboscript_mytools.toolset:MyTools"
```

```python
from pydantic import BaseModel
from roboscript.tools import ToolContext, Toolset, tool


class Answer(BaseModel):
    text: str


class MyTools(Toolset):
    name = "mytools"

    @tool(kind="pure", tier="D")
    async def echo(self, ctx: ToolContext, text: str) -> Answer:
        """Say the text back."""
        return Answer(text=text)
```

## skillsets

Interface: `roboscript.tools.skillsets.SkillSetPlugin` -- a `manifest` (`SkillSet`) and
`extensions` (`ToolsetExtension` classes). Validate with
`python -m roboscript.testing.validation mybench --embodiment myrobot`.

```toml
[project.entry-points."roboscript.skillsets"]
mybench = "roboscript_mybench.skills:MYBENCH_SKILLS"
```

## perception

Interface: `roboscript.perception.grounding.Grounder` and `Segmenter`. A backend with
`build(resources)` constructs itself from the toolset's resources.

```toml
[project.entry-points."roboscript.perception"]
mydetector = "roboscript_mydetector:MyDetector"
```

## planners

Interface: `roboscript.contracts.motion.Planner` -- `plan`, `plan_batch`, `ik`, `fk`. Serve with
`python -m roboscript.motion --config <file> --planner <name>`.

```toml
[project.entry-points."roboscript.planners"]
myplanner = "roboscript_myplanner:MyPlanner"
```

## brains

Interface: `roboscript.agent.Brain` -- `async run(session) -> BrainResult`. Built-ins: `responses`,
`scripted`, `replay`.

```toml
[project.entry-points."roboscript.brains"]
mybrain = "roboscript_mybrain:MyBrain"
```

## methods

Interface: `roboscript.learning.method.Declaration` and `Method` -- phases, roles, visibility
tables, artifact kind. Evaluation phases are enforced to tier D.

```toml
[project.entry-points."roboscript.methods"]
mymethod = "roboscript_mymethod.method:MyMethod"
```

## protocols

Interface: `roboscript.learning.context.LearningProtocol` -- `name`, `Config`,
`async run(ctx, config)`.

```toml
[project.entry-points."roboscript.protocols"]
myprotocol = "roboscript_mymethod.protocol:MyProtocol"
```

## memory_stores

Interface: `roboscript.memory.MemoryStore` -- `commit`, `load`, `versions`, `freeze`. Built-in:
`file`.

```toml
[project.entry-points."roboscript.memory_stores"]
mystore = "roboscript_mystore:MyMemoryStore"
```

## artifact_stores

Interface: `roboscript.contracts.artifact.ArtifactStore` -- `kind`, `load`, `versions`, `freeze`.
Built-ins: `task_memory`, `skillset`, `rsi`.

```toml
[project.entry-points."roboscript.artifact_stores"]
mykind = "roboscript_mykind:MyArtifacts"
```

## gateway_policies

Interface: `roboscript.gateway.router.RoutingPolicy` -- `order(model, tier, ready, *, pinned,
last_good, load)` returns endpoints in preference order.

```toml
[project.entry-points."roboscript.gateway_policies"]
mypolicy = "roboscript_mypolicy:MyPolicy"
```

## guards

Interface: `roboscript.hub.pipeline.Guard` -- `name`, `phase` (`pre`/`guards`/`around`/`post`),
`make(config)`. Reject a call by raising `Reject(code, detail)`.

```toml
[project.entry-points."roboscript.guards"]
mylimit = "roboscript_mylimit:MY_LIMIT"
```

## event_sinks

Interface: `roboscript.records.sinks.Sink` -- `name`, `Config`, `make(config, run_dir)` returning
an `EventSink`.

```toml
[project.entry-points."roboscript.event_sinks"]
mysink = "roboscript_mysink:MY_SINK"
```

## prechecks

Interface: `roboscript.learning.prechecks.Precheck` -- `check(facts, config)` returns one line per
problem. Built-ins: `env_capabilities`, `seed_split`, `skillsets`.

```toml
[project.entry-points."roboscript.prechecks"]
mycheck = "roboscript_mycheck:MY_CHECK"
```

## dispatch_transports

Interface: `roboscript.orchestrate.dispatch.transport.Transport` -- `run`, `pull`, `push`.
Built-ins: `ssh`, `local`.

```toml
[project.entry-points."roboscript.dispatch_transports"]
mytransport = "roboscript_mytransport:MyTransport"
```

## dispatch_policies

Interface: `roboscript.orchestrate.dispatch.assign.AssignmentPolicy` -- `assign`, `reassign`.
Built-in: `balanced`.

```toml
[project.entry-points."roboscript.dispatch_policies"]
mypolicy = "roboscript_mypolicy:MyAssignment"
```

## evaluations

Interface: `roboscript.contracts.jobs.EvaluationFactory` -- `settings`, `kinds`, and a call
returning the `Evaluation`.

```toml
[project.entry-points."roboscript.evaluations"]
mybench = "roboscript_mybench.eval:MyEvaluation"
```
