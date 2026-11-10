# How to add a simulated benchmark

A benchmark is an env server plus a `Benchmark` descriptor. The adapter lives in
`packages/benchmarks/<name>/`, configs and scripts in `examples/simBenchmarks/<Name>/`.
See `robotwin` for a complete example and `behavior` for an entry skeleton.

## 1. Declare capabilities

Write down what your benchmark offers before any code:

- **Reset**: `seed`, `initial_state` or `manual`
- **Success**: `latch` (checked after each action), `terminal_score` (at the end) or `external`
- **Reruns**: can one reset be opened twice? (default True)
- **Optional faces**: `truth`, `expert`, `demos`, `sensing`

The `env_capabilities` precheck refuses a method whose `requires` does not match.

## 2. Write the env server

Subclass `roboscript.envs.skeleton.SimBackend` with eight methods over your simulator. Only
`sim/official.py` imports the simulator (inside functions). The agent side never imports `sim/`.

```
packages/benchmarks/<name>/src/roboscript_<name>/
├── benchmark.py       Benchmark descriptor
├── config.py          env server config model
├── embodiments/       robot YAML
├── skills/            operation APIs (optional)
└── sim/
    ├── official.py    the ONLY file that imports the simulator
    ├── simulator.py   SimBackend subclass
    └── host.py        python -m roboscript_<name>.sim.host --config <file>
```

## 3. Write the descriptor and register

```python
MYBENCH = Benchmark(
    name="mybench",
    tasks={"lift_cube": TaskInfo(step_limit=400)},
    env_server="roboscript_mybench.sim.host",
    interpreter="sim",
    embodiment=HERE / "embodiments" / "myrobot.yaml",
)
```

```toml
[project.entry-points."roboscript.benchmarks"]
mybench = "roboscript_mybench.benchmark:MYBENCH"
```

## 4. Add evaluation (optional)

Register an `EvaluationFactory` under `roboscript.evaluations` if your benchmark has its own
official evaluation client.

## 5. Test

```bash
uv run pytest tests/benchmarks/<name>/
```

Run `BenchmarkConformance` and `EnvConformance` from `roboscript.testing`.
