# BEHAVIOR integration

Initial adapter for [BEHAVIOR](https://behavior.stanford.edu/) (OmniGibson). The adapter package
is `packages/benchmarks/behavior/`. Full integration is in progress — OmniGibson is not installed
and the current tests run against a stand-in module.

## Declarations

| | |
|---|---|
| Robot | R1 Pro: two arms, a torso (`joint_groups`) and a holonomic base (`actions.base`) |
| Reset | By instance index (`initial_state`) |
| Success | Terminal q-score; no latch |
| Runs per reset | One (`reruns=False`) |
| Expert / demos | None |
| Skills | To be built with the skill builder |

## Layout

```
packages/benchmarks/behavior/src/roboscript_behavior/
├── benchmark.py              Benchmark descriptor            (agent side)
├── config.py                 BehaviorConfig                  (agent side)
├── tasks.yaml                Task table (placeholder limits and scenes)
├── embodiments/              Robot profile
├── skills/                   Skillset (agent side)
└── sim/                      Simulator side (OmniGibson imports only here)
examples/simBenchmarks/BEHAVIOR/run/   Recipe bindings
```

## Try it

```bash
uv run pytest tests/benchmarks/behavior/test_behavior.py
```

## To integrate

1. Correct `sim/official.py`, the embodiment profile, camera sizes and q-score against OmniGibson source.
2. Implement `drive_base` for base actions.
3. Build operation APIs into `skills/` with the skill builder.
4. Run `EnvConformance` on the real simulator.
