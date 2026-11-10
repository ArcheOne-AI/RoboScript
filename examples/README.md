# Examples

How RoboScript is used, per environment. Nothing here is published or imported by a package. Tests run under
`tests/examples/`, and a benchmark's under `tests/benchmarks/<name>/`.

| Directory | What it holds | How to run it |
|---|---|---|
| `tutorials/` | Four walkthroughs on the Mock benchmark with a scripted brain: one episode, a toolset, a skillset, a plugin | see [tutorials/README.md](tutorials/README.md). No model, GPU or simulator needed |
| `simBenchmarks/RoboTwin/` | RoboTwin's configs, scripts and method bindings. The adapter is in `packages/benchmarks/robotwin/` | see [its README](simBenchmarks/RoboTwin/README.md) |
| `simBenchmarks/BEHAVIOR/` | BEHAVIOR integration (in development). The adapter is in `packages/benchmarks/behavior/` | `uv run pytest tests/benchmarks/behavior` |
| `realRobots/_template/` | The starting point for a real robot: configuration, embodiment, backend, server script, offline check and skillset skeleton | `uv run pytest tests/examples/real_robots` |

A guide for each of these is in [`docs/how-to/`](../docs/README.md#how-to-guides).
