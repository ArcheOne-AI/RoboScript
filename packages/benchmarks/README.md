# Benchmark plugins

Each directory is one benchmark adapter. The Mock benchmark lives in the core
(`roboscript.envs.mock`) for tutorials and tests.

A benchmark plugin provides a `Benchmark` descriptor, an env server, an embodiment profile, a task
table and optionally a skillset. The agent side imports nothing from `sim/`. Configs, scripts and
method bindings go in `examples/simBenchmarks/<Name>/`.

Guide: [docs/how-to/add-a-sim-benchmark.md](../../docs/how-to/add-a-sim-benchmark.md).
