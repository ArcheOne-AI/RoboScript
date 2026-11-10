# roboscript-robotwin

RoboTwin 2.0 integration: 50 bimanual tasks on `aloha_agilex`, clean and randomized. Provides the
benchmark descriptor, env server, embodiment profile, `robotwin` skillset, demonstration tools
and official evaluation (RoboTwin's own client driving the RoboScript Agent host).

Entry points: `robotwin` in `roboscript.benchmarks`, `roboscript.skillsets` and
`roboscript.evaluations`; `aloha_agilex` in `roboscript.embodiments`.

RoboTwin is a separate checkout (`ROBOTWIN_ROOT`) and is never patched. Setup and usage:
[examples/simBenchmarks/RoboTwin](../../../examples/simBenchmarks/RoboTwin/README.md).
