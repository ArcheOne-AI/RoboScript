"""Conformance checks for benchmark descriptors (`roboscript.benchmarks`,
docs/concepts/embodied-core.md). The env server a descriptor names has its own suite
(`roboscript.testing.env`).

A descriptor conforms when it loads by the name it is registered under (or by import path), is
the same descriptor after a JSON round trip, names tasks, says how its episodes are reset, an
embodiment profile that loads, an env config model that resolves and covers the keys of its
default env config, an env server module that imports where the agent's interpreter runs it, and
skillsets that load, fit its name and its embodiment (its arms by count and name, the kinds of
action it takes) and bind what they need.

Use them from pytest by subclassing `BenchmarkConformance` as a `Test*` class with a
`benchmark_target` fixture, or without pytest via `roboscript.testing.precheck`:

```python
@pytest.fixture(params=registered("roboscript.benchmarks", "roboscript_mybench"))
def benchmark_target(request):
    return BenchmarkTarget(request.param)

class TestMyBenchmark(BenchmarkConformance): ...
```
"""

import importlib.util
from dataclasses import dataclass

from roboscript.contracts.skills import Bindings
from roboscript.envs import Benchmark, load_benchmark
from roboscript.skills import check_interface
from roboscript.testing import unit
from roboscript.tools.skillsets import load_skillset


@dataclass(frozen=True)
class BenchmarkTarget:
    """`ref`: the benchmark's name in `roboscript.benchmarks`, or an import path
    `package.module:attribute`."""

    ref: str


class BenchmarkConformance:
    """The checks, as `test_*` methods taking a `BenchmarkTarget`."""

    @unit
    async def test_loads_and_round_trips(self, benchmark_target: BenchmarkTarget) -> None:
        ref = benchmark_target.ref
        benchmark = load_benchmark(ref)
        assert isinstance(benchmark, Benchmark)
        if ":" not in ref:
            assert benchmark.name == ref, f"registered as {ref!r}, named {benchmark.name!r}"
        assert benchmark.tasks, "a benchmark names its tasks"
        assert benchmark.capabilities.reset, "a benchmark says how its episodes are reset"
        assert Benchmark.model_validate_json(benchmark.model_dump_json()) == benchmark

    @unit
    async def test_the_embodiment_loads(self, benchmark_target: BenchmarkTarget) -> None:
        profile = load_benchmark(benchmark_target.ref).profile()
        assert profile.arms

    @unit
    async def test_the_env_config_model_covers_the_default_config(
        self, benchmark_target: BenchmarkTarget
    ) -> None:
        benchmark = load_benchmark(benchmark_target.ref)
        if benchmark.env_config_model is None:
            assert not benchmark.env_config, "an env config needs a model that validates it"
            return
        model = benchmark.config_model()
        assert set(benchmark.env_config) <= set(model.model_fields), "env_config keys unknown"

    @unit
    async def test_the_env_server_imports_where_the_agent_runs_it(
        self, benchmark_target: BenchmarkTarget
    ) -> None:
        benchmark = load_benchmark(benchmark_target.ref)
        if benchmark.interpreter == "agent":  # a sim server's modules live in its own env
            assert importlib.util.find_spec(benchmark.env_server) is not None, benchmark.env_server

    @unit
    async def test_its_skillsets_fit(self, benchmark_target: BenchmarkTarget) -> None:
        benchmark = load_benchmark(benchmark_target.ref)
        plugins = [load_skillset(ref) for ref in benchmark.skillsets]
        provided = Bindings()
        for plugin in plugins:
            provided = provided.merged(plugin.manifest.bindings)
        profile = benchmark.profile()
        for plugin in plugins:
            manifest = plugin.manifest
            assert not manifest.envs or benchmark.name in manifest.envs, manifest.name
            assert check_interface(manifest, profile, provided.provided()) == []
