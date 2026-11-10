"""The benchmark descriptor conformance suite (`roboscript.testing.benchmarks`) on every benchmark
the core registers in `roboscript.benchmarks`, loaded through the entry point as a third-party
package's would be."""

import pytest

from roboscript.envs.benchmark import GROUP
from roboscript.testing import registered
from roboscript.testing.benchmarks import BenchmarkConformance, BenchmarkTarget

BUILTIN = registered(GROUP)

pytestmark = pytest.mark.skipif(
    not BUILTIN, reason="roboscript is not installed here (the simulator env uses PYTHONPATH)"
)


@pytest.fixture(params=BUILTIN)
def benchmark_target(request: pytest.FixtureRequest) -> BenchmarkTarget:
    return BenchmarkTarget(request.param)


class TestBuiltinBenchmarks(BenchmarkConformance):
    pass


def test_the_core_registers_the_mock_benchmark() -> None:
    assert "mock" in BUILTIN
