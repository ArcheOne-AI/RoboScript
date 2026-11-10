"""The conformance suites of the extension points RoboTwin registers in: its benchmark descriptor
(`roboscript.benchmarks`) and its embodiment profile (`roboscript.embodiments`), loaded through
the entry points as the core's tests load the built-ins."""

import pytest

from roboscript.testing import registered
from roboscript.testing.benchmarks import BenchmarkConformance, BenchmarkTarget
from roboscript.testing.embodiments import EmbodimentConformance, EmbodimentTarget

pytestmark = pytest.mark.unit

BENCHMARKS = registered("roboscript.benchmarks", "roboscript-robotwin")
EMBODIMENTS = registered("roboscript.embodiments", "roboscript-robotwin")
installed = pytest.mark.skipif(
    not (BENCHMARKS and EMBODIMENTS), reason="roboscript-robotwin is not installed here"
)


@pytest.fixture(params=BENCHMARKS)
def benchmark_target(request: pytest.FixtureRequest) -> BenchmarkTarget:
    return BenchmarkTarget(request.param)


@pytest.fixture(params=EMBODIMENTS)
def embodiment_target(request: pytest.FixtureRequest) -> EmbodimentTarget:
    return EmbodimentTarget(request.param)


@installed
class TestRoboTwinBenchmark(BenchmarkConformance):
    pass


@installed
class TestRoboTwinEmbodiment(EmbodimentConformance):
    pass


@installed
def test_robotwin_registers_its_benchmark_and_robot() -> None:
    assert (BENCHMARKS, EMBODIMENTS) == (["robotwin"], ["aloha_agilex"])
