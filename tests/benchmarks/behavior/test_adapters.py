"""What every placeholder adapter shares: the agent side imports no official code, the official
import sits in `sim/official.py` alone, and a method that needs what an adapter does not offer is
refused before the run starts (wicl on BEHAVIOR's single-run instances in particular)."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.envs import Benchmark
from roboscript.learning.method import load_method
from roboscript.learning.prechecks import ENV_CAPABILITIES, RunFacts, run_prechecks
from roboscript.learning.runspec import RunSpec
from roboscript_behavior.benchmark import BEHAVIOR

SRC = Path(__file__).resolve().parents[3] / "packages" / "benchmarks" / "behavior" / "src"
ADAPTERS = [
    ("roboscript_behavior", SRC, {"omnigibson", "omni"}),
]


@pytest.mark.parametrize(("package", "root", "official"), ADAPTERS)
def test_the_agent_side_imports_no_official_or_sim_code(
    package: str, root: Path, official: set[str]
) -> None:
    code = (
        f"import sys, {package}.benchmark, {package}.config, {package}.skills\n"
        f"bad = {sorted(official)!r}\n"
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in bad or "
        f"m.startswith('{package}.sim')))"
    )
    paths = [str(root), *sys.path]
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": ":".join(paths)},
    )
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize(("package", "root", "official"), ADAPTERS)
def test_only_sim_official_imports_the_official_package(
    package: str, root: Path, official: set[str]
) -> None:
    importers = set()
    for path in sorted((root / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.args[0], ast.Constant)
            ):
                names = [str(node.args[0].value)]
            else:
                continue
            if any(name.split(".")[0] in official for name in names):
                importers.add(path.name)
    assert importers == {"official.py"}


def _facts(benchmark: Benchmark, required: EnvCapabilities) -> RunFacts:
    spec = RunSpec.model_validate(
        {
            "env": {"name": "mock"},
            "embodiment": {"name": "mock_bimanual"},
            "method": {"name": "wicl"},
            "tasks": [next(iter(benchmark.tasks))],
            "seeds": {"blocks": {}},
        }
    )
    return RunFacts(spec=spec, offered=benchmark.capabilities, required=required)


async def test_wicl_is_refused_on_behavior_with_a_line_for_each_gap() -> None:
    requires = load_method("wicl").requires
    assert requires.reruns, "wicl opens one reset again (S1's and S2's tries)"
    problems = await run_prechecks([(ENV_CAPABILITIES, {})], _facts(BEHAVIOR, requires))
    assert "env_capabilities: the env lacks reruns of one reset (the env opens each only once)" in (
        problems
    )
    assert "env_capabilities: the env lacks success judged by latch" in problems
    assert "env_capabilities: the env lacks reset by seed" in problems
