"""The Agent host process (`python -m roboscript.learning.agent_host`) serving wicl: it refuses to
start without the tools its P3 calls. The process rig is the core's
(`tests/core/learning/test_agent_host.py`)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from roboscript.kernel import entrypoints

pytestmark = pytest.mark.skipif(
    "scripted" not in entrypoints.names("roboscript.brains"),
    reason="the Agent host runs in the agent venv (roboscript installed, with entry points)",
)


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("core_test_agent_host", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load(Path(__file__).resolve().parents[2] / "core" / "learning" / "test_agent_host.py")
serve, TASK = _core.serve, _core.TASK


SCENE = [  # sensing mounts the Mock's skillset: its guide is what wicl's prompts read
    {"name": "sensing", "skillsets": ["mock"]},
    {"name": "worldstate"},
    {"name": "done"},
]


async def test_the_process_refuses_to_start_without_the_tools_the_method_calls(
    tmp_path: Path, artifacts_dir: Path
) -> None:
    """wicl's P3 calls `check_done` (`MethodRuntime.mounts`): a host without the done
    toolset stops as it starts, not at its first gated episode."""
    code, verdicts = await serve(
        tmp_path,
        artifacts_dir,
        method="wicl",
        toolsets=SCENE[:2],
        replies=[[{"name": "finish", "arguments": {"note": "done"}}]],
    )
    assert code != 0 and verdicts == []
    assert "p3: check_done" in (artifacts_dir / "host.log").read_text(encoding="utf-8")
