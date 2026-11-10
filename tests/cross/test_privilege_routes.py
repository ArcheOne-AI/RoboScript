"""The simulator's ground truth reaches agent-facing tools by one route only
(docs/architecture.md): the `privileged` toolset, tier P. No other toolset or
skillset source names the env's control plane (`EnvControl`) or its ground-truth snapshot
(`gt_snapshot`)."""

import re
from pathlib import Path
from typing import Final

import pytest

from roboscript.settings import repo_root

pytestmark = pytest.mark.unit

ROUTE: Final = re.compile(r"\b(EnvControl|gt_snapshot)\b")
PRIVILEGED: Final = Path("packages/core/src/roboscript/toolsets/privileged.py")


def agent_facing() -> list[Path]:
    """Every toolset and skillset module of every package: what lives under a `toolsets/` or
    `skills/` directory of a package's sources."""
    sources = (repo_root() / "packages").rglob("src/**/*.py")
    return sorted(p for p in sources if {"toolsets", "skills"} & set(p.parts))


def test_only_the_privileged_toolset_reaches_the_ground_truth() -> None:
    files = agent_facing()
    assert repo_root() / PRIVILEGED in files and len(files) > 10
    offenders = [
        f"{path.relative_to(repo_root())}: {match.group(0)}"
        for path in files
        if path != repo_root() / PRIVILEGED
        for match in ROUTE.finditer(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
