"""The builder's session backends (`skill_builder/session.py`): the fake backend run through the
real session machinery with no model and no network, its skill written and its case run."""

import textwrap
from pathlib import Path
from typing import Any

import pytest

from roboscript_methods.skill_builder.builder.clusters import Cluster, Gap
from roboscript_methods.skill_builder.builder.driver import Limits, SessionContext
from roboscript_methods.skill_builder.session import (
    FAKE_GATEWAY,
    SessionConfig,
    make_session,
    session_runtime,
)

pytestmark = pytest.mark.unit

GAP = "taskA/left place obj at base [10, 40]"
REPO = Path(__file__).resolve().parents[3]


def reports(tmp_path: Path) -> Path:
    directory = tmp_path / "reports"
    directory.mkdir(exist_ok=True)
    (directory / "taskA.json").write_text('{"calls": [], "explained": []}', encoding="utf-8")
    return directory


def cluster() -> Cluster:
    gap = Gap(id=GAP, task="taskA", segment=GAP.split("/", 1)[1], kind="place", arm="left",
              obj="obj", target="base", frames=[10, 40], cause="entry fails", failure="reach",
              attempts=[])  # fmt: skip
    return Cluster("place-reach", "place", "reach", [gap], [], "_a")


SKILL = """
from roboscript.contracts.skills import SkillSpec

NAME = "slide_place"
KINDS = ("place",)
SPEC = SkillSpec(name=NAME, description="a place", parameters={"type": "object"},
                 chain=["parameters", "planning"])


async def act(api, arm, obj, target, params):
    return []
"""


async def test_the_fake_backend_writes_its_skill_and_runs_its_case(tmp_path: Path) -> None:
    skills = tmp_path / "skills" / "slide_place"
    skills.mkdir(parents=True)
    (skills / "skill.py").write_text(textwrap.dedent(SKILL), encoding="utf-8")
    (skills / "fit.py").write_text("def fit(segment, recording, scene, base, poses):\n"
                                   "    return {}\n", encoding="utf-8")  # fmt: skip
    (skills / "cases.yaml").write_text(f"gaps: ['{GAP}', 'taskZ/not a gap']\n", encoding="utf-8")
    config = SessionConfig(backend="fake", fake_skills=tmp_path / "skills",
                           state_root=tmp_path / "state")  # fmt: skip
    runtime = session_runtime(config)
    assert runtime.gateway_url == FAKE_GATEWAY and runtime.token.get_secret_value() == "fake"
    session = make_session(config)
    asked: list[Any] = []

    async def case(request: Any) -> dict[str, Any]:
        asked.append(request)
        return {"explained": True}

    context = SessionContext(reports=reports(tmp_path), limits=Limits(case_runs=3))
    work = tmp_path / "work"
    outcome = await session(cluster(), work, case, [], context)

    assert outcome.status == "done" and outcome.case_runs == 1 and outcome.tokens == 0
    assert [(r.skill, r.gap) for r in asked] == [("slide_place_a", GAP)]
    written = (work / "skill" / "slide_place_a" / "skill.py").read_text(encoding="utf-8")
    assert 'NAME = "slide_place_a"' in written  # renamed with the cluster's suffix
    assert outcome.last_message == "slide_place_a: explained 1"
    assert outcome.detail["usage_source"] == "codex"
