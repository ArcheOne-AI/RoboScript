"""Skillset validation (docs/how-to/add-a-skillset.md) on MockEnv: the boundary check passes the
core's and the test skillset and catches a leaky one."""

import sys
import textwrap
from pathlib import Path

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.ids import TaskName
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.testing.boundary import check_boundary
from roboscript.testing.hold import Hold
from roboscript.testing.validation import (
    CeilingEpisode,
    CeilingPlanner,
    CeilingRun,
)
from roboscript.tools import LocalProvider
from roboscript.toolsets.manipulation import Manipulation
from roboscript.toolsets.perception import Perception

pytestmark = pytest.mark.unit

REACH = "roboscript.testing.reach:REACH_SKILLS"
RATIONALE = {"stage": "lift", "scene": "a red box", "progress": "starting", "plan": "lift it"}
PROFILE = load_embodiment(MOCK_BIMANUAL)


async def lift_red(episode: CeilingEpisode) -> None:
    """Grasp the red box where the truth says it is and lift it 10 cm."""
    assert episode.tools == ("reach",)  # the skillset's tools, nothing else
    x, y, z = episode.truth.objects["red"].p
    arm = "left" if x < 0 else "right"
    for tcp, gripper in [((x, y, z), "open"), ((x, y, z), "closed"), ((x, y, z + 0.1), "closed")]:
        done = await episode.call(
            "reach", rationale=RATIONALE, arm=arm, tcp=list(tcp), gripper=gripper
        )
        assert not done.is_error, done


async def idle(episode: CeilingEpisode) -> None:
    """Does nothing: no episode succeeds."""


def _run(tmp_path: Path, planner: CeilingPlanner = lift_red, min_ratio: float = 0.0) -> CeilingRun:
    blobs = LocalBlobStore(tmp_path / "blobs")
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs"), blobs=blobs)
    return CeilingRun(
        providers=lambda: [LocalProvider(Hold(), None, ["mock", REACH], PROFILE)],
        env=env,
        env_client=lambda _: env,
        task=TaskName("lift_red"),
        seeds=[0, 1, 2],
        planner=planner,
        embodiment=str(MOCK_BIMANUAL),
        min_ratio=min_ratio,
        blobs=blobs,
    )


def test_the_shipped_skillsets_keep_to_the_boundary() -> None:
    assert check_boundary(REACH) == []
    hosts = {"manipulation": Manipulation, "perception": Perception}
    assert check_boundary("manipulation", hosts) == []


def test_a_leaky_skillset_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "leaky"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helpers.py").write_text(
        "from roboscript.learning import runner\n", encoding="utf-8"
    )
    (package / "tools.py").write_text(
        textwrap.dedent(
            """
            from roboscript.envs.mock import TABLE_Z
            from roboscript.tools import ToolContext, tool
            from roboscript.tools.spec import ToolsetExtension
            from leaky import helpers
            from pydantic import BaseModel

            class Out(BaseModel):
                z: float

            class Peek(ToolsetExtension):
                extends = "hold"

                @tool(kind="pure", tier="P")
                async def peek(self, ctx: ToolContext) -> Out:
                    \"\"\"Peeks.\"\"\"
                    truth = await ctx.control.gt_snapshot(ctx.env_episode)
                    return Out(z=TABLE_Z)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "leaky", raising=False)
    from leaky.tools import Peek  # type: ignore[import-not-found]

    from roboscript.contracts.skills import SkillSet
    from roboscript.tools.skillsets import SkillSetPlugin, skill

    manifest = SkillSet(
        name="leaky", version="1", layer="built", skills=[skill(Peek.specs()[0], ["receipt"])]
    )
    lines = check_boundary(SkillSetPlugin(manifest, (Peek,)))
    assert lines == [
        "leaky.peek: tier P, a skillset's tools are tier D",
        "leaky: leaky.helpers imports roboscript.learning (roboscript.learning is off limits)",
        "leaky: leaky.tools imports roboscript.envs.mock (roboscript.envs is off limits)",
        "leaky: leaky.tools names gt_snapshot",
    ]
