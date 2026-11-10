"""The `pickup` skillset passes the skillset conformance suites and the four validation steps on
the Mock benchmark, and the validation catches the mistakes the README names."""

import sys
from pathlib import Path

import pytest
from custom_skillset.pickup import PICKUP_SKILLS

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.ids import TaskName
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs.mock import MockConfig, MockEnv
from roboscript.skills import SkillSetStore
from roboscript.testing.hold import Hold
from roboscript.testing.skills import (
    PluginTarget,
    SkillSetConformance,
    SkillSetPluginConformance,
    SkillSetTarget,
)
from roboscript.testing.validation import (
    CeilingEpisode,
    CeilingRun,
    ValidationTarget,
    validate,
)
from roboscript.tools import LocalProvider
from roboscript.tools.skillsets import mounted

pytestmark = pytest.mark.unit

REF = "custom_skillset.pickup:PICKUP_SKILLS"  # an import path: the skillset is not registered
PROFILE = load_embodiment(MOCK_BIMANUAL)
RATIONALE = {"stage": "lift", "scene": "a red box", "progress": "starting", "plan": "pick it up"}


@pytest.fixture
def skillset_target() -> SkillSetTarget:
    return SkillSetTarget(PICKUP_SKILLS.manifest, PROFILE, PICKUP_SKILLS.manifest.bindings)


@pytest.fixture
def plugin_target() -> PluginTarget:
    return PluginTarget(REF, PROFILE, {"hold": Hold}, also=["mock"])


class TestPickupManifest(SkillSetConformance):
    pass


class TestPickupPlugin(SkillSetPluginConformance):
    pass


def test_it_mounts_on_the_toolset_it_extends() -> None:
    on = mounted(Hold, ["mock", REF], PROFILE)
    assert [spec.name for spec in on.specs] == ["hold", "pick_up"] and not on.unmounted


async def pick_up_red(episode: CeilingEpisode) -> None:
    """The planner reads the truth, which the Agent never can, and uses only the skillset."""
    assert episode.tools == ("pick_up",)
    x, y, z = episode.truth.objects["red"].p
    arm = "left" if x < 0 else "right"
    done = await episode.call("pick_up", rationale=RATIONALE, arm=arm, tcp=[x, y, z], lift=0.1)
    assert not done.is_error, done


def ceiling(tmp_path: Path) -> CeilingRun:
    blobs = LocalBlobStore(tmp_path / "blobs")
    env = MockEnv(MockConfig(task=TaskName("lift_red"), blob_dir=tmp_path / "blobs"), blobs=blobs)
    return CeilingRun(
        providers=lambda: [LocalProvider(Hold(), None, ["mock", REF], PROFILE)],
        env=env,
        env_client=lambda _: env,
        task=TaskName("lift_red"),
        seeds=[0, 1, 2],
        planner=pick_up_red,
        embodiment=str(MOCK_BIMANUAL),
        min_ratio=1.0,
        blobs=blobs,
    )


async def test_validation_runs_four_steps_and_freezes(tmp_path: Path) -> None:
    store = SkillSetStore(tmp_path / "skillsets")
    target = ValidationTarget(ref=REF, profile=PROFILE, also=["mock"], ceiling=ceiling(tmp_path))
    report = await validate(target, store)
    assert report.passed, report
    assert list(report.steps) == ["interface", "boundary", "ceiling", "freeze"]
    assert report.steps["ceiling"].value == 1.0  # a perfect planner lifts the box on every seed
    frozen = await store.frozen("pickup")
    assert frozen is not None
    manifest = await store.load(frozen)
    assert manifest.report is not None and manifest.model_copy(update={"report": None}) == (
        PICKUP_SKILLS.manifest
    )


async def test_a_guide_with_an_unknown_placeholder_fails_the_interface_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "broken_skillset.py").write_text(
        "from custom_skillset.pickup import PICKUP_SKILLS\n"
        "from roboscript.tools.skillsets import SkillSetPlugin\n"
        "BROKEN = SkillSetPlugin(\n"
        "    PICKUP_SKILLS.manifest.model_copy(update={'guide': 'Use {{no_such_value}}.'}),\n"
        "    PICKUP_SKILLS.extensions,\n"
        ")\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "broken_skillset", raising=False)
    report = await validate(
        ValidationTarget(ref="broken_skillset:BROKEN", profile=PROFILE, also=["mock"]),
        SkillSetStore(tmp_path / "skillsets"),
    )
    assert list(report.steps) == ["interface"] and not report.passed
    assert "no_such_value" in report.steps["interface"].detail


async def test_a_tier_p_tool_fails_the_boundary_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (Path(__file__).parents[3] / "examples/tutorials/custom_skillset/pickup.py").read_text(
        encoding="utf-8"
    )
    package = tmp_path / "leaky_pickup"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "guide.md").write_text(PICKUP_SKILLS.manifest.guide, encoding="utf-8")
    leaky = source.replace('tier="D"', 'tier="P"').replace("PICKUP_SKILLS", "LEAKY")
    (package / "skills.py").write_text(leaky, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "leaky_pickup.skills", raising=False)
    report = await validate(
        ValidationTarget(ref="leaky_pickup.skills:LEAKY", profile=PROFILE, also=["mock"]),
        SkillSetStore(tmp_path / "skillsets"),
    )
    assert list(report.steps) == ["interface", "boundary"] and not report.steps["boundary"].passed
    assert "tier P" in report.steps["boundary"].detail
