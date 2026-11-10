"""The prompt renderer, and the method's templates it fills (recipes/wicl/prompts) with the
Mock benchmark's skillset guide (`roboscript.testing.mock_guide.md`)."""

from pathlib import Path

import pytest
import yaml

from roboscript.agent import placeholders, render_prompt
from roboscript.testing.skills import MOCK_SKILLS

PROMPTS = Path(__file__).parents[3] / "recipes" / "wicl" / "prompts"
NEEDS = {
    "actor.md": {"instruction", "memory", "stage_rules", "conventions", "robot"},
    "demo_reader.md": {"instruction", "memory", "tools", "conventions"},
    "reviser.md": {"instruction", "memory", "budget", "tools", "conventions"},
    "cleanup.md": {"instruction", "memory", "budget", "tools", "conventions"},
    "criteria.md": {"instruction", "memory", "budget", "conventions"},
}
GUIDE = MOCK_SKILLS.manifest.guide


def test_placeholders_are_filled_exactly() -> None:
    template = "Do {{ task }} with {{arm}}; then {{task}}."
    assert placeholders(template) == {"task", "arm"}
    values = {"task": "lift {{arm}}", "arm": "left"}  # values are inserted as they are
    assert render_prompt(template, values) == "Do lift {{arm}} with left; then lift {{arm}}."
    with pytest.raises(ValueError, match=r"needs values for \['arm'\]"):
        render_prompt(template, {"task": "x"})
    with pytest.raises(ValueError, match=r"no placeholders for \['extra'\]"):
        render_prompt(template, {"task": "x", "arm": "y", "extra": "z"})


def test_the_recipe_templates_render() -> None:
    assert {
        name: placeholders((PROMPTS / name).read_text(encoding="utf-8")) for name in NEEDS
    } == NEEDS
    rules = yaml.safe_load((PROMPTS / "stage_rules.yaml").read_text(encoding="utf-8"))
    assert set(rules) == {"s1_hinted", "s1", "s2", "p3"}
    assert GUIDE is not None and placeholders(GUIDE) == {"world_axes", "tcp_offset_m"}
    values = {"world_axes": "x right, y forward, z up", "tcp_offset_m": "0.12"}
    conventions = render_prompt(GUIDE, values)
    assert "World frame: x right, y forward, z up." in conventions
    assert "0.12 m ahead of the ee" in conventions
    common = {"instruction": "Lift the red box.", "memory": "(empty)", "conventions": conventions}
    extra = {
        "actor.md": {"stage_rules": rules["s2"], "robot": "a robot"},
        "demo_reader.md": {"tools": "- lift(color)"},
        "reviser.md": {"budget": "2 edits", "tools": "- lift(color)"},
        "cleanup.md": {"budget": "no budget", "tools": "- lift(color)"},
        "criteria.md": {"budget": "no budget"},
    }
    for name in ("actor.md", "demo_reader.md", "reviser.md", "cleanup.md", "criteria.md"):
        prompt = render_prompt(
            (PROMPTS / name).read_text(encoding="utf-8"), {**common, **extra.get(name, {})}
        )
        assert "{{" not in prompt and "Lift the red box." in prompt
