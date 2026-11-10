"""BEHAVIOR's skillset `behavior` (docs/how-to/add-a-skillset.md): a skeleton with no skills.

What the skill builder produces for BEHAVIOR lands here: the bindings its skills need as data, the
skills it adds, a `guide.md`. A skill that drives the base executes `base` actions, which the
`execute` wire does not carry: its `SkillNeeds.actions` says so. `Benchmark.skillsets` names this
by import path."""

from typing import Final

from roboscript.contracts.skills import SkillSet
from roboscript.tools.skillsets import SkillSetPlugin

BEHAVIOR_SKILLS: Final = SkillSetPlugin(
    manifest=SkillSet(
        name="behavior",
        version="0",
        layer="benchmark",
        envs=["behavior"],
        embodiments=["r1_pro_behavior"],
    )
)
