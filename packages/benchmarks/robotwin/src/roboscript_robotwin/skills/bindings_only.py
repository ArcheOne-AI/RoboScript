"""RoboTwin's binding data with none of its own skills: the core's entries bound for RoboTwin
(`roboscript_robotwin.bindings`: planner profile, executor tuning, support surface) and its guide,
the starting point of P1's skill builder on RoboTwin (`examples/simBenchmarks/RoboTwin/p1`), which
builds RoboTwin's skills on top of it. Named by import path,
`roboscript_robotwin.skills.bindings_only:BINDINGS_ONLY`."""

from pathlib import Path
from typing import Final

from roboscript.contracts.skills import SkillSet
from roboscript.tools.skillsets import SkillSetPlugin
from roboscript_robotwin.bindings import BINDINGS

GUIDE: Final = Path(__file__).with_name("guide.md")

BINDINGS_ONLY: Final = SkillSetPlugin(
    manifest=SkillSet(
        name="robotwin_bindings",
        version="1",
        layer="benchmark",
        envs=["robotwin"],
        embodiments=["aloha_agilex"],
        bindings=BINDINGS,
        guide=GUIDE.read_text(encoding="utf-8"),
        skills=[],
    ),
)
