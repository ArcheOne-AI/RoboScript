"""RoboTwin's skillset `robotwin` (docs/how-to/add-a-skillset.md), RoboTwin's P1 operation APIs as
built: the bindings it gives the core's entries, as data (`roboscript_robotwin.bindings`), and
the skills validated on RoboTwin only, which it adds to the core's toolsets (`rotate_about` to
`manipulation`, `estimate_hinge` and `measure_turn` to `perception`), and its
guide (`guide.md`: the world frame, the ee and the TCP, the grippers, arm and object names, the
rationale and the image overlays, filled in for the robot by `skills.render_guide`), which the
roles of a stack that mounts it read.
Registered in `roboscript.skillsets` as `robotwin`; where the package is only on `PYTHONPATH`
(the simulator env), named by import path, `roboscript_robotwin.skills:ROBOTWIN_SKILLS`
(`Benchmark.skillsets`)."""

from pathlib import Path
from typing import Final

from roboscript.contracts.skills import SkillSet
from roboscript.tools.skillsets import SkillSetPlugin, skill
from roboscript_robotwin.bindings import BINDINGS
from roboscript_robotwin.skills.actions import RoboTwinActions
from roboscript_robotwin.skills.perception import RoboTwinPerception

GUIDE: Final = Path(__file__).with_name("guide.md")
_SPECS: Final = {
    spec.name: spec for spec in [*RoboTwinActions.specs(), *RoboTwinPerception.specs()]
}

ROBOTWIN_SKILLS: Final = SkillSetPlugin(
    manifest=SkillSet(
        name="robotwin",
        version="2",
        layer="benchmark",
        envs=["robotwin"],
        embodiments=["aloha_agilex"],
        bindings=BINDINGS,
        guide=GUIDE.read_text(encoding="utf-8"),
        skills=[
            skill(
                _SPECS["rotate_about"],
                ["parameters", "perception", "planning", "executor", "env_execute", "receipt"],
                perception=["registered parts"],
            ),
            skill(_SPECS["estimate_hinge"], ["parameters", "perception", "receipt"]),
            skill(_SPECS["measure_turn"], ["parameters", "perception", "receipt"]),
        ],
    ),
    extensions=(RoboTwinActions, RoboTwinPerception),
)
