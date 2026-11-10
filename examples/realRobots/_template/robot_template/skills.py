"""This robot's skillset (docs/how-to/add-a-skillset.md): a skeleton with no skills.

What the skill builder produces on the robot lands here: the bindings its skills need as data, the
skills it adds, a `guide.md`. `layer="robot"` marks it as the robot's own. Name it in the run's
`skillsets`; a benchmark does not carry it, since a real robot is no benchmark.
"""

from typing import Final

from roboscript.contracts.skills import SkillSet
from roboscript.tools.skillsets import SkillSetPlugin

ROBOT_SKILLS: Final = SkillSetPlugin(
    manifest=SkillSet(name="my_robot", version="0", layer="robot", embodiments=["my_robot"])
)
