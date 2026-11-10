"""The core skillset `manipulation` (docs/how-to/add-a-skillset.md): the `manipulation` toolset's
tools as entries every env with the bindings they ask for mounts alike, registered in
`roboscript.skillsets`. Its tools are the toolset's own, so it adds no extension; an env's
skillset binds them (`contracts.skills.Bindings`) and may add tools of its own to the toolset.
`together` and `plan_arms` need two arms, and every entry takes arms named `left` and `right`
(their parameters' and receipts' `Arm`), so none mounts on a robot whose arms are named otherwise
(`skills.check_interface`)."""

from typing import Final, get_args

from roboscript.contracts.robot import Arm
from roboscript.contracts.skills import ChainStep, SkillSet
from roboscript.tools.skillsets import SkillSetPlugin, skill
from roboscript.toolsets.manipulation import Manipulation

_ACT: Final[list[ChainStep]] = ["parameters", "planning", "executor", "env_execute", "receipt"]
_SEEN: Final[list[ChainStep]] = ["parameters", "perception", *_ACT[1:]]
_PLANNED: Final[list[ChainStep]] = ["parameters", "perception", "planning", "receipt"]
_CHAINS: Final[dict[str, list[ChainStep]]] = {
    "move_to_pose": _ACT,
    "move_by": _ACT,
    "open_gripper": ["parameters", "executor", "env_execute", "receipt"],
    "close_gripper": ["parameters", "executor", "env_execute", "receipt"],
    "return_home": _ACT,
    "grasp_object": _SEEN,
    "place_object": _SEEN,
    "follow_path": _SEEN,
    "together": _SEEN,
    "nudge": _SEEN,
    "check_reach": _PLANNED,
    "plan_arms": _PLANNED,
}
_TWO_ARMS: Final = frozenset({"together", "plan_arms"})
_OBJECTS: Final = frozenset(
    {"grasp_object", "place_object", "follow_path", "together", "nudge", "plan_arms"}
)

MANIPULATION_SKILLS: Final = SkillSetPlugin(
    manifest=SkillSet(
        name="manipulation",
        version="2",
        layer="core",
        skills=[
            skill(
                spec,
                _CHAINS[spec.name],
                min_arms=2 if spec.name in _TWO_ARMS else 1,
                arms=get_args(Arm),
                perception=["registered objects"] if spec.name in _OBJECTS else [],
            )
            for spec in Manipulation.specs()
        ],
    )
)
