"""The operation API layer (docs/how-to/add-a-skillset.md): the skillsets an env mounts, their
verification and their guides.

The manifest and the bindings (`SkillSet`, `Bindings`) are data and live in `contracts.skills`;
how a provider mounts a skillset's tools on a core toolset is `tools.skillsets`; the core's
`manipulation` skillset is `toolsets.manipulation.skillset`; a skill builder is a method
(`learning.builder.SkillBuilder`); the validation tool (the boundary check, the reachable ceiling
and `validate`, which runs the four validation steps and freezes) is `roboscript.testing.boundary`
and `roboscript.testing.validation`.

- `verify`: `check_interface`, `check_mounts` (the precheck of a stack's mounts, `MountCheck`)
- `guide`: `render_guide`, a guide filled in for the robot it runs on
- `store`: `SkillSetStore`, frozen skillsets (`roboscript.artifact_stores` as `skillset`)
"""

from roboscript.skills.guide import GUIDE_VALUES, guide_values, render_guide, render_guides
from roboscript.skills.store import SkillSetStore, skillset_hash, skillset_ref
from roboscript.skills.verify import MountCheck, check_interface, check_mounts

__all__ = [
    "GUIDE_VALUES",
    "MountCheck",
    "SkillSetStore",
    "check_interface",
    "check_mounts",
    "guide_values",
    "render_guide",
    "render_guides",
    "skillset_hash",
    "skillset_ref",
]
