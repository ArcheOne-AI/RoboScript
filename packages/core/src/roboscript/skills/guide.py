"""A skillset's guide as a role reads it (docs/how-to/add-a-skillset.md).

A guide (`SkillSet.guide`) is Markdown the method renders into the prompts of the roles it shows a
mounted skillset to. It is a template over the robot it runs on: `{{world_axes}}`, the profile's
words for the world frame's axes, and `{{tcp_offset_m}}`, how far ahead of the ee the TCP is (the
first arm's, in meters, `%g`). These are the embodiment's values (its binding), so the same
guide reads right on every robot it mounts on.
"""

import re
from collections.abc import Sequence
from typing import Final

import numpy as np

from roboscript.embodiments import EmbodimentProfile

GUIDE_VALUES: Final = ("world_axes", "tcp_offset_m")
"""The placeholders a guide may use."""
PLACEHOLDER: Final = re.compile(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}")
"""A placeholder, written as in prompt templates (`agent.prompts`)."""


def guide_values(profile: EmbodimentProfile) -> dict[str, str]:
    """What `profile`'s robot fills a guide's placeholders with."""
    offset = next(iter(profile.arms.values())).ee_to_tcp
    return {"world_axes": profile.world_axes, "tcp_offset_m": f"{float(np.linalg.norm(offset)):g}"}


def render_guide(guide: str, profile: EmbodimentProfile) -> str:
    """`guide` filled in for `profile`'s robot, stripped; ValueError for a placeholder it may not
    use (`GUIDE_VALUES`)."""
    values = guide_values(profile)

    def value(match: re.Match[str]) -> str:
        if match[1] not in values:
            raise ValueError(f"a guide may use {list(GUIDE_VALUES)}, not {{{{{match[1]}}}}}")
        return values[match[1]]

    return PLACEHOLDER.sub(value, guide).strip()


def render_guides(guides: Sequence[str], profile: EmbodimentProfile) -> str:
    """The mounted skillsets' guides, each rendered (`render_guide`), a blank line apart."""
    return "\n\n".join(render_guide(guide, profile) for guide in guides)
