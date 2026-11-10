"""Prompt templates: Markdown with `{{name}}` placeholders that the runner fills. Templates are
method content (`recipes/wicl/prompts/actor.md`, ...); this is only the renderer. Every
placeholder needs a value and every value a placeholder, so a renamed field fails loudly."""

import re
from collections.abc import Mapping
from typing import Final

PLACEHOLDER: Final = re.compile(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}")


def placeholders(template: str) -> set[str]:
    """The names a template needs."""
    return set(PLACEHOLDER.findall(template))


def render_prompt(template: str, values: Mapping[str, str]) -> str:
    """`template` with each `{{name}}` replaced by `values[name]` (inserted as is)."""
    wanted = placeholders(template)
    if missing := sorted(wanted - set(values)):
        raise ValueError(f"the prompt needs values for {missing}")
    if unused := sorted(set(values) - wanted):
        raise ValueError(f"the prompt has no placeholders for {unused}")
    return PLACEHOLDER.sub(lambda match: values[match[1]], template)
