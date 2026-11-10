"""The RoboTwin pipeline's config as dispatch composes it (`DispatchConfig.composer`): the
machine's file layered over its env binding and recipe (`robotwin_pipeline.config.compose_layers`),
with the binding named relative to the repository, so that every worker resolves it in its own
code copy and the provenance is the same on every machine."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from robotwin_pipeline.config import compose_layers

from roboscript.settings import repo_root


def compose(path: Path, overrides: Sequence[str]) -> dict[str, Any]:
    values = compose_layers(str(path), overrides)
    binding = Path(values["binding"])
    if binding.is_absolute() and binding.is_relative_to(repo_root()):
        values["binding"] = binding.relative_to(repo_root()).as_posix()
    return values
