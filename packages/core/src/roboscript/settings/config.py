"""Machine-local YAML configs under `configs/`, validated by pydantic models."""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from roboscript.contracts.canonical import canonical_json
from roboscript.kernel.compose import compose
from roboscript.settings.env import repo_root

M = TypeVar("M", bound=BaseModel)


def load_config(
    name: str, model: type[M], *, root: Path | None = None, overrides: Sequence[str] = ()
) -> M:
    """Read `configs/<name>.yaml` (`name` may hold `/`, e.g. `stacks/dev-local`) under the
    repository root, apply dotted `overrides` and validate it as `model`."""
    path = (repo_root() if root is None else root) / "configs" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: copy configs/{name}.example.yaml to configs/{name}.yaml"
        )
    return load_config_file(path, model, overrides=overrides)


def load_config_ref(ref: str, model: type[M], *, overrides: Sequence[str] = ()) -> M:
    """A config from a YAML file (a path ending in `.yaml` or `.yml`), or by name from
    `configs/<name>.yaml` (e.g. `motion`): what services take as `--config`."""
    path = Path(ref)
    if path.suffix in (".yaml", ".yml"):
        return load_config_file(path, model, overrides=overrides)
    return load_config(ref, model, overrides=overrides)


def config_ref_path(ref: str) -> Path:
    """The file a config ref names (`load_config_ref`), which must exist."""
    path = Path(ref)
    if path.suffix not in (".yaml", ".yml"):
        path = repo_root() / "configs" / f"{ref}.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} does not exist: copy configs/{ref}.example.yaml to configs/{ref}.yaml"
            )
    return path


def read_mapping(path: Path) -> dict[str, Any]:
    """The YAML mapping in the file at `path` (empty file: an empty mapping)."""
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:  # an empty file
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    return data


def load_config_file(path: Path, model: type[M], *, overrides: Sequence[str] = ()) -> M:
    """Read the YAML file at `path`, apply dotted `overrides` and validate it as `model`."""
    data = read_mapping(path)
    try:
        return model.model_validate(compose(data, overrides=overrides))
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


def config_hash(model: BaseModel) -> str:
    """sha256 hex of the canonical JSON dump, secrets masked (they never reach a hash)."""
    return hashlib.sha256(canonical_json(model.model_dump(mode="json")).encode("utf-8")).hexdigest()
