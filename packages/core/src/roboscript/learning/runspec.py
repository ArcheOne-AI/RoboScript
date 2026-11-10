"""The run declaration and its layered config (docs/concepts/operations.md).

Every item a run uses is a name plus overrides. Config layers stack in `LAYERS` order: core
defaults, the method's defaults, the generic recipe (`recipes/<method>/`), the env's defaults and
binding (`examples/<...>/<env>/run/<method>/`), a variant, then this run's `--set` overrides. The
composed result is written to the run directory and a resumed run reads only that. An item named
with a `url` is a server already running there, not something to start in process: a toolset's
MCP endpoint (`python -m roboscript.integrations.mcp.serve`, which holds its config, so the item
has no overrides; `learning.mounts.run_providers` mounts it). A skillset has no server of its
own: its tools extend toolsets and its bindings travel as data, so a skillset served elsewhere is
the toolset it extends, served with it, and the skillset item takes no `url`. An env's `url`
(its server's control plane) is not read: a runner takes its env control planes from its
caller. Plugins run in process. The run's plugins (guards, event sinks, prechecks) are composed by
`roboscript.learning.compose`; a benchmark's pipeline builds its `RunSpec` from its own layered
config.
"""

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

from roboscript.contracts.ids import TaskName
from roboscript.contracts.phase import RoleName
from roboscript.kernel.compose import compose
from roboscript.learning.seeds import SeedSplit
from roboscript.settings import read_mapping, repo_root

Layer = Literal["core", "method", "recipe", "env", "variant", "site"]
LAYERS: Final[tuple[Layer, ...]] = ("core", "method", "recipe", "env", "variant", "site")
"""File layers, lowest first; the run's `--set` overrides apply after all of them. `site`: this
machine's deployment (interpreters, GPUs, directories; a git-ignored copy of a `*.example.yaml`),
which none of the other layers may name."""
PluginGroup = Literal["guards", "event_sinks", "prechecks"]


class RunItem(BaseModel):
    """A registered item by name, with overrides of its config; `url`: the address of a server
    that already runs it (None: the run starts or mounts it in process)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    overrides: dict[str, JsonValue] = {}
    url: str | None = None

    @field_validator("url")
    @classmethod
    def _url(cls, url: str | None) -> str | None:
        if url is not None and not url.startswith(("http://", "https://")):
            raise ValueError(f"url {url!r} is not an http(s) address")
        return url


class PluginItem(RunItem):
    """A plugin of the run by its entry-point group (`roboscript.<group>`) and its name there, or
    its import path `package.module:attribute`. Guards, event sinks and prechecks run in the run's
    own process: no `url`."""

    group: PluginGroup

    @field_validator("url")
    @classmethod
    def _in_process(cls, url: str | None) -> str | None:
        if url is not None:
            raise ValueError("guards, event sinks and prechecks run in process: no url")
        return url


class RunSpec(BaseModel):
    """One run: the env (simulated benchmark or robot) and embodiment, the skillsets and
    toolsets, a brain per role, the method, the tasks, the seed split and the plugins."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    env: RunItem
    embodiment: RunItem
    skillsets: list[RunItem] = []
    toolsets: list[RunItem] = []
    brains: dict[RoleName, RunItem] = {}
    method: RunItem
    tasks: list[TaskName]
    seeds: SeedSplit
    plugins: list[PluginItem] = []

    @field_validator("skillsets", "toolsets", "plugins")
    @classmethod
    def _unique(cls, items: list[RunItem]) -> list[RunItem]:
        names = [
            f"{item.group}:{item.name}" if isinstance(item, PluginItem) else item.name
            for item in items
        ]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"named twice: {twice}")
        return items

    @field_validator("toolsets")
    @classmethod
    def _remote_toolsets(cls, items: list[RunItem]) -> list[RunItem]:
        if remote := [item.name for item in items if item.url is not None and item.overrides]:
            raise ValueError(f"toolsets {remote}: a remote toolset's config is its server's")
        return items

    @field_validator("skillsets")
    @classmethod
    def _skillsets_in_toolsets(cls, items: list[RunItem]) -> list[RunItem]:
        if served := [item.name for item in items if item.url is not None]:
            raise ValueError(
                f"skillsets {served} have a url: a skillset extends toolsets and has no server of "
                "its own; serve the toolset it extends with it (the toolset server's `skillsets`) "
                "and give that toolset item the url"
            )
        return items

    @field_validator("tasks")
    @classmethod
    def _tasks(cls, tasks: list[TaskName]) -> list[TaskName]:
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError("tasks must be a non-empty list without repeats")
        return tasks


def recipe_of(binding: Path) -> Path:
    """The generic recipe an env binding binds: `recipes/<method>/` for a binding directory
    `.../run/<method>/`."""
    return repo_root() / "recipes" / binding.name


def binding_layers(binding: Path, file: str) -> dict[Layer, dict[str, Any]]:
    """The `recipe` and `env` layers of `file` (e.g. `protocols.yaml`) for the env binding
    directory `binding`: the generic recipe's file and the binding's, each when it exists."""
    layers: dict[Layer, dict[str, Any]] = {}
    for name, directory in (("recipe", recipe_of(binding)), ("env", binding)):
        path = directory / file
        if path.is_file():
            layers[name] = read_mapping(path)  # type: ignore[index]
    return layers


def resolve(
    layers: Mapping[Layer, dict[str, Any]],
    overrides: Sequence[str] = (),
    replace: Collection[str] = (),
) -> dict[str, Any]:
    """Stack the given file layers in `LAYERS` order (missing ones are skipped), then apply the
    run's `--set` overrides (`key.path=value`). `replace`: key paths the highest layer that names
    them sets whole, instead of merging into the lower layers' (`kernel.compose.compose`)."""
    unknown = sorted(set(layers) - set(LAYERS))
    if unknown:
        raise ValueError(f"unknown config layers {unknown}; the layers are {list(LAYERS)}")
    stacked = [layers[name] for name in LAYERS if name in layers]
    return compose({}, *stacked, overrides=overrides, replace=replace)
