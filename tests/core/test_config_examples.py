"""Every core example config (`configs/*.example.yaml`) validates against the model that reads it,
and those that say "without a file the defaults apply" hold exactly the defaults. Method,
benchmark and vision examples are checked with their own tests."""

from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from roboscript.agent.responses import ResponsesConfig
from roboscript.envs.mock import MockConfig
from roboscript.gateway.config import load
from roboscript.hub import HubConfig
from roboscript.hub.service import HubServerConfig
from roboscript.memory import MemoryConfig, StoreSpec
from roboscript.perception.config import PerceptionConfig
from roboscript.settings import load_config_file, repo_root
from roboscript.toolsets.manipulation import ManipulationConfig
from roboscript.toolsets.scene import SceneConfig
from roboscript.toolsets.sensing import SensingConfig

CONFIGS = repo_root() / "configs"
DEFAULTS: dict[str, type[BaseModel]] = {
    "hub": HubConfig,
    "perception": PerceptionConfig,
    "scene": SceneConfig,
    "sensing": SensingConfig,
}
"""Examples that restate the defaults."""
VALID: dict[str, type[BaseModel]] = {
    "agent": ResponsesConfig,
    "hub-server": HubServerConfig,
    "manipulation": ManipulationConfig,
    "memory": MemoryConfig,
    "mock_env": MockConfig,
}
"""Examples that set values of their own."""
ELSEWHERE = {"builder", "learning", "motion", "toolset-server", "vision"}
"""The methods' (builder, learning), the served toolset's and the vision service's examples are
checked by their own tests; motion's is a placeholder to fill in with a robot's values."""


def _data(name: str) -> Any:
    return yaml.safe_load((CONFIGS / f"{name}.example.yaml").read_text(encoding="utf-8"))


def test_every_example_is_checked() -> None:
    found = {path.name.removesuffix(".example.yaml") for path in CONFIGS.glob("*.example.yaml")}
    assert found == set(DEFAULTS) | set(VALID) | {"gateway"} | ELSEWHERE


@pytest.mark.parametrize("name", sorted(DEFAULTS))
def test_the_example_holds_the_defaults(name: str) -> None:
    model = DEFAULTS[name]
    assert model.model_validate(_data(name)) == model()


@pytest.mark.parametrize("name", sorted(VALID))
def test_the_example_validates(name: str) -> None:
    load_config_file(CONFIGS / f"{name}.example.yaml", VALID[name])


def test_the_manipulation_example_overrides_no_binding() -> None:
    config = ManipulationConfig.model_validate(_data("manipulation"))
    assert config.planner.url is not None and config.table_top_z is None  # the scene's to set
    assert config.model_copy(update={"executor": {}}) == ManipulationConfig(planner=config.planner)
    assert config.executor == {}  # a copy never shadows the binding


def test_the_agent_example_is_the_defaults_but_its_context_windows() -> None:
    config = ResponsesConfig.model_validate(_data("agent"))
    assert config.context.windows == {"brain-default": 272000}
    windows = config.context.model_copy(update={"windows": {}})
    assert config.model_copy(update={"context": windows}) == ResponsesConfig()


def test_the_memory_example_is_the_defaults_but_its_store_and_recipe() -> None:
    config = MemoryConfig.model_validate(_data("memory"))
    assert config.store == StoreSpec(name="file", options={"root": "runs/memory"})
    assert config.model_copy(update={"store": StoreSpec()}) == MemoryConfig(recipe=config.recipe)


def test_the_gateway_example_names_its_variables_in_the_env_api_example() -> None:
    example = load(CONFIGS / "gateway.example.yaml")
    assert example.endpoints and example.models
    for model in example.models.values():
        assert {d.endpoint for d in model.deployments} <= set(example.endpoints)
    assert {example.resolve(alias) for alias in example.aliases} <= set(example.models)
    variables = {e.url_env for e in example.endpoints.values()}
    variables |= {e.key_env for e in example.endpoints.values()}
    listed = (repo_root() / ".env.api.example").read_text(encoding="utf-8").splitlines()
    assert variables <= {line.split("=")[0] for line in listed if "=" in line}
