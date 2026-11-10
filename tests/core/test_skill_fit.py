"""A mounted tool fits the robot it mounts for (docs/how-to/add-a-skillset.md): the core entries
take arms named `left` and `right`, so on a one-armed robot whose arm is `arm` none of them mounts,
though every binding they ask for is there. On such a robot the two-armed entries are left out
and the run goes on, and a served Hub refuses to start on a toolset server mounted for another
robot."""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
import yaml
from pydantic import SecretStr

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.capabilities import EnvCapabilities
from roboscript.contracts.skills import Bindings, SkillSet
from roboscript.embodiments import MOCK_BIMANUAL, load_embodiment
from roboscript.envs import load_benchmark
from roboscript.hub.service import HubServerConfig, ToolsetMount
from roboscript.hub.tokens import Tokens, service_key
from roboscript.integrations.hub_server import compose
from roboscript.integrations.mcp.serve import ToolsetServer, ToolsetServerConfig
from roboscript.kernel import KernelError
from roboscript.learning.prechecks import (
    SKILLSETS,
    RunFacts,
    precheck_result,
    run_prechecks,
)
from roboscript.learning.runspec import RunSpec
from roboscript.testing.gateway import serving
from roboscript.testing.skills import MOCK_EXECUTOR
from roboscript.tools import LocalProvider
from roboscript.tools.provider import unfit_tools
from roboscript.tools.skillsets import SkillSetPlugin, mounted
from roboscript.toolsets.manipulation import Manipulation, ManipulationConfig, PlannerConfig

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
BINDS = "fake_single:BINDS"
SECRET = SecretStr("s" * 40)
MANIPULATION_TOOLS = {spec.name for spec in Manipulation.specs()}
PLANNED = ManipulationConfig(planner=PlannerConfig(url="http://127.0.0.1:1"))  # never called


@pytest.fixture
def single(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-armed mock robot whose arm is `arm`, and `fake_single:BINDS`, a skillset that binds
    what the core entries ask for (the executor's tuning) on any robot."""
    data: dict[str, Any] = yaml.safe_load(Path(MOCK_BIMANUAL).read_text(encoding="utf-8"))
    arm = data["arms"]["left"]
    data |= {
        "name": "mock_single",
        "action_dim": 7,
        "arms": {"arm": arm},
        "cameras": {"overhead": data["cameras"]["overhead"]},
    }
    path = tmp_path / "mock_single.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    module = ModuleType("fake_single")
    bindings = Bindings(executor=MOCK_EXECUTOR)
    binds = SkillSet(name="binds", version="1", layer="robot", bindings=bindings)
    module.BINDS = SkillSetPlugin(binds)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_single", module)
    return path


def test_no_core_entry_mounts_on_a_robot_whose_arm_is_named_otherwise(single: Path) -> None:
    profile = load_embodiment(single)
    unchecked = mounted(Manipulation, [BINDS])
    assert unchecked.specs and not unchecked.unmounted  # bindings alone let them all through
    found = mounted(Manipulation, [BINDS], profile)
    assert not found.specs and set(found.unmounted) == {s.name for s in Manipulation.specs()}
    assert found.unmounted["grasp_object"] == (
        "grasp_object: takes arms named ['left', 'right'], mock_single has ['arm']"
    )
    assert "together: needs 2 arms, mock_single has 1" in found.unmounted["together"]
    with pytest.raises(KernelError, match="mounts none of its tools"):
        LocalProvider(Manipulation(), PLANNED, [BINDS], profile)


def _facts(embodiment: str, **extra: Any) -> RunFacts:
    benchmark = load_benchmark("mock")
    spec = RunSpec.model_validate(
        {
            "env": {"name": "mock"},
            "embodiment": {"name": embodiment},
            "method": {"name": "a_method"},
            "tasks": list(benchmark.tasks),
            "seeds": {"blocks": {}},
        }
        | extra
    )
    return RunFacts(spec=spec, offered=benchmark.capabilities, required=EnvCapabilities())


@pytest.fixture
def left(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-armed mock robot whose arm is `left`, and `fake_left:BENCH`, the Mock benchmark with
    `fake_left:BINDS` (the executor's tuning, on any robot) for its skillsets."""
    data: dict[str, Any] = yaml.safe_load(Path(MOCK_BIMANUAL).read_text(encoding="utf-8"))
    data |= {"name": "mock_left", "action_dim": 7, "arms": {"left": data["arms"]["left"]}}
    path = tmp_path / "mock_left.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    module = ModuleType("fake_left")
    binds = SkillSet(
        name="binds", version="1", layer="robot", bindings=Bindings(executor=MOCK_EXECUTOR)
    )
    module.BINDS = SkillSetPlugin(binds)  # type: ignore[attr-defined]
    module.BENCH = load_benchmark("mock").model_copy(  # type: ignore[attr-defined]
        update={"skillsets": ["fake_left:BINDS"]}
    )
    monkeypatch.setitem(sys.modules, "fake_left", module)
    return path


async def test_on_a_one_armed_robot_the_two_armed_entries_are_left_out_and_the_run_goes_on(
    left: Path, caplog: pytest.LogCaptureFixture
) -> None:
    profile = load_embodiment(left)
    provider = LocalProvider(Manipulation(), PLANNED, ["fake_left:BINDS"], profile)
    names = [spec.name for spec in provider.specs()]
    assert "grasp_object" in names and "together" not in names and "plan_arms" not in names
    assert provider.unmounted["together"] == "together: needs 2 arms, mock_left has 1"
    assert unfit_tools([provider], profile) == []  # what it mounted fits
    spec = RunSpec.model_validate(
        {
            "env": {"name": "fake_left:BENCH"},
            "embodiment": {"name": str(left)},
            "method": {"name": "a_method"},
            "tasks": ["lift_red"],
            "seeds": {"blocks": {}},
            "toolsets": [{"name": "manipulation"}],
        }
    )
    facts = RunFacts(spec=spec, offered=EnvCapabilities(), required=EnvCapabilities())
    found = await precheck_result([(SKILLSETS, {})], facts)
    assert found.problems == []  # the run is not refused
    assert sorted(found.notes) == [
        "skillsets: manipulation.plan_arms: not mounted, plan_arms: needs 2 arms, mock_left has 1",
        "skillsets: manipulation.together: not mounted, together: needs 2 arms, mock_left has 1",
    ]
    with caplog.at_level(logging.INFO, logger="roboscript.learning.prechecks"):
        assert await run_prechecks([(SKILLSETS, {})], facts) == []
    assert "manipulation.together: not mounted" in caplog.text  # said, not refused


@asynccontextmanager
async def _toolset_server(root: Path, embodiment: str | None) -> AsyncIterator[str]:
    """The manipulation toolset with `fake_left:BINDS` on a toolset server for `embodiment`;
    yields its MCP endpoint."""
    config = ToolsetServerConfig(
        blobs=str(root / "blobs"),
        toolset=PLANNED.model_dump(mode="json"),
        skillsets=["fake_left:BINDS"],
        embodiment=embodiment,
    )
    tokens = Tokens({"service": service_key(SECRET)})
    async with httpx.AsyncClient(trust_env=False) as http:
        blobs = LocalBlobStore(root / "blobs")
        server = ToolsetServer(Manipulation(), config, tokens=tokens, blobs=blobs, http=http)
        async with server, serving(server.app(), root) as url:
            yield f"{url}/mcp"


def _served_hub(root: Path, url: str, embodiment: str | None) -> HubServerConfig:
    item = ToolsetMount(name="manipulation", url=url, skillsets=["fake_left:BINDS"])
    return HubServerConfig(
        method="roboscript.testing.phases:TestMethod",
        toolsets=[item],
        embodiment=embodiment,
        blobs=str(root / "blobs"),
    )


async def test_a_served_hub_refuses_to_start_on_a_toolset_server_mounted_for_another_robot(
    left: Path, tmp_path: Path
) -> None:
    """A server mounted for the one-armed robot leaves `together` and `plan_arms` out, which a
    Hub naming no robot would offer: the Hub does not start, as a run would not. Against a server
    that mounts as it does, it starts and offers what the server lists."""
    blobs = LocalBlobStore(tmp_path / "blobs")
    tokens = Tokens.from_secret(SECRET)
    async with _toolset_server(tmp_path, str(left)) as url:
        with pytest.raises(ValueError, match=r"it lacks \['together', 'plan_arms'\]"):
            async with compose(
                _served_hub(tmp_path, url, None),
                tokens=tokens,
                blobs=blobs,
                events=None,
                gateway=None,
            ):
                raise AssertionError("the Hub started")
    async with (
        _toolset_server(tmp_path, None) as url,
        compose(
            _served_hub(tmp_path, url, None),
            tokens=tokens,
            blobs=blobs,
            events=None,
            gateway=None,
        ) as service,
    ):
        names = [name for name in service.hub.mounted if name in MANIPULATION_TOOLS]
    assert names == [spec.name for spec in Manipulation.specs()]
