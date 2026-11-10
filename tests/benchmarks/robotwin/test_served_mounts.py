"""A served Hub and a toolset server mount RoboTwin's toolsets for its robot (`embodiment` in
their configs, `integrations.hub_server.mount`, `ToolsetServerConfig`): a served Hub mounting
manipulation from a toolset server for the same robot passes the start check (`check_server`).
That a robot the tools do not fit has them left out, and that a Hub and server mounted for
different robots refuse to start, is tested on a one-armed mock robot in
`tests/core/test_skill_fit.py`."""

import importlib.util
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from pydantic import SecretStr

from roboscript.blobs import LocalBlobStore
from roboscript.hub import Hub
from roboscript.hub.service import HubServerConfig, ToolsetMount
from roboscript.hub.tokens import Tokens, service_key
from roboscript.integrations.hub_server import compose
from roboscript.integrations.mcp.serve import ToolsetServer, ToolsetServerConfig
from roboscript.settings import repo_root
from roboscript.testing.gateway import serving
from roboscript.toolsets.manipulation import Manipulation
from roboscript_methods.wicl.method import Wicl
from roboscript_robotwin.benchmark import HERE

pytestmark = pytest.mark.unit

SCRIPT = HERE.parents[4] / "examples" / "simBenchmarks" / "RoboTwin" / "scripts" / "run_s2.py"
ALOHA = "aloha_agilex"
SECRET = SecretStr("s" * 40)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_s2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mounts(tmp: Path) -> tuple[list[ToolsetMount], Hub]:
    """The toolsets of a RoboTwin S1 stack (`run_s2.toolsets`, with demo) as served-Hub mounts,
    and the Hub of that stack as a run builds it."""
    script = _script()
    (tmp / "configs").mkdir(exist_ok=True)
    example = repo_root() / "examples/simBenchmarks/RoboTwin/configs" / "manipulation.example.yaml"
    (tmp / "configs" / "manipulation.yaml").write_text(
        example.read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = script.manipulation_config("http://127.0.0.1:41995", root=tmp)
    providers = script.toolsets(config, {"motion": "http://127.0.0.1:1"}, demo=True)
    items = [
        ToolsetMount(
            name=p.name, config=p.config.model_dump(mode="json"), skillsets=list(p.skillsets)
        )
        for p in providers
    ]
    return items, Hub(providers, blobs=LocalBlobStore(tmp / "blobs"), phases=Wicl.phases)


def _listed(hub: Hub, toolset: str | None = None) -> list[tuple[str, str]]:
    """The mounted tools (of `toolset`; None: all) with their descriptor hashes, in order."""
    return [
        (name, tool.descriptor.hash)
        for name, tool in hub.mounted.items()
        if toolset is None or tool.meta.toolset == toolset
    ]


async def test_a_served_hub_starts_on_a_toolset_server_mounted_for_the_same_robot(
    tmp_path: Path,
) -> None:
    items, run_stack = _mounts(tmp_path)
    manipulation = next(item for item in items if item.name == "manipulation")
    config = ToolsetServerConfig(
        blobs=str(tmp_path / "blobs"),
        toolset=manipulation.config,
        skillsets=manipulation.skillsets,
        embodiment=ALOHA,
    )
    blobs = LocalBlobStore(tmp_path / "blobs")
    async with httpx.AsyncClient(trust_env=False) as http:
        tokens = Tokens({"service": service_key(SECRET)})
        server = ToolsetServer(Manipulation(), config, tokens=tokens, blobs=blobs, http=http)
        async with server, serving(server.app(), tmp_path) as url:
            remote = manipulation.model_copy(update={"config": {}, "url": f"{url}/mcp"})
            hub_config = HubServerConfig(
                method="roboscript.testing.phases:TestMethod",
                toolsets=[remote],
                embodiment=ALOHA,
                blobs=str(tmp_path / "blobs"),
            )
            async with compose(
                hub_config,
                tokens=Tokens.from_secret(SECRET),
                blobs=blobs,
                events=None,
                gateway=None,
            ) as service:
                served = _listed(service.hub, "manipulation")
    in_process = _listed(run_stack, "manipulation")
    assert served == in_process and "rotate_about" in dict(in_process)  # RoboTwin's skillset
