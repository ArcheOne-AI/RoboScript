"""The planner service's lifecycle over a real socket, with the mock planner in place of cuRobo."""

import math
from functools import partial
from pathlib import Path

import anyio
import httpx
import pytest

from roboscript.contracts.errors import ProtocolError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import TIMEOUTS, Planner, PlanRequest
from roboscript.motion import MockPlanner
from roboscript.motion.service import run
from roboscript.rpc import Ready, aclose, client, read_ready
from roboscript.testing import precheck
from roboscript.testing.planner import PlannerConformance, PlannerTarget

WORLD = Pose(p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0))
HOME = [-0.25, -0.2, 0.95, 0.0, math.pi / 2, 0.0]
AWAY = [-0.1, -0.05, 0.9, 0.2, 1.2, 0.3]


class GatedMock(MockPlanner):
    """Warms up once the test opens the gate."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = anyio.Event()
        self.ready = False
        self.failure: str | None = None

    async def start(self) -> None:
        await self.gate.wait()
        self.ready = True


async def test_the_service_listens_first_and_serves_after_the_warm_up(tmp_path: Path) -> None:
    planner = GatedMock()
    ready_file = tmp_path / "ready" / "motion.json"
    async with anyio.create_task_group() as tg, httpx.AsyncClient(trust_env=False) as http:
        tg.start_soon(partial(run, planner, ready_file=ready_file))
        ready = await _read(ready_file)
        assert (await http.get(f"{ready.url}/readyz")).status_code == 503
        planner.gate.set()
        with anyio.fail_after(5):
            while True:
                if (await http.get(f"{ready.url}/readyz")).status_code == 200:
                    break
                await anyio.sleep(0.01)

        proxy = client(Planner, ready.url, token=None, timeout=10, timeouts=TIMEOUTS)
        target = PlannerTarget(proxy, WORLD, {"left": [HOME, AWAY]}, lock_drift_deg=1e-6)
        assert not any((await precheck(PlannerConformance(), target)).values())  # via JSON
        request = PlanRequest(arm="left", base=WORLD, start=HOME[:5], goal=WORLD)
        with pytest.raises(ProtocolError, match="has 6 joints"):
            await proxy.plan(request)
        await aclose(proxy)
        tg.cancel_scope.cancel()


async def _read(path: Path) -> Ready:
    with anyio.fail_after(10):
        while True:
            if (ready := read_ready(path)) is not None:
                return ready
            await anyio.sleep(0.01)
