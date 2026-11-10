"""The cuRobo planner's failure latch, without a GPU (fakes stand in for it): an error on its
thread takes it out of service, but for a ValueError in cuRobo's graph interpolation, which fails
only its request (a float edge in `linear_smooth`)."""

import threading
import types
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final

import anyio
import numpy as np
import pytest

from roboscript.contracts.errors import InfraError
from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import (
    FKRequest,
    FKResult,
    PlanRequest,
    PlanResult,
)
from roboscript.contracts.skills import PlannerProfile, PlannerTable
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.motion import ArmConfig, CuroboPlanner
from roboscript.motion.config import CuroboConfig
from roboscript.motion.curobo import _Arm

BASE = Pose(p=(0.0, -0.6, 0.0), q=(1.0, 0.0, 0.0, 0.0))
PLAN = PlanRequest(arm="left", base=BASE, start=[0.0] * 6, goal=BASE)
EDGE = (
    "A value (27.0) in x_new is above the interpolation range's maximum value (26.999999999999996)."
)


async def test_an_error_on_the_planner_thread_makes_the_planner_unready(tmp_path: Path) -> None:
    planner = _planner(tmp_path)
    calls = []

    def crash(*args: Any) -> FKResult:
        calls.append(args)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    planner._fk = crash  # type: ignore[method-assign]
    request = FKRequest(arm="left", base=BASE, joints=[[0.0] * 6])
    assert planner.ready
    with pytest.raises(InfraError, match=r"RuntimeError in crash.*restart") as failed:
        await planner.fk(request)
    assert failed.value.code == "not_ready"
    assert str(failed.value.__cause__) == "CUDA error: an illegal memory access was encountered"
    assert not planner.ready  # /readyz answers 503, and the service exits to be restarted
    assert planner.failure == "RuntimeError in crash"
    for call in (planner.info(), planner.fk(request)):
        with pytest.raises(InfraError, match=r"RuntimeError in crash.*restart") as caught:
            await call
        assert caught.value.code == "not_ready"
    assert len(calls) == 1  # nothing reached the planner thread after the failure


def _fake_arm() -> _Arm:
    return _Arm(
        joints=[f"fl_joint{i}" for i in range(1, 7)],
        links=["fl_link6"],
        lower=[-10.0] * 6,
        upper=[10.0] * 6,
        planner_T_root=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        single=None,
        batch=None,
        kinematics=None,
        ee_to_tcp=(0.12, 0.0, 0.0),
        world_base=BASE.as_list(),
        world=(),
    )


def _frame(module: str, name: str, inner: Callable[[Exception], None] | None) -> Any:
    """A function named `name` as if defined in `module` (its frames carry both), calling
    `inner` with the error it is given, or raising it if there is none."""
    code = _call.__code__.replace(co_name=name)
    return types.FunctionType(code, {"__name__": module, "_inner": inner}, name)


_inner: Callable[[Exception], None] | None = None
"""What `_call` calls: each `_frame`'s own, in its globals."""


def _call(error: Exception) -> None:
    if _inner is None:
        raise error
    _inner(error)


def _stack(*frames: tuple[str, str]) -> Callable[[Exception], None]:
    """Calls through `frames` (outer to inner) to raise the error it is given."""
    call = None
    for module, name in reversed(frames):
        call = _frame(module, name, call)
    assert call is not None
    return call  # type: ignore[no-any-return]


GRAPH_FRAMES: Final = (
    ("contextlib", "inner"),
    ("curobo.wrap.reacher.motion_gen", "graph_search"),
    ("torch.utils._contextlib", "decorate_context"),
    ("curobo.graph.graph_base", "find_paths"),
    ("curobo.graph.graph_base", "get_interpolated_trajectory"),
    ("curobo.util.trajectory", "get_interpolated_trajectory"),
    ("contextlib", "inner"),
    ("curobo.util.trajectory", "linear_smooth"),
    ("scipy.interpolate._polyint", "__call__"),
    ("scipy.interpolate._interpolate", "_evaluate"),
    ("scipy.interpolate._interpolate", "_check_bounds"),
)
"""The frames below `MotionGen._plan_from_solve_state` on a motion-service traceback
(`curobo.GRAPH_INTERPOLATION`)."""
GRAPH = _stack(*GRAPH_FRAMES)
"""cuRobo's graph planner interpolating a path it found, raising the error it is given."""


async def test_a_value_error_in_the_graph_interpolation_fails_only_its_request(
    tmp_path: Path,
) -> None:
    planner = _planner(tmp_path)
    calls: list[int] = []

    def _plan_single(*args: Any) -> PlanResult:
        calls.append(len(calls))
        if len(calls) == 1:
            GRAPH(ValueError(EDGE))
        return PlanResult(reason="no_path", dt=0.004)

    planner._plan_single = _plan_single  # type: ignore[method-assign]
    with pytest.raises(InfraError, match=r"failed this request .*graph interpolation") as caught:
        await planner.plan(PLAN)
    assert caught.value.code == "not_ready"  # the infrastructure's: never a plan_failed receipt
    assert isinstance(caught.value.__cause__, ValueError)
    assert planner.ready and planner.failure is None
    assert (await planner.plan(PLAN)).reason == "no_path"  # it goes on planning
    assert calls == [0, 1]


TRAJECTORY = [f for f in GRAPH_FRAMES if f[0] in ("curobo.util.trajectory", "contextlib")]
GRAPH_BASE = [f for f in GRAPH_FRAMES if f[0] != "curobo.util.trajectory"]
OTHER_VALUE = "operands could not be broadcast together with shapes (27,) (28,)"
"""Another `ValueError` from those frames (not scipy's out-of-range one)."""
OTHERS = [
    pytest.param(lambda: _stack(("roboscript", "plan")), ValueError, EDGE, id="outside cuRobo"),
    pytest.param(lambda: _stack(*TRAJECTORY), ValueError, EDGE, id="linear_smooth, no graph"),
    pytest.param(lambda: _stack(*GRAPH_BASE), ValueError, EDGE, id="graph_base, no linear_smooth"),
    pytest.param(
        lambda: _stack(*reversed(GRAPH_FRAMES)), ValueError, EDGE, id="the frames reversed"
    ),
    pytest.param(lambda: GRAPH, RuntimeError, EDGE, id="not a ValueError"),
    pytest.param(lambda: GRAPH, ValueError, OTHER_VALUE, id="another ValueError there"),
]


async def test_calls_queued_behind_the_failing_one_are_refused(tmp_path: Path) -> None:
    """A call queued on the planner thread while the one before it fails is refused there, not
    run on a planner whose state is suspect; the failure named is the first."""
    planner = _planner(tmp_path)
    entered, release = threading.Event(), threading.Event()
    ran: list[str] = []

    def _plan_single(*args: Any) -> PlanResult:
        ran.append("plan")
        entered.set()
        release.wait(10)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    def _fk(*args: Any) -> FKResult:
        ran.append("fk")
        raise KeyError("a second failure")

    planner._plan_single = _plan_single  # type: ignore[method-assign]
    planner._fk = _fk  # type: ignore[method-assign]
    outcomes: dict[str, BaseException] = {}

    async def call(name: str, coroutine: Awaitable[object]) -> None:
        try:
            await coroutine
        except Exception as error:
            outcomes[name] = error

    request = FKRequest(arm="left", base=BASE, joints=[[0.0] * 6])
    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg:
            tg.start_soon(call, "plan", planner.plan(PLAN))
            await anyio.to_thread.run_sync(entered.wait, 10)
            tg.start_soon(call, "fk", planner.fk(request))  # admitted: nothing failed yet
            await anyio.sleep(0.2)  # queued behind the plan
            release.set()
    assert ran == ["plan"]
    for name in ("plan", "fk"):
        error = outcomes[name]
        assert isinstance(error, InfraError) and error.code == "not_ready", name
        assert "RuntimeError in _plan_single" in str(error), name
    assert planner.failure == "RuntimeError in _plan_single"


def _planner(tmp_path: Path) -> CuroboPlanner:
    """A cuRobo planner as if warmed up, on a fake arm (no GPU, no curobo)."""
    planner = CuroboPlanner(
        CuroboConfig(
            embodiment=str(MOCK_BIMANUAL),
            arms={"left": ArmConfig(curobo_yml=Path("left.yml"), base=BASE)},
            planner_profile=PlannerProfile(table=PlannerTable(size=(0.7, 2.0, 0.04), top_z=0.7)),
            assets_root=tmp_path,
        )
    )
    planner._arms = {"left": _fake_arm()}
    return planner
