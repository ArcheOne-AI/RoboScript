"""roboscript.contracts.motion: JSON round trips, strict requests, derived fields, interface."""

import inspect
import typing
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import (
    MAX_GOALS,
    MAX_JOINT_VECTORS,
    TIMEOUTS,
    ArmInfo,
    BatchPlanRequest,
    FKRequest,
    FKResult,
    IKRequest,
    IKSolution,
    Planner,
    PlannerInfo,
    PlanRequest,
    PlanResult,
)
from roboscript.contracts.versions import VERSIONS

BASE = Pose(p=(0.0, -0.6, 0.0), q=(0.7071068, 0.0, 0.0, 0.7071068))
GOAL = Pose(p=(-0.2, -0.1, 0.9), q=(0.5, -0.5, 0.5, 0.5))
JOINTS = [0.0, 1.2, 1.0, 0.1, -0.2, 0.3]

SAMPLES: list[BaseModel] = [
    PlannerInfo(
        provider="mock",
        arms={
            "left": ArmInfo(
                joints=["j1"], lower=[-1.0], upper=[1.0], links=["l6"], ee_to_tcp=(0.12, 0, 0)
            )
        },
        dt=0.004,
    ),
    PlanRequest(arm="left", base=BASE, start=JOINTS, goal=GOAL, constraint="orientation"),
    BatchPlanRequest(arm="right", base=BASE, start=JOINTS, goals=[GOAL, GOAL], ee_path=True),
    PlanResult(reason="ok", path=[JOINTS, JOINTS], dt=0.004, ee_path=[GOAL, GOAL], time_s=0.05),
    PlanResult(reason="unreachable", detail="IK Fail", dt=0.004),
    IKRequest(arm="left", base=BASE, goals=[GOAL], seed=JOINTS),
    IKSolution(joints=JOINTS, position_error_m=0.0004, rotation_error_deg=0.01),
    IKSolution(joints=None),
    FKRequest(arm="left", base=BASE, joints=[JOINTS], links=["fl_link6"]),
    FKResult(ee=[GOAL], tcp=[GOAL], links={"fl_link6": [GOAL]}),
]


@pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
def test_models_round_trip_through_json(model: BaseModel) -> None:
    assert type(model).model_validate_json(model.model_dump_json()) == model


REQUESTS: dict[type[BaseModel], dict[str, Any]] = {
    PlanRequest: {"arm": "left", "base": BASE, "start": JOINTS, "goal": GOAL},
    BatchPlanRequest: {"arm": "left", "base": BASE, "start": JOINTS, "goals": [GOAL]},
    IKRequest: {"arm": "left", "base": BASE, "goals": [GOAL], "seed": JOINTS},
    FKRequest: {"arm": "left", "base": BASE, "joints": [JOINTS]},
}


@pytest.mark.parametrize("request_model", REQUESTS, ids=lambda m: m.__name__)
def test_requests_refuse_unknown_fields_and_empty_lists(request_model: type[BaseModel]) -> None:
    data = REQUESTS[request_model]
    request_model.model_validate(data)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        request_model.model_validate({**data, "seed_state": 1})
    for field in ("goals", "joints"):
        if field in data:
            with pytest.raises(ValidationError, match="at least 1 item"):
                request_model.model_validate({**data, field: []})


def test_request_lists_are_capped() -> None:
    goals = [GOAL] * (MAX_GOALS + 1)
    with pytest.raises(ValidationError, match=f"at most {MAX_GOALS} items"):
        BatchPlanRequest.model_validate({**REQUESTS[BatchPlanRequest], "goals": goals})
    with pytest.raises(ValidationError, match=f"at most {MAX_GOALS} items"):
        IKRequest.model_validate({**REQUESTS[IKRequest], "goals": goals})
    joints = [JOINTS] * (MAX_JOINT_VECTORS + 1)
    with pytest.raises(ValidationError, match=f"at most {MAX_JOINT_VECTORS} items"):
        FKRequest.model_validate({**REQUESTS[FKRequest], "joints": joints})


def test_requests_refuse_unknown_arms_constraints_and_non_finite_joints() -> None:
    data = REQUESTS[PlanRequest]
    for bad in ({"arm": "middle"}, {"constraint": "line"}, {"start": [float("nan")] * 6}):
        with pytest.raises(ValidationError):
            PlanRequest.model_validate({**data, **bad})


def test_status_and_success_are_derived_and_serialized() -> None:
    ok = PlanResult(reason="ok", path=[JOINTS], dt=0.004)
    failed = PlanResult(reason="no_path", detail="TrajOpt Fail", dt=0.004)
    assert (ok.status, failed.status) == ("success", "failed")
    assert '"status":"failed"' in failed.model_dump_json()
    assert IKSolution(joints=JOINTS).success and not IKSolution(joints=None).success
    assert '"success":false' in IKSolution(joints=None).model_dump_json()


def test_the_planner_interface_is_a_versioned_async_plane() -> None:
    assert Planner.PLANE == "motion" and VERSIONS["motion"] == Planner.VERSION
    methods = {
        name: member
        for name, member in vars(Planner).items()
        if inspect.isfunction(member) and not name.startswith("_")
    }
    assert set(methods) == {"info", "plan", "plan_batch", "ik", "fk"} == set(TIMEOUTS)
    for name, method in methods.items():
        assert inspect.iscoroutinefunction(method), name
        hints = typing.get_type_hints(method, include_extras=True)
        params = [p for p in inspect.signature(method).parameters if p != "self"]
        assert set(hints) == {*params, "return"}, name
