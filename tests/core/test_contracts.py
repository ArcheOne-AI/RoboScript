"""roboscript.contracts: validation, JSON round trips, secrets, interfaces."""

import inspect
import json
import typing
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from roboscript.contracts.action import ActionBatch, ActionStep, ExecuteResult, StepState
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.camera import CameraFrame, CameraModel
from roboscript.contracts.capabilities import EnvCapabilities, SensingCapabilities
from roboscript.contracts.env import (
    Diagnosis,
    EnvAgent,
    EnvControl,
    EnvDiscovery,
    EnvPublic,
    EpisodeOpened,
    EpisodeSpec,
    ExpertAction,
    ExpertCall,
    ExpertTrace,
    GtSnapshot,
    JudgeResult,
    ObjectTruth,
    ObserveRequest,
)
from roboscript.contracts.gateway import (
    GatewayAdmin,
    Permit,
    PermitIntake,
    SessionRegistration,
)
from roboscript.contracts.geometry import OrientedBox, Pose
from roboscript.contracts.hub import HubControl
from roboscript.contracts.ids import REVEAL
from roboscript.contracts.observation import Observation
from roboscript.contracts.receipt import (
    HARD_FAILURES,
    AchievedPose,
    ActionReceipt,
    CommandedPoses,
    GraspChoice,
    GripperReport,
    ReceiptReason,
)
from roboscript.contracts.robot import ArmState, RobotState
from roboscript.contracts.tools import (
    CallContext,
    EnvBinding,
    ImagePart,
    TextPart,
    ToolCall,
    ToolDescriptor,
    ToolMeta,
    ToolResult,
)
from roboscript.contracts.trace import TraceContext

MASK = "**********"

POSE = Pose(p=(0.1, -0.2, 0.9), q=(1.0, 0.0, 0.0, 0.0))
BLOB_RGB = BlobRef(
    id="a" * 64, size=921_600, media_type="image/png", dtype="uint8", shape=(480, 640, 3)
)
BLOB_DEPTH = BlobRef(
    id="b" * 64, size=1_228_928, media_type="application/x-npy", dtype="float32", shape=(480, 640)
)
CAMERA = CameraModel(
    name="overhead",
    width=640,
    height=480,
    K=((717.3, 0, 320), (0, 717.3, 240), (0, 0, 1)),
    cam2world=((1, 0, 0, 0), (0, -1, 0, -0.45), (0, 0, -1, 1.35), (0, 0, 0, 1)),
)
ARM = ArmState(ee=POSE, tcp=POSE, joints=None, joints_cmd=[0.0] * 6, gripper=None, gripper_cmd=1.0)
ROBOT = RobotState(step=3, step_limit=400, arms={"left": ARM, "right": ARM}, qpos_cmd=[0.0] * 14)

SAMPLES: list[BaseModel] = [
    POSE,
    OrientedBox(center=(0, 0, 0.8), size=(0.2, 0.1, 0.05), q=(1, 0, 0, 0)),
    BLOB_RGB,
    CAMERA,
    CameraFrame(camera=CAMERA, rgb=BLOB_RGB, depth=BLOB_DEPTH),
    ROBOT,
    Observation(
        frame_id="frame-0123456789ab",
        step=3,
        cameras={"overhead": CameraFrame(camera=CAMERA, rgb=BLOB_RGB)},
        robot=ROBOT,
        instruction="Pick up the hammer.",
    ),
    ActionBatch(batch_id="batch-1", steps=[ActionStep(qpos=[0.0] * 14, until_gripper_stable=True)]),
    ExecuteResult(
        batch_id="batch-1", states=[StepState(index=0, robot=ROBOT)], executed=1, stopped="success"
    ),
    ActionReceipt(
        reason=ReceiptReason.clamped,
        arm="left",
        commanded=CommandedPoses(ee=POSE, tcp=POSE),
        achieved=AchievedPose(tcp=POSE, orient_err_deg=1.5),
        deviation_m=0.004,
        steps_used=6,
        steps_left=394,
        gripper=GripperReport(commanded=0.0, measured=0.21, holding=True),
        grasp=GraspChoice(candidate=1, n_candidates=4),
        tcp_path={"left": [POSE, POSE]},
        note="clamped to 0.2 m",
    ),
    ToolDescriptor(
        name="grasp_object",
        description="Grasp an object.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema=None,
        meta=ToolMeta(
            toolset="manipulation",
            kind="execution",
            tier="D",
            stages=["s1", "s2", "p3"],
            uses_steps=True,
            needs_rationale=True,
            timeout_s=120,
        ),
        hash="c" * 64,
    ),
    ToolResult(
        content=[TextPart(text="obj_3 at the left"), ImagePart(blob=BLOB_RGB, label="view")]
    ),
    ToolCall(call_id="call-1", name="observe", arguments={"cameras": ["overhead"]}, origin="model"),
    CallContext(
        episode_ref="ep-0123456789ab",
        stage="s2",
        tier="D",
        env=EnvBinding(url="http://127.0.0.1:18100", episode_id="env-1", token="agent-secret"),
        embodiment="mock_bimanual",
        trace=TraceContext(trace_id="1" * 32, span_id="2" * 16),
    ),
    EnvDiscovery(
        contracts={"env.agent": "1.0", "env.control": "1.0", "env.public": "1.0"},
        benchmark="mock",
        embodiment="mock_bimanual",
        upstream_commit="abc1234",
        config_hash="d" * 64,
        capabilities=EnvCapabilities(
            success={"latch"},
            reset={"seed"},
            sensing=SensingCapabilities(
                scene_identity=True,
                measured_proprio=True,
                depth=True,
                cameras=["overhead", "wrist_left", "wrist_right"],
                step_unit="action",
            ),
        ),
    ),
    EpisodeSpec(task="lift_red", seed=10000, stage="s2"),
    EpisodeOpened(
        episode_id="env-1",
        agent_url="http://127.0.0.1:18101",
        agent_token="agent-secret",
        instruction="Pick up the hammer.",
        step_limit=400,
        embodiment="mock_bimanual",
        cameras=["overhead"],
        solvable=True,
    ),
    JudgeResult(episode_id="env-1", success=True, success_step=57, steps_used=57, step_limit=400),
    Diagnosis(
        episode_id="env-1",
        success=False,
        objects=[ObjectTruth(name="hammer", initial=POSE, final=POSE, displacement_m=0.0)],
        notes=["hammer never lifted"],
    ),
    GtSnapshot(episode_id="env-1", step=10, objects={"hammer": POSE}),
    EpisodeSpec(task="lift_red", seed=10000, prerun_attempts=3, trace_expert=True),
    ExpertTrace(
        episode_id="env-1",
        calls=[
            ExpertCall(
                kind="move",
                actions={
                    "left": [
                        ExpertAction(target=POSE),
                        ExpertAction(target=POSE, constraint="orientation"),
                        ExpertAction(gripper=0.0),
                    ]
                },
                api={"left": "grasp_object"},
                sim_steps=412,
            ),
            ExpertCall(kind="hold", sim_steps=600),
            ExpertCall(kind="move", actions={}, sim_steps=0, refused=True),
            ExpertCall(
                kind="move",
                actions={"right": [ExpertAction(target=POSE)]},
                api={"right": "grasp_object"},
                sim_steps=180,
                partial=True,
                openings={"right": 1.0},
            ),
        ],
    ),
    GtSnapshot(episode_id="env-1", step=0, objects={"hammer": POSE}, table_top_z=0.7),
    ObserveRequest(cameras=["overhead"]),
    Permit(episode_id="ep-1", tool="observe", args_hash="f" * 64, call_id=None),
    SessionRegistration(
        session_id="ep-1",
        permit_url="http://127.0.0.1:18200/permits",
        permit_token="permit-secret",
        expires_at=1.5e9,
        session_key_hash="d" * 64,
        tools={"e" * 64: "observe"},
    ),
]


@pytest.mark.parametrize("model", SAMPLES, ids=lambda m: type(m).__name__)
def test_models_round_trip_through_json(model: BaseModel) -> None:
    wire = model.model_dump_json(context=REVEAL)
    assert type(model).model_validate_json(wire) == model
    json.loads(model.model_dump_json())  # the default (masked) dump is JSON too


REQUESTS: dict[type[BaseModel], dict[str, Any]] = {
    EpisodeSpec: {"task": "lift_red", "seed": 1},
    ObserveRequest: {},
    ActionStep: {"qpos": [0.0]},
    ActionBatch: {"batch_id": "b", "steps": [{"qpos": [0.0]}]},
    ToolCall: {"call_id": "c", "name": "observe", "arguments": {}, "origin": "model"},
    EnvBinding: {"url": "http://x", "episode_id": "env-1", "token": "t"},
    CallContext: {"episode_ref": "ep-1", "stage": "s2", "tier": "D"},
    Permit: {"episode_id": "e", "tool": "t", "args_hash": "a", "call_id": None},
    SessionRegistration: {
        "session_id": "s",
        "permit_url": "u",
        "permit_token": "t",
        "expires_at": 0,
        "session_key_hash": "k",
        "tools": {},
    },
}


@pytest.mark.parametrize("request_model", REQUESTS, ids=lambda m: m.__name__)
def test_request_models_refuse_unknown_fields(request_model: type[BaseModel]) -> None:
    data = REQUESTS[request_model]
    request_model.model_validate(data)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        request_model.model_validate({**data, "surprise": 1})


def test_value_models_ignore_fields_from_newer_minor_versions() -> None:
    data = {
        "episode_id": "env-1",
        "success": True,
        "success_step": None,
        "steps_used": 3,
        "step_limit": None,
        "added_in_1_1": "x",
    }
    assert JudgeResult.model_validate(data).success


@pytest.mark.parametrize(
    "q", [(0, 0, 0, 0), (1, 1, 0, 0), (float("nan"), 0, 0, 1)], ids=["zero", "norm", "nan"]
)
def test_pose_refuses_non_unit_or_non_finite_quaternions(q: tuple[float, ...]) -> None:
    with pytest.raises(ValidationError):
        Pose(p=(0, 0, 0), q=q)


def test_only_episodes_of_no_agent_stage_trace_the_expert() -> None:
    EpisodeSpec(task="t", seed=1, trace_expert=True)  # control-plane tools
    EpisodeSpec(task="t", seed=1, stage="demo", trace_expert=True)
    for stage in ("s1", "s2", "p3"):
        with pytest.raises(ValidationError, match=f"no trace_expert in stage {stage}"):
            EpisodeSpec(task="t", seed=1, stage=stage, trace_expert=True)


def test_receipt_reasons_format_as_their_values() -> None:
    reason = ReceiptReason.plan_failed
    assert str(reason) == f"{reason}" == "plan_failed" == reason
    assert f"fail_fast:{reason}@grasp" == "fail_fast:plan_failed@grasp"
    assert {str(r) for r in HARD_FAILURES} == {
        "unreachable",
        "plan_failed",
        "empty_grasp",
        "lost_hold",
    }
    receipt = ActionReceipt(reason="empty_grasp", arm="right", steps_used=4)
    assert receipt.hard_failure
    assert not ActionReceipt(reason="stopped_short").hard_failure
    assert json.loads(receipt.model_dump_json())["reason"] == "empty_grasp"


@pytest.mark.parametrize("name", ["observe", "ext_acme_scan", "a" + "b" * 39])
def test_tool_names_that_fit_client_prefixes(name: str) -> None:
    descriptor = next(m for m in SAMPLES if isinstance(m, ToolDescriptor))
    ToolDescriptor.model_validate({**descriptor.model_dump(), "name": name})


@pytest.mark.parametrize("name", ["Observe", "1tool", "get-image", "a" * 41, ""])
def test_tool_names_that_do_not(name: str) -> None:
    descriptor = next(m for m in SAMPLES if isinstance(m, ToolDescriptor))
    with pytest.raises(ValidationError):
        ToolDescriptor.model_validate({**descriptor.model_dump(), "name": name})


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (EnvBinding(url="http://x", episode_id="env-1", token="s3cret"), "token"),
        (
            EpisodeOpened(
                episode_id="env-1",
                agent_url="http://x",
                agent_token="s3cret",
                instruction=None,
                step_limit=None,
                embodiment="mock_bimanual",
                cameras=[],
                solvable=False,
            ),
            "agent_token",
        ),
        (
            SessionRegistration(
                session_id="ep-1",
                permit_url="http://x",
                permit_token="s3cret",
                expires_at=0,
                session_key_hash="k",
                tools={},
            ),
            "permit_token",
        ),
    ],
    ids=["EnvBinding", "EpisodeOpened", "SessionRegistration"],
)
def test_wire_secrets_are_masked_in_json_unless_revealed(model: BaseModel, field: str) -> None:
    assert json.loads(model.model_dump_json())[field] == MASK
    assert model.model_dump(mode="json")[field] == MASK
    assert "s3cret" not in repr(model) and "s3cret" not in str(model)
    wire = model.model_dump_json(context=REVEAL)
    assert json.loads(wire)[field] == "s3cret"
    assert getattr(type(model).model_validate_json(wire), field).get_secret_value() == "s3cret"
    # Python-mode dumps keep the SecretStr, so model -> dict -> model keeps the real token.
    dumped = model.model_dump()
    assert isinstance(dumped[field], SecretStr)
    assert getattr(type(model).model_validate(dumped), field).get_secret_value() == "s3cret"


PROTOCOLS: list[Any] = [EnvPublic, EnvControl, EnvAgent, GatewayAdmin, PermitIntake, HubControl]


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda p: p.__name__)
def test_rpc_interfaces_are_async_and_fully_typed(protocol: Any) -> None:
    methods = {
        name: member
        for name, member in inspect.getmembers(protocol, inspect.isfunction)
        if not name.startswith("_")
    }
    assert methods
    for name, method in methods.items():
        assert inspect.iscoroutinefunction(method), name
        hints = typing.get_type_hints(method, include_extras=True)  # as the RPC layer reads them
        params = [p for p in inspect.signature(method).parameters if p != "self"]
        assert set(hints) == {*params, "return"}, name
