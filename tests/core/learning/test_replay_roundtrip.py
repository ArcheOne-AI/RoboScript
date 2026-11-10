"""Record a run on MockEnv, replay it on a fresh one, and compare (roboscript.records.replay):
the model (an Actor on the Responses API whose answers are random), a planner and a vision
service are answered from the tape; the env runs again for real; the comparison streams match.
A replay that sends the model another prompt misses, with the diff."""

import json
import random
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import openai
import pytest
from pydantic import BaseModel

from roboscript import rpc
from roboscript.agent import ResponsesBrain
from roboscript.agent.replay import ReplayBrain
from roboscript.agent.responses import ResponsesConfig
from roboscript.blobs import LocalBlobStore
from roboscript.contracts.geometry import Pose
from roboscript.contracts.motion import FKRequest, Planner
from roboscript.contracts.vision import SegmentPrompt, Vision
from roboscript.embodiments import MOCK_BIMANUAL
from roboscript.envs.mock import MockConfig
from roboscript.learning import Evaluate, EvaluateResult, LearningConfig, ProtocolRunner
from roboscript.memory import FileMemoryStore, VersionSource
from roboscript.motion import MockPlanner
from roboscript.records.replay import ReplayMiss, Tape, TapeConfig, compare, tape_client, use
from roboscript.testing.gateway import ToolCall, responses_payload
from roboscript.testing.perception import FakeVision
from roboscript.testing.phases import TestMethod, extend
from roboscript.testing.protocols import (
    LIFT_RED,
    RATIONALE,
    Lifter,
    ScriptedEnv,
    color_of,
    lift_memory,
)
from roboscript.tools import LocalProvider, ToolContext, Toolset, tool
from roboscript.toolsets.sensing import Sensing

RECIPE = Path(__file__).resolve().parents[3] / "recipes" / "wicl"
SEEDS = [20000, 20002]
BASE = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


class Probed(BaseModel):
    text: str


class Probe(Toolset):
    """A perception tool that asks the planner and the vision service, as the scene toolsets do:
    what it returns depends on their answers."""

    name = "probe"

    def __init__(self, planner: Planner, vision: Vision) -> None:
        self.planner = planner
        self.vision = vision

    @tool(kind="perception", tier="D")
    async def probe(self, ctx: ToolContext) -> Probed:
        """Measure the scene."""
        fk = await self.planner.fk(
            FKRequest(arm="left", base=BASE, joints=[[0.1, 0.2, 0.3, 0.0, 0.0, 0.0]])
        )
        image = await ctx.blobs.aput_array(np.zeros((8, 8, 3), dtype=np.uint8))
        found = await self.vision.segment(image, SegmentPrompt(text="box"))
        mask = await ctx.blobs.aget_array(found.masks[0].mask)
        return Probed(text=f"tcp {fk.tcp[0].p} mask {int(mask.sum())} px by {found.model}")


class NoService:
    """Stands for a planner and a vision service that a replay must never call."""

    async def fk(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a replay called the planner")

    async def segment(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a replay called the vision service")


def random_masks(rgb: Any, prompt: SegmentPrompt) -> list[Any]:
    mask = np.zeros((8, 8), dtype=bool)
    mask[: random.randint(1, 8), :3] = True  # differs per call: a replay must serve the record
    return [mask]


class Model:
    """An Actor's model: probe, then lift the instruction's box (a random height), then finish;
    every call id random. Its state is per episode (the prompt-cache key)."""

    def __init__(self) -> None:
        self.calls = 0
        self.turns: dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        step = self.turns[body["prompt_cache_key"]] = (
            self.turns.get(body["prompt_cache_key"], 0) + 1
        )
        call_id = f"call_{secrets.token_hex(4)}"
        if step == 1:
            answer = ToolCall("probe", "{}", call_id=call_id)
        elif step == 2:
            height = random.choice([0.06, 0.07, 0.08])
            args = {"rationale": RATIONALE, "color": color_of(body), "height_m": height}
            answer = ToolCall("lift", json.dumps(args), call_id=call_id)
        else:
            answer = ToolCall("finish", json.dumps({"note": "done"}), call_id=call_id)
        return httpx.Response(200, json=responses_payload(body["model"], answer))


def runner(
    run_dir: Path,
    blobs: LocalBlobStore,
    envs: Mapping[str, ScriptedEnv],
    brain: Any,
    probe: Probe,
    models: Any = None,
) -> ProtocolRunner:
    method = extend(TestMethod, {"actor": ["probe.probe"]}, after="lifter", only=["s1", "s2", "p3"])
    return ProtocolRunner(
        LearningConfig(run_dir=run_dir, method="wicl", recipe=RECIPE),
        providers=[LocalProvider(Sensing()), LocalProvider(Lifter(envs)), LocalProvider(probe)],
        blobs=blobs,
        controls=dict(envs),
        embodiment=MOCK_BIMANUAL,
        agent_env=lambda binding: next(
            env for env in envs.values() if env.episodes.current == binding.episode_id
        ),
        brains={"actor": lambda episode: brain},
        method=method,
        models=models,
        planner=MockPlanner(),
    )


async def run(
    tmp: Path, name: str, tape: Tape, live: bool, seeds: list[int] = SEEDS
) -> EvaluateResult:
    """Evaluate a frozen lift Memory on `seeds` in a fresh run directory and blob store."""
    blobs = LocalBlobStore(tmp / name / "blobs")
    config = MockConfig(task=LIFT_RED, blob_dir=tmp / name / "blobs")
    envs = {LIFT_RED: ScriptedEnv(config, blobs)}
    store = FileMemoryStore(tmp / name / "run")
    await store.commit(lift_memory(0.06), status="s1_draft", source=VersionSource(kind="s1"))
    await store.freeze(LIFT_RED, 0, reason="manual")
    brain_config = ResponsesConfig(model="m")
    if live:
        model = Model()
        vision = FakeVision(blobs, masks=random_masks)
        planner, seen = rpc.taped(Planner, MockPlanner()), rpc.taped(Vision, vision)  # type: ignore[type-abstract]

        def client(session: str) -> openai.AsyncOpenAI:
            http = tape_client(httpx.AsyncClient(transport=httpx.MockTransport(model)), tape)
            return openai.AsyncOpenAI(
                api_key=f"t:{session}", base_url="http://gw/v1", max_retries=0, http_client=http
            )

        made = runner(
            tmp / name / "run",
            blobs,
            envs,
            ResponsesBrain(brain_config),
            Probe(planner, seen),
            client,
        )  # type: ignore[arg-type]
    else:
        planner = rpc.taped(Planner, NoService())  # type: ignore[type-abstract, arg-type]
        seen = rpc.taped(Vision, NoService())  # type: ignore[type-abstract, arg-type]
        made = runner(
            tmp / name / "run", blobs, envs, ReplayBrain(brain_config), Probe(planner, seen)
        )  # type: ignore[arg-type]
    async with made:
        result = await made.run(Evaluate(), LIFT_RED, {"seeds": seeds})
    assert isinstance(result, EvaluateResult)
    return result


async def test_a_run_replays_bit_for_bit(tmp_path: Path) -> None:
    def tape(name: str, mode: str) -> Tape:
        source = None if mode == "record" else tmp_path / "tape_a"
        config = TapeConfig(
            mode=mode,  # type: ignore[arg-type]
            directory=tmp_path / f"tape_{name}",
            source=source,
            scope="lift_red/eval",
            blob_dir=tmp_path / name / "blobs",
            volatile={"run_root": str(tmp_path / name)},
        )
        return Tape(config)

    with use(tape("a", "record")) as recording:
        recorded = await run(tmp_path, "a", recording, live=True)
    assert recorded.n == len(SEEDS)
    with use(tape("b", "replay")) as replaying:
        replayed = await run(tmp_path, "b", replaying, live=False)
        summary = replaying.summary()
    assert [t.success for t in replayed.tries] == [t.success for t in recorded.tries]
    assert summary["misses"] == 0 and summary["unserved"] == {}
    found = compare(tmp_path / "tape_a", tmp_path / "tape_b", client_phases=["p3"])
    assert found.equal, found.text()
    kinds = {
        json.loads(line)["kind"]
        for path in (tmp_path / "tape_a").rglob("compare.*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    }
    assert {"episode", "tool", "env.agent.execute", "memory.commit"} <= kinds
    calls = [
        json.loads(line)["stream"]
        for path in (tmp_path / "tape_a").rglob("calls.*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert {"model:/v1/responses", "motion.fk", "vision.segment"} <= set(calls)

    # Other seeds: another scene, so another first prompt, which the tape has no answer for.
    with use(tape("c", "replay")), pytest.raises(ReplayMiss) as missed:
        await run(
            tmp_path,
            "c",
            Tape(TapeConfig(mode="record", directory=tmp_path / "x")),
            live=False,
            seeds=[20004],
        )
    assert "model:/v1/responses" in str(missed.value)
    assert "nearest recorded request" in str(missed.value)
