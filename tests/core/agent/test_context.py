"""The context budget: a Demo-Reader-like episode (one long turn of parallel image calls) never
sends a request estimated over its budget; older tool images become stubs that name the call that
shows them again, then the oldest calls of the turn fold into one summary message, the brief and
the newest exchange untouched; each shrink is a `model.context` event, and the log still rebuilds
every request as it was sent. The estimate itself: images at their billed cost for their shape
and detail."""

from typing import Any

import pytest

from roboscript.agent import (
    ContextBudget,
    HistoryPolicy,
    LoopConfig,
    ScriptedBrain,
    call,
    read_lines,
    rebuild_requests,
    think,
)
from roboscript.agent.context import estimate
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.tools import TextPart
from roboscript.records import read_events

CAMERAS = ["overhead", "wrist_left", "wrist_right"]
ACT = {"rationale": {"stage": "grasp", "scene": "a red box", "progress": "starting", "plan": "act"}}


def reading(calls: int) -> list[Any]:
    """`calls` replies of an observe and three image calls each (six images), then finish."""
    views = [call("get_image", camera=camera, frame="frame_1") for camera in CAMERAS]
    reply = [think("compare the views"), call("observe"), *views]
    return [reply] * calls + [[call("finish", note="read")]]


def sent(session: Any) -> list[tuple[int, dict[str, Any], dict[str, BlobRef]]]:
    """Each logged request (images as blob URLs) with its call number and blobs."""
    lines = read_lines(session.recorder.calls)
    contexts = {line["sha256"]: line for line in lines if line["kind"] == "context"}
    return [
        (
            line["call"],
            {
                **line["request"],
                "instructions": contexts[line["context"]]["instructions"],
                "tools": contexts[line["context"]]["tools"],
            },
            {blob: BlobRef.model_validate(ref) for blob, ref in line["blobs"].items()},
        )
        for line in lines
        if line["kind"] == "request"
    ]


def shrinks(rig: Any, session: Any) -> list[dict[str, Any]]:
    return [
        event.data
        for event in read_events(rig.events.path)
        if event.episode_id == session.episode_id and event.kind == "model.context"
    ]


def outputs(request: dict[str, Any]) -> list[Any]:
    return [
        item["output"] for item in request["input"] if item.get("type") == "function_call_output"
    ]


BUDGET = ContextBudget(windows={"scripted": 30_000})


async def test_a_long_reading_turn_stays_within_its_budget(rig: Any) -> None:
    brief = [TextPart(text="The demonstration: " + "a long description. " * 40)]
    config = LoopConfig(
        observe=False, history=HistoryPolicy(image_exchanges=40, tool_images=12), context=BUDGET
    )
    session = await rig.session(toolsets=["sensing"], brief=brief)
    requests: list[dict[str, Any]] = []

    async def reply(request: dict[str, Any]) -> Any:
        requests.append(request)
        return reading(40)[len(requests) - 1]

    result = await ScriptedBrain(reply, config).run(session)
    assert (result.end.reason, result.turns, result.model_calls) == ("finish", 1, 41)

    budget = BUDGET.budget("scripted", overflowed=False)
    sizes = [estimate(request, refs, BUDGET) for _, request, refs in sent(session)]
    assert max(sizes) <= budget
    assert sizes[-1] > budget * BUDGET.shrink_to * 0.8  # it levels off, it does not collapse
    events = shrinks(rig, session)
    assert events and all(e["before"] > budget >= e["after"] and not e["over"] for e in events)
    assert all(e["reason"] == "budget" for e in events)
    assert events[0]["to"]["tool_images"] is not None  # images go first
    assert events[-1]["to"]["fold_exchanges"] > 0  # then the oldest calls of the turn
    folds = [e["to"]["fold_exchanges"] for e in events]
    assert folds == sorted(folds)  # only one way

    last = requests[-1]["input"]
    assert last[0]["content"][0]["text"].startswith("The demonstration: ")  # the brief stays
    folded = last[1]["content"][0]["text"]
    assert folded.startswith("Earlier calls of this turn in brief")
    assert "call 1: observe({}) -> " in folded and "; get_image({" in folded
    newest = outputs(requests[-1])[-4:]  # the newest exchange's four outputs, images and all
    assert all(isinstance(output, list) for output in newest)
    stubs = [o for o in outputs(requests[-1]) if isinstance(o, str)]
    assert stubs and stubs[0].endswith("shows them again]")
    assert "images left out to save context (" in stubs[0]

    assert await rebuild_requests(session.recorder.calls, rig.blobs) == requests


@pytest.mark.parametrize(
    ("shape", "detail", "tokens"),
    [((480, 640, 3), "auto", 300), ((480, 640, 3), "low", 85), (None, "high", 1536)],
)
def test_an_image_costs_its_patches(shape: Any, detail: str, tokens: int) -> None:
    ref = BlobRef(id="a" * 64, size=1, media_type="image/png", shape=shape)
    image = {"type": "input_image", "image_url": "rs-blob:" + ref.id, "detail": detail}
    text = len('{"input":[{"type":"input_image"}]}') // 4 + 1  # what is left of it, as text
    assert estimate({"input": [image]}, {ref.id: ref}, ContextBudget()) == text + tokens
