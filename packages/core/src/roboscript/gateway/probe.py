"""Endpoint probe: `python -m roboscript.gateway.probe --model M [--reps N] --out DIR`.

Sends the same small vision questions to each deployment of a model, one deployment at a time
through the admin-only `x-rs-deployment` header of the running gateway (`gateway_url()`, with
`RS_GATEWAY_TOKEN` and `RS_GATEWAY_ADMIN_TOKEN` from `.env`), and measures per endpoint: the
input tokens billed for one image in a user message and for the same image inside a tool's output
(`function_call_output`), which shows whether the endpoint bills a tool-output image as its base64
text (`tool_output_images: false` in the gateway config); latency; errors; and whether the answers
are right. The images are synthetic (colored shapes on a noisy background, counts known), so the
probe needs no data. Placement `views` puts five labeled views in one tool output the way the agent
builds them (`agent.history`, `detail: auto`); it runs once, not per rep. Calls per endpoint:
3 images x (`user`, `tool`) x `--reps`, plus 1 for `views`.

`--in-process` builds the gateway app inside this process instead (no port; its request log and
state under `--out`), with no gateway running; it reads `.env.api` as the
gateway does. Writes `calls.jsonl` and `table.md` to `--out`; never a key or token.
"""

import argparse
import base64
import io
import json
import re
import secrets
import statistics
import time
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anyio
import httpx
import numpy as np
from PIL import Image, ImageDraw
from pydantic import SecretStr

from roboscript.gateway.app import Gateway
from roboscript.gateway.client import gateway_url
from roboscript.gateway.config import env_api_file, load, read_upstreams
from roboscript.gateway.proxy import upstream_client
from roboscript.settings import EnvSettings

INSTRUCTIONS = "Answer with one integer and nothing else."
TOOL = {
    "type": "function",
    "name": "look",
    "description": "Returns the current camera image.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True,
}
COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60)}


@dataclass(frozen=True)
class Scene:
    """A synthetic image and a counting question with its answer."""

    question: str
    answer: int
    jpeg: bytes


@dataclass
class Result:
    endpoint: str
    placement: str  # "user" (image in a user message), "tool" (in a function_call_output), "views"
    scene: int
    rep: int
    status: int
    seconds: float
    input_tokens: int | None
    answer: str | None
    correct: bool
    error: str | None


def scenes(seed: int = 0) -> list[Scene]:
    """Three 640x480 JPEG frames (about the size of a camera view) of colored circles and
    squares on a noisy gray background; each asks for the count of one color and shape."""
    rng = np.random.default_rng(seed)
    specs = [("red", "circle", 3), ("blue", "square", 5), ("green", "circle", 4)]
    out = []
    for color, shape, count in specs:
        noise = rng.normal(150, 18, (480, 640, 3)).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(noise)
        draw = ImageDraw.Draw(image)
        shapes = [(color, shape)] * count
        shapes += [(c, s) for c in COLORS for s in ("circle", "square") if (c, s) != (color, shape)]
        cells = rng.permutation(48)[: len(shapes)]  # an 8 x 6 grid of 80 px cells: no overlap
        for (c, s), cell in zip(shapes, cells, strict=True):
            x, y = 80 * (cell % 8) + 15, 80 * (cell // 8) + 15
            box = (x, y, x + 50, y + 50)
            (draw.ellipse if s == "circle" else draw.rectangle)(box, fill=COLORS[c])
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        question = f"How many {color} {shape}s are in the image?"
        out.append(Scene(question, count, buffer.getvalue()))
    return out


def request(
    model: str, scene: Scene, placement: str, others: Sequence[Scene] = ()
) -> dict[str, Any]:
    """The Responses body asking `scene`'s question with its image in a user message or as the
    output of a `look` call the model made; for `views`, that output holds `scene` and `others`
    as labeled views."""
    image = _image(scene, "high")
    if placement == "views":
        views = [scene, *others]
        parts = [{"type": "input_text", "text": f"{len(views)} camera views:"}]
        for index, view in enumerate(views):
            parts += [{"type": "input_text", "text": f"[view{index}]"}, _image(view, "auto")]
        question = f"Call look, then answer about view0: {scene.question}"
        items: list[dict[str, Any]] = [
            {"role": "user", "content": question},
            {"type": "function_call", "call_id": "call_look", "name": "look", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_look", "output": parts},
        ]
    elif placement == "user":
        content = [{"type": "input_text", "text": scene.question}, image]
        items = [{"role": "user", "content": content}]
    else:
        items = [
            {"role": "user", "content": f"Call look, then answer: {scene.question}"},
            {"type": "function_call", "call_id": "call_look", "name": "look", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "call_look",
                "output": [{"type": "input_text", "text": "the camera image:"}, image],
            },
        ]
    return {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": items,
        "tools": [TOOL],
        "reasoning": {"effort": "low"},
        "store": False,
    }


async def ask(
    http: httpx.AsyncClient,
    headers: dict[str, str],
    endpoint: str,
    body: dict[str, Any],
    scene: Scene,
    placement: str,
    index: int,
    rep: int,
) -> Result:
    started = time.monotonic()
    error, tokens, text = None, None, None
    try:
        response = await http.post(
            "/responses", json=body, headers={**headers, "x-rs-deployment": endpoint}
        )
        status = response.status_code
    except httpx.HTTPError as failure:
        status, error = 0, type(failure).__name__
    seconds = round(time.monotonic() - started, 2)
    if status == 200:
        payload = response.json()
        tokens = (payload.get("usage") or {}).get("input_tokens")
        text = _output_text(payload)
    elif status:
        error = _error(response)
    number = re.search(r"-?\d+", text or "")
    correct = number is not None and int(number.group()) == scene.answer
    return Result(endpoint, placement, index, rep, status, seconds, tokens, text, correct, error)


async def probe(
    http: httpx.AsyncClient,
    headers: dict[str, str],
    model: str,
    endpoints: Sequence[str],
    reps: int,
) -> list[Result]:
    """Every endpoint in parallel, each one's calls one after another."""
    frames = scenes()
    others = [*frames[1:], *scenes(seed=1)[:2]]
    results: list[Result] = []

    async def one(endpoint: str) -> None:
        body = request(model, frames[0], "views", others)
        results.append(await ask(http, headers, endpoint, body, frames[0], "views", 0, 0))
        for rep in range(reps):
            for index, scene in enumerate(frames):
                for placement in ("user", "tool"):
                    body = request(model, scene, placement)
                    args = (endpoint, body, scene, placement, index, rep)
                    results.append(await ask(http, headers, *args))

    async with anyio.create_task_group() as group:
        for endpoint in endpoints:
            group.start_soon(one, endpoint)
    return sorted(results, key=lambda r: (r.endpoint, r.rep, r.scene, r.placement))


def table(results: list[Result]) -> str:
    """Per endpoint: median billed input tokens per placement, their ratio, median latency,
    errors and right answers."""
    lines = [
        "| endpoint | user-image input tokens | tool-image input tokens | tool/user | "
        "5-view tool input tokens | median latency s | errors | right (user) | right (tool) | "
        "right (views) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for endpoint in sorted({r.endpoint for r in results}):
        mine = [r for r in results if r.endpoint == endpoint]

        def tokens(placement: str, mine: list[Result] = mine) -> float | None:
            values = [r.input_tokens for r in mine if r.placement == placement and r.input_tokens]
            return statistics.median(values) if values else None

        def right(placement: str, mine: list[Result] = mine) -> str:
            ours = [r for r in mine if r.placement == placement]
            return f"{sum(r.correct for r in ours)}/{len(ours)}"

        user, tool = tokens("user"), tokens("tool")
        ratio = f"{tool / user:.1f}" if user and tool else "-"
        ok = [r.seconds for r in mine if r.status == 200]
        latency = f"{statistics.median(ok):.1f}" if ok else "-"
        errors = [f"{r.placement}:{r.status}:{r.error}" for r in mine if r.status != 200]
        shown = f"{len(errors)}/{len(mine)}" + (
            f" ({'; '.join(sorted(set(errors)))})" if errors else ""
        )
        cells = [
            endpoint,
            _fmt(user),
            _fmt(tool),
            ratio,
            _fmt(tokens("views")),
            latency,
            shown,
            right("user"),
            right("tool"),
            right("views"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m roboscript.gateway.probe", description=__doc__)
    parser.add_argument("--model", required=True, help="a model or alias of the gateway config")
    parser.add_argument("--endpoints", nargs="*", help="default: every deployment of the model")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--config", default="gateway", help="the gateway's config (YAML or name)")
    parser.add_argument("--in-process", action="store_true", help="build the gateway app here")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = anyio.run(_run, args)
    with (args.out / "calls.jsonl").open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result)) + "\n")
    text = table(results)
    (args.out / "table.md").write_text(text, encoding="utf-8")
    print(text, end="")


async def _run(args: argparse.Namespace) -> list[Result]:
    config = load(args.config)
    if args.model not in config.models:
        raise SystemExit(f"{args.model} is not a model of {args.config}")
    endpoints = args.endpoints or [d.endpoint for d in config.models[args.model].deployments]
    async with AsyncExitStack() as stack:
        if args.in_process:
            config = config.model_copy(
                update={
                    "log_file": args.out / "gateway.jsonl",
                    "state_file": args.out / "gateway-state.json",
                }
            )
            token, admin = SecretStr(secrets.token_hex(16)), SecretStr(secrets.token_hex(16))
            upstream = await stack.enter_async_context(upstream_client())
            gateway = Gateway(
                config,
                read_upstreams(config, env_api_file()),
                token=token,
                admin_token=admin,
                http=upstream,
            )
            stack.push_async_callback(gateway.aclose)
            transport: httpx.AsyncBaseTransport = httpx.ASGITransport(gateway.app)
            base = "http://gateway/v1"
        else:
            settings = EnvSettings()
            if settings.gateway_token is None or settings.gateway_admin_token is None:
                raise SystemExit("set RS_GATEWAY_TOKEN and RS_GATEWAY_ADMIN_TOKEN in .env")
            token, admin = settings.gateway_token, settings.gateway_admin_token
            transport, base = httpx.AsyncHTTPTransport(), gateway_url()
        client = httpx.AsyncClient(transport=transport, base_url=base, timeout=300, trust_env=False)
        http = await stack.enter_async_context(client)
        headers = {
            "authorization": f"Bearer {token.get_secret_value()}",
            "x-rs-admin-token": admin.get_secret_value(),
        }
        return await probe(http, headers, args.model, endpoints, args.reps)


def _image(scene: Scene, detail: str) -> dict[str, Any]:
    url = "data:image/jpeg;base64," + base64.b64encode(scene.jpeg).decode()
    return {"type": "input_image", "image_url": url, "detail": detail}


def _output_text(payload: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in payload.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


def _error(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return response.text[:120]
    code = error.get("code") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else str(error)
    return f"{code or '-'}: {str(message)[:120]}"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


if __name__ == "__main__":
    main()
