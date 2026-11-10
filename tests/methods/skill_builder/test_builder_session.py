"""The builder session (`skill_builder/builder`: `codex.py`, `brief.py`,
`prompts/builder.md`) with a fake Codex and no network or model: the Codex config it gets pins
the model, gateway and sandbox, case requests are served through `bin/case` and the skill
collected, and the wall clock stops Codex."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from roboscript_methods.skill_builder.builder.brief import Brief
from roboscript_methods.skill_builder.builder.clusters import Cluster, Gap
from roboscript_methods.skill_builder.builder.codex import (
    Budget,
    CaseRequest,
    Runtime,
    codex_config,
    run_session,
)

pytestmark = pytest.mark.unit

TOKEN = "test-gateway-token"
MODEL = "test-model"
GAPS = ("taskA/left place obj at base [10, 40]", "taskB/right grasp thing [1, 20]")

FAKE = """
import json, os, subprocess, sys, time
args = sys.argv[1:]
mode = args[args.index("--mode") + 1]
work = args[args.index("-C") + 1]
last = args[args.index("-o") + 1]
prompt = sys.stdin.read()
os.chdir(work)
pid = open(os.path.join(os.path.dirname(work), os.path.basename(work) + ".codex", "pid")).read()
with open(".tmp/seen.json", "w") as f:
    json.dump({"prompt": prompt, "env": dict(os.environ), "argv": args, "pid": pid,
               "pgid": os.getpgid(0), "me": os.getpid()}, f)
def emit(event):
    print(json.dumps(event), flush=True)
emit({"type": "thread.started", "thread_id": "t"})
emit({"type": "turn.started"})
if mode in ("sleep", "silent"):
    time.sleep(60)
if mode == "stall":
    emit({"type": "error", "message": "stream disconnected before completion: idle timeout"})
    emit({"type": "turn.failed", "error": {"message": "stream disconnected before completion"}})
    sys.exit(1)
if mode == "crash":
    emit({"type": "turn.failed", "error": {"message": "the skill directory is invalid"}})
    sys.exit(1)
if mode == "tokens":
    emit({"type": "turn.completed", "usage": {"input_tokens": 900, "cached_input_tokens": 100,
                                              "output_tokens": 200}})
    time.sleep(60)
os.makedirs("skill/lift_slide", exist_ok=True)
for name, text in (("skill.py", "NAME = 'lift_slide'\\n"), ("fit.py", "def fit(): pass\\n"),
                   ("cases.yaml", "gaps: []\\n")):
    with open(os.path.join("skill/lift_slide", name), "w") as f:
        f.write(text)
if mode == "link":
    os.symlink("/etc/hostname", "skill/lift_slide/data.txt")
status = json.load(open(".case/status.json"))
runs = []
asked = (json.loads(args[args.index("--cases") + 1]) if "--cases" in args else None) or [
    ["lift_slide", status["gaps"][0]], ["lift_slide", status["gaps"][1]]]
if "--submit" in args:  # all queued first, then waited for together
    ids = []
    for request in asked:
        done = subprocess.run(["bin/case", "--submit", *request], capture_output=True, text=True)
        ids.append(done.stdout.split()[1].rstrip(":"))
        with open("skill/lift_slide/skill.py", "a") as f:
            f.write("# edited after a submit\\n")
    done = subprocess.run(["bin/case", "--wait", *ids], capture_output=True, text=True)
    runs.append({"code": done.returncode, "out": done.stdout, "ids": ids})
for request in [] if "--submit" in args else asked:
    done = subprocess.run(["bin/case", *request], capture_output=True, text=True)
    runs.append({"code": done.returncode, "out": done.stdout})
    with open("skill/lift_slide/skill.py", "a") as f:
        f.write("# edited after a run\\n")
with open(".tmp/runs.json", "w") as f:
    json.dump(runs, f)
listed = subprocess.run(["bin/case", "--list"], capture_output=True, text=True)
with open(".tmp/list.txt", "w") as f:
    f.write(listed.stdout)
emit({"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "done"}})
emit({"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 400,
                                          "output_tokens": 50}})
with open(last, "w") as f:
    f.write("lift_slide: explained 1 of 2")
"""


def brief(code_root: Path) -> Brief:
    gaps = [
        Gap(id=g, task=g.split("/")[0], segment=g.split("/", 1)[1], kind="place", arm="left",
            obj="obj", target="base", frames=[10, 40], cause="entry fails", failure="reach",
            attempts=[{"form": "target", "skill": "place_object", "receipt": "place",
                       "reason": "unreachable", "note": "no plan to the target",
                       "why": "place unreachable: no plan; off_m 0.3"}])
        for g in GAPS
    ]  # fmt: skip
    return Brief.from_cluster(Cluster("place-reach", "place", "reach", gaps), []).model_copy(
        update={"code_root": code_root}
    )


def runtime(tmp_path: Path, mode: str, **knobs: Any) -> Runtime:
    fake = tmp_path / "fake_codex.py"
    fake.write_text(FAKE, encoding="utf-8")
    return Runtime(
        codex=[sys.executable, str(fake), "--mode", mode],
        model=MODEL,
        effort="high",
        gateway_url="http://127.0.0.1:9/v1",
        token=SecretStr(TOKEN),
        codex_root=tmp_path / "codex-state",
        poll_s=0.05,
        grace_s=2.0,
        **knobs,
    )


class Cases:
    def __init__(self) -> None:
        self.requests: list[CaseRequest] = []

    async def __call__(self, request: CaseRequest) -> dict[str, Any]:
        self.requests.append(request)
        snapshot = (request.skill_dir / "skill.py").read_text(encoding="utf-8")
        return {"segment": request.segment, "explained": request.n == 1, "source": snapshot}


async def test_a_session_serves_its_case_runs_and_collects_the_skill(tmp_path: Path) -> None:
    cases, work = Cases(), tmp_path / "work"
    result = await run_session(brief(tmp_path), work, Budget(), cases, runtime(tmp_path, "ok"))

    assert result.status == "done" and result.exit_code == 0
    assert result.skills == [work / "skill" / "lift_slide"]
    assert result.case_runs == 2 and [c["explained"] for c in result.cases] == [True, False]
    assert [r.gap for r in cases.requests] == list(GAPS)
    first = cases.requests[0]
    assert first.task == "taskA" and first.segment == "left place obj at base [10, 40]"
    assert first.skill == "lift_slide" and not first.skill_dir.is_relative_to(work)
    # each run saw the skill as it was when asked (the fake edits it after each run)
    assert "edited" not in (first.skill_dir / "skill.py").read_text(encoding="utf-8")
    runs = json.loads((work / ".tmp" / "runs.json").read_text(encoding="utf-8"))
    assert [r["code"] for r in runs] == [0, 0]
    assert json.loads(runs[0]["out"].split("\n", 1)[1])["row"]["explained"] is True
    listed = (work / ".tmp" / "list.txt").read_text(encoding="utf-8")
    assert listed.startswith("case runs used 2 of 12; about ") and "minutes" in listed
    assert "0 running (at most 4 at once)" in listed
    assert listed.rstrip().endswith(GAPS[1])
    assert result.tokens == 1050 and result.usage_source == "codex"
    assert result.last_message == "lift_slide: explained 1 of 2"
    assert (
        result.transcript.is_file()
        and len(result.transcript.read_text(encoding="utf-8").splitlines()) == 4
    )
    assert (work.with_name("work.codex") / "result.json").is_file()
    assert (
        len((work.with_name("work.codex") / "cases.jsonl").read_text(encoding="utf-8").splitlines())
        == 2
    )

    seen = json.loads((work / ".tmp" / "seen.json").read_text(encoding="utf-8"))
    assert "12 case runs" in seen["prompt"] and "3 hours" in seen["prompt"]
    assert "Up to 4 run at once" in seen["prompt"]
    assert "$" not in seen["prompt"]
    assert seen["argv"][-1] == "-" and "--json" in seen["argv"]
    # the gateway key is in the config file only, never in Codex's environment
    assert not any(TOKEN in v for v in seen["env"].values())
    assert Path(seen["env"]["HOME"]) != Path.home()
    home = Path(seen["env"]["CODEX_HOME"])
    assert home.parent.parent == tmp_path / "codex-state" and not home.is_relative_to(work)
    assert os.listdir(tmp_path / "codex-state") == []  # the session's state is gone, key and all
    assert (work / "BRIEF.md").read_text(encoding="utf-8").startswith("# Brief: cluster")
    assert (work / ".tmp" / "rs.env").read_text(encoding="utf-8") == ""


async def test_the_wall_clock_stops_codex(tmp_path: Path) -> None:
    rt = runtime(tmp_path, "sleep")
    result = await run_session(brief(tmp_path), tmp_path / "w", Budget(wall_s=0.5), Cases(), rt)
    assert result.status == "timeout" and result.exit_code != 0
    assert result.wall_s < 10 and result.skills == []


def test_the_codex_config_pins_model_gateway_and_sandbox(tmp_path: Path) -> None:
    tomllib = pytest.importorskip("tomllib")
    work = tmp_path / "work"
    work.mkdir()
    accepted = tmp_path / "accepted"
    (tmp_path / "repo" / "configs").mkdir(parents=True)
    (tmp_path / "repo" / ".env").write_text("RS_GATEWAY_TOKEN=x\n", encoding="utf-8")
    (tmp_path / "repo" / ".env.local").write_text("RS_GATEWAY_TOKEN=x\n", encoding="utf-8")
    (tmp_path / "keys").write_text("x\n", encoding="utf-8")
    (tmp_path / "repo" / ".env.api").symlink_to(tmp_path / "keys")  # outside the read roots
    b = brief(tmp_path / "repo").model_copy(update={"accepted": [accepted]})
    rt = runtime(tmp_path, "ok", read_roots=[tmp_path / "python"])
    config = tomllib.loads(codex_config(b, work, rt, "build-1"))

    assert config["model"] == MODEL  # the deployment's, from its config
    assert config["model_reasoning_effort"] == "high"
    plain = runtime(tmp_path, "ok").model_copy(update={"effort": None})
    assert "model_reasoning_effort" not in tomllib.loads(codex_config(b, work, plain, "build-1"))
    provider = config["model_providers"][config["model_provider"]]
    assert provider["base_url"] == rt.gateway_url and provider["wire_api"] == "responses"
    assert provider["experimental_bearer_token"] == f"{TOKEN}:build-1"
    profile = config["permissions"][config["default_permissions"]]
    assert profile["network"] == {"enabled": False}
    fs = profile["filesystem"]
    assert [p for p, mode in fs.items() if mode == "write"] == [str(work.resolve())]
    repo = (tmp_path / "repo").resolve()
    assert fs[str(repo)] == "read" and fs[str(accepted.resolve())] == "read"
    assert fs[str(repo / ".env")] == fs[str(repo / "configs")] == "none"
    assert fs[str(repo / ".env.local")] == "none"  # every .env*
    assert str(repo / ".env.api") not in fs and str(tmp_path / "keys") not in fs
    assert str(repo / ".dev") not in fs  # missing: bubblewrap could not mask either
    assert config["project_doc_max_bytes"] == 0 and config["web_search"] == "disabled"
    assert config["features"]["multi_agent"] is False
    env = config["shell_environment_policy"]["set"]
    assert "PYTHONPATH" not in env
    # the masked .env is unreadable: settings read an empty file in the workspace instead
    assert env["RS_ENV_FILE"] == str(work / ".tmp" / "rs.env")
