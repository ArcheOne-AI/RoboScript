"""The sandbox for processes that run a builder Agent's code (`skill_builder/builder/
sandbox.py`): credentials never enter the environment, and on Linux with bubblewrap a process
run in it reads no secrets and writes only its write roots."""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from roboscript_methods.skill_builder.builder.sandbox import (
    REPO,
    wrap,
)

pytestmark = pytest.mark.unit
needs_bwrap = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bwrap") is None, reason="bubblewrap on Linux"
)
HOME_SEED = (".cache/rs-sandbox-test-seed",)


def test_credentials_never_enter_the_environment(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RS_GATEWAY_TOKEN"):
        wrap(["true"], write=[tmp_path], read=[], env={"RS_GATEWAY_TOKEN": "x"})
    with pytest.raises(FileNotFoundError):
        wrap(["true"], write=[tmp_path / "missing"], read=[], env={})


@needs_bwrap
def test_inside_it_reads_no_secrets_and_writes_only_its_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, data = tmp_path / "out", tmp_path / "data"
    out.mkdir()
    data.mkdir()
    (data / ".env.secret").write_text("SECRET", encoding="utf-8")
    (data / "input.txt").write_text("input", encoding="utf-8")
    monkeypatch.setenv("RS_LEAK", "leak")
    scratch = data / "scratch"
    cache = tmp_path / "cache"
    seeded = cache / "generated"  # under a writable root, but code: a private copy
    seeded.mkdir(parents=True)
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    script = f"""
import json, os, pathlib
def readable(p):
    try:
        return pathlib.Path(p).read_text()
    except OSError:
        return None
def writable(d):
    try:
        (pathlib.Path(d) / "probe").write_text("x")
        return True
    except OSError:
        return False
print(json.dumps({{
    "secret": readable("{data / ".env.secret"}"), "input": readable("{data / "input.txt"}"),
    "repo_env": readable("{REPO / ".env"}"), "home": sorted(os.listdir("{Path.home()}")),
    "write_out": writable("{out}"), "write_data": writable("{data}"),
    "write_repo": writable("{REPO}"), "write_tmp": writable(os.environ["TMPDIR"]),
    "env": sorted(os.environ), "scratch": os.listdir("{scratch}"),
    "write_scratch": writable("{scratch}"), "write_home": writable(os.environ["HOME"]),
    "pids": len([p for p in os.listdir("/proc") if p.isdigit()]),
    "seeded": readable("{seeded / "gen.py"}"), "write_seeded": writable("{seeded}"),
    "net": reach({port}), "settings": settings(),
    "home_seed": os.path.isdir(os.path.join(os.environ["HOME"], "{HOME_SEED[0]}")),
}}))
"""
    script = (
        "import socket\n"
        "def reach(port):\n"
        "    try:\n"
        "        socket.create_connection(('127.0.0.1', port), timeout=2).close()\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
        "def settings():\n"
        "    from roboscript.settings import EnvSettings\n"
        "    return 'ok' if EnvSettings().gateway_token is None else 'leak'\n"
    ) + script
    scratch = data / "scratch"
    scratch.mkdir()
    (scratch / "shared.json").write_text("{}", encoding="utf-8")
    (seeded / "gen.py").write_text("x = 1", encoding="utf-8")
    seed = Path.home() / HOME_SEED[0]
    seed_made = not seed.exists()
    seed.mkdir(parents=True, exist_ok=True)
    command = wrap(
        [sys.executable, "-c", script], write=[out, cache], read=[data], env={"X": "1"},
        scratch=[scratch], seeded=[seeded], home_seed=HOME_SEED,
    )  # fmt: skip
    try:
        done = subprocess.run(command, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        seen = json.loads(done.stdout)
    finally:
        if seed_made:
            seed.rmdir()
    assert seen["seeded"] == "x = 1" and seen["write_seeded"] and not (seeded / "probe").exists()
    assert not seen["net"]  # no network unless asked
    online = wrap([sys.executable, "-c", script], write=[out, cache], read=[data], env={},
                  network=True)  # fmt: skip
    assert json.loads(subprocess.run(online, capture_output=True, text=True, check=True).stdout)[
        "net"]  # fmt: skip
    listener.close()
    assert seen["secret"] in ("", None) and seen["input"] == "input"  # masked: empty or refused
    assert seen["repo_env"] in ("", None)
    assert set(seen["home"]) <= {".local", ".cache"}  # the interpreter, the seeded cache
    assert seen["home_seed"]
    # the masked .env stays unreadable; settings read the empty RS_ENV_FILE instead
    assert seen["settings"] == "ok" and "RS_ENV_FILE" in seen["env"]
    assert seen["write_out"] and seen["write_tmp"]
    assert not seen["write_data"] and not seen["write_repo"]
    assert "RS_LEAK" not in seen["env"] and "X" in seen["env"] and seen["pids"] < 5
    assert (out / "probe").exists() and not (data / "probe").exists()
    assert seen["scratch"] == [] and seen["write_scratch"] and seen["write_home"]
    assert sorted(os.listdir(scratch)) == ["shared.json"]  # nothing written there was kept
    assert os.environ["RS_LEAK"] == "leak"
