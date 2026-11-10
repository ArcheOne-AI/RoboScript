"""Conformance checks for dispatch transports (`roboscript.orchestrate.dispatch.transport`).

A transport conforms when it runs a command in the directory asked with the input given and
returns its exit status and output (a failing command is a result, not an error), copies exactly
the listed files both ways (creating directories, keeping bytes), and raises `TransportError`
when it times out. Use them from pytest by subclassing `TransportConformance` as a `Test*` class
with a `transport_target` fixture, or without pytest via `roboscript.testing.precheck`.
"""

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from roboscript.orchestrate.dispatch.transport import Transport, TransportError
from roboscript.testing import unit


@dataclass(frozen=True)
class TransportTarget:
    """`remote`: an empty directory on the transport's host (its path there); `local`: an empty
    local directory; `python`: an interpreter on the host."""

    transport: Transport
    remote: str
    local: Path
    python: str = sys.executable


class TransportConformance:
    """The checks, as `test_*` methods taking a `TransportTarget`."""

    @unit
    async def test_run(self, transport_target: TransportTarget) -> None:
        t = transport_target
        code = "import os, sys; print(os.getcwd()); print(sys.stdin.read().upper())"
        found = await t.transport.run(
            [t.python, "-c", code], cwd=t.remote, input=b"hi", timeout_s=60
        )
        assert found.returncode == 0, found.stderr
        cwd, echoed = found.stdout.decode().splitlines()
        assert Path(cwd).name == Path(t.remote).name  # the remote host resolves its own links
        assert echoed == "HI"
        failed = await t.transport.run(
            [t.python, "-c", "raise SystemExit(3)"], cwd=t.remote, timeout_s=60
        )
        assert failed.returncode == 3

    @unit
    async def test_timeout(self, transport_target: TransportTarget) -> None:
        t = transport_target
        try:
            await t.transport.run(
                [t.python, "-c", "import time; time.sleep(30)"], cwd=t.remote, timeout_s=1
            )
        except TransportError:
            return
        raise AssertionError("a command past its timeout must raise TransportError")

    @unit
    async def test_copy(self, transport_target: TransportTarget) -> None:
        t = transport_target
        source = t.local / "out"
        (source / "a" / "b").mkdir(parents=True)
        payload = bytes(range(256)) * 300
        (source / "a" / "b" / "data.bin").write_bytes(payload)
        (source / "note.txt").write_text("note", encoding="utf-8")
        (source / "unlisted.txt").write_text("stays", encoding="utf-8")
        await t.transport.push(
            source, f"{t.remote}/tree", ["a/b/data.bin", "note.txt"], timeout_s=60
        )
        back = t.local / "back"
        await t.transport.pull(f"{t.remote}/tree", back, ["a/b/data.bin", "note.txt"], timeout_s=60)
        digest = hashlib.sha256((back / "a" / "b" / "data.bin").read_bytes()).hexdigest()
        assert digest == hashlib.sha256(payload).hexdigest()
        assert (back / "note.txt").read_text(encoding="utf-8") == "note"
        assert not (back / "unlisted.txt").exists()
        await t.transport.pull(f"{t.remote}/tree", t.local / "none", [], timeout_s=60)
