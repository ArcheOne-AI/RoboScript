"""Fixtures of the integration tests: a blob store, and the Hub's event log (kept in a failing
test's case directory). The MCP tests need the `mcp` extra, which the simulator env lacks: there
they are not collected."""

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.records import EventWriter

collect_ignore_glob = [] if importlib.util.find_spec("mcp") else ["test_*.py"]


@pytest.fixture
def blobs(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
def events(artifacts_dir: Path) -> Iterator[EventWriter]:
    with EventWriter(artifacts_dir / "events.jsonl", "hub") as writer:
        yield writer
