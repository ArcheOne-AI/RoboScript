"""Fixtures for the Hub tests: a blob store, the event log (kept in a failing test's case
directory), the Mock toolset, Hubs over it, and a maker of model calls."""

import itertools
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from roboscript.blobs import LocalBlobStore
from roboscript.contracts.tools import ToolCall
from roboscript.hub import Hub
from roboscript.records import EventWriter
from roboscript.testing.mock import Mock
from roboscript.testing.phases import dev_phases
from roboscript.tools import LocalProvider, ToolProvider

MakeHub = Callable[..., Hub]


@pytest.fixture
def blobs(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


@pytest.fixture
def events(artifacts_dir: Path) -> Iterator[EventWriter]:
    with EventWriter(artifacts_dir / "events.jsonl", "hub") as writer:
        yield writer


@pytest.fixture
def mock() -> Mock:
    return Mock()


@pytest.fixture
def make_hub(blobs: LocalBlobStore, events: EventWriter, mock: Mock) -> MakeHub:
    """A Hub over the Mock toolset (actions take `act_seconds`) and any other providers, under
    `TEST_PHASES` with every role of phase `dev` seeing all of the other providers' tools (or
    under the `phases` given)."""

    def make(*providers: ToolProvider, act_seconds: float = 0.05, **options: Any) -> Hub:
        toolset = LocalProvider(mock, {"act_seconds": act_seconds})
        options.setdefault("phases", dev_phases([provider.name for provider in providers]))
        return Hub([toolset, *providers], blobs=blobs, events=events, **options)

    return make


@pytest.fixture
async def hub(make_hub: MakeHub) -> AsyncIterator[Hub]:
    async with make_hub() as started:
        yield started


@pytest.fixture
def model() -> Callable[..., ToolCall]:
    """`model("act", rationale=...)`: a call as a model makes it, with a fresh call id."""
    ids = itertools.count(1)

    def make(tool: str, /, **arguments: Any) -> ToolCall:
        return ToolCall(call_id=f"call-{next(ids)}", name=tool, arguments=arguments, origin="model")

    return make
