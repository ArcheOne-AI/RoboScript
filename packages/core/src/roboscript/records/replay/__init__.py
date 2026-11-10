"""Record and replay a run's external edges, and compare two runs
(docs/concepts/operations.md): the tool that proves a change left a run's behavior the
same, bit for bit. Off unless `RS_REPLAY` (or `use`) switches it on.

- `canonical`: the canonical form of a request and its key (what is stripped, listed there);
- `tape`: `Tape`, the process's tape (`active`, `use`), `ReplayMiss`;
- `http`: `TapeTransport`, the model edge under the gateway's OpenAI client;
- `compare`: `compare(baseline, replay, client_phases=[...])` of two runs' comparison streams
  (`client_phases`: the method's phases whose episodes an outside evaluation client ends, e.g.
  wicl's `["p3"]`).

The RPC edge (motion, vision; the env's planes for comparison only) is `roboscript.rpc.taped`.
"""

from roboscript.records.replay.canonical import STRIPPED_KEYS, canonicalize, request_key
from roboscript.records.replay.compare import Comparison, Divergence, compare
from roboscript.records.replay.http import TapeTransport, replay_client, tape_client
from roboscript.records.replay.tape import (
    OBSERVED_PLANES,
    REPLAYED,
    ReplayMiss,
    Tape,
    TapeConfig,
    active,
    use,
)

__all__ = [
    "OBSERVED_PLANES",
    "REPLAYED",
    "STRIPPED_KEYS",
    "Comparison",
    "Divergence",
    "ReplayMiss",
    "Tape",
    "TapeConfig",
    "TapeTransport",
    "active",
    "canonicalize",
    "compare",
    "replay_client",
    "request_key",
    "tape_client",
    "use",
]
