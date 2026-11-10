"""Debug hooks of the protocols (docs/concepts/learning.md): subclass `Hooks` and override what
you need, e.g. to alarm when a cache value strays three trust radii from M's. The protocols await
them in order; a hook that raises stops the protocol (a resumed practice starts again at the
seed it was on)."""

from roboscript.learning.episode import EpisodeOutcome
from roboscript.memory import EditRecord, FreezeRecord, MemoryDiff, MergeRecord, VersionMeta


class Hooks:
    """No-ops; `seed` is S1's demo seed in `imitate`."""

    async def on_seed_start(self, task: str, seed: int, memory: VersionMeta) -> None:
        """A seed starts with the Memory at version `memory`."""

    async def on_try_end(self, task: str, seed: int, attempt: int, outcome: EpisodeOutcome) -> None:
        """A try (an attempt in S1) ended; voided infrastructure runs are not reported."""

    async def on_revise(
        self, task: str, seed: int, revision: int, records: list[EditRecord], diff: MemoryDiff
    ) -> None:
        """A Reviser run made `records` (rejected ones too); `diff` is what it changed."""

    async def on_merge(
        self, task: str, seed: int, record: MergeRecord, version: VersionMeta | None
    ) -> None:
        """A cache that made its seed succeed was merged into `version` (None: a jitter)."""

    async def on_skip(self, task: str, seed: int, reason: str) -> None:
        """A seed was skipped: `unsolvable` (not counted) or `discarded` (every try failed)."""

    async def on_stop(self, task: str, reason: str, frozen: FreezeRecord | None) -> None:
        """A protocol stopped: `converged`, `not converged` (the cap), `paused`, S1's `passed`
        or `attempts exhausted`; `frozen` is the freeze record of S2."""
