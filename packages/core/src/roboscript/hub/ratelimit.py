"""`rate_limit`, an example guard (off by default; `roboscript.guards`): at most `max_calls`
model calls per episode in any `window_s` seconds, rejected with `rate_limited` beyond that. It
runs at the end of the `pre` phase, so a rejected call takes no action lock and no turn."""

import time
from collections import deque
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from roboscript.hub.pipeline import Call, Guard, Reject, Step


class RateLimitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_calls: PositiveInt = 60
    window_s: PositiveFloat = 60.0


def rate_limit(config: RateLimitConfig) -> Step:
    """A fresh step with its own counts per episode (one Hub's)."""
    seen: dict[str, deque[float]] = {}

    async def step(call: Call) -> None:
        now = time.monotonic()
        times = seen.setdefault(call.episode_id, deque())
        while times and now - times[0] >= config.window_s:
            times.popleft()
        if len(times) >= config.max_calls:
            detail = f"More than {config.max_calls} calls in {config.window_s:g} s; wait a moment."
            raise Reject("rate_limited", detail)
        times.append(now)

    step.__name__ = "rate_limit"
    return step


def _make(config: Any) -> Step:
    return rate_limit(config)


RATE_LIMIT: Final = Guard(name="rate_limit", phase="pre", make=_make, Config=RateLimitConfig)
