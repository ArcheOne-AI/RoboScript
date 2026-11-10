"""Load statistics for `GET /admin/stats` (docs/concepts/agent-runtime.md): per deployment the calls
it answered and its errors by kind since start, and call latency and queue-wait percentiles over
the last `window_s` seconds, and the attempts' failure rate over the same window, which health
reads; per model its overflows, moves and queue timeouts."""

import time
from collections import Counter, defaultdict, deque
from typing import Any

_Key = tuple[str, str]  # (model, endpoint)
_Sample = tuple[float, ...]  # (time.monotonic(), values...)


class Stats:
    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._served: Counter[_Key] = Counter()
        self._errors: defaultdict[_Key, Counter[str]] = defaultdict(Counter)
        self._calls: defaultdict[_Key, deque[_Sample]] = defaultdict(deque)
        self._waits: defaultdict[_Key, deque[_Sample]] = defaultdict(deque)
        self._events: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._outcomes: defaultdict[_Key, deque[_Sample]] = defaultdict(deque)  # 1.0: failed

    def served(self, model: str, endpoint: str, total_s: float, first_byte_s: float) -> None:
        """A call the deployment answered; `total_s` until its answer was through."""
        self._served[(model, endpoint)] += 1
        self._push(self._calls[(model, endpoint)], total_s, first_byte_s)

    def error(self, model: str, endpoint: str, kind: str) -> None:
        """A failed attempt (its outcome: an upstream status, `timeout`, ...) or a broken answer."""
        self._errors[(model, endpoint)][kind] += 1

    def waited(self, model: str, endpoint: str, seconds: float) -> None:
        """A request that waited `seconds` for its slot on the deployment."""
        self._push(self._waits[(model, endpoint)], seconds)

    def count(self, model: str, event: str) -> None:
        """`overflow` (an attempt sent to a lower tier because the better one was full),
        `move_<reason>` (a session re-pinned elsewhere) or `queue_timeout`."""
        self._events[model][event] += 1

    def attempted(self, model: str, endpoint: str, *, failed: bool) -> None:
        """An attempt's outcome: answered, or failed before the client got anything."""
        self._push(self._outcomes[(model, endpoint)], float(failed))

    def failure_rate(self, model: str, endpoint: str) -> tuple[int, float]:
        """The attempts in the window and the share of them that failed (0 without any)."""
        outcomes = self._recent(self._outcomes[(model, endpoint)])
        return len(outcomes), (sum(s[1] for s in outcomes) / len(outcomes) if outcomes else 0.0)

    def forget_outcomes(self, model: str, endpoint: str) -> None:
        """Start the failure rate afresh (a deployment back in rotation)."""
        self._outcomes[(model, endpoint)].clear()

    def median_s(self, model: str, endpoint: str) -> float | None:
        """The deployment's median call latency over the window; None without calls."""
        return _pct([s[1] for s in self._recent(self._calls[(model, endpoint)])], 50)

    def latency_quantile(self, model: str, q: float) -> float | None:
        """The q-th percentile of `model`'s answered calls over the window, all deployments
        together; None without any."""
        recent = [self._recent(calls) for key, calls in self._calls.items() if key[0] == model]
        values = [sample[1] for calls in recent for sample in calls]
        return _pct(values, q)

    def deployment(self, model: str, endpoint: str) -> dict[str, Any]:
        key = (model, endpoint)
        calls, waits = self._recent(self._calls[key]), self._recent(self._waits[key])
        calls_n, rate = self.failure_rate(model, endpoint)
        return {
            "served": self._served[key],
            "attempts": calls_n,
            "failure_rate": round(rate, 3),
            "errors": dict(self._errors[key]),
            "latency_s": _summary([s[1] for s in calls]),
            "first_byte_s": _summary([s[2] for s in calls]),
            "queue_wait_s": _summary([s[1] for s in waits]),
        }

    def model(self, model: str) -> dict[str, Any]:
        return {"events": dict(self._events[model])}

    def _push(self, samples: deque[_Sample], *values: float) -> None:
        samples.append((time.monotonic(), *values))
        self._recent(samples)

    def _recent(self, samples: deque[_Sample]) -> deque[_Sample]:
        """`samples` without those older than the window."""
        oldest = time.monotonic() - self.window_s
        while samples and samples[0][0] < oldest:
            samples.popleft()
        return samples


def _summary(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": _pct(values, 50), "p90": _pct(values, 90)}


def _pct(values: list[float], q: float) -> float | None:
    """The q-th percentile, interpolated between neighbours; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * q / 100
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (at - low), 3)
