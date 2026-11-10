"""The one canonical JSON text, for values that different processes must hash alike."""

import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys, no whitespace and non-ASCII kept; NaN and infinities are refused."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
