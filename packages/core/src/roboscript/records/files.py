"""Files replaced whole: what services write for others to read (ready files, endpoints, state,
Memory versions)."""

import os
import threading
from pathlib import Path


def write_atomic(path: Path, data: bytes) -> None:
    """Replace `path` with `data` at once: readers see the old file or the new, never a part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
