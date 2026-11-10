"""`wicl`'s Task Memory content on the core's document engine (docs/concepts/learning.md,
docs/how-to/add-a-method.md): the edit validators it opens every `Editor` with (`validators`): the
done floors (`floors`) and the box-faces rule (`faces.BoxFaces`); and `arms.unbind_arms`, the
deterministic rewrite of a Memory that binds a side into arms by rule, applied as S1, S2,
`evaluate` and P3 load it (`unbound` is wicl's `MethodRuntime.frozen`).
"""

from collections.abc import Mapping

from roboscript.memory.edits import EditValidator
from roboscript_methods.wicl.memory.faces import BoxFaces, Grasped
from roboscript_methods.wicl.memory.floors import DoneFloors


def validators(
    grasped: Mapping[str, Grasped] | None = None, required: Mapping[str, str] | None = None
) -> list[EditValidator]:
    """The validators of every editor `wicl` opens; `grasped` and `required` as `BoxFaces`
    takes them (none: nothing seen yet, nothing required)."""
    return [DoneFloors(), BoxFaces(dict(grasped or {}), dict(required or {}))]


__all__ = ["DoneFloors", "validators"]
