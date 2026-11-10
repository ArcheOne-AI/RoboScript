"""Done criteria are read as perception can resolve them (docs/concepts/perception.md,
docs/concepts/learning.md): the edit validator `wicl` registers with every `Editor` it opens
(`DoneFloors`), refusing an edit that asks a done check for a band narrower than perception's error,
and, under `EditRules.strict_done`, one that loosens a done criterion.
"""

from collections.abc import Collection

from roboscript.memory.calls import parse_calls
from roboscript.memory.doc import MemoryDoc
from roboscript.memory.edits import (
    EPSILON,
    AddStage,
    DefineParam,
    EditOp,
    Editor,
    EditValidator,
    RemoveParam,
    SetCategory,
    SetDone,
    SetNumber,
    show,
)
from roboscript.memory.params import DoneSpecDef, DoneSpecParam, NumberParam
from roboscript.perception.criteria import FLOORS, TOL_FLOOR_DEG, UNITS, floored


class DoneFloors(EditValidator):
    """`below_floor` on every edit, then `loosens_done` where the rules are `strict_done`."""

    def rules(self, editor: Editor, op: EditOp) -> str | None:
        if why := below_floor(editor.doc, op, editor.lint.carried_ends):
            return why
        if editor.rules.strict_done:
            return loosens_done(editor.base, editor.doc, op)
        return None


def below_floor(doc: MemoryDoc, op: EditOp, held_ends: Collection[str]) -> str | None:
    """Why `op` would ask a done check for more than perception can tell, or None: a done
    criterion's tolerance below perception's error (`perception.criteria.FLOORS`), or a stage's done
    line with a `yaw_aligned` tolerance below `TOL_FLOOR_DEG` or an absolute yaw (`ref_deg`, a
    number for ref) on a role the demonstration ends holding where its arm carried it
    (`held_ends`, `LintConfig.carried_ends`). The check reads such a band at the floor anyway
    (a stored one too), so the edit would change nothing it measures; and the demonstration's
    end gives no yaw for an object it ends holding (only one set down ends at a yaw), so a world
    yaw written for one is a guess the Actor would turn the held object to."""
    spec = None
    if isinstance(op, DefineParam) and isinstance(op.param, DoneSpecDef):
        spec, tol = op.param.value, op.param.value.tol
    elif isinstance(op, SetNumber) and op.field == "tol" and not isinstance(op.value, tuple):
        param = doc.params.get(op.param)
        if isinstance(param, DoneSpecParam):
            spec, tol = param.value, float(op.value)
    if spec is not None and (floor := FLOORS.get(spec.metric)) is not None and tol < floor:
        unit = UNITS[spec.metric][0]
        return (
            f"a {spec.metric} tolerance of {tol:g} {unit} is below perception's error "
            f"({floor:g} {unit}): the check cannot tell such a band from noise and reads it at "
            f"{floor:g} anyway; keep the tolerance at {floor:g} or above"
        )
    if isinstance(op, SetDone):
        text = op.done
    elif isinstance(op, AddStage):
        text = op.stage.done
    else:
        return None
    for call in parse_calls(text):
        if call.name != "yaw_aligned":
            continue
        role = call.keywords.get("obj", call.values[0] if call.values else "").strip()
        given = call.keywords.get("tol", call.values[2] if len(call.values) > 2 else "")
        band = _number(doc, given)
        if band is not None and band < TOL_FLOOR_DEG:
            return (
                f"yaw_aligned's tol {band:g} deg is below perception's error on a yaw "
                f"({TOL_FLOOR_DEG:g} deg): the check reads it at {TOL_FLOOR_DEG:g} anyway; keep it "
                f"at {TOL_FLOOR_DEG:g} or above"
            )
        ref = call.keywords.get("ref", call.values[1] if len(call.values) > 1 else "")
        absolute = "ref_deg" in call.keywords or _number(doc, ref) is not None
        if absolute and role in held_ends:
            return (
                f"yaw_aligned with an absolute yaw on {role}, which the demonstration ends holding "
                "where its arm carried it: the demonstration's end gives no yaw for an object it "
                "ends holding, so a world yaw written for it is a guess, and the check leaves it "
                f"undecided; where {role} ends is check_done's (its lift, its place relative to "
                "the robot's base). Compare its yaw with another object's (ref=<role>) if that "
                "matters, or leave yaw out"
            )
    return None


def _number(doc: MemoryDoc, text: str) -> float | None:
    """A number written in a done line (a literal or a `{number}` parameter), else None."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        param = doc.params.get(text[1:-1])
        return float(param.value) if isinstance(param, NumberParam) else None
    try:
        return float(text)
    except ValueError:
        return None


def loosens_done(base: MemoryDoc, doc: MemoryDoc, op: EditOp) -> str | None:
    """Why `op` on `doc` would loosen a done criterion (`EditRules.strict_done`), or None: its
    band against the band it had in `base`, the document the editor opened (a criterion defined
    since is the editor's own)."""
    name = op.name if isinstance(op, RemoveParam) else getattr(op, "param", None)
    was = base.params.get(name) if isinstance(name, str) else None
    param = doc.params.get(name) if isinstance(name, str) else None
    if not isinstance(was, DoneSpecParam) or not isinstance(param, DoneSpecParam):
        return None
    if isinstance(op, RemoveParam | SetCategory):
        what = "removing it" if isinstance(op, RemoveParam) else f"changing its {op.field}"
        return (
            f"{name} is a done criterion: they are only tightened, never loosened, and {what} "
            "drops a condition the exit had; keep it, narrow its band, or define another criterion"
        )
    if not isinstance(op, SetNumber) or op.field not in ("value", "tol"):
        return None
    if isinstance(op.value, tuple):
        return None  # a vector for a scalar field: its own checks refuse it
    # bands as the check reads them: a tolerance below perception's error is read at it
    spec, ref = param.value, floored(was.value)
    new = float(op.value)
    value, tol = (new, spec.tol) if op.field == "value" else (spec.value, new)
    tol = max(tol, FLOORS.get(spec.metric, 0.0))
    low, high = ref.value - ref.tol, ref.value + ref.tol
    if value - tol >= low - EPSILON and value + tol <= high + EPSILON:
        return None
    return (
        f"{name} is a done criterion: they are only tightened, never loosened (the official check "
        "alone decides success, and a gate too strict costs no success, a loose one a false "
        f"finish). Its band was {low:.4g} to {high:.4g}; {op.field} {show(op.value)} would make "
        f"it {value - tol:.4g} to {value + tol:.4g}. Keep it inside that band (narrow the "
        "tolerance first to move the value), or leave it"
    )
