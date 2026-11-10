"""Box grasps stay square to the box: a Reviser edit may not tilt a grasp's
closing axis off the faces of a box it grasps, nor move the grasp point along that axis far off
the box's middle.

Fingers closing along an axis off a box's face normals meet it on an edge first and turn it as
they close; a grasp point off the middle along that axis puts one pad on the box before the other,
with the same effect. The turn is not seen until the object is measured again, so the next stage
works from a stale pose. When an object turns in the grasp the fix is in how the gripper comes in,
never a compensating tilt.

What a grasp spec grasps comes from the workflow (the `obj` its calls pass with it) and what
that object is from the tries the Reviser reads (`Grasped`: perception's shape class and box
size, by role and alias). A spec whose object is not a box, or unknown, is not checked. An edit
that brings a tilted spec nearer the faces, or nearer the middle, is always allowed. A spec on
a part (a drawer's handle on a cabinet, also a pose grasp whose x points to the part) is not
checked either: its fingers close on the part, whose box the tries do not record, and the
object's box says nothing of it. The
demonstration's own fit is never refused, and where it already sits off the faces (or off the
middle) the check stands aside for that spec: perception's box is the object's bounding box, and
a demonstration that grasped a bowl's rim, a fan or a handle on a box-shaped body that way shows
the box's faces are not what the fingers close on.
"""

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from roboscript.manipulation.specs import GEOMETRIC_ENDS, OTHER_ARM, GraspSpec
from roboscript.manipulation.synthesis import contact
from roboscript.memory.calls import parse_calls
from roboscript.memory.doc import MemoryDoc
from roboscript.memory.edits import (
    DefineParam,
    EditOp,
    Editor,
    EditValidator,
    FitSpec,
    RemoveParam,
    RenameParam,
    RotateSpec,
    SetCategory,
    SetNumber,
    grasp_params,
)
from roboscript.memory.linter import FITTED_SOURCE
from roboscript.memory.params import demo_fit

BOX: Final = "box"
FACE_TILT_DEG: Final = 5.0
"""A box grasp's closing axis lies at most this far (deg) from the nearest face normal."""
CENTER_SHARE: Final = 0.5
"""A box grasp's point lies at most this share of the box's half-width off its middle along the
closing axis."""
_SLACK_DEG: Final = 0.5  # a tilt this near the one before is that tilt (axes shown to 4 decimals)
_SLACK_SHARE: Final = 0.02


@dataclass(frozen=True)
class Grasped:
    """An object a try saw, as perception fitted it: its shape class (`box`, `cylinder`, ...)
    and its extents along its own axes (m), which a pose grasp's frame shares."""

    shape: str
    size: tuple[float, float, float]


@dataclass(frozen=True)
class FaceFit:
    """How a grasp spec sits on a box: its closing axis's angle (deg) from the nearest face
    normal and its point's distance off the middle along that axis, as a share of the half-width
    there (the worst over the approaches the spec may take)."""

    tilt_deg: float
    share: float


def names_part(spec: GraspSpec) -> bool:
    """Whether `spec` grasps a part of its object: a part other than the whole, or a pose grasp
    whose frame's x points to a registered part (`x_toward` naming no geometric end and not the
    other arm). Its fingers close on that part, not on the object's box."""
    if spec.part != "whole":
        return True
    named = spec.x_toward is not None and spec.x_toward not in (OTHER_ARM, *GEOMETRIC_ENDS)
    return spec.anchor == "pose" and named


def face_fit(spec: GraspSpec, size: tuple[float, float, float]) -> FaceFit | None:
    """`spec` on a box of `size` (None for a spec on a part, `names_part`: a part's box is not
    the object's, and the tries record none; or one whose contact depends on the surface: an
    apex)."""
    if names_part(spec):
        return None
    extents = np.asarray(size, dtype=float)
    approaches = (
        ["top"]
        if spec.anchor == "pose"
        else (["top", "side"] if spec.approach == "auto" else [spec.approach])
    )
    tilt = share = 0.0
    found = False
    for approach in approaches:
        matrix = contact(spec, approach, extents)  # type: ignore[arg-type]
        if matrix is None:
            continue
        found = True
        z = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
        tilt = max(tilt, math.degrees(math.acos(min(1.0, float(np.abs(z).max())))))
        half = float(extents @ np.abs(z)) / 2
        if half > 0:
            share = max(share, abs(float(matrix[:3, 3] @ z)) / half)
    return FaceFit(tilt, share) if found else None


_OBJ: Final = re.compile(r"^\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")


def grasped_roles(doc: MemoryDoc, name: str) -> list[str]:
    """The objects the workflow's calls grasp with the spec `name` (their `obj`), in order."""
    roles: list[str] = []
    for stage in doc.workflow:
        for call in parse_calls(stage.call):
            if not any(
                _OBJ.match(v.strip()) and v.strip().strip("{}") == name
                for v in call.keywords.values()
            ):
                continue
            found = _OBJ.match(call.keywords.get("obj", "").strip().strip("'\""))
            if found and found[1] not in roles:
                roles.append(found[1])
    return roles


def off_faces(before: MemoryDoc, after: MemoryDoc, grasped: Mapping[str, Grasped]) -> str | None:
    """Why the edit from `before` to `after` is refused (a grasp spec it changes or defines
    would sit off a box it grasps, further than before), or None."""
    for name, param in after.params.items():
        if not isinstance(param.value, GraspSpec):
            continue
        old = before.params.get(name)
        was_spec = old.value if old is not None and isinstance(old.value, GraspSpec) else None
        if was_spec == param.value:
            continue
        fitted = demo_fit(param)
        for role in grasped_roles(after, name):
            obj = grasped.get(role)
            if obj is None or obj.shape != BOX:
                continue
            now = face_fit(param.value, obj.size)
            if now is None:
                continue
            was = None if was_spec is None else face_fit(was_spec, obj.size)
            fit = None if fitted is None else face_fit(fitted, obj.size)
            if fitted == param.value:
                continue  # the demonstration's fit itself
            square = fit is None or fit.tilt_deg <= FACE_TILT_DEG
            centered = fit is None or fit.share <= CENTER_SHARE
            if (
                square
                and now.tilt_deg > FACE_TILT_DEG
                and (was is None or now.tilt_deg > was.tilt_deg + _SLACK_DEG)
            ):
                since = "" if was is None else f" (it was {was.tilt_deg:.0f} deg)"
                return (
                    f"{name} would close its fingers along an axis {now.tilt_deg:.0f} deg off "
                    f"the nearest face of {role}{since}, a box: fingers closing off a box's faces "
                    "meet an edge first and turn the box as they close, and nothing sees the "
                    f"turn until it is measured again. Keep the closing axis within "
                    f"{FACE_TILT_DEG:g} deg of a face normal. Where the box turned in the grasp, "
                    "align the closing axis with its faces (turn it back toward them), or change "
                    "how the gripper comes in (the approach, the depth along it, the pre-grasp "
                    "distance); do not tilt it further"
                )
            if (
                centered
                and now.share > CENTER_SHARE
                and (was is None or now.share > was.share + _SLACK_SHARE)
            ):
                return (
                    f"{name} would put the grasp point {now.share:.0%} of {role}'s half-width off "
                    "its middle along the closing axis (a box): one finger meets it well before "
                    "the other and turns or pushes it as they close. Keep the point within "
                    f"{CENTER_SHARE:.0%} of the half-width of the middle along the closing axis; "
                    "move it along the approach or across instead"
                )
    return None


def face_lines(doc: MemoryDoc, grasped: Mapping[str, Grasped], *, refit: bool) -> list[str]:
    """The Memory's grasp specs that sit off a box they grasp now, one line each with the fix
    (`refit`: the editor can refit a spec at its demonstration keypoint, `fit_spec`); a spec
    whose demonstration fit sits off the faces too is left out (see the module)."""
    lines = []
    for name, param in doc.params.items():
        if not isinstance(param.value, GraspSpec):
            continue
        fitted = demo_fit(param)
        for role in grasped_roles(doc, name):
            obj = grasped.get(role)
            if obj is None or obj.shape != BOX:
                continue
            now = face_fit(param.value, obj.size)
            fit = None if fitted is None else face_fit(fitted, obj.size)
            if now is None:
                continue
            where = param.source.removeprefix(f"{FITTED_SOURCE} ")
            back = (
                f"refit it as the demonstration had it (fit_spec {name} at {where}) or "
                if refit and fitted is not None
                else ""
            )
            if (fit is None or fit.tilt_deg <= FACE_TILT_DEG) and now.tilt_deg > FACE_TILT_DEG:
                was = (
                    ""
                    if fit is None
                    else f"; the demonstration's fit closed {fit.tilt_deg:.0f} deg"
                )
                lines.append(
                    f"- {name} on {role} (a box) closes {now.tilt_deg:.0f} deg off its nearest "
                    f"face normal{was}: {back}turn it back toward the faces (rotate_spec for a "
                    "pose grasp, else yaw_deg or lean_deg); a further tilt does not fix it"
                )
            elif (fit is None or fit.share <= CENTER_SHARE) and now.share > CENTER_SHARE:
                lines.append(
                    f"- {name} on {role} (a box) grasps {now.share:.0%} of the half-width off its "
                    f"middle along the closing axis: {back}move its point back toward the middle "
                    "along that axis"
                )
    return lines


def off_rule(doc: MemoryDoc, name: str, grasped: Mapping[str, Grasped]) -> str | None:
    """The role of a box `name` grasps where it now sits off the rule and its demonstration fit
    did not (a tilt off the faces or a point off the middle), or None."""
    param = doc.params.get(name)
    if param is None or not isinstance(param.value, GraspSpec):
        return None
    fitted = demo_fit(param)
    for role in grasped_roles(doc, name):
        obj = grasped.get(role)
        if obj is None or obj.shape != BOX:
            continue
        now = face_fit(param.value, obj.size)
        fit = None if fitted is None else face_fit(fitted, obj.size)
        if now is None:
            continue
        if (fit is None or fit.tilt_deg <= FACE_TILT_DEG) and now.tilt_deg > FACE_TILT_DEG:
            return role
        if (fit is None or fit.share <= CENTER_SHARE) and now.share > CENTER_SHARE:
            return role
    return None


def required_fixes(
    doc: MemoryDoc,
    grasped: Mapping[str, Grasped],
    tipped: Iterable[tuple[str | None, str]],
    *,
    refit: bool,
) -> dict[str, str]:
    """The grasp specs a revision must fix before any other grasp edit, each with the edit that
    fixes it: those named by a close that tipped a box (`tipped`: spec and object as the call
    named them, from the try's ground truth) and that sit off the rule where their demonstration
    fit did not. The fix is the refit at its keypoint where the editor can refit (`refit`), else
    turning it back within the rule."""
    required: dict[str, str] = {}
    for spec, obj in tipped:
        if spec is None or spec in required:
            continue
        role = off_rule(doc, spec, grasped)
        named = grasped.get(obj)
        if role is None or (named is not None and named is not grasped.get(role)):
            continue
        param = doc.params[spec]
        where = param.source.removeprefix(f"{FITTED_SOURCE} ")
        if refit and demo_fit(param) is not None:
            required[spec] = f'fit_spec(name="{spec}", at="{where}")'
        else:
            required[spec] = (
                f"rotate_spec / set_number on {spec} turning its closing axis back within "
                f"{FACE_TILT_DEG:g} deg of {role}'s face normals and its point within "
                f"{CENTER_SHARE:.0%} of the half-width of the middle"
            )
    return required


@dataclass
class BoxFaces(EditValidator):
    """The rule as an editor's validator: `grasped`, the objects the tries under review saw, by
    role and alias (`wicl.roles.grasped_objects`; the Revisers add to it as they read tries),
    against which a grasp spec on a box must stay square and centered (`off_faces`); `required`,
    grasp specs whose close tipped a box while they sat off its faces (`required_fixes`: S1's
    ground truth), each with the edit that fixes it: until it no longer breaks the rule, any other
    edit to a grasp the workflow passes (a spec or a number its grasp calls take) is refused,
    naming that edit."""

    grasped: dict[str, Grasped] = field(default_factory=dict)
    required: dict[str, str] = field(default_factory=dict)

    def pending(self, editor: Editor, op: EditOp) -> str | None:
        """Why `op` waits for a required fix: it edits a grasp the workflow passes while a spec
        that tipped a box still sits off its faces, and is not an edit of that spec."""
        pending = {
            name: fix
            for name, fix in self.required.items()
            if off_rule(editor.doc, name, self.grasped) is not None
        }
        if not pending:
            return None
        if isinstance(op, FitSpec | DefineParam):
            target = op.name
            if isinstance(op, DefineParam) and op.param.kind != "grasp_spec":
                return None
        elif isinstance(op, SetNumber | SetCategory | RotateSpec | RemoveParam | RenameParam):
            target = op.name if isinstance(op, RemoveParam | RenameParam) else op.param
            if target not in grasp_params(editor.doc):
                return None
        else:
            return None
        if target in pending:
            return None
        named = "; ".join(f"{name}: {fix}" for name, fix in pending.items())
        return (
            "fix the grasp that tipped the box first: the try's ground truth shows its close "
            "turned a box while its closing axis (or its point) sat off the box's faces, where "
            "the demonstration's grasp did not; no other grasp edit is judged until "
            f"it holds square. Send this edit first: {named}"
        )

    def applied(self, editor: Editor, doc: MemoryDoc, op: EditOp) -> str | None:
        return off_faces(editor.doc, doc, self.grasped) if self.grasped else None


def find_box_faces(editor: Editor) -> BoxFaces | None:
    """The editor's `BoxFaces` validator, None when it has none (the rule is not on)."""
    return next((v for v in editor.validators if isinstance(v, BoxFaces)), None)


def box_faces(editor: Editor) -> BoxFaces:
    """The editor's `BoxFaces` validator, to record grasps into; `LookupError` when it has none
    (a copy made here would hold grasps no editor ever checks)."""
    found = find_box_faces(editor)
    if found is None:
        raise LookupError("the editor has no BoxFaces validator")
    return found
