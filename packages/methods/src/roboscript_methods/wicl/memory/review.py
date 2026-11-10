"""The S1 Reviser's edit validators that read the record of the seed so far (`imitate.revise`
opens its `Editor` with them): `EditStreak` refuses another nudge of a target the last revisions
kept nudging while the attempts passed no more of the demonstration, and `AfterFirst` refuses an
edit whose only evidence is calls the Actor improvised after the review's first divergence.
"""

from collections.abc import Mapping

from roboscript.memory.edits import (
    AXES,
    EditOp,
    Editor,
    EditValidator,
    RotateSpec,
    SetCategory,
    SetNumber,
)

Target = tuple[str, str | None]


def target_of(op: object) -> Target | None:
    """The numeric or categorical target an edit nudges (a rotation: the grasp's `AXES`); None
    for another edit (a definition, a refit, text)."""
    if isinstance(op, SetNumber | SetCategory):
        return op.param, op.field
    if isinstance(op, RotateSpec):
        return op.param, AXES
    return None


def _shown(target: Target) -> str:
    param, field = target
    return param if field is None else f"{param}.{field}"


class EditStreak(EditValidator):
    """`streaks`: by target, the revisions in a row that edited it with the next attempt passing
    no more of the demonstration's keypoints than the best before it (`imitate.edit_streaks`);
    `limit`: from this many on, another edit of it is refused (a refit, `fit_spec`, is not: it
    puts the whole spec back where the demonstration has it). `fits`: by spec the demonstration
    fitted, where it refits (named in the refusal); a parameter without one (a number, a spec
    written otherwise) has no refit to offer and is never refused."""

    def __init__(self, streaks: Mapping[Target, int], limit: int, fits: Mapping[str, str]) -> None:
        self.streaks = dict(streaks)
        self.limit = limit
        self.fits = dict(fits)

    def rules(self, editor: Editor, op: EditOp) -> str | None:
        target = target_of(op)
        if target is None or self.limit <= 0 or self.streaks.get(target, 0) < self.limit:
            return None
        param = target[0]
        if param not in self.fits:
            return None
        refit = f'fit_spec(name="{param}", at="{self.fits[param]}")'
        return (
            f"{_shown(target)} was edited in the last {self.streaks[target]} revisions and the "
            "attempts after them passed no more of the demonstration's keypoints: another nudge "
            f"repeats that; refit {param} as the demonstration has it ({refit}), or change "
            "another parameter"
        )


class AfterFirst(EditValidator):
    """`improvised`: the parameters the review names only at stages after its first divergence
    whose calls the Actor improvised (no workflow stage of the Memory's, or arguments the
    stage's call does not name), each with the first such stage (`imitate.improvised_only`): an
    edit of one is refused, since what ran there followed from the earlier failure and the
    Actor's own call, not from the Memory."""

    def __init__(self, improvised: Mapping[str, str], first: str) -> None:
        self.improvised = dict(improvised)
        self.first = first

    def rules(self, editor: Editor, op: EditOp) -> str | None:
        target = target_of(op)
        if target is None or target[0] not in self.improvised:
            return None
        return (
            f"{target[0]} is named only at {self.improvised[target[0]]}, after the first "
            f"divergence ({self.first}), by a call the Actor improvised there: that call followed "
            "from the earlier failure, not from the Memory; change what the first divergence "
            "names"
        )


__all__ = ["AfterFirst", "EditStreak", "target_of"]
