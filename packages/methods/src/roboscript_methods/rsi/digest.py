"""A deterministic digest of one evaluated episode for the reflection (docs/concepts/learning.md):
what a reflector reads first of an episode, every line pointing to a turn, from the D parts of the
records' view (`learning.eval_view`) and the official success only, never the truth diagnosis.

`digest_episode` reads, per episode: its key, arm and seed; the official success; how it ended (the
end's reason, turns and model calls, the env steps the Agent's actions used, the official step
count and limit); each failed or refused tool call, grouped by tool and reason with the turns it
happened in (a hard-failure receipt: its reason, with the measured gripper opening of an empty
grasp or a lost hold; an error: its code); refused finishes grouped by cause (read from the
refusal's text by `ReflectRules.finish_refusals`); a false finish (the done check passed, or
`finish` was refused, or the episode ended by `finish`, and the success never latched; never for an
episode flagged as infrastructure's); the infrastructure flags: the host's `infra`, the end it
voided (an end after an unanswered infrastructure error), and for records without that flag an
episode whose last tool call failed for infrastructure reasons and was then finished, which is the
infrastructure's, never a strategy failure; the keyframes a reflector can ask images of (start,
before and after each grasp, after each place, last: the frame taken last in or before the call's
turn, and the first one after it); the entry ids the Actor cited in its scratchpad; how the episode
ended at each arm (`ArmEnd`): the last gripper state a receipt of the episode reported for it
(open, closed on an object, closed empty), read from the receipt's text
(`toolsets.manipulation.results.receipt_text`, the sensing tools' arm lines), or, when the success
latched during a call that opens its arm's gripper at its end (`RELEASING`), released by it; the
arm the instruction names, if any (`ReflectRules.arm_named`), and each arm choice
(`ReflectRules.arm_tool`'s calls: the object, the answer's first sentence, the rationale the Actor
gave with that turn's action), so a reflection can tell an arm chosen against the instruction or by
distance alone from one it names. `successes_ended` is the line a brief gives of how a task's
successes ended. `render_digest` is its text, `headline` its one line; `render_turns` pages through
the turns themselves (observations, the model's text, calls and results, images by label).
"""

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from roboscript.learning.eval_view import EpisodeOutcome, EvalEpisode, RecordPart
from roboscript_methods.rsi.doc import EpisodeKey
from roboscript_methods.rsi.recipe import ReflectRules

INFRA_ERROR: Final = "infra_error"
ENDED: Final = "episode_ended"
"""A call after the success latched mid-batch: the episode's end, no failure."""
OWN: Final = frozenset({"finish", "scratchpad"})
"""The Hub's own tools: answering one is no sign the infrastructure is back (`Hub`)."""
RECEIPT: Final = re.compile(r"^Receipt: (\w+) \(hard failure\)")
OPENING: Final = re.compile(r"opening (\d+(?:\.\d+)?)")
CITED: Final = re.compile(r"\b(?:G|T-[A-Za-z0-9_]+)-[0-9]{2,}\b")
DETAIL: Final = 140
GRIPPER: Final = re.compile(
    r"^(?P<arm>[a-z][a-z0-9_]*) (?:gripper: commanded|arm: .*; gripper command) "
    r"(?P<command>\d+(?:\.\d+)?), opening (?:not measured|\d+(?:\.\d+)?), "
    r"holding (?P<holding>something|nothing|unknown)\.?$",
    re.MULTILINE,
)
"""An arm's gripper line in a receipt: the manipulation receipts' and the sensing tools'."""
CLOSED_BELOW: Final = 0.5
"""A gripper command below this is closed (`manipulation.receipts`)."""
GripperEnd = Literal[
    "open",
    "open, holding something",
    "closed on an object",
    "closed empty",
    "closed, holding unknown",
    "released by the last call",
]
OPENED: Final = frozenset({"open", "released by the last call"})
ALL_OPEN: Final = "every arm open or released, nothing held"
RELEASING: Final[dict[str, tuple[str, bool | float]]] = {
    "place_object": ("is_open", True),
    "open_gripper": ("pos", 1.0),
    "nudge": ("release", True),
}
"""Calls that open their arm's gripper at their end, by the argument that says so and its
default (the manipulation toolset's `place_object`, `open_gripper` and `nudge`): when the success
latched during one, no receipt reports the arm's end, and the call itself says the gripper
opens."""


class FailureGroup(BaseModel):
    """Calls of one tool that failed alike: `kind` (`hard_failure`, a receipt's, or `error`, a
    code), the reason, the turns, and the first one's detail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str
    kind: str
    reason: str
    turns: tuple[NonNegativeInt, ...]
    detail: str = ""


class FinishRefusal(BaseModel):
    """Refused `finish` calls of one cause (`ReflectRules.finish_refusals`, or `other`), their
    turns, and the first one's detail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cause: str
    turns: tuple[NonNegativeInt, ...]
    detail: str = ""


class Keyframe(BaseModel):
    """A frame the reflector can ask images of: its alias, step and turn, and its names."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame: str
    step: NonNegativeInt
    turn: NonNegativeInt | None
    names: tuple[str, ...]


class ArmEnd(BaseModel):
    """How an arm ended: the last gripper state a receipt reported for it, and that turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: str
    state: GripperEnd
    turn: NonNegativeInt | None

    @property
    def text(self) -> str:
        return f"{self.arm} {self.state}"


class ArmPick(BaseModel):
    """One arm choice of the Actor (`ReflectRules.arm_tool`): its turn, the object it asked about,
    the answer's first sentence (or the error), and the rationale of that turn's action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: NonNegativeInt | None
    obj: str
    answer: str
    rationale: str = ""


class EpisodeDigest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: EpisodeKey
    arm: str
    seed: NonNegativeInt
    instruction: str
    instruction_arm: str = ""
    """The arm the instruction names (`ReflectRules.arm_named`'s match), "" when it names none."""
    arm_picks: tuple[ArmPick, ...] = ()
    success: bool
    outcome: EpisodeOutcome
    failures: tuple[FailureGroup, ...] = ()
    failures_left_out: NonNegativeInt = 0
    done_passed: tuple[NonNegativeInt, ...] = ()
    finish_refused: tuple[FinishRefusal, ...] = ()
    false_finish: str = ""
    infra: tuple[str, ...] = ()
    keyframes: tuple[Keyframe, ...] = ()
    frames_taken: NonNegativeInt = 0
    actions_left_out: NonNegativeInt = 0
    cited: tuple[str, ...] = ()
    events_from: str = "episode"
    end: tuple[ArmEnd, ...] = ()


@dataclass
class _Call:
    turn: int | None
    name: str
    receipt: dict[str, Any] | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


def digest_episode(
    episode: EvalEpisode, key: EpisodeKey, arm: str, rules: ReflectRules
) -> EpisodeDigest:
    """The digest of `episode` (of the view; its diagnosis part, if any, is not read)."""
    if key.task != episode.task:
        raise ValueError(f"key {key.label()} names another task than {episode.task}")
    outcome = episode.outcome or EpisodeOutcome()
    calls = _calls(episode.parts)
    failures = _failures(calls, rules)
    done = tuple(
        c.turn
        for c in calls
        if c.name == rules.done_tool and c.turn is not None and _text(c).startswith(rules.done_pass)
    )
    refused = _finish_refusals(calls, rules)
    infra = _infra(episode.success, outcome, calls)
    keyframes, frames, left_out = _keyframes(episode.parts, calls, rules)
    shown = failures[: rules.digest_failures]
    instruction = _instruction(episode.parts)
    named = re.search(rules.arm_named, instruction, re.IGNORECASE)
    return EpisodeDigest(
        key=key,
        arm=arm,
        seed=episode.seed,
        instruction=instruction,
        instruction_arm=named.group() if named else "",
        arm_picks=_arm_picks(episode.parts, calls, rules),
        success=episode.success,
        outcome=outcome,
        failures=tuple(shown),
        failures_left_out=len(failures) - len(shown),
        done_passed=done,
        finish_refused=refused,
        # an infrastructure failure is no strategy failure: nothing of it reads as a false finish
        false_finish="" if episode.success or infra else _false_finish(outcome, done, refused),
        infra=infra,
        keyframes=keyframes,
        frames_taken=frames,
        actions_left_out=left_out,
        cited=_cited(episode.parts),
        events_from=episode.events_from,
        end=_end(calls),
    )


def headline(digest: EpisodeDigest) -> str:
    """One line: key, seed, arm, the outcome and what went wrong."""
    o = digest.outcome
    verdict = "success" if digest.success else "FAILURE"
    bits = [f"end {o.end or '?'}", f"{_n(o.turns)} turns", f"{_n(o.steps)} steps"]
    hard = sum(len(g.turns) for g in digest.failures if g.kind == "hard_failure")
    errors = sum(len(g.turns) for g in digest.failures if g.kind == "error")
    if hard:
        bits.append(f"{hard} hard failures")
    if errors:
        bits.append(f"{errors} refused or erred calls")
    if digest.false_finish:
        bits.append("false finish")
    if digest.infra:
        bits.append("INFRA")
    return f"{digest.key.label()} seed {digest.seed} [{digest.arm}]: {verdict}; {', '.join(bits)}"


def render_digest(digest: EpisodeDigest, rules: ReflectRules) -> str:
    """The digest as a reflector reads it (at most `rules.digest_turns` turns per list)."""
    cap = rules.digest_turns
    o = digest.outcome
    lines = [f"# {headline(digest)}", f"instruction: {digest.instruction}"]
    named = digest.instruction_arm
    lines.append(f"instruction names an arm: {named!r}" if named else "instruction names no arm")
    if digest.arm_picks:
        lines.append("arm choices (turn, object: answer; the turn's rationale):")
        for pick in digest.arm_picks[:cap]:
            turn = "?" if pick.turn is None else str(pick.turn)
            why = f"; rationale: {pick.rationale}" if pick.rationale else ""
            lines.append(f"- turn {turn}, {pick.obj}: {pick.answer}{why}")
        if len(digest.arm_picks) > cap:
            lines.append(f"- ... {len(digest.arm_picks) - cap} more (rsi_episode lists turns)")
    calls = f"{_n(o.model_calls)} model calls" + (
        f" of {o.max_model_calls}" if o.max_model_calls else ""
    )
    official = f"{_n(o.official_steps)} of {_n(o.step_limit)} official steps"
    lines.append(
        f"ended: {o.end or '?'} after {_n(o.turns)} turns ({calls}); the Agent's actions used "
        f"{_n(o.steps)} steps, {official}"
    )
    lines += [f"infra: {flag}" for flag in digest.infra]
    lines.append(f"end (each arm's last reported gripper): {end_text(digest.end)}")
    if digest.false_finish:
        lines.append(f"false finish: {digest.false_finish}")
    if digest.done_passed:
        lines.append(f"the done check passed at turns {_turns(digest.done_passed, cap)}")
    if digest.finish_refused:
        lines.append("refused finishes (cause: turns; first one's detail):")
        for refusal in digest.finish_refused:
            turns = _turns(refusal.turns, cap)
            lines.append(
                f"- {refusal.cause} x{len(refusal.turns)}: turns {turns}; {refusal.detail}"
            )
    if digest.failures:
        lines.append("failed or refused calls (tool, reason: turns; first one's detail):")
        for group in digest.failures:
            turns = _turns(group.turns, cap)
            detail = f"; {group.detail}" if group.detail else ""
            count = len(group.turns)
            lines.append(f"- {group.tool} {group.reason} x{count}: turns {turns}{detail}")
        if digest.failures_left_out:
            lines.append(f"- ... {digest.failures_left_out} more groups (rsi_episode lists turns)")
    else:
        lines.append("failed or refused calls: none")
    if digest.events_from == "host":
        lines.append("events: the episode kept no event log; read from the host's, by its id")
    elif digest.events_from == "none":
        lines.append("events: none recorded for this episode (no keyframes)")
    lines.append(f"keyframes ({digest.frames_taken} frames taken; ask rsi_frames by alias):")
    for frame in digest.keyframes:
        turn = "" if frame.turn is None else f", turn {frame.turn}"
        lines.append(f"- {frame.frame} (step {frame.step}{turn}): {', '.join(frame.names)}")
    if digest.actions_left_out:
        lines.append(f"- ... {digest.actions_left_out} grasp or place calls in between left out")
    if digest.cited:
        lines.append(f"entries the Actor cited: {', '.join(digest.cited)}")
    return "\n".join(lines)


def render_turns(episode: EvalEpisode, first: int, last: int, max_chars: int) -> tuple[str, int]:
    """Turns `first` to `last` of `episode` as text (the observation, the model's text, each call
    and its result; images by label), cut at a turn boundary before `max_chars`; and the turn the
    page stopped before (`last + 1` when it holds them all)."""
    by_turn: dict[int, list[RecordPart]] = {}
    for part in episode.parts:
        if part.turn is not None and part.kind != "keyframe" and part.tier == "D":
            by_turn.setdefault(part.turn, []).append(part)
    out: list[str] = []
    size = 0
    for turn in range(first, last + 1):
        if turn not in by_turn:
            continue
        block = [f"## turn {turn}"] + [_part_line(part) for part in by_turn[turn]]
        text = "\n".join(block)
        if out and size + len(text) > max_chars:
            return "\n\n".join(out), turn
        out.append(text[:max_chars])
        size += len(text)
    return "\n\n".join(out), last + 1


def _part_line(part: RecordPart) -> str:
    content = part.content
    if part.kind == "observation":
        return f"observation (step {part.step}):\n{content}"
    if part.kind == "assistant":
        return f"model: {content}"
    if part.kind == "tool_call":
        assert isinstance(content, dict)
        arguments = json.dumps(content["arguments"], ensure_ascii=False, sort_keys=True)
        return f"-> {content['name']}({_cut(arguments, 800)})"
    if part.kind == "receipt":
        assert isinstance(content, dict)
        error = f" error {content['error']}" if content.get("error") else ""
        return f"<- {content.get('status')}{error}: {_content_text(content.get('content'))}"
    if part.kind in OWN:
        return f"-> {part.kind}({json.dumps(content, ensure_ascii=False, sort_keys=True)})"
    return f"{part.kind}: {_cut(json.dumps(content, ensure_ascii=False), 800)}"


def _content_text(content: Any) -> str:
    texts = []
    for item in content or []:
        if item.get("type") == "text":
            texts.append(item["text"])
        else:
            blob = item.get("blob") or {}
            label = item.get("label") or (blob.get("meta") or {}).get("camera") or "image"
            texts.append(f"[image: {label}]")
    return "\n".join(texts)


def _calls(parts: Sequence[RecordPart]) -> list[_Call]:
    """Calls in order (tools and the Hub's own), each with its receipt (an own call's only when
    it was refused)."""
    calls: list[_Call] = []
    for part in parts:
        if part.kind == "tool_call":
            assert isinstance(part.content, dict)
            arguments = part.content.get("arguments")
            calls.append(
                _Call(
                    part.turn,
                    str(part.content["name"]),
                    arguments=dict(arguments) if isinstance(arguments, dict) else {},
                )
            )
        elif part.kind in OWN:
            calls.append(_Call(part.turn, part.kind))
        elif part.kind == "receipt" and calls:
            assert isinstance(part.content, dict)
            calls[-1].receipt = dict(part.content)
    return calls


def _text(call: _Call) -> str:
    return _content_text((call.receipt or {}).get("content"))


def _failures(calls: Sequence[_Call], rules: ReflectRules) -> list[FailureGroup]:
    groups: dict[tuple[str, str, str], list[tuple[int | None, str]]] = {}
    for call in calls:
        if call.receipt is None or call.name in OWN:
            continue  # a refused finish is a false finish, below
        text = _text(call)
        error = call.receipt.get("error")
        if error == ENDED:
            continue
        if error:
            kind, reason, detail = "error", str(error), _first_line(text, error)
        elif found := RECEIPT.match(text):
            kind, reason = "hard_failure", found.group(1)
            detail = _hard_detail(text)
        else:
            continue
        groups.setdefault((call.name, kind, reason), []).append((call.turn, detail))
    found_groups = [
        FailureGroup(
            tool=tool,
            kind=kind,
            reason=reason,
            turns=tuple(t for t, _ in seen if t is not None),
            detail=seen[0][1],
        )
        for (tool, kind, reason), seen in groups.items()
    ]
    # the most frequent first, then by first turn: what recurs is what a reflector should read
    return sorted(found_groups, key=lambda g: (-len(g.turns), g.turns[:1], g.tool, g.reason))


def _first_line(text: str, error: str) -> str:
    line = text.split("\n", 1)[0]
    line = line.removeprefix(f"Error ({error}): ")
    return _cut(line, DETAIL)


def _hard_detail(text: str) -> str:
    """The measured gripper opening (an empty grasp, a lost hold) and the receipt's note."""
    bits = []
    for line in text.split("\n"):
        if "gripper:" in line and (opening := OPENING.search(line)):
            bits.append(f"gripper opening {opening.group(1)}")
        elif line.startswith("Note: "):
            bits.append(_cut(line.removeprefix("Note: "), DETAIL))
    return "; ".join(bits)


def _finish_refusals(calls: Sequence[_Call], rules: ReflectRules) -> tuple[FinishRefusal, ...]:
    """Refused `finish` calls grouped by cause, read from the refusal's text: the Hub refuses
    one for different reasons (no done check after the last action, a failing done check, a
    success that has not latched), and only the text says which."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for call in calls:
        if call.name != "finish" or call.receipt is None or call.turn is None:
            continue
        error = str(call.receipt.get("error") or "")
        text = _text(call).split("\n", 1)[0].removeprefix(f"Error ({error}): ")
        cause = next((c for c, p in rules.finish_refusals.items() if text.startswith(p)), "other")
        groups.setdefault(cause, []).append((call.turn, _cut(text, DETAIL)))
    return tuple(
        FinishRefusal(cause=cause, turns=tuple(t for t, _ in seen), detail=seen[0][1])
        for cause, seen in sorted(groups.items(), key=lambda g: (-len(g[1]), g[1][0][0]))
    )


def _false_finish(
    outcome: EpisodeOutcome, done: tuple[int, ...], refused: tuple[FinishRefusal, ...]
) -> str:
    said = []
    if done:
        said.append(f"the done check passed {len(done)} times")
    if refused:
        causes = ", ".join(f"{r.cause} x{len(r.turns)}" for r in refused)
        said.append(f"finish was refused {sum(len(r.turns) for r in refused)} times ({causes})")
    if outcome.end == "finish":
        said.append("the episode ended by finish")
    return "; ".join(said) + " and the success never latched" if said else ""


def _infra(success: bool, outcome: EpisodeOutcome, calls: Sequence[_Call]) -> tuple[str, ...]:
    flags = []
    if outcome.infra:
        flags.append("the host marked infrastructure trouble in this episode")
    if outcome.voided:
        flags.append(f"the host voided its end ({outcome.voided}) after an unanswered infra error")
    unanswered = None  # as the Hub keeps it: an infra error until a later tool answers
    for call in calls:
        error = (call.receipt or {}).get("error")
        if error == INFRA_ERROR:
            unanswered = (call.name, call.turn)
        elif call.name not in OWN and not error:
            unanswered = None
    if unanswered and outcome.end == "finish" and not success and not outcome.voided:
        name, turn = unanswered
        flags.append(
            f"infra: the last tool call ({name}, turn {turn}) failed for infrastructure reasons "
            "and the episode was then finished: not a strategy failure"
        )
    return tuple(flags)


def _keyframes(
    parts: Sequence[RecordPart], calls: Sequence[_Call], rules: ReflectRules
) -> tuple[tuple[Keyframe, ...], int, int]:
    frames = [p for p in parts if p.kind == "keyframe"]
    if not frames:
        return (), 0, 0
    names: dict[int, list[str]] = {0: ["start"]}
    actions = []
    grasps = places = 0
    for call in calls:
        if call.turn is None or call.name not in (*rules.grasp_tools, *rules.place_tools):
            continue
        error = (call.receipt or {}).get("error")
        failed = RECEIPT.match(_text(call)) or (error if error != ENDED else None)
        tag = f" ({failed.group(1) if isinstance(failed, re.Match) else failed})" if failed else ""
        if call.name in rules.grasp_tools:
            grasps += 1
            actions.append((call.turn, [f"before_grasp_{grasps}{tag}"], [f"after_grasp_{grasps}"]))
        else:
            places += 1
            actions.append((call.turn, [], [f"after_place_{places}{tag}"]))
    keep = rules.digest_actions
    left_out = max(0, len(actions) - keep)
    if left_out:
        actions = actions[: keep // 2] + actions[len(actions) - (keep - keep // 2) :]
    for turn, before, after in actions:
        at_or_before = [i for i, f in enumerate(frames) if (f.turn or 0) <= turn]
        later = [i for i, f in enumerate(frames) if (f.turn or 0) > turn]
        if before and at_or_before:
            names.setdefault(at_or_before[-1], []).extend(before)
        if later:
            names.setdefault(later[0], []).extend(after)
    names.setdefault(len(frames) - 1, []).append("last")
    named = []
    for index in sorted(names):
        part = frames[index]
        assert isinstance(part.content, dict) and part.step is not None
        named.append(
            Keyframe(
                frame=str(part.content["frame"]),
                step=part.step,
                turn=part.turn,
                names=tuple(names[index]),
            )
        )
    return tuple(named), len(frames), left_out


def end_text(end: Sequence[ArmEnd]) -> str:
    """How an episode ended at its arms, as a digest reads it."""
    if not end:
        return "no arm reported"
    return "; ".join(f"{a.text}" + ("" if a.turn is None else f" (turn {a.turn})") for a in end)


def end_pattern(end: Sequence[ArmEnd]) -> str:
    """How an episode ended at its arms, turns left out: `ALL_OPEN` when every reported arm was
    open holding nothing."""
    if not end:
        return "no arm reported"
    if all(a.state in OPENED for a in end):
        return ALL_OPEN
    return ", ".join(a.text for a in end)


def successes_ended(digests: Iterable[EpisodeDigest]) -> str:
    """How a task's successful episodes ended at their arms, counted by pattern (infrastructure
    episodes left out): the line a brief gives a reflector and the reviewer."""
    ends = [end_pattern(d.end) for d in digests if d.success and not d.infra]
    if not ends:
        return "successes ended: no success to show how an episode ends"
    counts = Counter(ends)
    shown = "; ".join(f"{n}/{len(ends)} with {pattern}" for pattern, n in counts.most_common())
    return f"successes ended: {shown}"


def _end(calls: Sequence[_Call]) -> tuple[ArmEnd, ...]:
    last: dict[str, ArmEnd] = {}
    for call in calls:
        if call.receipt is None:
            continue
        for match in GRIPPER.finditer(_text(call)):
            closed = float(match["command"]) < CLOSED_BELOW
            holding = match["holding"]
            state: GripperEnd
            if holding == "unknown":
                state = "closed, holding unknown" if closed else "open"
            elif closed:
                state = "closed on an object" if holding == "something" else "closed empty"
            else:
                state = "open, holding something" if holding == "something" else "open"
            last[match["arm"]] = ArmEnd(arm=match["arm"], state=state, turn=call.turn)
    latched = next(
        (
            c
            for c in calls
            if c.name not in OWN and c.receipt is not None and c.receipt.get("error") == ENDED
        ),
        None,
    )
    if latched is not None and latched.name in RELEASING:
        name, default = RELEASING[latched.name]
        value = latched.arguments.get(name, default)
        arm = latched.arguments.get("arm")
        if isinstance(arm, str) and isinstance(value, (bool, int, float)) and value >= CLOSED_BELOW:
            last[arm] = ArmEnd(arm=arm, state="released by the last call", turn=latched.turn)
    return tuple(last[arm] for arm in sorted(last))


def _arm_picks(
    parts: Sequence[RecordPart], calls: Sequence[_Call], rules: ReflectRules
) -> tuple[ArmPick, ...]:
    rationales = {p.turn: p.content for p in parts if p.kind == "rationale" and p.turn is not None}
    picks = []
    for call in calls:
        if call.name != rules.arm_tool:
            continue
        error = (call.receipt or {}).get("error")
        text = _text(call)
        answer = _first_line(text, str(error)) if error else _sentence(text)
        rationale = rationales.get(call.turn)
        if isinstance(rationale, dict):
            rationale = "; ".join(
                f"{k}: {v}" for k, v in rationale.items() if k in ("scene", "plan")
            )
        picks.append(
            ArmPick(
                turn=call.turn,
                obj=str(call.arguments.get("obj", "")),
                answer=answer,
                rationale=_cut(str(rationale or ""), 2 * DETAIL),
            )
        )
    return tuple(picks)


def _sentence(text: str) -> str:
    """The first sentence of a result's first line."""
    line = text.split("\n", 1)[0]
    head, dot, _ = line.partition(". ")
    return _cut(head + dot.strip() if dot else line, DETAIL)


def _instruction(parts: Sequence[RecordPart]) -> str:
    return next((str(p.content) for p in parts if p.kind == "instruction"), "")


def _cited(parts: Sequence[RecordPart]) -> tuple[str, ...]:
    notes = [json.dumps(p.content, ensure_ascii=False) for p in parts if p.kind == "scratchpad"]
    return tuple(dict.fromkeys(m for note in notes for m in CITED.findall(note)))


def _turns(turns: Sequence[int], cap: int) -> str:
    shown = ", ".join(str(t) for t in turns[:cap])
    return shown + (f", ... +{len(turns) - cap}" if len(turns) > cap else "")


def _n(value: int | None) -> str:
    return "?" if value is None else str(value)


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
