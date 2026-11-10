"""Identify once: the objects of a task's roles, found on the initial scene and registered
(docs/concepts/perception.md).

The roles are the Memory's OBJECTS, abstract (what each does in the task, what kind of object it
is); which object plays a role comes from the episode's live instruction, which every prompt
carries. Without an instruction (the Demo Reader's `locate`) each role comes with a description.

1. Proposals: the frame's geometric object candidates, as the caller computes them; with the
   vision service, merged ones split (`split`: an object against its box).
2. Set-of-Mark: one VLM call picks a mark (a proposal) for every role, or none, and gives the
   instruction's words for each role (`vlm.SomGrounder`). Beside it, with the live instruction and
   two roles or more (or one, when the instruction names an arm), one text-only call gives each
   role's words, noun, the color, size or place words that single it out and the arm the
   instruction has act on its object (`vlm.role_words`; the arm is stored on the registered object
   for `plan_arms`), and geometry re-orders the marks of roles
   that share a noun by what those words name (`reorder`). Size and place are measured: the roles
   they order keep those marks, which count as the tie-break's vote (`measured`).
3. Cross-check: GroundingDINO's top box for each role's words (`vision.VisionDetector`), lifted;
   its next boxes are kept for step 7.
4. Look-alikes: the roles of one kind of object (the OBJECTS line's clause after ';') with
   distinct marks, unless their words all order them by size or place (crops lose both), go to
   one VLM call that shows each mark cut out and enlarged, with the wrist camera's closer view
   where it sees the object 1.3 times larger (`vlm.compare`). An answer that gives the group's
   roles distinct marks decides the group: those marks stand, and count as the tie-break's vote.
   Roles named alike but for a number whose words are all the same (phrase, color, size and
   place), or whose objects look alike (`order.alike`: words that differ cannot be checked on
   identical objects), have nothing a comparison could tell apart: they skip the call (a role's
   name is part of what it compares) and take their marks in a fixed layout order instead
   (`_lay_out`: the listed roles, nearest first, then the robot's left, `order.order_roles`), so
   the demo and every new scene bind them alike; those marks stand the same way (`layout`).
5. Box: one VLM call (`vlm.BoxGrounder.locate_roles`) for the roles Set-of-Mark left without a
   mark (the fallback) and the other roles whose mark GroundingDINO contradicts (the tie-break);
   boxes are lifted with depth, and a box on nothing standing above the table shows nothing.
6. Vote, per role (`_decide`): with a mark, GroundingDINO agreeing makes it `ok` and showing
   nothing leaves it `unconfirmed`; showing another object calls the box (or the look-alike
   call, or the measurement), and two of the three agreeing decide (`ok`, the dissent
   recorded); only when all three differ (or the box shows nothing either) is it `ambiguous`:
   the mark's object, flagged, with the candidates. Without a mark the box answers, confirmed
   (`ok`), unconfirmed or contradicted (`ambiguous`) by GroundingDINO; with neither the role is
   `not_found` (a cross-check candidate is reported). A role the words say the instruction
   leaves out (it names fewer objects of that kind than the roles list: one item of two) is
   `not_named`, nothing registered: the workflow has fewer objects this episode; never where
   the instruction refers to them collectively ("both of them", "the rest", a plural:
   `collective`), which names every role of the kind.
7. Claims (`settle`): answers are registered best evidence first (`ok`, `unconfirmed`, then
   `ambiguous`; role order within), each from every view (`locate`; a box that names no proposal
   is carried into the other views). A role whose answer shows an object registered already
   (earlier in the call, or before) falls back to its other answers (its mark, the box,
   GroundingDINO's boxes in their order) that show no registered object: `unconfirmed`, with a
   note; with none left it is `ambiguous` and names that object (`same_as`). A role whose answer
   cannot be lifted is `not_found` with a note. A backend call the model refuses leaves its
   roles without its answer; the call fails (`ToolError`) only when the model refused every
   request, and an infrastructure failure (`InfraError`) always stops it.

Confidence: 1 for `ok`, 0.5 for `unconfirmed` and `ambiguous`.

Answers name objects through proposals: a mark is one; a box names the proposal its lifted pixels
match (IoU >= `MATCH` in the anchor camera), else only its lifted pixels; a split piece names its
anchor-view mask (a blob). Two answers show the same object when they name one proposal, or when
at least half the anchor pixels of the smaller lie in the other; a registered object's anchor
pixels are its points projected there. Every step is logged as a `perception.step` event
(`proposals`, `split`, `som`, `words`, `reorder`, `detect`, `compare`, `box`, `decide`, `claim`,
then `locate`'s `lift`, `fit`, `register`), with the exact prompts and answers.
"""

import functools
import itertools
import re
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeVar

import anyio
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy import ndimage

from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import ToolError
from roboscript.contracts.geometry import Vec3
from roboscript.contracts.robot import Arm, RobotState
from roboscript.contracts.vision import Detection
from roboscript.perception.config import DEFAULTS, OrderConfig, PerceptionConfig
from roboscript.perception.fit import summarize
from roboscript.perception.lift import box_mask, lift
from roboscript.perception.locate import locate
from roboscript.perception.order import alike, order_roles
from roboscript.perception.overlay import crop_sheet, som_overlay
from roboscript.perception.proposals import Proposal, propose, view_mask
from roboscript.perception.registry import SceneRegistry, Shown, Step, object_geometry, silent
from roboscript.perception.reorder import reorder, seen_of
from roboscript.perception.split import kind, kinds, split
from roboscript.perception.views import FrameGeometry, View
from roboscript.perception.vision import Detected, VisionDetector, VisionSegmenter
from roboscript.perception.vlm import (
    Boxes,
    BoxGrounder,
    Marks,
    RoleWords,
    SomGrounder,
    compare,
    role_words,
)
from roboscript.spatial.camera import project

MATCH: Final = 0.5
"""A box's lifted pixels name a proposal whose pixels they match with this IoU."""
SAME: Final = 0.5
"""Two answers show one object when this share of the smaller one's pixels lies in the other."""
CONFIRMED: Final = 1.0
UNCONFIRMED: Final = 0.5
RANK: Final = {"ok": 0, "unconfirmed": 1, "ambiguous": 2}
"""Claims on one object go to the best evidence."""
LATER: Final = 2
PLURAL_BOXES: Final = 8
"""GroundingDINO's boxes for a plural role's words (`ask`'s `plural`; the others get `1 + LATER`,
the first boxes of the same call): enough for every look-alike of a group."""
"""GroundingDINO's boxes after its top one that a role falls back to (step 7)."""
CLOSER: Final = 1.3
"""A candidate's crop gets the wrist view that shows it with this many times its anchor pixels."""
_ARM: Final = re.compile(r"\b(?:left|right)[- ](?:hand|arm|gripper)\b(?![- ]side)", re.IGNORECASE)
"""An instruction that names an arm (`left arm`, `right hand`, not a `left-hand side`): its words
call says which role's object each arm acts on, even with a single role. Without one, no arm
the words call reads counts: `left of` an object is where something goes, not an arm."""

Reason = Literal["ok", "unconfirmed", "ambiguous", "not_found", "not_named"]
PixelBox = tuple[float, float, float, float]
_T = TypeVar("_T")


class Answer(BaseModel):
    """One backend's answer for a role in `camera`: a mark (Set-of-Mark, or the look-alike call,
    `compare`) or a box (the VLM, with its point; GroundingDINO, with its score, which ranks and
    is never thresholded); the proposal it names (or, for a split piece, its anchor-view mask
    blob) and the center of its lifted points (all None: it shows nothing)."""

    model_config = ConfigDict(frozen=True)

    backend: str
    camera: str
    mark: int | None = None
    box: PixelBox | None = None
    point: tuple[float, float] | None = None
    score: float | None = None
    proposal: int | None = None
    mask: BlobRef | None = None
    center: Vec3 | None = None


class Identified(BaseModel):
    """A role's identification: the registered object (None when there is none), its confidence,
    whether the backends disagreed, the distinct objects they showed (the chosen one first), each
    backend's answer (`som`, `grounding_dino`, `vlm`, `compare`, `layout`, `measured`), the
    backends outvoted (`dissent`), the instruction's words for it (`phrase`), the registered
    object its answer showed when that object was taken (`same_as`), why an answer led
    nowhere or elsewhere (`note`) and the arm the instruction has act on it (`arm`, from the
    words call; None: it says none)."""

    model_config = ConfigDict(frozen=True)

    role: str
    description: str
    phrase: str | None = None
    arm: Arm | None = None
    object: str | None = None
    confidence: float = 0.0
    ambiguous: bool = False
    reason: Reason
    candidates: list[Answer] = []
    evidence: dict[str, Answer] = {}
    dissent: list[str] = []
    same_as: str | None = None
    note: str = ""


@dataclass(frozen=True, eq=False)
class _Hit:
    """An answer and the object it shows: its pixels in each view (the proposal's, else the
    lifted box's, carried into the other views) and the proposal, if it names one."""

    answer: Answer
    masks: dict[str, NDArray[np.bool_]]
    proposal: Proposal | None


@dataclass(eq=False)
class Evidence:
    """What the backends answered for each role of a frame, before anything is registered: the
    proposals asked about (split ones with their mask blobs, `pieces`), the mark's,
    GroundingDINO's and the box's objects (None: none), GroundingDINO's next boxes (`later`),
    each backend's answers, the instruction's words per role, the model calls made and refused,
    the roles the instruction asks for no object in (`unnamed`) and the arm it has act on each
    role's object (`arms`, the words call's reading)."""

    geometry: FrameGeometry
    roles: dict[str, str]
    phrases: dict[str, str | None]
    som: dict[str, _Hit | None]
    check: dict[str, _Hit | None]
    boxes: dict[str, _Hit | None]
    answers: dict[str, dict[str, Answer]]
    proposals: list[Proposal] = field(default_factory=list)
    pieces: dict[int, BlobRef] = field(default_factory=dict)
    later: dict[str, list[Detection]] = field(default_factory=dict)
    calls: int = 0
    refused: list[str] = field(default_factory=list)
    unnamed: set[str] = field(default_factory=set)
    arms: dict[str, Arm] = field(default_factory=dict)


async def ask(
    geometry: FrameGeometry,
    roles: Mapping[str, str],
    proposals: Sequence[Proposal],
    *,
    instruction: str | None = None,
    som: SomGrounder | None = None,
    box: BoxGrounder | None = None,
    detector: VisionDetector | None = None,
    segmenter: VisionSegmenter | None = None,
    config: PerceptionConfig = DEFAULTS,
    log: Step = silent,
    plural: Collection[str] = (),
) -> Evidence:
    """Every backend's answer for each role (role -> what it does and is, with the live
    `instruction`; else a description to find it by), nothing registered; backends left out are
    skipped (the split needs `detector` and `segmenter`, the words and the look-alike call `som`'s
    model); a role of `plural` (`order: rest`) gets up to `PLURAL_BOXES` GroundingDINO boxes, the
    others `1 + LATER` of the same call. ToolError when the model refused every request;
    InfraError propagates."""
    camera = geometry.anchor
    rgb = geometry.view(camera).image.rgb
    steps = config.identify
    pieces: dict[int, BlobRef] = {}
    if steps.split and detector is not None and segmenter is not None:
        proposals, pieces = await split(
            geometry, proposals, kinds(roles), detector, segmenter, config, log
        )
    log("proposals", frame=geometry.frame, camera=camera, proposals=_listed(proposals, camera))
    refused: list[str] = []
    calls = 0

    async def refusable(what: str, call: Callable[[], Awaitable[_T]]) -> _T | None:
        nonlocal calls
        calls += 1
        try:
            return await call()
        except ToolError as error:
            refused.append(f"{what}: {error.detail}")
            return None

    choose = (
        None
        if som is None
        else functools.partial(som.choose, geometry, proposals, roles, camera, instruction)
    )
    words_wanted = (
        som is not None
        and instruction is not None
        and steps.words
        and (len(roles) > 1 or _ARM.search(instruction) is not None)
    )
    ask_words = (
        None
        if not words_wanted
        else functools.partial(
            role_words,
            som.client,
            roles,
            instruction,
            som.config,
            som.blobs,  # type: ignore[union-attr]
        )
    )
    marks, said = await _both(
        None if choose is None else functools.partial(refusable, "Set-of-Mark", choose),
        None if ask_words is None else functools.partial(refusable, "words", ask_words),
    )
    if marks is not None:
        _log_som(log, geometry, camera, roles, marks)
    words: dict[str, RoleWords] = {}
    arms: dict[str, Arm] = {}
    measured: set[str] = set()
    unnamed: set[str] = set()
    if said is not None:
        words, asked = said
        log("words", frame=geometry.frame, words={r: w.model_dump() for r, w in words.items()},
            asked=asked.model_dump(mode="json"))  # fmt: skip
        unnamed = _unnamed(words, instruction)
        words = {r: w for r, w in words.items() if r not in unnamed}
        if instruction and _ARM.search(instruction):  # "left of the box" names a side, no arm
            arms = {r: w.arm for r, w in words.items() if w.arm is not None}
        if marks is not None:
            chosen, changes, measured = reorder(words, marks.chosen, proposals, rgb, camera)
            if changes or measured:
                log("reorder", frame=geometry.frame, changes=changes, measured=sorted(measured))
                marks = Marks(chosen, marks.phrases, marks.shown, marks.overlay, marks.asked)
    phrases: dict[str, str | None] = {role: None for role in roles}
    if marks is not None:
        phrases.update(marks.phrases)
    if instruction is not None:  # the instruction's words for a role, never the description's
        said_phrases = {r: w.phrase for r, w in words.items()}
        phrases = {
            role: instruction_phrase(instruction, phrases[role], said_phrases.get(role))
            for role in roles
        }
    queries = {role: phrases[role] or _words(role, roles[role], instruction) for role in roles}
    detected: Detected | None = None
    if detector is not None:
        unique = list(dict.fromkeys(queries.values()))
        wide = any(role in plural for role in roles)
        detected = await detector.detect(rgb, unique, PLURAL_BOXES if wide else 1 + LATER)
        log("detect", frame=geometry.frame, camera=camera, **_detections(detected))
    first = functools.partial(
        _first, geometry, proposals, pieces, roles, marks, detected, queries, config
    )
    som_hits, check_hits, answers = await anyio.to_thread.run_sync(first)
    later: dict[str, list[Detection]] = {}
    if detected is not None:
        for r, q in queries.items():
            later[r] = detected.boxes.get(q, [])[1 : None if r in plural else 1 + LATER]
    box_hits: dict[str, _Hit | None] = dict.fromkeys(roles)
    decided: set[str] = set()
    for role in measured:  # their marks are measured: the tie-break's vote is the measurement
        hit = som_hits[role]
        if hit is not None:
            answer = hit.answer.model_copy(update={"backend": "measured"})
            box_hits[role] = _Hit(answer, hit.masks, hit.proposal)
            answers[role]["measured"] = answer
            decided.add(role)
    named = {role: text for role, text in roles.items() if role not in unnamed}
    grouped = _twins(named, som_hits, words, rgb, camera, steps.alike)
    twins = [group for group, _ in grouped]
    laid = {role for group in twins for role in group}
    alike = _look_alikes(named, som_hits, words) if steps.compare else []
    groups = [[r for r in g if r not in laid] for g in alike]
    groups = [g for g in groups if len(g) > 1]
    if groups and instruction is not None and som is not None:
        picked = await refusable(
            "compare",
            functools.partial(
                _compare, som, geometry, groups, som_hits, roles, phrases, instruction, log
            ),
        )
        for group in groups:
            ids = [None if picked is None else picked.get(r) for r in group]
            if None in ids or len(set(ids)) < len(group):  # not one object each: undecided
                continue
            for role, hit in zip(group, ids, strict=True):
                assert hit is not None
                answer = hit.answer.model_copy(update={"backend": "compare"})
                som_hits[role] = box_hits[role] = _Hit(answer, hit.masks, hit.proposal)
                answers[role]["compare"] = answer
                decided.add(role)
    if twins:
        placed = _lay_out(twins, som_hits, rgb, camera)
        log("layout", frame=geometry.frame, groups=twins, why=[why for _, why in grouped],
            marks={r: h.answer.mark for r, h in placed.items()})  # fmt: skip
        for role, hit in placed.items():
            answer = hit.answer.model_copy(update={"backend": "layout"})
            som_hits[role] = box_hits[role] = _Hit(answer, hit.masks, hit.proposal)
            answers[role]["layout"] = answer
            decided.add(role)
    fallback = [role for role in roles if som_hits[role] is None and role not in unnamed]
    tie_break = [
        role
        for role in roles
        if role not in decided | unnamed and _disagree(som_hits[role], check_hits[role], camera)
    ]
    wanted = [role for role in roles if role in fallback or role in tie_break]
    if box is not None and wanted:
        asked_for = {
            role: roles[role] if instruction is None else _with_phrase(roles[role], phrases[role])
            for role in wanted
        }
        found = await refusable(
            "boxes",
            functools.partial(
                box.locate_roles, rgb, asked_for, camera, camera, instruction,
            ),
        )  # fmt: skip
        if found is not None:
            _log_boxes(log, geometry, found, fallback, tie_break)
            hits = functools.partial(_box_hits, geometry, proposals, pieces, found, config)
            for role, (hit, answer) in (await anyio.to_thread.run_sync(hits)).items():
                box_hits[role] = hit
                answers[role]["vlm"] = answer
    if calls and len(refused) == calls:
        raise ToolError("the grounding model refused every request: " + "; ".join(refused))
    return Evidence(
        geometry=geometry,
        roles=dict(roles),
        phrases=phrases,
        som=som_hits,
        check=check_hits,
        boxes=box_hits,
        answers=answers,
        proposals=list(proposals),
        pieces=pieces,
        later=later,
        calls=calls,
        refused=refused,
        arms=arms,
        unnamed=unnamed,
    )


def _unnamed(words: Mapping[str, RoleWords], instruction: str | None = None) -> set[str]:
    """The roles the instruction asks for no object in: those its words say it leaves out, where
    it names another role of the same kind (so an answer that names nothing drops no role), and
    it does not refer to objects of that kind collectively (`collective`: "both of them", "the
    rest"), which names every role of the kind."""
    return {
        role
        for role, said in words.items()
        if not said.named
        and said.noun.strip()
        and any(w.named and w.noun == said.noun for r, w in words.items() if r != role)
        and not collective(instruction, said.noun)
    }


_COLLECTIVE: Final = re.compile(
    r"\b(?:both|all|each|every|pair|two|three|four|five|six|seven|eight|nine|ten"
    r"|the rest|the others|the other ones|remaining)\b",
    re.IGNORECASE,
)
"""Words with which an instruction refers to several objects at once."""


def collective(instruction: str | None, noun: str) -> bool:
    """Whether `instruction` refers to objects collectively (a number word, both, all, each,
    every, a pair, the rest, the others, the remaining ones) or names `noun` in the plural: then
    it names every role of that kind, whatever a words call said of one."""
    if not instruction:
        return False
    if _COLLECTIVE.search(instruction):
        return True
    noun = noun.strip().lower()
    plurals = {noun + "s", noun + "es"}
    if noun.endswith("y"):
        plurals.add(noun[:-1] + "ies")
    return bool(plurals & set(re.findall(r"[a-z]+", instruction.lower())))


async def _both(
    first: Callable[[], Awaitable[_T]] | None, second: Callable[[], Awaitable[Any]] | None
) -> tuple[_T | None, Any]:
    """Two calls at once (either may be None: not made); an error of either is raised after
    both end, the first's first, so an infrastructure failure is not wrapped in a group."""
    results: list[Any] = [None, None]
    errors: list[BaseException | None] = [None, None]

    async def run(i: int, call: Callable[[], Awaitable[Any]]) -> None:
        try:
            results[i] = await call()
        except Exception as error:  # re-raised below, unwrapped
            errors[i] = error

    async with anyio.create_task_group() as group:
        for i, call in enumerate((first, second)):
            if call is not None:
                group.start_soon(run, i, call)
    for error in errors:
        if error is not None:
            raise error
    return results[0], results[1]


def _look_alikes(
    roles: Mapping[str, str], marks: Mapping[str, _Hit | None], words: Mapping[str, RoleWords]
) -> list[list[str]]:
    """Groups of roles of one kind of object whose marks are distinct objects, unless their words
    all order them by size or place (geometry does that, `reorder`; crops lose both)."""
    by_kind: dict[str, list[str]] = {}
    for role, text in roles.items():
        by_kind.setdefault(kind(text), []).append(role)
    groups = []
    for group in by_kind.values():
        hits = [marks[r] for r in group]
        if len(group) < 2 or any(h is None or h.proposal is None for h in hits):
            continue
        shown = {h.proposal.id for h in hits if h is not None and h.proposal is not None}
        ordered = all(r in words and (words[r].size or words[r].place) for r in group)
        if len(shown) == len(group) and not ordered:
            groups.append(group)
    return groups


def _twins(
    roles: Mapping[str, str],
    marks: Mapping[str, _Hit | None],
    words: Mapping[str, RoleWords],
    rgb: NDArray[np.uint8],
    camera: str,
    config: OrderConfig,
) -> list[tuple[list[str], str]]:
    """Roles of one kind of object whose names differ only by a number (`bowl1`, `bowl2`; not
    `left_part`, `right_part`, whose names the comparison reads) and whose marks are distinct
    objects, when nothing the instruction or a crop shows tells them apart, so a comparison's
    answer would be a guess: their words are all the same (phrase, color, size, place: `words`),
    or their objects look alike (`order.alike`: words that differ cannot be checked, `looks`).
    Each group with why."""
    numbered: dict[tuple[str, str], list[str]] = {}
    for role, text in roles.items():
        hit = marks[role]
        if role in words and hit is not None and hit.proposal is not None:
            stem = re.sub(r"[\d_\W]+", "", role.lower())
            numbered.setdefault((kind(text), stem), []).append(role)
    out: list[tuple[list[str], str]] = []
    for group in numbered.values():
        seen = [seen_of(h.proposal, rgb, camera) for r in group if (h := marks[r]) and h.proposal]
        shown = {h.proposal.id for r in group if (h := marks[r]) and h.proposal}
        if len(group) < 2 or len(shown) < len(group):
            continue
        same: dict[tuple[Any, ...], list[str]] = {}
        for role in group:
            same.setdefault(_said(words[role]), []).append(role)
        if len(same) > 1 and all(alike(a, b, config) for a, b in itertools.combinations(seen, 2)):
            out.append((group, "looks"))
            continue
        out += [(g, "words") for g in same.values() if len(g) > 1]
    return out


def _said(w: RoleWords) -> tuple[Any, ...]:
    return (" ".join(w.phrase.lower().split()), w.color, w.size, w.place)


def _lay_out(
    groups: list[list[str]], marks: Mapping[str, _Hit | None], rgb: NDArray[np.uint8], camera: str
) -> dict[str, _Hit]:
    """Each group's roles, in their listed order, take the group's marks in the layout order
    (`order.order_roles`: nearest first, then the robot's left): one fixed convention, so the
    demo and every new scene bind the same roles alike."""
    out: dict[str, _Hit] = {}
    for group in groups:
        placed = [h for h in (marks[r] for r in group) if h is not None]
        seen = [seen_of(h.proposal, rgb, camera) for h in placed if h.proposal is not None]
        found = order_roles(group, seen, "layout")
        out.update({role: placed[i] for role, i in found.items()})
    return out


async def _compare(
    som: SomGrounder,
    geometry: FrameGeometry,
    groups: list[list[str]],
    marks: Mapping[str, _Hit | None],
    roles: Mapping[str, str],
    phrases: Mapping[str, str | None],
    instruction: str,
    log: Step,
) -> dict[str, _Hit | None]:
    """The look-alike call: each group's marks cut out and lettered (with a closer wrist view),
    the whole view numbered alike; which one each role's object is."""
    camera = geometry.anchor
    hits: list[_Hit] = []
    letters: dict[str, list[str]] = {}
    for group in groups:
        shown = [marks[role] for role in group]
        hits += [h for h in shown if h is not None and h not in hits]
        for role in group:
            letters[role] = [chr(ord("A") + hits.index(h)) for h in shown if h is not None]
    views = [_crop_views(geometry, hit) for hit in hits]
    wrists = any(len(v) > 1 for v in views)
    lettered = [chr(ord("A") + i) for i in range(len(hits))]
    sheet = await anyio.to_thread.run_sync(crop_sheet, views, lettered)
    overview, _ = await anyio.to_thread.run_sync(
        som_overlay,
        geometry.view(camera).image.rgb,
        {i + 1: hit.masks[camera] for i, hit in enumerate(hits)},
    )
    asked_roles = {
        role: (roles[role], phrases.get(role) or role.replace("_", " "), letters[role])
        for group in groups
        for role in group
    }
    picked, asked = await compare(
        som.client, sheet, overview, asked_roles, instruction, wrists, som.config, som.blobs,
        som.pipeline.anchor_words,
    )  # fmt: skip
    log("compare", frame=geometry.frame, groups=groups, letters=letters, picked=picked,
        asked=asked.model_dump(mode="json"))  # fmt: skip
    return {r: None if p is None else hits[ord(p) - ord("A")] for r, p in picked.items()}


def _crop_views(
    geometry: FrameGeometry, hit: _Hit
) -> list[tuple[NDArray[np.uint8], NDArray[np.bool_]]]:
    """A candidate's anchor view, and the other view that shows it with at least `CLOSER` times
    as many pixels, if one does."""
    camera = geometry.anchor
    views = [(geometry.view(camera).image.rgb, hit.masks[camera])]
    size = int(hit.masks[camera].sum())
    others = [c for c in geometry.views if c != camera and c in hit.masks]
    best = max(others, key=lambda c: int(hit.masks[c].sum()), default=None)
    if best is not None and hit.masks[best].sum() >= CLOSER * size:
        views.append((geometry.view(best).image.rgb, hit.masks[best]))
    return views


def settle(
    evidence: Evidence,
    registry: SceneRegistry,
    *,
    robot: RobotState | None = None,
    region: Callable[[Answer], str | None] | None = None,
    config: PerceptionConfig = DEFAULTS,
    log: Step = silent,
    carried: Collection[str] = (),
) -> dict[str, Identified]:
    """Decide each role (the vote), then register the answers best evidence first, none on an
    object registered already (a role falls back to its other answers; `robot`: the frame's
    robot state, for held objects). `region` names an answer's region for the registry's
    candidates; a role of `carried` is not registered to a flat answer
    (`IdentifyConfig.flat_carried`). Plain numpy work: async callers run it in a worker thread,
    holding the registry for it."""
    geometry = evidence.geometry
    camera = geometry.anchor
    decided = {role: _decide(evidence, role) for role in evidence.roles}
    for identified, _ in decided.values():
        log("decide", frame=geometry.frame, **identified.model_dump(mode="json"))
    taken = _registered(registry, geometry.view(camera), robot, geometry.frame)
    order = sorted(
        (role for role, (_, hit) in decided.items() if hit is not None),
        key=lambda role: RANK[decided[role][0].reason],
    )
    out = {role: identified for role, (identified, _) in decided.items()}
    flat = config.proposals.min_height
    for role in order:
        identified, hit = decided[role]
        assert hit is not None
        owner = next((obj for obj, mask, pid in taken if _shows(hit, mask, pid, camera)), None)
        if owner is not None:
            other = _fallback(evidence, role, hit, taken, config)
            shown = None if other is None else other.answer.model_dump(mode="json")
            log("claim", frame=geometry.frame, role=role, same_as=owner, fallback=shown)
            if other is None:
                taken_by = {"reason": "ambiguous", "ambiguous": True, "confidence": 0.0}
                out[role] = identified.model_copy(update={**taken_by, "same_as": owner})
                continue
            note = f"its first answer shows {owner}, registered for another role"
            candidates = [other.answer, *(a for a in identified.candidates if a != other.answer)]
            identified = identified.model_copy(
                update={
                    "reason": "unconfirmed",
                    "confidence": UNCONFIRMED,
                    "ambiguous": False,
                    "candidates": candidates,
                    "note": note,
                }
            )
            hit = other
        shown = [
            Shown(backend=a.backend, region=None if region is None else region(a), center=a.center)
            for a in identified.candidates
        ]
        try:
            record = locate(
                registry,
                geometry,
                role,
                hit.proposal if hit.proposal is not None else hit.masks,
                config=config,
                confidence=identified.confidence,
                ambiguous=identified.ambiguous,
                candidates=shown if identified.ambiguous else (),
                log=log,
            )
        except (LookupError, ValueError) as error:
            note = f"its answer could not be measured: {error}"
            out[role] = identified.model_copy(
                update={"reason": "not_found", "confidence": 0.0, "ambiguous": False, "note": note}
            )
            continue
        height = float(record.fit.obb.size[2])
        if role in carried and config.identify.flat_carried and height <= flat:
            del registry.objects[record.id]
            note = (
                f"its answer is a flat patch on the table ({height * 1000:.1f} mm high), nothing "
                "an arm carries"
            )
            log("flat", frame=geometry.frame, role=role, height=height)
            out[role] = identified.model_copy(
                update={"reason": "not_found", "confidence": 0.0, "ambiguous": False, "note": note}
            )
            continue
        if identified.arm is not None:
            record = record.model_copy(update={"arm": identified.arm})
            registry.objects[record.id] = record
        out[role] = identified.model_copy(update={"object": record.id})
        pid = None if hit.proposal is None else hit.proposal.id
        taken.append((record.id, hit.masks[camera], pid))
    return out


def _fallback(
    evidence: Evidence,
    role: str,
    hit: _Hit,
    taken: Sequence[tuple[str, NDArray[np.bool_], int | None]],
    config: PerceptionConfig,
) -> _Hit | None:
    """A role's other answer that shows no registered object: its mark, the box, GroundingDINO's
    top box, then its next boxes, in that order (None: none)."""
    geometry = evidence.geometry
    camera = geometry.anchor
    options = [evidence.som[role], evidence.boxes[role], evidence.check[role]]
    for detection in evidence.later.get(role, []):
        lifted = _lift_box(
            geometry, evidence.proposals, evidence.pieces, "grounding_dino", detection.box,
            config, score=detection.score,
        )  # fmt: skip
        options.append(lifted)
    for other in options:
        if other is None or _same(other, hit, camera):
            continue
        if not any(_shows(other, mask, pid, camera) for _, mask, pid in taken):
            return other
    return None


async def identify_scene(
    geometry: FrameGeometry,
    roles: Mapping[str, str],
    registry: SceneRegistry,
    *,
    proposals: Sequence[Proposal] | None = None,
    instruction: str | None = None,
    som: SomGrounder | None = None,
    box: BoxGrounder | None = None,
    detector: VisionDetector | None = None,
    segmenter: VisionSegmenter | None = None,
    robot: RobotState | None = None,
    region: Callable[[Answer], str | None] | None = None,
    config: PerceptionConfig = DEFAULTS,
    log: Step = silent,
) -> dict[str, Identified]:
    """`ask`, then `settle` into `registry`, in role order (`proposals`: computed with `config`
    unless given)."""
    if proposals is None:
        proposals = await anyio.to_thread.run_sync(propose, geometry, config)
    evidence = await ask(
        geometry,
        roles,
        proposals,
        instruction=instruction,
        som=som,
        box=box,
        detector=detector,
        segmenter=segmenter,
        config=config,
        log=log,
    )
    decide = functools.partial(
        settle, evidence, registry, robot=robot, region=region, config=config, log=log
    )
    return await anyio.to_thread.run_sync(decide)


def _decide(evidence: Evidence, role: str) -> tuple[Identified, _Hit | None]:
    """The vote of one role: its decision and the answer to register (None: nothing)."""
    camera = evidence.geometry.anchor
    mark, check, boxed = evidence.som[role], evidence.check[role], evidence.boxes[role]
    base: dict[str, Any] = {
        "role": role,
        "description": evidence.roles[role],
        "phrase": evidence.phrases[role],
        "arm": evidence.arms.get(role),
        "evidence": evidence.answers[role],
    }
    if role in evidence.unnamed:
        note = "the instruction asks for no object in this role"
        return Identified(reason="not_named", note=note, **base), None

    def decision(
        reason: Reason, hit: _Hit, others: Sequence[_Hit], dissent: Sequence[str] = ()
    ) -> tuple[Identified, _Hit | None]:
        confidence = CONFIRMED if reason == "ok" else UNCONFIRMED
        shown = [hit.answer, *(o.answer for o in others if not _same(o, hit, camera))]
        identified = Identified(
            reason=reason,
            confidence=confidence,
            ambiguous=reason == "ambiguous",
            candidates=shown,
            dissent=list(dissent),
            **base,
        )
        return identified, hit

    if mark is None:
        if boxed is None:
            candidates = [] if check is None else [check.answer]
            return Identified(reason="not_found", candidates=candidates, **base), None
        if check is None:
            return decision("unconfirmed", boxed, [])
        if _same(boxed, check, camera):
            return decision("ok", boxed, [])
        return decision("ambiguous", boxed, [check])
    if check is None:
        return decision("unconfirmed", mark, [])
    if _same(mark, check, camera):
        return decision("ok", mark, [])
    if boxed is not None and _same(boxed, mark, camera):
        return decision("ok", mark, [check], ["grounding_dino"])
    if boxed is not None and _same(boxed, check, camera):
        return decision("ok", check, [mark], ["som"])
    return decision("ambiguous", mark, [check, *([] if boxed is None else [boxed])])


def _first(
    geometry: FrameGeometry,
    proposals: Sequence[Proposal],
    pieces: Mapping[int, BlobRef],
    roles: Mapping[str, str],
    marks: Marks | None,
    detected: Detected | None,
    queries: Mapping[str, str],
    config: PerceptionConfig,
) -> tuple[dict[str, _Hit | None], dict[str, _Hit | None], dict[str, dict[str, Answer]]]:
    """The objects the marks and GroundingDINO's top boxes show, and the answers so far."""
    camera = geometry.anchor
    by_id = {p.id: p for p in proposals}
    som: dict[str, _Hit | None] = {}
    check: dict[str, _Hit | None] = {}
    answers: dict[str, dict[str, Answer]] = {}
    for role in roles:
        given: dict[str, Answer] = {}
        mark = None if marks is None else marks.chosen.get(role)
        som[role] = None
        if mark is not None:
            p = by_id[mark]
            named = _names(p, pieces)
            answer = Answer(
                backend="som", camera=camera, mark=mark, center=p.summary.center, **named
            )
            som[role] = _Hit(answer, p.masks, p)
            given["som"] = answer
        elif marks is not None:
            given["som"] = Answer(backend="som", camera=camera)
        check[role] = None
        if detected is not None:
            top = detected.boxes.get(queries[role]) or []
            best = top[0] if top else None
            if best is not None:
                check[role] = _lift_box(
                    geometry,
                    proposals,
                    pieces,
                    "grounding_dino",
                    best.box,
                    config,
                    score=best.score,
                )
            hit = check[role]
            given["grounding_dino"] = (
                hit.answer if hit is not None else _empty("grounding_dino", camera, best)
            )
        answers[role] = given
    return som, check, answers


def _box_hits(
    geometry: FrameGeometry,
    proposals: Sequence[Proposal],
    pieces: Mapping[int, BlobRef],
    found: Boxes,
    config: PerceptionConfig,
) -> dict[str, tuple[_Hit | None, Answer]]:
    out = {}
    for role, box in found.boxes.items():
        hit = None
        if box is not None:
            hit = _lift_box(geometry, proposals, pieces, "vlm", box, config, found.points[role])
        empty = Answer(backend="vlm", camera=geometry.anchor, box=box)
        out[role] = (hit, empty if hit is None else hit.answer)
    return out


def _lift_box(
    geometry: FrameGeometry,
    proposals: Sequence[Proposal],
    pieces: Mapping[int, BlobRef],
    backend: str,
    box: PixelBox,
    config: PerceptionConfig,
    point: tuple[float, float] | None = None,
    score: float | None = None,
) -> _Hit | None:
    """What a box in the anchor camera shows: the proposal its lifted pixels match, else those
    pixels, carried into the other views (None: nothing stands in the box)."""
    camera = geometry.anchor
    lifted = lift(geometry, {camera: box_mask(box, geometry.view(camera).image.shape)}, config)
    if lifted is None:
        return None
    seen = lifted.masks[camera]
    best, best_iou = None, 0.0
    for p in proposals:
        inside = int((p.masks[camera] & seen).sum())
        if inside:
            iou = inside / int((p.masks[camera] | seen).sum())
            if iou > best_iou:
                best, best_iou = p, iou
    named = best if best is not None and best_iou >= MATCH else None
    summary = summarize(lifted.points, geometry.table.z)
    answer = Answer(
        backend=backend,
        camera=camera,
        box=box,
        point=point,
        score=score,
        center=summary.center,
        **({} if named is None else _names(named, pieces)),
    )
    if named is not None:
        return _Hit(answer, named.masks, named)
    masks, cut = dict(lifted.masks), lifted.cut[camera]
    for name, view in geometry.views.items():
        if name not in masks:
            masks[name] = view_mask(
                lifted.points, summary, lifted.kind, view, geometry.table, config, cut
            )
    return _Hit(answer, masks, None)


def _names(proposal: Proposal, pieces: Mapping[int, BlobRef]) -> dict[str, Any]:
    """How an answer names a proposal: by its id, or a split piece by its mask blob."""
    if proposal.id in pieces:
        return {"mask": pieces[proposal.id]}
    return {"proposal": proposal.id}


def _disagree(mark: _Hit | None, check: _Hit | None, camera: str) -> bool:
    """Whether a mark and the cross-check show different objects (both showing one)."""
    return mark is not None and check is not None and not _same(mark, check, camera)


def _same(a: _Hit, b: _Hit, camera: str) -> bool:
    if a.proposal is not None and b.proposal is not None:
        return a.proposal.id == b.proposal.id
    return _overlap(a.masks[camera], b.masks[camera])


def _shows(hit: _Hit, mask: NDArray[np.bool_], proposal: int | None, camera: str) -> bool:
    """Whether a hit shows a taken object (its anchor mask, and its proposal if it named one)."""
    if hit.proposal is not None and proposal is not None:
        return hit.proposal.id == proposal
    return _overlap(hit.masks[camera], mask)


def _overlap(x: NDArray[np.bool_], y: NDArray[np.bool_]) -> bool:
    smaller = min(int(x.sum()), int(y.sum()))
    return smaller > 0 and int((x & y).sum()) >= SAME * smaller


def claimed(
    registry: SceneRegistry,
    geometry: FrameGeometry,
    masks: Mapping[str, NDArray[np.bool_]],
    robot: RobotState | None,
    exclude: str | None = None,
) -> str | None:
    """The registered object (but `exclude`) that pixels of `geometry`'s frame (camera -> mask)
    show, by `settle`'s claim check: in the anchor view, else the first view they are in; None
    when they show none. `robot`: the frame's robot state, for held objects."""
    cameras = [c for c in (geometry.anchor, *masks) if c in masks and masks[c].any()]
    if not cameras:
        return None
    camera = cameras[0]
    taken = _registered(registry, geometry.view(camera), robot, geometry.frame)
    return next((o for o, mask, _ in taken if o != exclude and _overlap(masks[camera], mask)), None)


def rest_instances(
    registry: SceneRegistry,
    geometry: FrameGeometry,
    obj: str,
    proposals: Sequence[Proposal],
    *,
    robot: RobotState | None = None,
    boxes: Sequence[PixelBox] | None = None,
    config: PerceptionConfig = DEFAULTS,
    log: Step = silent,
) -> list[str]:
    """A plural role's other objects (`order: rest`): every proposal of `geometry`'s frame no
    registered object claims (`claimed`) that one of `boxes` shows (GroundingDINO's boxes for
    the role's words in the identification, `detector_boxes`, lifted as its answers are; None:
    no detector answered, every proposal) and whose size, its sides sorted, is within the
    measure's `size_tol` (log ratio, each side) of `obj`'s, registered under `obj`'s name with
    its confidence, in proposal order; their ids."""
    first = registry.get(obj)
    size = np.sort(np.maximum(np.asarray(first.fit.obb.size), 0.01))
    shown = None
    if boxes is not None:
        lifted = [_lift_box(geometry, proposals, {}, "grounding_dino", b, config) for b in boxes]
        shown = {h.proposal.id for h in lifted if h is not None and h.proposal is not None}
    added = []
    for proposal in proposals:
        if shown is not None and proposal.id not in shown:
            continue
        sides = np.sort(np.maximum(np.asarray(proposal.summary.size), 0.01))
        if float(np.abs(np.log(sides / size)).max()) > config.measure.size_tol:
            continue
        if claimed(registry, geometry, proposal.masks, robot) is not None:
            continue
        record = locate(
            registry, geometry, first.name, proposal, config=config,
            confidence=first.confidence, log=log,
        )  # fmt: skip
        log("rest", frame=geometry.frame, role=first.name, object=record.id, proposal=proposal.id)
        added.append(record.id)
    return added


def detector_boxes(evidence: Evidence, role: str) -> list[PixelBox] | None:
    """GroundingDINO's boxes for a role's words in `evidence`, best first (its top box and the
    next ones, `later`); None: it gave no answer for the role (no detector)."""
    top = evidence.answers.get(role, {}).get("grounding_dino")
    if top is None:
        return None
    boxes = [] if top.box is None else [top.box]
    return boxes + [d.box for d in evidence.later.get(role, [])]


def _registered(
    registry: SceneRegistry, view: View, robot: RobotState | None, frame: str
) -> list[tuple[str, NDArray[np.bool_], int | None]]:
    """The registered objects' pixels in a view of `frame`: their points projected, gaps closed.
    An object measured in that frame shows where it was measured (it may have been carried off
    since); the others where they are now (held ones where the robot holds them; skipped
    without it)."""
    image = view.image
    h, w = image.shape
    out: list[tuple[str, NDArray[np.bool_], int | None]] = []
    for record in registry.objects.values():
        if record.frame == frame:
            points = np.asarray(record.points)
        elif record.grasps and robot is None:
            continue
        else:
            points = object_geometry(record, robot).points
        uv, _, front = project(points, image.K, image.cam2world)
        pixel = np.floor(uv[front & np.isfinite(uv).all(axis=1)]).astype(np.int64)
        pixel = pixel[(pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0)]
        pixel = pixel[pixel[:, 1] < h]
        mask = np.zeros((h, w), dtype=bool)
        mask[pixel[:, 1], pixel[:, 0]] = True
        out.append((record.id, ndimage.binary_dilation(mask, iterations=2), None))
    return out


_STOP: Final = frozenset(
    ("a", "an", "the", "of", "on", "in", "into", "onto", "to", "at", "by", "with", "from", "for")
)
_STOP |= {"and", "or", "its", "it", "this", "that", "one"}


def _content(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


def in_instruction(phrase: str, instruction: str) -> bool:
    """Whether every content word of `phrase` is in `instruction`, as a word (a plural `s` or
    `es` either way) or inside its words run together (`playing cards` in `playingcards`): words
    the model copied from it, not from a role's description."""
    said = set(_content(instruction))
    joined = "".join(_content(instruction))
    words = _content(phrase)

    def found(word: str) -> bool:
        stems = {word, word.removesuffix("es"), word.removesuffix("s"), word + "s", word + "es"}
        return bool(stems & said) or (len(word) >= 3 and word in joined)

    return bool(words) and all(found(w) for w in words)


def instruction_phrase(instruction: str, *phrases: str | None) -> str | None:
    """The first of `phrases` (Set-of-Mark's, then the words call's) the instruction holds
    (`in_instruction`); None when none does: a phrase taken from a role's description names
    another episode's object, which the cross-check would then look for."""
    for phrase in phrases:
        if phrase is not None and in_instruction(phrase, instruction):
            return phrase
    return None


def _with_phrase(text: str, phrase: str | None) -> str:
    """A role's description for the box call, with the instruction's words for it."""
    return text if phrase is None else f'{text}; the instruction\'s words for it: "{phrase}"'


def _words(role: str, text: str, instruction: str | None) -> str:
    """A cross-check query when Set-of-Mark gave none: the description, or with an instruction
    (abstract roles) the role's name as words."""
    return text if instruction is None else role.replace("_", " ")


def _empty(backend: str, camera: str, detection: Any) -> Answer:
    """An answer that shows nothing: no box, or a box with nothing above the table in it."""
    return Answer(backend=backend, camera=camera, box=None if detection is None else detection.box)


def _listed(proposals: Sequence[Proposal], camera: str) -> list[dict[str, Any]]:
    return [
        {
            "id": p.id,
            "kind": p.kind,
            "center": list(p.summary.center),
            "size": list(p.summary.size),
            "pixels": int(p.masks[camera].sum()),
        }
        for p in proposals
    ]


def _log_som(
    log: Step, geometry: FrameGeometry, camera: str, roles: Mapping[str, str], marks: Marks
) -> None:
    asked = None if marks.asked is None else marks.asked.model_dump(mode="json")
    keys = {f"r{i}": role for i, role in enumerate(roles, 1)}
    found = {"shown": marks.shown, "keys": keys, "chosen": marks.chosen, "asked": asked}
    log("som", frame=geometry.frame, camera=camera, phrases=marks.phrases, **found)


def _log_boxes(
    log: Step, geometry: FrameGeometry, found: Boxes, fallback: list[str], tie_break: list[str]
) -> None:
    asked = found.asked.model_dump(mode="json")
    wanted = {"fallback": fallback, "tie_break": tie_break}
    answer = {"boxes": found.boxes, "points": found.points}
    log("box", frame=geometry.frame, roles=list(found.boxes), **wanted, **answer, asked=asked)


def _detections(detected: Detected) -> dict[str, Any]:
    boxes = {
        query: [{"box": list(d.box), "score": d.score, "label": d.label} for d in found]
        for query, found in detected.boxes.items()
    }
    return {
        "model": detected.model,
        "time_s": detected.time_s,
        "latency_s": round(detected.latency_s, 3),
        "boxes": boxes,
    }
