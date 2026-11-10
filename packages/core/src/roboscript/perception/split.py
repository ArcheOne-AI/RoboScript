"""Splitting a proposal that holds several objects before Set-of-Mark
(docs/concepts/perception.md).

Height-map proposals merge objects that touch (an object against its box, one proposal for both, so
one of the two roles has nothing to mark). GroundingDINO's boxes
for each role's kind of object (`kind`: the OBJECTS line's clause after its first ';') are lifted;
one whose object's pixels (those the proposals hold) lie, by `SHARE` of them at least, inside one
proposal at least `RATIO` times their size is cut out of that proposal with SAM 3 (its box as the
prompt, the mask kept inside the proposal): the piece and the rest of the proposal replace it. They
are not the frame's proposals, which every process recomputes alike and region ids name, so each
comes with its mask in the anchor view as a blob: an answer that names one names that mask. A
proposal that a better box of the kind (boxes come best first) covers whole is that object (a large
body, then a box on one of its sub-parts), and the kind's lesser boxes in it are its parts, not
objects to cut out.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from roboscript.contracts.blob import BlobRef
from roboscript.perception.config import DEFAULTS, PerceptionConfig
from roboscript.perception.fit import summarize
from roboscript.perception.grounding import SegmentPrompt
from roboscript.perception.lift import box_mask, lift, touches_border
from roboscript.perception.proposals import Proposal, view_mask
from roboscript.perception.registry import Step, silent
from roboscript.perception.views import FrameGeometry
from roboscript.perception.vision import VisionDetector, VisionSegmenter

SHARE: Final = 0.8
RATIO: Final = 1.8
TOP_K: Final = 3
"""GroundingDINO's boxes per kind tried."""
MIN_PIXELS: Final = 200
"""A box whose object shows fewer pixels in the proposals is not tried."""
MIN_POINTS: Final = 50
"""A piece or rest with fewer lifted points is dropped."""
_EIGHT: Final = np.ones((3, 3), dtype=bool)


def kind_clause(description: str) -> str:
    """The clause of an OBJECTS description that says what kind of object the role is: the one
    after its first ';' (the whole description without one), up to its first ',' or '.'. What
    follows adds the parts grasped or placed at and their symmetry (`- <role>: <what it does>;
    <what kind>, <its parts>;
    ...`)."""
    parts = description.split(";")
    clause = parts[1] if len(parts) > 1 else parts[0]
    return re.split(r"[,.]", clause, maxsplit=1)[0].strip()


def kind(description: str) -> str:
    """The kind of object an OBJECTS description names (`kind_clause`), lowercase, without an
    article."""
    words = kind_clause(description).lower()
    for article in ("a ", "an ", "the "):
        if words.startswith(article):
            return words[len(article) :]
    return words


async def split(
    geometry: FrameGeometry,
    proposals: Sequence[Proposal],
    kinds: Sequence[str],
    detector: VisionDetector,
    segmenter: VisionSegmenter,
    config: PerceptionConfig = DEFAULTS,
    log: Step = silent,
) -> tuple[list[Proposal], dict[int, BlobRef]]:
    """The proposals with merged ones split (new ids after the frame's), and the anchor-view
    mask blob of each new one."""
    camera = geometry.anchor
    rgb = geometry.view(camera).image.rgb
    found = await detector.detect(rgb, sorted(set(kinds)), TOP_K)
    current = list(proposals)
    made: list[tuple[str, Any]] = []
    for query, detections in found.boxes.items():
        whole: list[Proposal] = []  # by identity: pieces are all id 0 until numbered
        for detection in detections:
            host, is_whole = _fit(geometry, current, detection.box, config)
            if host is not None and is_whole:
                whole.append(host)
            if host is None or is_whole or any(host is w for w in whole):
                continue
            answer = await segmenter.segment(geometry, camera, SegmentPrompt(box=detection.box))
            piece = answer.mask & host.masks[camera]
            rest = host.masks[camera] & ~ndimage.binary_dilation(piece, _EIGHT, iterations=2)
            parts = [_proposal(geometry, m, config) for m in (piece, rest)]
            if parts[0] is None:
                continue
            current.remove(host)
            current += [p for p in parts if p is not None]
            made.append((query, host.id))
    if not made:
        return list(proposals), {}
    top = max(p.id for p in proposals)
    new = [p for p in current if p.id == 0]
    numbered = [p for p in current if p.id != 0]
    for i, p in enumerate(new, top + 1):
        numbered.append(Proposal(i, p.kind, p.camera, p.points, p.summary, p.masks))
    blobs = {
        p.id: await segmenter.blobs.aput_array(p.masks[camera]) for p in numbered if p.id > top
    }
    log("split", frame=geometry.frame, split=[{"kind": q, "proposal": h} for q, h in made],
        new=sorted(blobs))  # fmt: skip
    return sorted(numbered, key=lambda p: p.id), blobs


def _fit(
    geometry: FrameGeometry,
    proposals: Sequence[Proposal],
    box: Sequence[float],
    config: PerceptionConfig,
) -> tuple[Proposal | None, bool]:
    """The proposal a box's object is (True) or lies in, when it is at least `RATIO` times the
    object (False); None when neither."""
    camera = geometry.anchor
    shape = geometry.view(camera).image.shape
    lifted = lift(geometry, {camera: box_mask(box, shape)}, config)
    if lifted is None:
        return None, False
    held = np.zeros(shape, dtype=bool)
    for p in proposals:
        held |= p.masks[camera]
    seen = lifted.masks[camera] & held
    if seen.sum() < MIN_PIXELS:
        return None, False
    host = max(proposals, key=lambda p: int((p.masks[camera] & seen).sum()))
    inside = int((host.masks[camera] & seen).sum())
    union = int((host.masks[camera] | seen).sum())
    if inside / union >= 0.5:
        return host, True
    if inside < SHARE * seen.sum() or host.masks[camera].sum() < RATIO * seen.sum():
        return None, False
    return host, False


def _proposal(
    geometry: FrameGeometry, mask: NDArray[np.bool_], config: PerceptionConfig
) -> Proposal | None:
    """The object an anchor-view mask shows, as a proposal (id 0 until numbered)."""
    camera = geometry.anchor
    lifted = lift(geometry, {camera: mask}, config)
    if lifted is None or len(lifted.points) < MIN_POINTS:
        return None
    summary = summarize(lifted.points, geometry.table.z)
    masks: dict[str, NDArray[np.bool_]] = {camera: lifted.masks[camera]}
    for name, view in geometry.views.items():
        if name != camera:
            masks[name] = view_mask(
                lifted.points,
                summary,
                lifted.kind,
                view,
                geometry.table,
                config,
                touches_border(lifted.masks[camera]),
            )
    return Proposal(0, lifted.kind, camera, lifted.points, summary, masks)


def kinds(roles: Mapping[str, str]) -> list[str]:
    """The kinds of the roles' objects, each once."""
    return sorted({kind(text) for text in roles.values()})
