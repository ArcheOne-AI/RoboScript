"""A re-recording's keypoints against the keypoints of the trajectory it replayed.

The same motion recorded twice should give the same keypoints. A difference is flagged for
review (`CompareConfig`) when the counts differ (a keypoint without a counterpart), a counterpart
lies farther than `flag_ratio` times the referee's own tolerance of the reference keypoint
(horizontally or vertically: what an S1 attempt following our keypoints would miss), or it comes
more than `flag_offset_frames` frames early or late. Only keypoints before the re-recording's
success
frame are compared: the replay stops at the success latch, and the referee requires no later
keypoint. The reference is cut where that frame maps to. A re-recording's frame maps to the
reference's through its `source_frames` (the identity when it sent every frame). Per arm,
keypoints pair up in order by a dynamic program over frame offsets: a gripper event only with the
same event, a motion cue only with a motion cue, and a pair more than `2 * GAP` frames apart costs
more than leaving both alone.
"""

from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import ArrayLike
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from roboscript.contracts.robot import Arm
from roboscript.demos.keypoints import GRIP_CUES, Keypoint, KeypointSet, Phase

GAP: Final = 10
"""Frames a keypoint without a counterpart costs in the pairing."""


class CompareConfig(BaseModel):
    """What the keypoint comparison flags for review, besides counts that differ: a pair farther
    apart than `flag_ratio` times the reference keypoint's tolerance, or more than
    `flag_offset_frames` frames (the reference's) apart."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    flag_ratio: PositiveFloat = 1.0
    flag_offset_frames: NonNegativeInt = 5


class KeypointPair(BaseModel):
    """A keypoint of ours and its counterpart in the reference (None when unpaired); `frame` is
    ours (the reference's when ours is None), `offset` our frame minus the reference's on the
    reference's frames, `xy_mm` and `dz_mm` how far ours lies from it horizontally and
    vertically, `ratio` the larger of them as a share of the reference keypoint's tolerance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: Arm
    ours: PositiveInt | None
    reference: PositiveInt | None
    phase: Phase
    reference_phase: Phase | None
    frame: NonNegativeInt
    offset: int | None
    xy_mm: NonNegativeFloat | None
    dz_mm: float | None
    ratio: NonNegativeFloat | None
    flagged: bool


class KeypointDiff(BaseModel):
    """`counts`: per arm, our keypoints and the reference's that were compared; `later`: those
    at or after the success frame (ours, the reference's), left out; `flags`: what to review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    counts: dict[Arm, tuple[NonNegativeInt, NonNegativeInt]]
    later: dict[Arm, tuple[NonNegativeInt, NonNegativeInt]]
    pairs: list[KeypointPair]
    flags: list[str]

    @property
    def worst(self) -> float | None:
        """The largest tolerance share of a pair (None without pairs)."""
        return max((p.ratio for p in self.pairs if p.ratio is not None), default=None)


def compare_keypoints(
    ours: KeypointSet,
    reference: KeypointSet,
    source_frames: ArrayLike | None = None,
    config: CompareConfig | None = None,
) -> KeypointDiff:
    """`ours` from a re-recording that sent `reference`'s frames `source_frames`, flagged by
    `config` (default: `CompareConfig()`)."""
    config = config or CompareConfig()
    to_reference = _mapping(source_frames)
    latch = ours.success_frame
    cut = None if latch is None else to_reference(latch)
    counts, later, pairs, flags = {}, {}, [], []
    for arm in sorted({*ours.arms, *reference.arms}):
        mine = [k for k in ours.arms.get(arm, []) if latch is None or k.frame < latch]
        theirs = [k for k in reference.arms.get(arm, []) if cut is None or k.frame < cut]
        counts[arm] = (len(mine), len(theirs))
        later[arm] = (
            len(ours.arms.get(arm, [])) - len(mine),
            len(reference.arms.get(arm, [])) - len(theirs),
        )
        frames = [to_reference(k.frame) for k in mine]
        arm_pairs = []
        for a, b in _align(frames, mine, theirs):
            if a is not None and b is not None:
                arm_pairs.append(_pair(arm, mine[a], theirs[b], frames[a], config))
            elif a is not None:
                arm_pairs.append(_unpaired(arm, mine[a], ours=True))
            elif b is not None:
                arm_pairs.append(_unpaired(arm, theirs[b], ours=False))
        pairs += arm_pairs
        unpaired = [p for p in arm_pairs if p.ratio is None]
        if unpaired:
            flags.append(
                f"{arm}: {len(mine)} keypoints before the success frame, the reference "
                f"{len(theirs)}; without a counterpart: " + ", ".join(map(_name, unpaired))
            )
        flags += [_flag(p) for p in arm_pairs if p.flagged and p.ratio is not None]
    return KeypointDiff(counts=counts, later=later, pairs=pairs, flags=flags)


def _mapping(source_frames: ArrayLike | None) -> Callable[[int], int]:
    if source_frames is None:
        return int
    frames = np.asarray(source_frames)
    return lambda t: int(frames[t])


def _compatible(a: Keypoint, b: Keypoint) -> bool:
    grip = a.cue in GRIP_CUES
    return grip == (b.cue in GRIP_CUES) and (not grip or a.cue == b.cue)


def _align(
    frames: list[int], mine: list[Keypoint], theirs: list[Keypoint]
) -> list[tuple[int | None, int | None]]:
    """Order-preserving pairs of indices (None: unpaired) at the least total cost."""
    n, m = len(mine), len(theirs)

    def pair_cost(i: int, j: int) -> float:
        fits = _compatible(mine[i], theirs[j])
        return abs(frames[i] - theirs[j].frame) if fits else np.inf

    cost = np.zeros((n + 1, m + 1))
    cost[0, :] = GAP * np.arange(m + 1)
    cost[:, 0] = GAP * np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(
                cost[i - 1, j - 1] + pair_cost(i - 1, j - 1),
                cost[i - 1, j] + GAP,
                cost[i, j - 1] + GAP,
            )
    out: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j and cost[i, j] == cost[i - 1, j - 1] + pair_cost(i - 1, j - 1):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i, j] == cost[i - 1, j] + GAP:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def _pair(arm: Arm, a: Keypoint, b: Keypoint, frame: int, config: CompareConfig) -> KeypointPair:
    """`a` of ours at `frame` on the reference's frames, and `b` of the reference."""
    d = np.subtract(a.tcp.p, b.tcp.p)
    xy, dz = float(np.linalg.norm(d[:2])), float(d[2])
    ratio = max(xy / b.tolerance.xy_m, abs(dz) / b.tolerance.z_m)
    offset = frame - b.frame
    return KeypointPair(
        arm=arm,
        ours=a.index,
        reference=b.index,
        phase=a.phase,
        reference_phase=b.phase,
        frame=a.frame,
        offset=offset,
        xy_mm=1000.0 * xy,
        dz_mm=1000.0 * dz,
        ratio=ratio,
        flagged=ratio > config.flag_ratio or abs(offset) > config.flag_offset_frames,
    )


def _unpaired(arm: Arm, k: Keypoint, ours: bool) -> KeypointPair:
    return KeypointPair(
        arm=arm,
        ours=k.index if ours else None,
        reference=None if ours else k.index,
        phase=k.phase,
        reference_phase=None,
        frame=k.frame,
        offset=None,
        xy_mm=None,
        dz_mm=None,
        ratio=None,
        flagged=True,
    )


def _name(p: KeypointPair) -> str:
    whose = f"K{p.ours}" if p.ours is not None else f"reference K{p.reference}"
    return f"{whose} ({p.phase}, frame {p.frame})"


def _flag(p: KeypointPair) -> str:
    assert p.xy_mm is not None and p.dz_mm is not None and p.ratio is not None
    return (
        f"{p.arm} K{p.ours} ({p.phase}) is {p.xy_mm / 10:.1f} cm off horizontally and "
        f"{p.dz_mm / 10:+.1f} cm vertically from reference K{p.reference}, {p.ratio:.1f}x its "
        f"tolerance ({p.offset:+d} frames)"
    )
