"""Choosing a task's demonstration: rank its candidate episodes, with a reason per criterion.

Every criterion measures a candidate against the task's other candidates, so what is usual for
the task sets the bar: the closes per arm (a re-grasp shows as a count off the task's usual one;
which arm does what may mirror between episodes), the path length and duration (the usual ones
are best), smoothness (the peak jerk, which the planner's occasional jumps and fast swings drive
up), how far each grasp's approach direction is from the task's usual one, and the number of
keypoints. Grasps compare by the gripper's pointing axis (along the embodiment's ee->TCP offset):
its turn about that axis follows the object's random yaw, and a parallel gripper is the same after
half a turn. Episodes that do with one arm what others do with the other compare as mirror images
across the plane the caller names. The costs add up; the lowest score ranks first and the episode
index breaks ties. Offline only: the ranking picks one demonstration per task before any
learning, and the other episodes are never shown to an agent.
"""

from collections import Counter
from collections.abc import Iterable
from statistics import median
from typing import Annotated

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from roboscript.contracts.geometry import Vec3
from roboscript.contracts.robot import Arm
from roboscript.demos.demo import Array, Demo
from roboscript.demos.keypoints import KeypointSet, pointing_axis, timed_frames
from roboscript.spatial.rotation import quat_rotate


class SelectionConfig(BaseModel):
    """Cost weights. A candidate whose closes per arm differ from the task's usual counts costs
    `signature_penalty`; path length and duration cost their weight per unit of |ln(value / task
    median)|, jerk per unit of ln(value / median) above the median; a grasp `grasp_weight` per
    degree of approach direction from the task's usual one; the keypoint count
    `keypoints_weight` per unit of relative deviation from the median. The review shows the top
    `keep`; `review_notes` uses the rest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signature_penalty: PositiveFloat = 10.0
    path_weight: NonNegativeFloat = 1.0
    duration_weight: NonNegativeFloat = 0.5
    jerk_weight: NonNegativeFloat = 1.0
    grasp_weight: NonNegativeFloat = 0.05
    keypoints_weight: NonNegativeFloat = 2.0
    keep: PositiveInt = 3
    review_score: NonNegativeFloat = 0.25
    variant_share: Annotated[float, Field(gt=0, le=1)] = 0.2
    close_margin: NonNegativeFloat = 0.05


class Grasp(BaseModel):
    """A close: the arm, the frame and the direction the gripper points there (world axes)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: Arm
    frame: NonNegativeInt
    approach: Vec3


class EpisodeFeatures(BaseModel):
    """What the ranking reads from one candidate. `path_m` sums both arms' TCP paths; `jerk` is
    the peak TCP jerk in meters per physics step cubed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode: NonNegativeInt
    seed: int
    frames: PositiveInt
    frame_steps: PositiveInt
    closes: dict[Arm, NonNegativeInt]
    path_m: NonNegativeFloat
    jerk: NonNegativeFloat
    grasps: list[Grasp]
    keypoints: dict[Arm, NonNegativeInt]

    @property
    def signature(self) -> tuple[int, ...]:
        """Closes per arm, most first: the same whichever arm the episode used."""
        return tuple(sorted(self.closes.values(), reverse=True))


class Criterion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    cost: NonNegativeFloat
    note: str


class RankedEpisode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    episode: NonNegativeInt
    seed: int
    rank: PositiveInt
    score: NonNegativeFloat
    criteria: list[Criterion]

    @property
    def reasons(self) -> list[str]:
        return [c.note for c in self.criteria]

    def criterion(self, name: str) -> Criterion:
        return next(c for c in self.criteria if c.name == name)


def episode_features(demo: Demo, keypoints: KeypointSet) -> EpisodeFeatures:
    meta = demo.meta
    if meta.provenance.episode is None:
        raise ValueError(f"{meta.provenance.path}: a candidate needs its episode number")
    timed = timed_frames(demo.arms.values())
    steps3 = float(meta.frame_steps) ** 3
    jerk = [
        float(np.linalg.norm(np.diff(t.tcp[timed, :3], n=3, axis=0), axis=1).max()) / steps3
        for t in demo.arms.values()
        if timed.sum() > 3
    ]
    kps = {arm: keypoints.arms.get(arm, []) for arm in demo.arms}
    grasps = [
        Grasp(arm=arm, frame=kp.frame, approach=_direction(kp.tcp.q, meta.ee_to_tcp[arm]))
        for arm, ks in kps.items()
        for kp in ks
        if kp.cue == "close"
    ]
    return EpisodeFeatures(
        episode=meta.provenance.episode,
        seed=meta.seed,
        frames=meta.frames,
        frame_steps=meta.frame_steps,
        closes={arm: sum(kp.cue == "close" for kp in ks) for arm, ks in kps.items()},
        path_m=sum(_path_m(t.tcp) for t in demo.arms.values()),
        jerk=max(jerk, default=0.0),
        grasps=sorted(grasps, key=lambda g: (g.frame, g.arm)),
        keypoints={arm: len(ks) for arm, ks in kps.items()},
    )


def rank_episodes(
    features: list[EpisodeFeatures], config: SelectionConfig, mirror: Vec3 | None
) -> list[RankedEpisode]:
    """All candidates of one task, best first. `mirror` is the normal of the plane across which
    the two arms mirror each other (None: grasps compare as they are)."""
    signatures = Counter(f.signature for f in sorted(features, key=lambda f: f.episode))
    modal = signatures.most_common(1)[0][0]
    usual = _usual_grasps([f for f in features if f.signature == modal], mirror)
    path = _median(f.path_m for f in features)
    frames = _median(f.frames for f in features)
    jerk = _median(f.jerk for f in features)
    count = _median(sum(f.keypoints.values()) for f in features)
    scored = []
    for f in features:
        closes = ", ".join(f"{arm} {n}" for arm, n in sorted(f.closes.items()))
        typical = f.signature == modal
        slots = _grasp_slots(f, mirror)
        devs = [_angle_deg(axis, usual[k]) for k, axis in slots.items() if k in usual]
        n = sum(f.keypoints.values())
        criteria = [
            Criterion(
                name="closes",
                cost=0.0 if typical else config.signature_penalty,
                note=f"closes {closes}" + ("" if typical else f", not the usual {modal}"),
            ),
            _log_ratio("path", f.path_m / path, config.path_weight, f"path {f.path_m:.2f} m"),
            _log_ratio("duration", f.frames / frames, config.duration_weight, f"{f.frames} frames"),
            Criterion(
                name="jerk",
                cost=config.jerk_weight * max(0.0, float(np.log(max(f.jerk, 1e-300) / jerk))),
                note=f"peak jerk {f.jerk / jerk:.2f}x the task median",
            ),
            Criterion(
                name="grasp",
                cost=config.grasp_weight * max(devs, default=0.0),
                note=f"grasp approach {max(devs):.1f} deg from the usual" if devs else "no grasp",
            ),
            Criterion(
                name="keypoints",
                cost=config.keypoints_weight * abs(n - count) / count,
                note=f"{n} keypoints (task median {count:g})",
            ),
        ]
        scored.append((sum(c.cost for c in criteria), f, criteria))
    scored.sort(key=lambda s: (s[0], s[1].episode))
    return [
        RankedEpisode(episode=f.episode, seed=f.seed, rank=r, score=score, criteria=criteria)
        for r, (score, f, criteria) in enumerate(scored, start=1)
    ]


def margin(ranked: list[RankedEpisode]) -> float | None:
    """How much worse the runner-up scored than the pick (None without a runner-up)."""
    return ranked[1].score - ranked[0].score if len(ranked) > 1 else None


def review_notes(
    ranked: list[RankedEpisode],
    features: list[EpisodeFeatures],
    config: SelectionConfig,
    chosen: int,
) -> list[str]:
    """What a reviewer should look at about the `chosen` episode: other per-arm close counts in
    at least `variant_share` of the episodes (one demonstration shows one of the task's
    strategies), a chosen episode that still costs `review_score`, and, for the automatic pick, a
    runner-up within `close_margin` that differs in its close counts or keypoint count. Near-ties
    between equally usual candidates are not worth a note."""
    by = {f.episode: f for f in features}
    pick = by[chosen]
    ranking = next(r for r in ranked if r.episode == chosen)
    counts = Counter(f.signature for f in features).most_common()
    other = [(sig, n) for sig, n in counts if sig != pick.signature]
    others = sum(n for _, n in other)
    notes = []
    if others >= config.variant_share * len(features):
        variants = ", ".join(f"{sig} in {n}" for sig, n in other)
        notes.append(
            f"{others} of {len(features)} episodes close differently per arm ({variants}); "
            f"episode {chosen} closes {pick.signature}"
        )
    if ranking.score >= config.review_score:
        costly = "; ".join(c.note for c in ranking.criteria if c.cost >= 0.05)
        notes.append(f"episode {chosen} (rank {ranking.rank}) costs {ranking.score:.2f}: {costly}")
    close = len(ranked) > 1 and ranked[1].score - ranking.score < config.close_margin
    if ranking.rank == 1 and close:
        second = by[ranked[1].episode]
        n, m = sum(pick.keypoints.values()), sum(second.keypoints.values())
        if second.signature != pick.signature or n != m:
            notes.append(
                f"runner-up episode {second.episode} scores within {config.close_margin:g} with "
                f"{m} keypoints and closes {second.signature} (the pick: {n}, {pick.signature})"
            )
    return notes


def _log_ratio(name: str, ratio: float, weight: float, what: str) -> Criterion:
    return Criterion(
        name=name,
        cost=weight * abs(float(np.log(max(ratio, 1e-300)))),
        note=f"{what}, {ratio:.2f}x the task median",
    )


def _median(values: Iterable[float]) -> float:
    # A task whose candidates all measure 0 on something compares them as equal on it.
    return max(float(median(values)), 1e-300)


def _path_m(tcp: Array) -> float:
    return float(np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=1).sum())


def _direction(q: tuple[float, float, float, float], ee_to_tcp: Vec3) -> Vec3:
    x, y, z = quat_rotate(q, pointing_axis(ee_to_tcp)).tolist()
    return x, y, z


def _grasp_slots(f: EpisodeFeatures, mirror: Vec3 | None) -> dict[tuple[int, int], Array]:
    """Grasp approach directions by (arm role, grasp number). Role 0 is the arm that closes first;
    when that is the right arm and a `mirror` plane is given, directions are reflected across
    it, so role 0 always plays the left arm and mirrored episodes compare."""
    if not f.grasps:
        return {}
    first = f.grasps[0].arm
    order = [first, "right" if first == "left" else "left"]
    seen: Counter[str] = Counter()
    slots = {}
    for g in f.grasps:
        d = np.asarray(g.approach)
        if first == "right" and mirror is not None:
            n = np.asarray(mirror) / np.linalg.norm(mirror)
            d = d - 2 * (d @ n) * n
        slots[(order.index(g.arm), seen[g.arm])] = d
        seen[g.arm] += 1
    return slots


def _usual_grasps(
    features: list[EpisodeFeatures], mirror: Vec3 | None
) -> dict[tuple[int, int], Array]:
    """Per grasp slot, the medoid direction (least total angle to the others)."""
    by_slot: dict[tuple[int, int], list[Array]] = {}
    for f in features:
        for key, axis in _grasp_slots(f, mirror).items():
            by_slot.setdefault(key, []).append(axis)
    usual = {}
    for key, axes in by_slot.items():
        a = np.array(axes)
        usual[key] = a[int(np.argmin(np.degrees(np.arccos(np.clip(a @ a.T, -1, 1))).sum(axis=1)))]
    return usual


def _angle_deg(a: Array, b: Array) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
