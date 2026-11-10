"""Seed splits and their guard (docs/concepts/learning.md, docs/concepts/operations.md).

A benchmark declares its evaluation blocks; the env binding declares demonstration, learning and
internal evaluation blocks. No two blocks overlap.

P4 reflects in rounds, and a split names them as a chain (`reflects`, in round order, which the
evaluation blocks' starts must follow): round r reads the records of its blocks R_r and is
evaluated on block E_r (`reflects[E_r] = R_r`); the blocks in `final` are test blocks no
reflection ever reads. Every round evaluates on a block of its own, never a final one, and no
round up to r reads E_r (the document of round r carries what every round up to r read); a later
round may (round 2 reads block 2, round 1's evaluation). The one exception: E_r may be read (or
overlapped) by a round up to r only when the run says `transductive` (reflecting on the evaluated
block: an upper bound, never called generalization). `extend_chain` appends rounds on the
next unused blocks of a benchmark's order, so earlier rounds and their final blocks stay as they
were (three rounds 1->2, 2->3, 3->4, final 5; two more: 4->6, 6->7, final 8, and block 5 stays a
test point of round 3's document).

The `seed_split` precheck (`learning.prechecks`) checks the run's blocks with the benchmark's
before a run starts; `check_reflection` checks, as a reflection starts and as its evaluation runs,
that the seeds it actually reads and evaluates lie in the blocks the split names, so that records
from elsewhere cannot leak into an evaluation (P4, `roboscript_methods.rsi`). The learning
blocks' numbers are config defaults of the protocols (`practice.PracticeConfig.first_seed`, ...),
which the precheck holds to their blocks.
"""

from collections.abc import Collection, Iterable

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self


class SeedRange(BaseModel):
    """Seeds `start` .. `start + count - 1`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: NonNegativeInt
    count: PositiveInt

    @property
    def stop(self) -> int:
        return self.start + self.count

    def overlaps(self, other: "SeedRange") -> bool:
        return self.start < other.stop and other.start < self.stop

    def holds(self, seed: int) -> bool:
        return self.start <= seed < self.stop


class SeedSplit(BaseModel):
    """Named seed blocks. `reflects`: the reflection chain in round order, each round's evaluation
    block -> the blocks whose records it read (P4: `p4_eval` -> `p3`; one name stands for a
    one-block tuple); `final`: test blocks no reflection reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocks: dict[str, SeedRange]
    reflects: dict[str, tuple[str, ...]] = {}
    final: tuple[str, ...] = ()
    transductive: bool = False

    @field_validator("reflects", mode="before")
    @classmethod
    def _one_block(cls, value: object) -> object:
        if isinstance(value, dict):
            return {k: (v,) if isinstance(v, str) else v for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _chain(self) -> Self:
        named = set(self.reflects) | {b for reads in self.reflects.values() for b in reads}
        if unknown := sorted((named | set(self.final)) - set(self.blocks)):
            raise ValueError(f"reflects or final names unknown blocks {unknown}")
        if empty := sorted(e for e, reads in self.reflects.items() if not reads):
            raise ValueError(f"evaluation blocks {empty} reflect on no block")
        starts = [self.blocks[e].start for e in self.reflects]
        if starts != sorted(starts):
            raise ValueError(
                f"rounds {list(self.reflects)} are not in the order of their evaluation blocks"
            )
        if len(set(self.final)) != len(self.final):
            raise ValueError(f"final names a block twice: {list(self.final)}")
        if both := sorted(set(self.final) & set(self.reflects)):
            raise ValueError(f"final blocks {both} are also a round's evaluation block")
        if read := sorted({b for reads in self.reflects.values() for b in reads} & set(self.final)):
            raise ValueError(f"final blocks {read} are read by a reflection")
        if not self.transductive:
            for evaluation in self.reflects:
                if evaluation in self._read_up_to(evaluation):
                    raise ValueError(
                        f"seed block {evaluation!r} is read by its own or an earlier round "
                        "(a reflection's evaluation; set transductive for scheme C)"
                    )
        names = sorted(self.blocks)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if not self.blocks[a].overlaps(self.blocks[b]):
                    continue
                reflected = self._related(a, b) or self._related(b, a)
                if reflected and self.transductive:
                    continue
                why = " (a reflection's evaluation; set transductive for scheme C)"
                why = why if reflected else ""
                raise ValueError(f"seed blocks {a!r} and {b!r} overlap{why}")
        return self

    def _read_up_to(self, evaluation: str) -> set[str]:
        """The blocks the rounds up to and including the one evaluated on `evaluation` read."""
        read: set[str] = set()
        for name, reads in self.reflects.items():
            read |= set(reads)
            if name == evaluation:
                break
        return read

    def _related(self, evaluation: str, read: str) -> bool:
        return evaluation in self.reflects and read in self._read_up_to(evaluation)


def extend_chain(
    split: SeedSplit,
    rounds: int,
    candidates: Iterable[tuple[str, SeedRange]],
    *,
    reads: Collection[str] = (),
    final: bool = True,
) -> SeedSplit:
    """`split` with `rounds` more rounds, each on the next of `candidates` (a benchmark's blocks
    in order, name and range) the split does not name yet, and then a new final block when
    `final`. The first new round reads `reads`, by default the last round's evaluation block; each
    later one reads the one before it. Earlier rounds and final blocks stay as they were."""
    if rounds < 1:
        raise ValueError(f"extend a chain by at least one round, not {rounds}")
    source = tuple(reads) or tuple(split.reflects)[-1:]
    if not source:
        raise ValueError("a chain without rounds starts from the blocks its first round reads")
    blocks, reflects = dict(split.blocks), dict(split.reflects)
    fresh = ((name, block) for name, block in candidates if name not in blocks)
    picked = []
    for _ in range(rounds + int(final)):
        try:
            name, block = next(fresh)
        except StopIteration:
            raise ValueError(f"the candidates ran out after blocks {picked}") from None
        blocks[name] = block
        picked.append(name)
    for name in picked[:rounds]:
        reflects[name] = source
        source = (name,)
    added = tuple(picked[rounds:])
    return SeedSplit(
        blocks=blocks,
        reflects=reflects,
        final=split.final + added,
        transductive=split.transductive,
    )


def check_reflection(
    split: SeedSplit,
    evaluation: str,
    reflected: Collection[int],
    evaluated: Collection[int] = (),
) -> None:
    """Raise ValueError unless `split` names the blocks the evaluation block `evaluation` reflects
    on (`SeedSplit.reflects`), every seed the reflection read (`reflected`) lies in one of them
    and every seed the evaluation runs (`evaluated`) lies in `evaluation`. The blocks were checked
    when the split was made (overlap and reading an evaluation block allowed only
    `transductive`), so seeds in them are never both reflected on and evaluated otherwise."""
    if evaluation not in split.reflects:
        raise ValueError(
            f"seed block {evaluation!r} reflects on no block: name it in the split's `reflects`"
        )
    names = split.reflects[evaluation]
    for blocks, seeds in ((names, reflected), ((evaluation,), evaluated)):
        ranges = [split.blocks[name] for name in blocks]
        if outside := sorted({s for s in seeds if not any(r.holds(s) for r in ranges)}):
            shown = ", ".join(
                f"{n!r} ({r.start}..{r.stop - 1})" for n, r in zip(blocks, ranges, strict=True)
            )
            plural = "s" if len(blocks) > 1 else ""
            raise ValueError(f"seeds {outside} lie outside seed block{plural} {shown}")
