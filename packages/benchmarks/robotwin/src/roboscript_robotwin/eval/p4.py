"""P4's evaluation blocks on RoboTwin (docs/concepts/learning.md, `p4_eval`): a P3 block of a frozen
Task Memory in one arm of a P4 run, run by the official client and server exactly as P3 runs them.

A unit prepared for P4 (a pipeline unit whose Task Memory is a base run's, frozen) holds its block
in `P3/p4.json` (`P4Block`): the arm, the round whose RSI document the arm reads (none: the
baseline), the official block, its modes, episodes and shards, the Task Memory the base run's P3
read, and with a document the Agent host's method and the document by reference. With a document
the unit keeps the artifact stack its Actor reads in `P3/overlay/stack.json` (the method's
composition: the Task Memory, then the document's parts) and the document's store under
`P3/overlay/store/` (frozen, copied in). Then:

- `p4_launcher`: the P3 launcher config with the arm's Agent host method and, with a document,
  its overlay (`learning.agent_host.OverlayConfig`); the baseline's is P3's as it is, so its
  Actor's prompt is P3's byte for byte;
- `p4_provenance`: what every episode summary records (`HostConfig.provenance`): the code, the
  config `p4:<arm>` and the sha256 of the job's own provenance with the block, the document's
  hash included, so that each episode names its arm and document;
- `check_settings`: the pipeline's P3 settings must be the block's (one block, its episodes and
  shards), so that a unit never evaluates another block than it was prepared for.

The benchmark knows no method: the document is an artifact of any kind, loaded through the
artifact store registered under `P4Block.store` (default: its kind).
"""

import hashlib
import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Self

from roboscript.contracts.artifact import ArtifactRef
from roboscript.contracts.canonical import canonical_json
from roboscript.contracts.provenance import Provenance
from roboscript.memory.artifact import KIND as TASK_MEMORY
from roboscript.settings import source_sha
from roboscript_robotwin.eval.blocks import P3Settings
from roboscript_robotwin.eval.launch import Mode, P3Config

BLOCK_FILE: Final = "P3/p4.json"
"""A prepared unit's block, under the unit's directory (`<task>/<variant>/`)."""
OVERLAY_DIR: Final = "P3/overlay"
STACK_FILE: Final = f"{OVERLAY_DIR}/stack.json"
STORE_DIR: Final = f"{OVERLAY_DIR}/store"


class P4Block(BaseModel):
    """One unit's P4 block: `run` and `stage` name the P4 run and its evaluation (`baseline/block2`,
    say); `arm` the arm; `round` the round whose document it reads (None: the baseline, no
    document); `block` the official seed block (100000 x (1 + block) on); `modes`, `episodes`,
    `shards` as P3's settings; `memory` the Task Memory the base run's P3 read (its manifest's
    version and hash); `method` the Agent host's method (None: the launcher's); `doc` the
    document by reference, loaded from the artifact store registered as `store` (None: as its
    kind); `pins` the directory, on the machine that evaluates the unit, of the expert's verdicts
    on the block's seeds that every arm's unit of the block shares (`P3Run.pins`: whichever arm
    checks a seed first decides, so the arms evaluate the same seeds at the same indices; None:
    each unit pins its own)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    arm: str = Field(pattern=r"^[a-z0-9_-]+$")
    round: PositiveInt | None = None
    block: NonNegativeInt
    modes: tuple[Mode, ...] = Field(min_length=1)
    episodes: int = Field(ge=1, le=100)
    shards: PositiveInt
    memory: ArtifactRef
    method: str | None = None
    doc: ArtifactRef | None = None
    store: str | None = None
    pins: Path | None = None

    @model_validator(mode="after")
    def _arm(self) -> Self:
        if (self.round is None) != (self.doc is None):
            raise ValueError("a block reads a round's document (round and doc), or none")
        if self.memory.kind != TASK_MEMORY:
            raise ValueError(f"memory is a {self.memory.kind}, not the {TASK_MEMORY} P3 read")
        if self.doc is not None and self.doc.kind == TASK_MEMORY:
            raise ValueError("the document read beside the Task Memory is not a Task Memory")
        if self.store is not None and self.doc is None:
            raise ValueError("store names where the document is loaded from: no document")
        if self.pins is not None and not self.pins.is_absolute():
            raise ValueError(f"pins {self.pins}: an absolute directory on the evaluating machine")
        if len(set(self.modes)) < len(self.modes):
            raise ValueError(f"modes repeat: {self.modes}")
        return self


def read_block(unit_dir: Path) -> P4Block | None:
    """The unit's P4 block (None: a unit of no P4 run, P3 as always)."""
    path = unit_dir / BLOCK_FILE
    if not path.is_file():
        return None
    return P4Block.model_validate_json(path.read_bytes())


def write_block(unit_dir: Path, block: P4Block) -> None:
    path = unit_dir / BLOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(block.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")


def check_settings(p3: P3Settings, block: P4Block) -> None:
    """Raise ValueError unless the pipeline's P3 settings evaluate the block: its one block, its
    episodes (each of its modes) and its shards."""
    wrong = []
    if p3.blocks != [block.block]:
        wrong.append(f"blocks {p3.blocks} (the unit's: [{block.block}])")
    for mode in block.modes:
        if p3.episodes_of(mode) != block.episodes:
            wrong.append(f"{mode} episodes {p3.episodes_of(mode)} ({block.episodes})")
    if p3.shards != block.shards:
        wrong.append(f"shards {p3.shards} ({block.shards})")
    if wrong:
        raise ValueError(f"P4 {block.stage}: the P3 settings differ from the unit's block: {wrong}")


def p4_launcher(launcher: P3Config, unit_dir: Path, block: P4Block) -> P3Config:
    """`launcher` for the block: the arm's Agent host method, and with a document the overlay of
    the unit's stack and store; the baseline's (no method, no document) is `launcher` itself."""
    host = dict(launcher.host)
    if block.method is not None:
        host["method"] = block.method
    if block.doc is not None:
        stack, store = unit_dir / STACK_FILE, unit_dir / STORE_DIR
        if not stack.is_file() or not store.is_dir():
            raise ValueError(
                f"{unit_dir}: P4 {block.stage} reads a document: no {stack} or {store}"
            )
        name = block.store or block.doc.kind
        host["overlay"] = {
            "stack": str(stack),
            "stores": {block.doc.kind: {"name": name, "options": {"root": str(store)}}},
        }
    if host == launcher.host:
        return launcher
    return P3Config.model_validate({**launcher.model_dump(mode="json"), "host": host})


def p4_provenance(job: Provenance | None, block: P4Block) -> Provenance:
    """What the block's episodes record: the code, `p4:<arm>`, and the sha256 of the job's own
    provenance (None: the code alone) with the block (its document's hash included)."""
    base = job or Provenance(source=source_sha())
    content = {"job": base.model_dump(mode="json"), "p4": block.model_dump(mode="json")}
    digest = hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()
    return Provenance(source=base.source, config=f"p4:{block.arm}", config_hash=digest)
