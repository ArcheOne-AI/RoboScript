"""Install P3's policy into a RoboTwin checkout, reversibly
(docs/concepts/embodied-core.md): symbolic links only, recorded in a manifest in this
repository (`.roboscript/robotwin-p3.json`); no RoboTwin file is modified and nothing else is
written there.

- `ROBOTWIN_ROOT/policy/RoboScript` -> `roboscript_robotwin/eval/policy` (the official policy
  package)
- `ROBOTWIN_ROOT/task_config/rs_agentic_{clean,randomized}_p3.yml` -> our P3 task configs

`install` first checks the checkout (`check_robotwin`): its commit, the official scripts P3 relies
on, and no uncommitted change to a tracked file. Installing again is a no-op; installing into
another checkout while one is recorded is refused; `uninstall` removes exactly the recorded links.
`check_installed` repeats the checks before each evaluation, against what was recorded.

```python
install(robotwin_root())
uninstall(robotwin_root())
```
"""

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from roboscript.records import write_atomic
from roboscript.settings import repo_root
from roboscript_robotwin.benchmark import HERE, P3_TASK_CONFIGS

POLICY: Final = "RoboScript"
"""The policy's name in RoboTwin: its directory under `policy/` and `policy_name`."""


class Expected(BaseModel):
    """The RoboTwin commit and the sha256 of the official files that P3 relies on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str
    files: dict[str, str]


C364C8E: Final = Expected(
    commit="c364c8ecfc59b8b4920eb23679cf26b59c52fb86",
    files={
        # The episode loop, ModelClient (30 s socket timeout), the frames, the result file.
        "script/eval_policy_client.py": (
            "8a41acbfa5011a7d120aec719463c6c100bd89aef7d3cdf722153a7b6fa590d3"
        ),
        # The server calling the model object's methods by name, `method(obs)` or `method()`.
        "script/policy_model_server.py": (
            "50187c3c1d82145c9ef8f3f15b0177eddea502c7cdecd004c83cf6b6bb6b1c8c"
        ),
    },
)


class Link(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path  # in the RoboTwin checkout
    target: Path


class InstallRecord(BaseModel):
    """What `install` checked (the commit and the files' digests) and the links it made."""

    model_config = ConfigDict(frozen=True, extra="forbid", serialize_by_alias=True)

    schema_: Literal["rs.robotwin.p3-install/1"] = Field(
        default="rs.robotwin.p3-install/1", alias="schema"
    )
    root: Path
    commit: str
    files: dict[str, str]
    links: list[Link]
    installed_at: str


def record_path() -> Path:
    return repo_root() / ".roboscript" / "robotwin-p3.json"


def links(root: Path) -> list[Link]:
    """The links P3 needs in the checkout at `root`."""
    configs = [
        Link(path=root / "task_config" / config.name, target=config)
        for config in P3_TASK_CONFIGS.values()
    ]
    return [Link(path=root / "policy" / POLICY, target=HERE / "eval" / "policy"), *configs]


def check_robotwin(root: Path, expected: Expected = C364C8E) -> None:
    """Refuse a checkout whose commit or relied-on files differ from `expected`, or that has
    uncommitted changes to tracked files (our links are untracked)."""
    commit = _git(root, "rev-parse", "HEAD")
    if commit != expected.commit:
        raise ValueError(f"{root} is at {commit}, not RoboTwin {expected.commit}")
    for name, digest in expected.files.items():
        found = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if found != digest:
            raise ValueError(f"{root / name} changed (sha256 {found[:12]}, expected {digest[:12]})")
    # --no-optional-locks: runs in parallel read the checkout without writing its index
    if changed := _git(
        root, "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError(f"{root} has uncommitted changes to tracked files:\n{changed}")


def check_installed(root: Path, *, record: Path | None = None) -> InstallRecord:
    """The install record of the checkout at `root`, checked again: its links in place and the
    checkout as `install` found it (`check_robotwin` against the recorded commit and files)."""
    path = record or record_path()
    if not path.is_file():
        raise RuntimeError(
            f"P3 is not installed: call roboscript_robotwin.eval.install.install({root})"
        )
    done = InstallRecord.model_validate_json(path.read_bytes())
    if done.root != root.resolve():
        raise RuntimeError(f"P3 is installed in {done.root}, not {root}")
    if not installed(root):
        raise RuntimeError(f"P3's links in {root} are missing or changed: install again")
    check_robotwin(done.root, Expected(commit=done.commit, files=done.files))
    return done


def installed(root: Path) -> bool:
    """Whether every link is in place and points to this checkout's files."""
    return all(_ours(link) for link in links(root))


def install(
    root: Path, *, expected: Expected = C364C8E, record: Path | None = None
) -> InstallRecord:
    """Check the checkout and add the links (those in place already stay); a path in the way that
    is not our link, or an install recorded for another checkout, is an error, and nothing is
    added then."""
    root = root.resolve()
    path = record or record_path()
    if path.is_file():
        other = InstallRecord.model_validate_json(path.read_bytes())
        if other.root != root:
            raise FileExistsError(f"P3 is installed in {other.root}: uninstall it there first")
    check_robotwin(root, expected)
    wanted = links(root)
    if blocked := [str(link.path) for link in wanted if _exists(link.path) and not _ours(link)]:
        raise FileExistsError(f"not our links, left alone: {blocked}; uninstall or remove them")
    for link in wanted:
        if not _ours(link):
            link.path.symlink_to(link.target)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    done = InstallRecord(
        root=root, commit=expected.commit, files=expected.files, links=wanted, installed_at=stamp
    )
    write_atomic(path, (done.model_dump_json(indent=2) + "\n").encode())
    return done


def uninstall(root: Path, *, record: Path | None = None) -> list[Path]:
    """Remove the links the record lists that still point where they did, then the record; the
    paths removed. A recorded path that is now something else is an error, and stays."""
    path = record or record_path()
    if not path.is_file():
        return []
    done = InstallRecord.model_validate_json(path.read_bytes())
    if done.root != root.resolve():
        raise ValueError(f"the install record is for {done.root}, not {root}")
    if changed := [str(link.path) for link in done.links if _exists(link.path) and not _ours(link)]:
        raise ValueError(f"no longer our links, left alone: {changed}")
    removed = [link.path for link in done.links if _exists(link.path)]
    for gone in removed:
        gone.unlink()
    path.unlink()
    return removed


def _ours(link: Link) -> bool:
    return link.path.is_symlink() and Path(os.readlink(link.path)) == link.target


def _exists(path: Path) -> bool:
    return path.is_symlink() or path.exists()


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise ValueError(f"{root}: git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()
