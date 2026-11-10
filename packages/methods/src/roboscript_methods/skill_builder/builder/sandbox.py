"""The sandbox for the processes that execute a builder Agent's code (README, "The gate"): the
gate's case runs and the sweeps import the written skills, and everything they start (the
benchmark's coverage script, the simulator it spawns) runs inside it.

`wrap(argv, write=..., read=..., env=..., ...)` is a command line. With `mode="bwrap"` (the
default), inside it:

- the root file system is read-only, with `hidden` (by default `HIDDEN`: every home, `/data`,
  `/tmp`, ...) replaced by empty tmpfs; the repository, the base interpreter and the `read` roots
  are bound back read-only, each one's `.env*` files and the repository's `configs/` and `.dev`
  masked; what a symbolic link near the top of a read root points to is read too (`linked`: a
  benchmark checkout whose assets link to another disk); a conda env's own roots (itself and its
  editable installs' checkouts) are `env_reads`, for the caller to pass as `read`;
- only the `write` directories are writable, and as private tmpfs (written, never kept) the
  `scratch` ones, `/tmp` (`TMPDIR`, whatever `hidden` says) and `HOME` (the caller's home path,
  empty but for a copy of each `home_seed` directory made as the sandbox starts: caches a
  simulator would otherwise rebuild; the originals are never written); the `seeded` directories
  are private copies of themselves the same way (what is kept across runs and writable holds
  data, never code);
- `/dev` is minimal plus the GPU nodes (`/dev/nvidia*`, `/dev/dri`); with `network` the network
  is the host's (a case run reaches its GPU's motion service on host loopback), else none
  (`--unshare-net`: the validator); the PID namespace is its own, so no other process (or its
  environment) is visible;
- the environment is `env` plus `HOME`, `TMPDIR` and `RS_ENV_FILE` (an empty file in the
  sandbox's `/tmp`), nothing inherited; a key that looks like a credential is refused. The
  repository's `.env` is masked, so reading it fails (EACCES where the kernel refuses the masked
  path); `RS_ENV_FILE` points the settings loader (`roboscript.settings.EnvSettings`) at an empty
  file instead, so code that reads settings inside works and sees no secret.

Linux only, bubblewrap without overlays (other caches under `HOME` start cold).

`mode="none"` is for debugging only: `argv` runs on the host with the same environment and no
isolation at all. A run that uses it says so (`record_mode` in its `run.json`, `mark_summaries` on
its `summary.md` files), and what it built is never frozen (`skill_builder.freeze`).
"""

import ast
import fnmatch
import json
import os
import shlex
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, get_args

SandboxMode = Literal["bwrap", "none"]
"""`bwrap`: bubblewrap (the default); `none`: no sandbox, debugging only."""
MODES: Final = get_args(SandboxMode)
REPO: Final = Path(__file__).resolve().parents[6]
"""The repository this file is in."""
HIDDEN: Final = ("/home", "/root", "/data", "/mnt", "/media", "/srv", "/tmp", "/var/tmp",
                 "/run/user")  # fmt: skip
"""Replaced by an empty tmpfs by default: user data, other users' files, sockets of user
sessions."""
MASKED: Final = (".env*",)
"""Masked in every readable root (secrets: the gateway token, upstream API keys)."""
MASKED_REPO: Final = ("configs", ".dev")
"""Masked in the repository: deployment configs, developer notes."""
CREDENTIAL: Final = ("*TOKEN*", "*SECRET*", "*KEY*", "*PASSWORD*", "*CREDENTIAL*")
GPU: Final = ("/dev/nvidia*", "/dev/dri")
LINK_DEPTH: Final = 3
"""How deep `linked` looks for symbolic links out of a read root."""
SEED: Final = "/tmp/.rs-seed"
"""Where the `home_seed` and `seeded` directories are bound read-only inside, for the copy."""
ENV_FILE: Final = "/tmp/rs-sandbox.env"
"""The empty settings file inside (`RS_ENV_FILE`)."""
PATH: Final = "/usr/local/bin:/usr/bin:/bin"
MARK: Final = "UNSANDBOXED"
"""What a summary of an unsandboxed run starts with."""
UNSANDBOXED: Final = (
    f"{MARK}: this run executed Agent-written code without a sandbox (debugging only); "
    "its skills are never frozen."
)
"""The first line of every `summary.md` of a run in mode `none` (`mark_summaries`)."""
RUN_FILE: Final = "run.json"
SUMMARY_FILE: Final = "summary.md"


def env_reads(python: Path) -> list[Path]:
    """What `python`'s env needs readable: the env, and the source checkout of each editable
    install in it (`__editable___*_finder.py`'s `MAPPING`), each as its git root (a package that
    reads files beside its sources, or its version from git)."""
    env = python.absolute().parent.parent
    found = [env]
    for finder in sorted(env.glob("lib/python3*/site-packages/__editable___*_finder.py")):
        tree = ast.parse(finder.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
                continue
            if node.target.id == "MAPPING" and node.value is not None:
                for source in ast.literal_eval(node.value).values():
                    found += git_root(Path(source))
    return list(dict.fromkeys(found))


def git_root(path: Path) -> list[Path]:
    """`path`'s checkout, and for a submodule its git directory in the superproject (versions
    are read from git: setuptools-scm)."""
    for parent in [path, *path.parents]:
        link = parent / ".git"
        if link.is_file() and (text := link.read_text(encoding="utf-8")).startswith("gitdir:"):
            return [parent, (parent / text.removeprefix("gitdir:").strip()).resolve()]
        if link.exists():
            return [parent]
    return [path]


def linked(
    roots: list[Path], depth: int = LINK_DEPTH, hidden: Sequence[str] = HIDDEN
) -> list[Path]:
    """Targets of the symbolic links within `depth` levels of `roots` that `hidden` would hide
    and no root holds (a checkout's assets link to another disk)."""
    resolved = [r.resolve() for r in roots]
    found: list[Path] = []
    level = [Path(r) for r in roots]
    for _ in range(depth):
        below = []
        for directory in level:
            try:
                entries = list(os.scandir(directory))
            except OSError:  # unreadable to this user: nothing to bind from it
                continue
            for entry in entries:
                if any(fnmatch.fnmatch(entry.name, m) for m in MASKED):
                    continue  # a secret: masked, never followed
                if entry.is_symlink():
                    target = Path(entry.path).resolve()
                    unseen = any(target.is_relative_to(h) for h in hidden)
                    held = any(target.is_relative_to(r) for r in [*resolved, *found])
                    if target.exists() and unseen and not held:
                        found.append(target)
                elif entry.is_dir():
                    below.append(Path(entry.path))
        level = below
    return found


def masked(roots: list[Path], repo: Path = REPO, hide: tuple[Path, ...] = ()) -> list[Path]:
    """The paths to mask inside `roots`: `MASKED` in each root, `MASKED_REPO` in `repo`, and
    `hide`, each as its target, kept only where that target is inside a root (one outside is not
    mounted, and bubblewrap cannot mask a path it does not see)."""
    resolved = [r.resolve() for r in roots]
    found = [p for r in roots for pattern in MASKED for p in Path(r).glob(pattern)]
    found += [repo / d for d in MASKED_REPO] + list(hide)
    targets = [p.resolve() for p in found]
    return [
        p for p in dict.fromkeys(targets)
        if p.exists() and any(p.is_relative_to(r) for r in resolved)
    ]  # fmt: skip


def wrap(
    argv: list[str],
    *,
    write: list[Path],
    read: list[Path],
    env: dict[str, str],
    scratch: Sequence[Path] = (),
    seeded: Sequence[Path] = (),
    home_seed: Sequence[str] = (),
    network: bool = False,
    hidden: Sequence[str] = HIDDEN,
    repo: Path = REPO,
    mode: SandboxMode = "bwrap",
) -> list[str]:
    """`argv` run in the sandbox (see the module): `write` writable, `read` (beside `repo` and the
    base interpreter) read-only, `scratch` writable but private and not kept, `seeded` the same
    but starting as a copy of what is there, `home_seed` (paths relative to the caller's home)
    copied into the empty `HOME`, `hidden` replaced by empty directories, the host's network only
    with `network`, the environment `env` plus `HOME`, `TMPDIR` and `RS_ENV_FILE`. With `mode`
    `none`: `argv` on the host with the environment `env` (debugging only)."""
    if mode not in MODES:
        raise ValueError(f"sandbox mode {mode!r} is not one of {list(MODES)}")
    refused = [k for k in env if any(fnmatch.fnmatch(k.upper(), c) for c in CREDENTIAL)]
    if refused:
        raise ValueError(f"credentials do not enter the sandbox: {refused}")
    missing = [p for p in [*write, *read, *scratch, *seeded] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"sandbox roots that do not exist: {missing}")
    if mode == "none":
        inside = {"PATH": PATH} | env | {"HOME": str(Path.home()), "PYTHONDONTWRITEBYTECODE": "1"}
        return ["/usr/bin/env", "-i", *[f"{k}={v}" for k, v in inside.items()], *argv]
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise FileNotFoundError(
            "bwrap (bubblewrap) is not installed: install it (Linux), or debug with sandbox none"
        )
    reads = [repo, Path(sys.base_prefix), *read]
    reads += linked(reads, hidden=hidden)
    command = [bwrap, "--ro-bind", "/", "/"]
    for hide in dict.fromkeys([*hidden, "/tmp"]):  # /tmp always: TMPDIR, the seeds' mounts
        if Path(hide).is_dir():
            command += ["--tmpfs", hide]
    command += ["--proc", "/proc", "--dev", "/dev"]
    for pattern in GPU:
        for node in sorted(Path("/").glob(pattern.lstrip("/"))):
            command += ["--dev-bind", str(node), str(node)]
    for root in dict.fromkeys(p.absolute() for p in reads):
        # at its real path, which links inside the sandbox resolve to, and at the path given
        for at in dict.fromkeys([root.resolve(), root]):
            command += ["--ro-bind", str(root.resolve()), str(at)]
    for path in masked(reads, repo):
        command += (
            ["--tmpfs", str(path)] if path.is_dir() else ["--ro-bind", "/dev/null", str(path)]
        )
    home = Path.home()
    copies = [((home / c).resolve(), f"{SEED}/home/{k}", f"$HOME/{shlex.quote(c)}")
              for k, c in enumerate(home_seed) if (home / c).is_dir()]  # fmt: skip
    copies += [(p.resolve(), f"{SEED}/{k}", shlex.quote(str(p.absolute())))
               for k, p in enumerate(seeded)]  # fmt: skip
    for source, at, _ in copies:
        command += ["--ro-bind", str(source), at]
    for root in dict.fromkeys(p.absolute() for p in write):
        command += ["--bind", str(root.resolve()), str(root)]
    for root in dict.fromkeys(p.absolute() for p in [*scratch, *seeded]):
        command += ["--tmpfs", str(root)]
    command += ["--dir", str(home), "--unshare-pid", "--die-with-parent", "--new-session"]
    if not network:
        command += ["--unshare-net"]
    command += ["--clearenv"]
    inside = {"PATH": PATH} | env
    inside |= {"HOME": str(home), "TMPDIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1",
               "RS_ENV_FILE": ENV_FILE}  # fmt: skip
    for key, value in inside.items():
        command += ["--setenv", key, value]
    steps = [f": > {ENV_FILE}"]  # the settings loader needs a regular file
    steps += [f"mkdir -p {target} && cp -a {at}/. {target}/" for _, at, target in copies]
    script = " && ".join([*steps, 'exec "$@"'])
    return [*command, "--", "/bin/sh", "-c", script, "sh", *argv]


def record_mode(run: Path, mode: SandboxMode) -> None:
    """Write `mode` into `<run>/run.json` as `"sandbox"`; once a run has been `none`, it stays
    `none` (a resumed run's skills may hold unsandboxed work). Call it after the driver wrote
    `run.json` (`Driver.start`) and before the rounds run."""
    if mode not in MODES:
        raise ValueError(f"sandbox mode {mode!r} is not one of {list(MODES)}")
    path = run / RUN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: start the driver first (Driver.start)")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("sandbox") == "none":
        mode = "none"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta | {"sandbox": mode}, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_mode(run: Path) -> SandboxMode | None:
    """The sandbox mode `<run>/run.json` records, None when it records none."""
    path = run / RUN_FILE
    if not path.is_file():
        return None
    found = json.loads(path.read_text(encoding="utf-8")).get("sandbox")
    if found is not None and found not in MODES:
        raise ValueError(f"{path}: sandbox {found!r} is not one of {list(MODES)}")
    return found  # type: ignore[no-any-return]


def mark_summaries(run: Path, mode: SandboxMode) -> list[Path]:
    """With `mode` `none`, put `UNSANDBOXED` as the first line of every `summary.md` under `run`
    that lacks it (call it whenever the driver has written summaries: after each round and at the
    end); the files changed. With `bwrap`, nothing."""
    if mode == "bwrap":
        return []
    changed = []
    for path in sorted(run.rglob(SUMMARY_FILE)):
        text = path.read_text(encoding="utf-8")
        if not text.startswith(MARK):
            path.write_text(f"{UNSANDBOXED}\n\n{text}", encoding="utf-8")
            changed.append(path)
    return changed


def unsandboxed(run: Path) -> list[str]:
    """Why `run`'s output is not sandboxed work, one line each (empty: it is): its `run.json`
    records no mode or `none`, or a `summary.md` under it starts with `UNSANDBOXED`."""
    lines = []
    mode = run_mode(run)
    if mode is None:
        lines.append(f"{run / RUN_FILE} records no sandbox mode (`sandbox.record_mode`)")
    elif mode == "none":
        lines.append(f"{run / RUN_FILE} records sandbox none")
    for path in sorted(run.rglob(SUMMARY_FILE)):
        with path.open(encoding="utf-8") as file:
            if file.readline().startswith(MARK):
                lines.append(f"{path} is marked UNSANDBOXED")
    return lines
