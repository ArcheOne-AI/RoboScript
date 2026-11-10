"""RoboTwin's own episode setup, reproduced from `script/eval_policy.py`: the
arguments of `setup_demo`, the expert pre-run that decides whether a seed counts, and the
instruction. RoboTwin's modules are imported from `ROBOTWIN_ROOT`, never copied or patched."""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from roboscript_robotwin.sim.expert import Recorder

log = logging.getLogger(__name__)


def use_robotwin(root: Path) -> None:
    """Import RoboTwin as its scripts do: from its checkout as working directory (assets and
    embodiments are found by relative paths), with `description/utils` importable too."""
    os.chdir(root)
    for path in (root / "description" / "utils", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def make_task(name: str) -> Any:
    """`envs.<name>.<name>()`, as `class_decorator` in eval_policy.py does."""
    return getattr(importlib.import_module(f"envs.{name}"), name)()


def setup_args(root: Path, task: str, task_config: Path, *, eval_mode: bool) -> dict[str, Any]:
    """The keyword arguments of `setup_demo` as eval_policy.py builds them (main(), lines 78-122
    and 153-154; eval_policy(), line 216), read from our task config instead of `task_config/`."""
    args: dict[str, Any] = yaml.safe_load(task_config.read_text(encoding="utf-8"))
    args.update(
        task_name=task,
        task_config=task_config.stem,
        ckpt_setting=None,
        policy_name="roboscript",
        eval_mode=eval_mode,
    )
    configs = root / "task_config"
    embodiments = yaml.safe_load((configs / "_embodiment_config.yml").read_text(encoding="utf-8"))
    cameras = yaml.safe_load((configs / "_camera_config.yml").read_text(encoding="utf-8"))
    head = cameras[args["camera"]["head_camera_type"]]
    args["head_camera_h"], args["head_camera_w"] = head["h"], head["w"]
    kinds = args["embodiment"]
    files = [embodiments[kind]["file_path"] for kind in kinds[:2]]
    if len(kinds) == 1:
        args.update(left_robot_file=files[0], right_robot_file=files[0], dual_arm_embodied=True)
    elif len(kinds) == 3:
        args.update(left_robot_file=files[0], right_robot_file=files[1], embodiment_dis=kinds[2])
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"{task_config}: embodiment lists 1 or 3 items, got {kinds}")
    for side in ("left", "right"):  # relative to the working directory, ROBOTWIN_ROOT
        config = Path(args[f"{side}_robot_file"]) / "config.yml"
        args[f"{side}_embodiment_config"] = yaml.safe_load(config.read_text(encoding="utf-8"))
    return args


def expert_prerun(
    task_env: Any,
    seed: int,
    args: dict[str, Any],
    episode: int,
    recorder: Recorder | None = None,
) -> Any | None:
    """eval_policy.py lines 222-252: the expert's episode info if it solves `seed`, else None.

    Also sets the attributes some success checks read that only `play_once` assigns (e.g.
    `arm_tag` in open_laptop), for the episode set up next on the same task object. The official
    loop counts any exception as "unsolvable"; here only objects that do not settle
    (`UnStableError`) and a failed plan do (RoboTwin's planning asserts that a target pose exists,
    `envs/utils/action.py:76`; and once a plan failed, `grasp_actor` and its kin return no
    actions, which some `play_once` scripts index regardless, so an error raised after
    `plan_success` fell is that failed plan's). Other errors propagate: the host reports them as
    `sim_fault`. A `recorder` traces the expert's API calls during `play_once`.
    """
    from envs.utils.create_actor import UnStableError

    try:
        task_env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **args)
        if recorder is not None:
            recorder.install(task_env)  # after setup_demo, which opens the grippers itself
        info = task_env.play_once()
    except (UnStableError, AssertionError) as error:
        log.info("expert pre-run of seed %d failed: %s: %s", seed, type(error).__name__, error)
        return None
    except Exception as error:
        if task_env.plan_success:
            raise
        log.info(
            "expert pre-run of seed %d failed: a plan failed, then %s: %s",
            seed,
            type(error).__name__,
            error,
        )
        return None
    finally:
        if recorder is not None:
            recorder.uninstall()
        task_env.close_env()
    if not task_env.plan_success:
        log.info("expert pre-run of seed %d failed: a plan failed", seed)
        return None
    if not task_env.check_success():
        log.info("expert pre-run of seed %d failed: its plans ran but it did not succeed", seed)
        return None
    return info


def instruction(task: str, info: Any, kind: Literal["seen", "unseen"]) -> str:
    """eval_policy.py lines 257-259: 100 descriptions generated from the expert's episode info,
    then one drawn with numpy's generator, which `setup_demo` seeded."""
    from generate_episode_instructions import generate_episode_descriptions

    descriptions = generate_episode_descriptions(task, [info["info"]], 100)
    return str(np.random.choice(descriptions[0][kind]))
