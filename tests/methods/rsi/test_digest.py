"""The per-episode digest the reflection reads first (docs/concepts/learning.md): infrastructure
flags, and that turn pages and the diagnosis are never read."""

from pathlib import Path

from roboscript.learning.eval_view import DIAGNOSIS, read_eval_episode
from roboscript_methods.rsi.digest import (
    digest_episode,
    headline,
    render_digest,
    render_turns,
)
from roboscript_methods.rsi.recipe import load_rsi_recipe
from tests.methods.rsi.records import (
    Episode,
    Turn,
    episodes,
    failed_episode,
    made,
)

RULES = load_rsi_recipe(Path("recipes/rsi")).reflect

HEAD = (
    "# block1/lift_red/ep003 seed 200003 [baseline]: FAILURE; end finish, 5 turns, 50 steps, "
    "1 hard failures, false finish\n"
    "instruction: move the part\n"
    "instruction names no arm\n"
    "ended: finish after 5 turns (5 model calls of 60); the Agent's actions used 50 steps, "
    "50 of 400 official steps\n"
    "end (each arm's last reported gripper): no arm reported\n"
    "false finish: the done check passed 1 times; the episode ended by finish and the success "
    "never latched\n"
)


def test_infrastructure_flags(tmp_path: Path) -> None:
    """The last tool call failed for infrastructure reasons and the Agent finished:
    flagged, never a strategy failure. A tool answering after the error clears it. The host's
    own flags (infra, a voided end) are shown as they are."""
    infra = Turn([made("place_object", "Error (infra_error): planner down", "infra_error")])
    finish = Turn([made("finish", note="giving up")])
    answered = Turn([made("look")])
    specs = [
        Episode("lift_red", 0, False, [Turn([made("look")]), infra, finish]),
        Episode("lift_red", 1, False, [Turn([made("look")]), infra, answered, finish]),
        Episode(
            "lift_red", 2, False, [Turn([made("look")]), infra], end="max_turns", voided="finish"
        ),
    ]
    found, keys, _ = episodes(tmp_path, specs)
    digests = [digest_episode(e, k, "baseline", RULES) for e, k in zip(found, keys, strict=True)]
    assert digests[0].infra == (
        "infra: the last tool call (place_object, turn 1) failed for infrastructure reasons and "
        "the episode was then finished: not a strategy failure",
    )
    assert "INFRA" in headline(digests[0])
    assert digests[1].infra == ()
    assert digests[2].infra == ("the host voided its end (finish) after an unanswered infra error",)
    group = digests[0].failures[0]
    assert (group.tool, group.kind, group.reason, group.turns) == (
        "place_object",
        "error",
        "infra_error",
        (1,),
    )
    assert group.detail == "planner down"


def test_turn_pages_and_the_diagnosis_never_read(tmp_path: Path) -> None:
    (episode,), (key,), _ = episodes(tmp_path, [failed_episode("lift_red", 3)])
    assert episode.directory is not None
    (episode.directory / DIAGNOSIS).write_text('{"miss": "SECRET"}', encoding="utf-8")
    privileged = read_eval_episode(episode.directory, episode.seed)
    assert privileged is not None and any(p.kind == "diagnosis" for p in privileged.parts)
    digest = digest_episode(privileged, key, "baseline", RULES)
    assert "SECRET" not in render_digest(digest, RULES)
    page, stopped = render_turns(privileged, 1, 2, 10_000)
    assert stopped == 3 and "SECRET" not in page
    assert page.startswith("## turn 1\nobservation (step 10):\nturn 1: the scene\n")
    assert '-> grasp_object({"actor": "part"})' in page
    assert "<- error" not in page and "Receipt: empty_grasp (hard failure)" in page
    short, stopped = render_turns(privileged, 0, 4, 200)
    assert short.startswith("## turn 0") and stopped == 1  # cut at a turn boundary
