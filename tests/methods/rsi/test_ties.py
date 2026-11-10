"""The reflection recipe's texts tied to the code that writes them: the done check's passing
head (wicl's `render_done`) and the start of each of the Hub's finish refusals (the Hub's
`finish` step and the Agent host's latch refusal), so the digest cannot drift from them."""

import inspect
from pathlib import Path

from roboscript.agent.host import LATCH_REFUSAL
from roboscript.contracts.tools import TextPart
from roboscript.hub import steps
from roboscript_methods.rsi.recipe import load_rsi_recipe
from roboscript_methods.wicl.toolsets.done import CheckDoneReport, render_done

RULES = load_rsi_recipe(Path("recipes/rsi")).reflect


def _head(passed: bool) -> str:
    report = CheckDoneReport(passed=passed, failing=[], frame="f", lines=[], remeasured={})
    first = render_done(report)[0]
    assert isinstance(first, TextPart)
    return first.text.split("\n", 1)[0]


def test_done_pass_is_the_head_of_a_passed_done_check() -> None:
    assert RULES.done_tool == "check_done"
    assert _head(True).startswith(RULES.done_pass)
    assert not _head(False).startswith(RULES.done_pass)


def test_finish_refusal_starts_are_the_hubs_texts() -> None:
    source = inspect.getsource(steps.finish)
    starts = RULES.finish_refusals
    assert set(starts) == {
        "done check not called after the last action",
        "done check failed",
        "success not latched",
    }
    assert 'f"' + starts["done check not called after the last action"] + "{check} after" in source
    assert 'f"' + starts["done check failed"] + "{check} did not pass" in source
    assert LATCH_REFUSAL.startswith(starts["success not latched"])
    # no start is a prefix of another's text: each refusal falls under one cause
    texts = ["Call check_done after", "The last check_done did not pass", LATCH_REFUSAL]
    for text in texts:
        assert sum(text.startswith(start) for start in starts.values()) == 1
