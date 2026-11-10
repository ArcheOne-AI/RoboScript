"""roboscript.kernel.compose: deep merges and dotted overrides."""

import copy

import pytest

from roboscript.kernel import compose


def test_patches_merge_deeply_in_order_without_touching_the_inputs() -> None:
    base = {
        "gateway": {"listen": "127.0.0.1:18790", "cooldown_s": 30, "endpoints": {"ep1": {"u": 1}}},
        "cameras": ["head", "left"],
        "scalar": 1,
    }
    first = {"gateway": {"cooldown_s": 60, "endpoints": {"ep2": {"u": 2}}}, "cameras": ["head"]}
    second = {"gateway": {"listen": None}, "scalar": {"now": "a mapping"}}
    saved = copy.deepcopy([base, first, second])
    result = compose(base, first, second)
    assert result == {
        "gateway": {
            "listen": None,
            "cooldown_s": 60,
            "endpoints": {"ep1": {"u": 1}, "ep2": {"u": 2}},
        },
        "cameras": ["head"],
        "scalar": {"now": "a mapping"},
    }
    result["gateway"]["endpoints"]["ep2"]["u"] = 99
    result["scalar"]["now"] = "changed"
    assert [base, first, second] == saved


def test_overrides_set_dotted_paths_to_yaml_values() -> None:
    overrides = [
        "gateway.cooldown_s=60",
        "gateway.stream=true",
        "gateway.state_file=null",
        "models.default=some-model",
        "sensing.cameras=[overhead, wrist_left]",
        "planner.tolerance_m=0.005",
        "run.tag='007'",
        "run.note=",
        "new.nested.key=on the fly",
    ]
    result = compose({"gateway": {"cooldown_s": 30, "listen": "x"}}, overrides=overrides)
    assert result == {
        "gateway": {"cooldown_s": 60, "listen": "x", "stream": True, "state_file": None},
        "models": {"default": "some-model"},
        "sensing": {"cameras": ["overhead", "wrist_left"]},
        "planner": {"tolerance_m": 0.005},
        "run": {"tag": "007", "note": None},
        "new": {"nested": {"key": "on the fly"}},
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("no_equals_sign", "key.path=value"),
        ("a..b=1", "key.path=value"),
        ("=1", "key.path=value"),
        ("a.b.c=2", "a.b is not a mapping"),
        ("a.x=[1,", "not valid YAML"),
    ],
)
def test_bad_overrides_say_which_and_why(override: str, message: str) -> None:
    with pytest.raises(ValueError, match=message) as error:
        compose({"a": {"b": 1}}, overrides=[override])
    assert override in str(error.value)


def test_replaced_paths_take_the_last_patch_that_names_them_whole() -> None:
    base = {
        "variants": {"default": {}, "ablation": {"practice": {"x": 1}}},
        "learning": {
            "brains": {"actor": {"effort": "low"}, "reviser": {"effort": "low", "parallel": True}},
            "reruns": 3,
        },
    }
    site = {"variants": {"default": {}}, "learning": {"brains": {"reviser": {"effort": "high"}}}}
    replace = ("variants", "learning.brains.*")
    assert compose(base, site, replace=replace) == {
        "variants": {"default": {}},
        "learning": {
            "brains": {"actor": {"effort": "low"}, "reviser": {"effort": "high"}},
            "reruns": 3,
        },
    }
    assert compose(base, {}, replace=replace) == base
    assert compose(base, site)["learning"]["brains"]["reviser"]["parallel"] is True  # merged
