"""The guard extension point: the built-in pipeline as registered guards, in the same order as the
default pipeline."""

from roboscript.hub import DEFAULT_GUARDS, DEFAULT_PIPELINE
from roboscript.hub import steps as s
from roboscript.testing.guards import GuardConformance

# The guard pipeline, in its fixed order.
BEFORE = (
    (s.episode_open, s.visibility, s.validate, s.resolve_aliases),
    (s.action_lock, s.budget, s.stage_rules, s.retry_guard, s.count_actions),
    (s.timeout, s.metering),
    (
        s.check_output,
        s.count_steps,
        s.unmoved,
        s.fail_fast,
        s.recovery,
        s.give_up,
        s.retry_record,
        s.done_state,
        s.finish,
        s.sanitize,
        s.render,
        s.action_notes,
    ),
)


def test_the_default_pipeline_is_the_builtin_guards_in_the_same_order() -> None:
    phases = (DEFAULT_PIPELINE.pre, DEFAULT_PIPELINE.guards, DEFAULT_PIPELINE.around)
    assert (*phases, DEFAULT_PIPELINE.post) == BEFORE
    assert tuple(step.__name__ for phase in BEFORE for step in phase) == DEFAULT_GUARDS


# The suite over every other built-in guard too.
for _name in DEFAULT_GUARDS:
    if _name != "budget":
        _test = f"TestBuiltin_{_name}"
        globals()[_test] = type(_test, (GuardConformance,), {"guard_name": _name})
