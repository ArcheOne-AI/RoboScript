# Plugins of your own: a guard and an event sink

Goal: change what the Hub allows and what a run exports, without touching the core.

```bash
uv run pytest tests/examples/tutorials/test_custom_plugin.py -v   # from the repository root
```

Code: `custom_plugin/plugins.py`; test: `tests/examples/tutorials/test_custom_plugin.py`.

## Plugins run in the run's own process

Three extension points are in-process plugins, not servers: **guards** (`roboscript.guards`),
**event sinks** (`roboscript.event_sinks`) and **prechecks** (`roboscript.prechecks`). Each is
plain data: a name, a pydantic `Config` and a function that makes the working part from the
validated config. A run lists the ones it adds as `plugins`; the built-ins are always there.

## A guard

A guard is one step of the Hub's pipeline that every tool call passes. `forbid_arm` refuses calls
whose `arm` argument is on a list the run gives:

```python
FORBID_ARM = Guard(name="forbid_arm", phase="pre", make=..., Config=ForbidArmConfig)
```

- `phase` is `pre` (before the action lock and the budget, so a refused call costs no turn),
  `guards`, `around` (wraps the tool: takes the call and `proceed`) or `post`. `after="budget"`
  would place it right after a named step; by default it goes to the end of its phase.
- A step refuses by raising `Reject(code, message)`. The code is stable, the message is what the
  model reads; the call comes back to the model as an error and the episode goes on.
- A step has no module-level state: `make` builds a fresh one per Hub, so keep counters inside it
  (`roboscript.hub.ratelimit` is the complete example).

## An event sink

A sink gets a copy of every event the run writes, beside the run's own logs, which stay the source
of truth. `tool_tally` counts the tools the model called and writes `tool_tally.json` when the run
closes:

```python
TOOL_TALLY = Sink(name="tool_tally", Config=TallyConfig, make=_make_tally)
```

`make(config, run_dir)` returns an object with `write(event)`, `flush()` and `close()`. A sink
writes only inside `run_dir`; the conformance suite checks it.

## Register and test

A package registers both in its `pyproject.toml`:

```toml
[project.entry-points."roboscript.guards"]
forbid_arm = "roboscript_tutorial.plugins:FORBID_ARM"

[project.entry-points."roboscript.event_sinks"]
tool_tally = "roboscript_tutorial.plugins:TOOL_TALLY"
```

and tests them with the suites in `roboscript.testing`:

```python
class TestForbidArm(GuardConformance):
    guard_name = "forbid_arm"


class TestToolTally(SinkConformance):
    sink_name = "tool_tally"
```

The suites load the plugin as a run does: by the name it is registered under, or by its import
path `package.module:attribute`. The tutorial is not installed, so its test names both by import
path (`guard_name = "custom_plugin.plugins:FORBID_ARM"`) and the rest reads the same. Without
pytest, `await precheck(Suite())` runs the same checks, so a plugin can ship them as its own
self-check.

## Use them in a run

In a run spec, the plugins are items with their group, name (or import path) and config:

```python
items = [  # installed, they would be name="forbid_arm" and name="tool_tally"
    PluginItem(
        group="guards", name="custom_plugin.plugins:FORBID_ARM", overrides={"arms": ["right"]}
    ),
    PluginItem(group="event_sinks", name="custom_plugin.plugins:TOOL_TALLY"),
]
async with compose_run(items) as composition:
    ...  # composition.pipeline, composition.sinks, composition.prechecks
```

`compose_run` validates every config before anything runs, so a misspelled key stops the run at
the start. The test builds the Hub from `composition.pipeline` and an `EventRouter` with the
opened sinks, which is what the runner does, and checks that the right arm is refused, the left
arm lifts the box, and the tally has both calls.

## A precheck is the third kind

`Precheck(name, check, Config)`: `check(facts, config)` returns one line per problem (empty:
fine), run before any job or model call. Test it with `PrecheckConformance` and `precheck_name`.
