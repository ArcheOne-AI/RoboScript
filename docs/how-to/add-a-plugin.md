# How to add a plugin

A run can add three kinds of plugin through its `plugins` list: guards, event sinks and
prechecks. Each is an entry-point group with a conformance suite.

## Guards

A guard is a step in the Hub's tool-call pipeline. It runs in one of four phases: `pre`,
`guards`, `around` or `post`.

```python
from roboscript.hub.pipeline import Guard, Reject

MY_LIMIT = Guard(
    name="mylimit",
    phase="guards",
    make=lambda config: my_step,
)
```

Register under `roboscript.guards`:

```toml
[project.entry-points."roboscript.guards"]
mylimit = "roboscript_mylimit:MY_LIMIT"
```

## Event sinks

A sink receives a copy of every event a run writes, beside the run's own logs.

```python
from roboscript.records.sinks import Sink

MY_SINK = Sink(name="mysink", Config=MySinkConfig, make=open_my_sink)
```

Register under `roboscript.event_sinks`:

```toml
[project.entry-points."roboscript.event_sinks"]
mysink = "roboscript_mysink:MY_SINK"
```

## Prechecks

A precheck runs before a run starts and returns one line per problem (empty means fine).

```python
from roboscript.learning.prechecks import Precheck

MY_CHECK = Precheck(name="mycheck", check=my_check_fn)
```

Register under `roboscript.prechecks`:

```toml
[project.entry-points."roboscript.prechecks"]
mycheck = "roboscript_mycheck:MY_CHECK"
```

## Testing

Each kind has a conformance suite in `roboscript.testing`: `GuardConformance`,
`SinkConformance`, `PrecheckConformance`. Set the class attribute with the registered name
and subclass it as a `Test*` class.
