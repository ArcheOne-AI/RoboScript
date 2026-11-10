# A toolset of your own

Goal: add tools to the Agent, mount them, and check them with the conformance suite.

```bash
cd examples/tutorials
uv run python -m custom_toolset.demo
```

Code: `custom_toolset/tape.py` (the toolset), `custom_toolset/demo.py` (mounting it and calling
it), `tests/examples/tutorials/test_custom_toolset.py` (the conformance test).

## The toolset

`tape` is a tape measure on the robot's own arms. It is a subclass of `roboscript.tools.Toolset`:

- `name = "tape"`: the toolset's name. Tools are listed to a phase as `tape.arm_gap`, or all at
  once as `tape.*` (in the order the class declares them).
- `Config = TapeConfig`: a pydantic model with `extra="forbid"` for the toolset's section of a run
  config. The Hub builds the class with no arguments and sets `self.config` before the first
  call. Conformance checks that an unknown key is refused.
- `@tool(kind=..., tier=...)` on `async` methods. `tier="D"` means the tool uses only what a real
  robot has, here proprioception through `ctx.env`. A tier-P tool (ground truth) would not be
  listed to any evaluation phase. `kind` says what the tool is: `pure`, `perception`,
  `execution` (a physical action, which takes a `rationale` and ends the turn), `memory`,
  `control`.
- The first paragraph of a tool's docstring is what the model reads. The parameters after `ctx`
  become its strict JSON schema (`Annotated[..., Field(description=...)]` for their text). The
  return annotation is the result model.
- `ctx.state` is a dict for this toolset and this episode, dropped when the episode closes
  (`mark` keeps the arms' positions there). A `ToolError` becomes an error result (code `tool_error`) the
  model can read and recover from; any other exception is logged in full and the model sees only
  `internal_error`.

## Mounting and calling

`demo.py` mounts `LocalProvider(Tape(), {"unit": "cm"})` in a Hub over `MockEnv`. The second
argument is the toolset's config as plain data, the same mapping a run's config file gives. It
then calls the tools the way a model does, with `hub.call`, so visibility, validation and the
event log apply. With `structured=True` in `OpenEpisode`, an outcome also carries its result as
data.

## The conformance test

```python
@pytest.fixture
def toolset_target() -> ToolsetTarget:
    return ToolsetTarget("tape", Tape, {"unit": "cm"})


class TestTape(ToolsetConformance):
    pass
```

The suite checks the class is a `Toolset` of its name, the config validates and refuses unknown
keys, every tool is well formed (strict schema, survives JSON, exports to the Responses, Chat and
MCP forms), a provider mounts it, a toolset server would list the same tools as a local provider,
and a Hub starts it and lists the right tools to a tier-P and a tier-D episode. Without pytest,
`await roboscript.testing.precheck(ToolsetConformance(), target)` runs the same checks and
returns what failed.

## From tutorial to package

A package registers the class and tests it through the entry point:

```toml
[project.entry-points."roboscript.toolsets"]
tape = "roboscript_tape.toolset:Tape"
```

```python
ToolsetTarget.registered("tape", {"unit": "cm"})
```

A registered toolset can run as its own server (`docs/how-to/remote-toolsets.md`); the Agent then
sees the same tools. Which roles see a tool is the method's phase table, never the toolset's.

## Limits

Nothing in this tutorial runs a toolset as a server. The conformance suite checks that a served one
would list the same tools, and `docs/how-to/remote-toolsets.md` shows how to serve one.
