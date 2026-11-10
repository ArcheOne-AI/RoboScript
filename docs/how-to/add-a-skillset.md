# How to add a skillset

A skillset is the operation API of one environment: extra tools added to core toolsets, plus data
binding the core's entries to the environment. It registers under `roboscript.skillsets`.

## 1. Create the manifest

A `SkillSet` declares the skillset's identity and bindings:

```python
from roboscript.contracts.skills import Bindings, SkillSet
from roboscript.tools.skillsets import SkillSetPlugin

MY_SKILLS = SkillSetPlugin(
    manifest=SkillSet(
        name="mybench",
        version="1",
        layer="benchmark",
        envs=["mybench"],
        embodiments=["myrobot"],
        bindings=Bindings(executor=MY_EXECUTOR),
    )
)
```

`Bindings` carries `PlannerProfile`, `ExecutorConfig` and `SupportSurface` as data. These bind
the core's manipulation, motion and perception entries to your environment.

## 2. Add tools via extensions

Write a `ToolsetExtension` to add tools to a core toolset:

```python
from roboscript.tools import ToolContext, tool
from roboscript.tools.skillsets import ToolsetExtension

class MyManipExtension(ToolsetExtension):
    extends = "manipulation"

    @tool(kind="execution", tier="D")
    async def rotate_about(self, ctx: ToolContext, arm: str, ...) -> ActionReceipt:
        """Turn a held object about a hinge."""
        ...
```

Add extension classes to `SkillSetPlugin(extensions=[MyManipExtension])`.

## 3. Write the guide

The `guide` field is Markdown shown to roles that use the skillset. It is templated over the
robot: `{{world_axes}}`, `{{tcp_offset_m}}`. `wicl` renders it at `{{conventions}}` in prompts.

## 4. Register

```toml
[project.entry-points."roboscript.skillsets"]
mybench = "roboscript_mybench.skills:MY_SKILLS"
```

Name the skillset in `Benchmark.skillsets` so stacks mount it.

## 5. Validate

`roboscript.testing.validation.validate` runs four steps: interface check, information boundary
check, an optional scripted ceiling, and freezing.

```bash
python -m roboscript.testing.validation mybench --embodiment myrobot --also manipulation \
    --forbidden roboscript_mybench.sim
```

## 6. Building skillsets with an agent

The `skill_builder` method lets a coding agent write skills in a sandbox. The driver gates them
through static checks, cases and a transfer check. Accepted skills are frozen as a skillset.
See `examples/simBenchmarks/RoboTwin/p1/` for the glue.
