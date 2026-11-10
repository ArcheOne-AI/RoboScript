# A skillset of your own

Goal: write an operation API (a skill the Agent calls to move the robot), validate it, and see how
validation catches mistakes.

```bash
uv run pytest tests/examples/tutorials/test_custom_skillset.py -v   # from the repository root
```

Code: `custom_skillset/pickup.py` (the skillset), `custom_skillset/guide.md` (its guide),
`tests/examples/tutorials/test_custom_skillset.py` (conformance, validation and the mistakes).

## What a skillset is

An operation API for one environment: a set of Agent-facing physical tools plus data that says
what they fit. In code it is a `SkillSetPlugin(manifest, extensions)`:

- the **manifest** (`contracts.skills.SkillSet`) is plain data: `name`, `version`, `layer`
  (`core`, `benchmark`, `robot`, `built`), the `envs` and `embodiments` it fits, its `skills`,
  the `bindings` it gives the core's entries (planner profile, executor tuning, support
  surface), its `guide` and its verification `report`. It never records how it was built.
- the **extensions** are classes of tools (`ToolsetExtension`), each naming the core toolset it
  `extends`. Their tools are mounted when that toolset is, and run as part of it.

## The skillset

`pickup.py` adds one tool, `pick_up(arm, tcp, lift)`, to the `hold` toolset: open, close on the
point, lift, as one counted action. `skill(spec, chain, arms=...)` builds the manifest entry from
the tool itself (name, description and schema are taken from the spec, so they cannot drift) and
declares what the caller must say: the **chain** of implementation steps the skill goes through
(`parameters`, `perception`, `planning`, `executor`, `env_execute`, `receipt`, in that order), and
what it **needs** of the robot (`arms=["left", "right"]` says its parameters take those arm names,
so it will not mount on a one-armed robot).

The skill reads only the robot's own state (`ctx.env.robot`) and executes through `ctx.env`:
it is tier D, as every skill must be. The guide, `guide.md`, is Markdown for the roles that use the
skillset; it is a template over the robot (`{{world_axes}}`, `{{tcp_offset_m}}`), and another
placeholder fails validation.

This skillset extends `hold`, a test toolset that executes through the env, because the Mock
benchmark mounts no planner. A skillset for a real benchmark extends `manipulation`, takes its
planner and executor from its bindings, and goes through `planning` and `executor`
(`docs/how-to/add-a-skillset.md`).

## The two conformance suites

```python
class TestPickupManifest(SkillSetConformance): ...  # a manifest: round trip, interface


class TestPickupPlugin(SkillSetPluginConformance): ...  # a registered plugin
```

The plugin suite checks that its manifest describes exactly the tools it mounts, that they mount on
the toolsets they extend, and that they mount alike served and in process. Each takes a fixture: `skillset_target` (a `SkillSetTarget`) and
`plugin_target` (`PluginTarget(ref, profile, hosts, also)`, where `also` lists the skillsets mounted
with it, whose bindings count).

## Validation: four steps

`roboscript.testing.validation.validate(ValidationTarget(...), SkillSetStore(...))` runs, each
only if those before it passed:

1. **interface**: the manifest is valid; every skill fits the robot (arm count and names, action
   kinds), the bindings it asks for are given, and the guide uses only known placeholders.
2. **boundary**: it reads, without importing, the code of the skillset and what it imports, and
   fails on an import of the privileged toolset, `envs`, `learning`, `orchestrate` or the demo
   referee, on a use of `EnvControl`, `GtSnapshot`, `gt_snapshot` or `Diagnosis`, and on any tool
   that is not tier D.
3. **ceiling**: a scripted planner that may read the ground truth calls only the skillset's tools
   on some seeds. The share of episodes the env judges a success is an upper bound on what an
   Agent reaches with them. With the planner in `test_custom_skillset.py`, `pick_up` lifts the
   box on every seed.
4. **freeze**: the manifest and its report are committed to a `SkillSetStore` and frozen; the same
   content gives the same version.

The test also shows two failures: a guide with an unknown placeholder stops at step 1, and a
skillset whose tool is tier P stops at step 2. In both, the report lists the steps that ran and
says why the last one failed.

## From tutorial to package

Register the plugin and name it in the benchmark (or the run):

```toml
[project.entry-points."roboscript.skillsets"]
pickup = "roboscript_mybench.skills:PICKUP_SKILLS"
```

`docs/how-to/add-a-skillset.md` has the rest: bindings for the core's entries, the guide, a
benchmark's own ceiling, and when a skillset moves into the core.

## Limits

A ceiling needs a `CeilingRun` you build (an env, a planner, seeds), also for the command
`python -m roboscript.testing.validation` (`--ceiling`, a function that returns it). The
support-surface binding is declared data that no core entry reads.
