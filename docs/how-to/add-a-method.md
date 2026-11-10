# How to add a method

A method declares its phases, roles, tool visibility and the artifact it learns. The core's
harness runs it. Its content (prompts, lint vocabularies, parameters) lives in
`recipes/<method>/`.

## 1. Declare the method

Subclass `roboscript.learning.method.Method`:

- **`name`**: the registered name
- **`phases`**: each a `PhaseSpec` with a kind (`learn` or `eval`), default rules and a
  visibility table mapping roles to the tools they see
- **`roles`**: each a `RoleSpec`
- **`requires`**: what capabilities the env must have
- **`artifact`**: the kind of frozen artifact (e.g. `task_memory`)

A visibility entry is `toolset.tool`, `toolset.*` (whole toolset) or `toolset.+` (tools from
mounted skillsets). An `eval` or `d_only` phase gives every role tier D only.

## 2. Implement the runtime

`Method.runtime(recipe)` returns a `MethodRuntime` with the hooks the core calls:

- `opening`: runs when an episode opens
- `finish`: the finish gate
- `prompts`: each role's prompt factory
- `demo`: demonstration views
- `explain`: failure explanation
- `frozen`: how a frozen Memory is read
- `evaluate_phase` / `host_phase`: which of your phases the core's protocols use

Each field defaults to nothing, so set only what you use.

## 3. Write the recipe

Put prompts, lint vocabularies and stage parameters in `recipes/<method>/`. The recipe holds no
paths, ports, GPUs or secrets.

## 4. Register

```toml
[project.entry-points."roboscript.methods"]
mymethod = "roboscript_mymethod.method:MyMethod"
```

Also register any protocols under `roboscript.protocols` and toolsets under
`roboscript.toolsets`.

## 5. Test

Run `MethodConformance` from `roboscript.testing`:

```bash
uv run pytest tests/methods/<name>/
```
