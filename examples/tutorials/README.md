# Tutorials

Four short walkthroughs, each a README plus code that runs on the Mock benchmark with a scripted
brain: no model, no GPU, no simulator. Each has a test in `tests/examples/tutorials/` that runs it
in CI, so a tutorial that stops working fails the build. They are not packages: the tests put this
directory on `sys.path`; to run the scripts from a shell, `cd examples/tutorials` first. The
tests run from the repository root (`uv run pytest tests/examples/tutorials`).

| Tutorial | You learn | Run |
|---|---|---|
| [quickstart_mock](quickstart_mock/README.md) | one episode: env, Hub, phase, brain, event log | `uv run python -m quickstart_mock.run` |
| [custom_toolset](custom_toolset/README.md) | write a toolset, check it with the conformance suite | `uv run python -m custom_toolset.demo` |
| [custom_skillset](custom_skillset/README.md) | write an operation API, validate it in four steps | `uv run pytest tests/examples/tutorials/test_custom_skillset.py` (from the root) |
| [custom_plugin](custom_plugin/README.md) | write a Hub guard and an event sink, register them, use them in a run | `uv run pytest tests/examples/tutorials/test_custom_plugin.py` (from the root) |

Read them in order: the later ones reuse the earlier ones' pieces (the plugin tutorial guards the
skill of the skillset tutorial). The how-to guides in `docs/how-to/` say how to do the same for a
real package (`add-a-skillset.md`, `add-a-plugin.md`, `add-a-method.md`), and
`docs/reference/extension-points.md` lists every extension point.
