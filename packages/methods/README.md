# roboscript-methods

The built-in learning methods. A method declares phases, roles and tool visibility; the core's
harness runs it. Method content (prompts, lint vocabularies, parameters) lives in `recipes/<method>/`.

| Method | What it does |
|---|---|
| `wicl` | Learns a task from one demo (S1 imitate, S2 practice), freezes a Task Memory, then evaluates |
| `rsi` | Reflects on any method's evaluation records, writes an RSI document read by the next evaluation |
| `wicl+rsi` | `wicl` with `rsi` attached |
| `skill_builder` | A coding agent writes skills gated by a driver; the frozen artifact is a skillset |

```bash
uv sync --all-packages   # from the repository root, Python 3.10+
```
