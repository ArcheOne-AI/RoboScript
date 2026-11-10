# roboscript

The core of RoboScript: contracts, plugin kernel, model gateway, tool Hub, embodied core
(embodiments, envs, motion, manipulation, perception), multi-stage learning harness and
orchestration. Methods, benchmarks, robots and the vision service plug in through
`roboscript.*` entry-point groups. The core imports none of them.

```bash
uv sync --all-packages   # from the repository root
```

Python 3.10+. Extras: `mcp`, `testing`.

Start at [docs/architecture.md](../../docs/architecture.md).
