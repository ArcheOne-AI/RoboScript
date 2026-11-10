# Contributing to RoboScript

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
uv sync --all-packages
uv run pytest
```

Copy `.env.example` to `.env` and `configs/*.example.yaml` to `configs/*.yaml`. Model API keys go in
`.env.api` (read only by the gateway). Benchmark simulators and the vision service have their own
environments, described in each package's README.

## Layers

The core's modules form layers checked by import-linter (`uv run lint-imports`). A module may import
only the layers below it (see `docs/architecture.md`). The core never imports torch, sapien or
curobo at module top level. Methods, benchmarks and robots depend on the core and never on each
other. A benchmark's agent side never imports its simulator side.

## Code style

Ruff handles formatting and linting (line length 100). The code must run on Python 3.10 -- avoid
`asyncio.TaskGroup`, `typing.Self`, `tomllib` and `ExceptionGroup`; use `anyio` and
`typing_extensions` instead.

Service code is asyncio. Blocking work goes to `anyio.to_thread`. Validate at boundaries only (config
files, network input, model JSON). Inside a process, trust the types.

Everything crossing a process boundary is a pydantic model from `roboscript.contracts`, versioned per
plane, as JSON. Large arrays go through the blob store by reference. Nothing is pickled. API keys
live in `.env.api` and use `SecretStr`.

## Tests

```bash
uv run pytest                    # unit tests (no simulator, GPU or network)
uv run pytest -m sim             # needs a benchmark's simulator
uv run pytest -m gpu             # needs a GPU
uv run pytest -m model           # calls real model endpoints
```

Results go to `test-results/latest/summary.json`. Mock only models, network and clock. Assert on
the outside world (files, env state, logs), not an agent's report.

## Pull requests

Before opening a PR:

```bash
uv run ruff check && uv run ruff format --check && uv run lint-imports && uv run pytest
```

Commit titles: `[type] summary` where type is `feat`, `fix`, `refactor`, `test`, `docs` or `chore`.
Update docs alongside code changes. Never commit `.env*` (except `*.example`), real configs,
`test-results/`, run outputs, data or weights.

Contributions are licensed under the [Apache License 2.0](LICENSE).
