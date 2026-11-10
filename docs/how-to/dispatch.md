# Dispatch: running on several machines

Dispatch spreads a pipeline's tasks over machines with no shared filesystem.

## Setup

The coordinator assigns tasks once (policy `balanced` by default) and starts one unit per
worker. Each unit is a pipeline launch over that worker's tasks and GPUs. One task's stages
never leave its unit.

The pipeline config is composed once on the coordinator. Workers get a copy that differs only in
tasks, GPUs and `local_keys`. A `provenance` hash ensures every worker runs the same version.

## Running

```bash
# On the coordinator
python -m roboscript.orchestrate.dispatch \
    --config dispatch.yaml --pipeline-config pipeline.yaml

# Resume after a failure
python -m roboscript.orchestrate.dispatch --resume <state_dir>
```

Worker commands run over a transport (`ssh` or `local`):

```bash
python -m roboscript.orchestrate.dispatch.agent <command>
```

The agent writes only inside its unit directory, signals only the pid it recorded and never
reads a token.

## Collecting results

Finished tasks move to the coordinator, checked against a manifest. The worker's copy is deleted
only after a verified copy. If a worker is lost, its unfinished tasks go to other workers. If
the coordinator dies, units keep running and `--resume` picks up from `state.json`.

## Transport

Register custom transports under `roboscript.dispatch_transports` and assignment policies under
`roboscript.dispatch_policies`.
