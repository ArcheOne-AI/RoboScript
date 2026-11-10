# configs

Only `*.example.yaml` files are tracked here; they arrive together with the component they
configure. To use one, copy it to the same name without `.example` (for example
`gateway.example.yaml` to `gateway.yaml`) and edit the copy. The copies are git-ignored
because they hold machine-specific settings such as ports, GPUs and interpreter paths;
secrets never go here but into `.env` and `.env.api`.
