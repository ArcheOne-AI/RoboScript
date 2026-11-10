# roboscript-vision

Runs GroundingDINO (open-vocabulary boxes) and SAM 3 (masks from text, boxes or points) on one
GPU, behind the typed `Vision` RPC. Images and masks travel as blob refs.

## Vision environment

The service runs in its own Python 3.11 venv. torch stays out of `uv.lock`.

```bash
uv venv --python 3.11 /path/to/roboscript-vision
uv pip sync --no-config --require-hashes --python /path/to/roboscript-vision/bin/python \
  packages/vision/requirements-gpu.txt
```

To update the lock, edit `requirements-gpu.in` and compile with the command in the header of
`requirements-gpu.txt`. The environment takes about 5.5 GB.

## Weights

| Model | Repository | Revision | `model.safetensors` sha256 |
|---|---|---|---|
| GroundingDINO base | `IDEA-Research/grounding-dino-base` | `12bdfa3120f3e7ec7b434d90674b3396eccf88eb` | `5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21` |
| SAM 3 | `facebook/sam3` (gated) | `f8ee5d16d5c30d0e87b1d22cbd358ce97761df56` on ModelScope | `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a` |

Fetch GroundingDINO with `hf download <repository> --revision <revision>` from the vision
environment. For SAM 3, accept the licence on the hub or copy files from a mirror into a directory
and point `sam3.model` at it.

## Running

```bash
CUDA_VISIBLE_DEVICES=0 RS_BLOB_DIR=/path/to/blobs PYTHONPATH=packages/core/src:packages/vision/src \
  /path/to/roboscript-vision/bin/python -m roboscript_vision --config vision [--ready-file PATH]
```

Both models together take about 8 GiB of GPU memory. In a stack, configure it as a service with
interpreter `vision`.

## Tests

`tests/vision/` uses numpy stand-ins and runs in every interpreter.
