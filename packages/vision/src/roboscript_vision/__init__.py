"""The vision service: GroundingDINO and SAM 3 on a GPU behind the typed `Vision` RPC
(`roboscript.contracts.vision`; docs/concepts/perception.md).

- `config`: `VisionConfig` (`configs/vision.example.yaml`)
- `backends`: the `Detector` / `Segmenter` interfaces and the transformers models behind them
- `service`: `VisionService`, `run` and `main` (`python -m roboscript_vision --config ...`)

It runs in the vision interpreter (packages/vision/README.md). Importing it does not import torch:
the models load on the service's GPU thread.
"""

__version__ = "0.1.0.dev0"
