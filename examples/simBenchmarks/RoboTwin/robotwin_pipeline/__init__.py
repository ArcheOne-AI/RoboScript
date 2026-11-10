"""The S1 -> S2 -> P3 pipeline (docs/concepts/operations.md): each task's S1 `imitate`,
then each variant's S2 `practice` (which freezes) and P3's official evaluation, as jobs in worker
processes of their own, in parallel on the GPUs' simulator places, resumable, respawned when
infrastructure fails them. `examples/simBenchmarks/RoboTwin/scripts/run_pipeline.py` runs it from
`configs/pipeline.yaml`.

- `config`: `PipelineConfig` (configs/pipeline.example.yaml over its env binding,
  `compose_config`), `load_directory`; P3's settings are
  `PipelineP3` (`roboscript_robotwin.eval.blocks.P3Settings` and `unconverged`)
- `run`: the jobs (`Host`, what their bodies get of the pipeline; `RunEnv`, `run_keys`; job keys
  and states are `roboscript.orchestrate.jobs`) and what they run with
  besides the config (`Wiring`)
- `worker`: a job in a process of its own (`main`; the generic worker and its command line are
  `roboscript.orchestrate.worker`, the job bodies wicl's and RoboTwin's `kinds`)
- `schedule`: `run_pipeline`, `plan` (a launch's dry run), `new_directory`, `preflight`,
  `leftovers`, `GatewayCheck`; GPU placement is `roboscript.orchestrate.slots`, the GPUs'
  memory, processes and shared claims `roboscript.stack.gpus`
- `report`: the status and summary (`write_status`, `write_summary`): the core's
  `roboscript.orchestrate.report` of wicl's and RoboTwin P3's parts

    <root>/<stamp>/  pipeline.yaml  launches.jsonl  status.{json,md}  summary.{json,md}
                     services/gpu<n>/          each GPU's services (logs, pid record)
                     <task>/S1/                the task's S1 (`robotwin_pipeline.run`)
                     <task>/<variant>/         a variant's S2 and P3
"""

from robotwin_pipeline.config import PipelineConfig, compose_config, load_directory
from robotwin_pipeline.report import status, summary, write_status, write_summary
from robotwin_pipeline.run import RunEnv, Wiring, run_keys
from robotwin_pipeline.schedule import (
    GatewayCheck,
    leftovers,
    new_directory,
    plan,
    plan_text,
    preflight,
    run_pipeline,
)

__all__ = [
    "GatewayCheck",
    "PipelineConfig",
    "RunEnv",
    "Wiring",
    "compose_config",
    "leftovers",
    "load_directory",
    "new_directory",
    "plan",
    "plan_text",
    "preflight",
    "run_keys",
    "run_pipeline",
    "status",
    "summary",
    "write_status",
    "write_summary",
]
