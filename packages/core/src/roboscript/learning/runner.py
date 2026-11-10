"""The `ProtocolRunner` (docs/concepts/learning.md, docs/concepts/operations.md): the
orchestrator's learning side.

It owns a run: one Hub for all tasks (toolsets and their backends start once; episodes keep their
state by id, so tasks share it safely), the memory toolset whose editors the protocols attach,
the env control planes (one env server per task), the Memory store (by default the run
directory, so versions land in `<run_dir>/<task>/memory/`), the recipe (prompts, Memory rules,
protocol parameters), a brain per role and the run's event log. Protocols (entry-point group
`roboscript.protocols`) run per task; tasks run in parallel (`run_tasks`), a task's protocols,
seeds and tries in sequence.

```python
async with ProtocolRunner(config, providers=[LocalProvider(Sensing())], blobs=blobs,
                          controls={task: control}, embodiment=profile_path) as runner:
    await runner.run("imitate", task, {"demo": "data/demos/lift_red/package"})
    await runner.run("practice", task)
    await runner.run("evaluate", task, {"count": 20})
```
"""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import httpx
from pydantic import BaseModel, SecretStr
from typing_extensions import Self

import roboscript
from roboscript import rpc
from roboscript.agent.session import Brain, ModelClients
from roboscript.blobs import BlobStore
from roboscript.contracts.capabilities import EnvCapabilities, EpisodeReset, episode_reset
from roboscript.contracts.env import EnvAgent, EnvControl
from roboscript.contracts.ids import RoleName, TaskName
from roboscript.contracts.motion import Planner
from roboscript.contracts.phase import PhaseSpec
from roboscript.contracts.tools import EnvBinding
from roboscript.contracts.trace import new_trace, use_trace
from roboscript.embodiments import load_embodiment
from roboscript.envs.cards import TaskCard
from roboscript.hub import DEFAULT_PIPELINE, Hub, HubConfig, Pipeline
from roboscript.hub.lock import catalog_lock, lock_hash
from roboscript.hub.visibility import Phases, absent, mount
from roboscript.kernel import entrypoints
from roboscript.learning.config import (
    BrainSpec,
    LearningConfig,
    ProtocolRecipe,
    RoleConfig,
    load_protocol_recipe,
)
from roboscript.learning.context import GROUP, LearningProtocol, TaskContext
from roboscript.learning.episode import BrainFactory, EpisodeDriver, Trajectories
from roboscript.learning.events import EventRouter
from roboscript.learning.hooks import Hooks
from roboscript.learning.method import Declaration, load_method
from roboscript.learning.mounts import remote_toolsets
from roboscript.learning.runtime import check_mounted
from roboscript.memory import FileMemoryStore, MemoryStore, ToolSig, load_recipe
from roboscript.records import write_atomic
from roboscript.records.sinks import EventSink
from roboscript.settings import require_source_sha
from roboscript.tools import LocalProvider, ToolProvider, ToolSpec, tool_signatures
from roboscript.tools.provider import provider_guides, unfit_tools
from roboscript.tools.skillsets import fits_skills, load_skillset
from roboscript.toolsets.memory import MemoryTools, SkillMemoryTools

log = logging.getLogger(__name__)

TaskPlan = Sequence[tuple["str | LearningProtocol", Mapping[str, Any] | None]]
"""A task's protocols in order, each with the config values that replace the recipe's."""


def actor_tools(
    providers: Sequence[ToolProvider], phases: Sequence[PhaseSpec], role: RoleName = "actor"
) -> list[ToolSpec]:
    """The tools `role` may call (tier D, not system-only) in every phase that runs it at tier D
    only, in the first such phase's order: what a Memory's call templates, which those phases
    carry out, may name (wicl: the S2 and P3 Actor's, not the S1 Actor's extras, which a
    hinted S1 Actor alone has)."""
    mounted = mount(providers)
    declared = Phases(phases, mounted, absent(providers))
    lists: list[tuple[str, ...]] = [
        declared.tools(phase.name, role)
        for phase in phases
        if role in phase.visibility and phase.tier(role) == "D"
    ]
    if not lists:
        return []
    shared = set.intersection(*(set(names) for names in lists))
    return [
        mounted[name].spec
        for name in lists[0]
        if name in shared and mounted[name].meta.tier == "D" and not mounted[name].meta.system_only
    ]


def load_protocol(name: str) -> LearningProtocol:
    """The protocol registered as `name` in `roboscript.protocols`."""
    protocol: LearningProtocol = entrypoints.load(GROUP, name)()
    return protocol


def brain_maker(spec: BrainSpec) -> Callable[[], Brain]:
    """A new brain per call, from `spec` (validated now): how S1, S2 and P3's Agent host make
    their episodes' brains."""
    cls = entrypoints.load("roboscript.brains", spec.name)
    config = cls.Config.model_validate(spec.config)
    make = getattr(cls, "from_config", cls)  # the scripted brain builds its script from config
    return lambda: make(config)  # type: ignore[no-any-return]


def brain_factory(spec: BrainSpec) -> BrainFactory:
    """A new brain per episode, from `spec` (validated now)."""
    make = brain_maker(spec)
    return lambda _episode: make()


def control_client(
    url: str, token: SecretStr, config: LearningConfig, http: httpx.AsyncClient | None = None
) -> EnvControl:
    """An env server's control plane at `url` (bearer `RS_ENV_TOKEN`)."""
    return rpc.client(
        EnvControl,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
        url,
        token=token,
        http=http,
        timeout=config.env_timeout_s,
        timeouts={"open": config.open_timeout_s},
    )


class ProtocolRunner:
    """`providers` are the toolsets besides `memory`, which the runner adds (in this process, or
    remote ones served by their own servers, checked as the runner starts: `learning.mounts`;
    `run_providers` makes them from a `RunSpec`); `controls` the env
    control plane of each task; `agent_env` makes the agent-plane client of an episode's binding
    (default: typed RPC); `brains` replace the config's; `models` is the gateway client factory
    (`roboscript.gateway.openai_client`), for brains and toolsets; `store` defaults to the file
    store in the run directory; `roles` replace the recipe's bounds of those roles; `planner`
    gives S1's referee its FK (the motion service, or `MockPlanner` for MockEnv); `pipeline` the
    Hub's guard pipeline (default: the built-in guards, `roboscript.hub.DEFAULT_PIPELINE`);
    `sinks` get a copy of every event of the run (`roboscript.learning.compose`); `recipe` the
    protocol recipe already composed (a run directory's, layered over the env binding; default:
    the recipe directory's `protocols.yaml`); `method` replaces the config's
    (`LearningConfig.method`), whose phases the Hub runs and whose roles get bounds and brains;
    `offered` is what the run's env offers (its `Benchmark` descriptor's capabilities, or
    `envs.offered_by`), whose reset kind the protocols open episodes by (`TaskContext.reset_by`,
    `contracts.capabilities.episode_reset`; None: by seed); `guides` the skillset guides the
    method's prompts get (default: those of the skillsets the providers mount, `provider_guides`;
    a stack whose toolsets mount none of its env's skillsets, as the core's learning tests on the
    Mock, names its env's); `cards` the tasks' success condition cards (its `Benchmark`
    descriptor's, `Benchmark.task_cards`; none: no task has one), which every role of a task
    reads under its instruction. A task runs one protocol at a time: its env serves one episode
    at a time."""

    def __init__(
        self,
        config: LearningConfig,
        *,
        providers: Sequence[ToolProvider],
        blobs: BlobStore,
        controls: Mapping[TaskName, EnvControl],
        embodiment: str | Path,
        agent_env: Callable[[EnvBinding], EnvAgent] | None = None,
        brains: Mapping[RoleName, BrainFactory] | None = None,
        models: ModelClients | None = None,
        store: MemoryStore | None = None,
        roles: Mapping[RoleName, RoleConfig] | None = None,
        planner: Planner | None = None,
        hub_config: HubConfig | None = None,
        method: type[Declaration] | None = None,
        hooks: Hooks | None = None,
        pipeline: Pipeline = DEFAULT_PIPELINE,
        sinks: Sequence[EventSink] = (),
        recipe: ProtocolRecipe | None = None,
        offered: EnvCapabilities | None = None,
        guides: Sequence[str] | None = None,
        cards: Mapping[TaskName, TaskCard] | None = None,
    ) -> None:
        if any(provider.name == MemoryTools.name for provider in providers):
            raise ValueError("the runner mounts the memory toolset itself: leave it out")
        self.method = load_method(config.method) if method is None else method
        self.runtime = self.method.runtime(config.recipe)
        names = [role.name for role in self.method.roles]
        config = config.for_roles(names)
        self.config = config
        self.run_dir = config.run_dir
        self.blobs = blobs
        self.controls = dict(controls)
        self.reset_by: EpisodeReset = "seed" if offered is None else episode_reset(offered)
        self.embodiment = str(embodiment)
        self.profile = load_embodiment(embodiment)
        # a stack that fits written skills is shown `skill_spec` in `memory_edit`; others are not
        fitting = fits_skills([provider.skillsets for provider in providers])
        memory = SkillMemoryTools if fitting else MemoryTools
        self.memory_tools = memory(self.runtime.edit_checks)
        self.providers = [*providers, LocalProvider(self.memory_tools)]
        if unfit := unfit_tools(self.providers, self.profile):
            raise ValueError(f"mounted tools do not fit {self.profile.name}: {'; '.join(unfit)}")
        self.guides = provider_guides(self.providers) if guides is None else list(guides)
        self._recipe_memory = load_recipe(config.recipe, config.lint_overlay)
        self._recipe_memory.check_phases([phase.name for phase in self.method.phases])
        self.use_tools(tool_signatures(actor_tools(self.providers, self.method.phases)))
        recipe = recipe if recipe is not None else load_protocol_recipe(config.recipe)
        recipe = recipe.model_copy(update={"roles": {**recipe.roles, **(roles or {})}})
        self.recipe = recipe.for_roles(names)
        self.brains = dict(brains or {r: brain_factory(s) for r, s in config.brains.items()})
        self.models = models
        self.store = store if store is not None else FileMemoryStore(config.run_dir)
        self.hooks = hooks if hooks is not None else Hooks()
        self.hub_config = hub_config if hub_config is not None else HubConfig()
        self.planner = planner
        self._locks: dict[TaskName, anyio.Lock] = {}
        self._agent_env = agent_env
        self._pipeline = pipeline
        self._sinks = tuple(sinks)
        self._cards = dict(cards or {})
        self._stack: AsyncExitStack | None = None

    def use_tools(self, tools: Mapping[str, ToolSig]) -> None:
        """Set the tools a Memory's calls may name: the linter's, and the list the roles that
        write templates read (the method's prompts, `MethodRuntime.prompts`, with the guides of
        the skillsets mounted on the providers). The constructor sets what the method's phases
        show an Actor; tests set their own."""
        self.memory = self._recipe_memory.with_tools(tools)
        self.prompts = self.runtime.prompts(tools, self.guides)

    async def __aenter__(self) -> Self:
        async with AsyncExitStack() as stack:
            self.events = stack.enter_context(
                EventRouter(self.run_dir / "events.jsonl", sinks=self._sinks)
            )
            http = await stack.enter_async_context(httpx.AsyncClient(trust_env=False))
            agent = self._agent_env or _rpc_agent_env(http, self.hub_config.env_timeout_s)
            trajectories = Trajectories(agent)
            self.hub = await stack.enter_async_context(
                Hub(
                    self.providers,
                    blobs=self.blobs,
                    config=self.hub_config,
                    events=self.events,
                    phases=self.method.phases,
                    env_client=trajectories.client,
                    models=self.models,
                    pipeline=self._pipeline,
                )
            )
            await stack.enter_async_context(remote_toolsets(self.hub, self.providers))
            learn = [phase for phase in self.method.phases if phase.kind == "learn"]
            check_mounted(self.runtime, learn, self.hub.mounted)
            self.driver = EpisodeDriver(
                hub=self.hub,
                memory=self.memory_tools,
                events=self.events,
                blobs=self.blobs,
                trajectories=trajectories,
                controls=self.controls,
                brains=self.brains,
                roles=self.recipe.roles,
                embodiment=self.embodiment,
                models=self.models,
                max_infra_reruns=self.config.max_infra_reruns,
                infra_backoff_s=self.config.infra_backoff_s,
                prompt_digest=self.prompts.digest,
                provenance=self.config.provenance,
                runtime=self.runtime,
                cards=self._cards,
            )
            self._locks = {task: anyio.Lock() for task in self.controls}
            self._manifest()
            self._stack = stack.pop_all()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()

    def context(self, task: TaskName) -> TaskContext:
        if task not in self.controls:
            raise ValueError(f"no env control plane for task {task!r}: {sorted(self.controls)}")
        return TaskContext(
            task=task,
            directory=self.run_dir / task,
            driver=self.driver,
            store=self.store,
            memory=self.memory,
            prompts=self.prompts,
            profile=self.profile,
            hooks=self.hooks,
            toolsets=tuple(provider.name for provider in self.providers),
            events=self.events,
            planner=self.planner,
            reset_by=self.reset_by,
        )

    async def run(
        self,
        protocol: str | LearningProtocol,
        task: TaskName,
        config: Mapping[str, Any] | None = None,
    ) -> BaseModel:
        """Run `protocol` (registered in `roboscript.protocols`, or given: nothing is registered
        where the core is only on PYTHONPATH, as in the simulator env) on `task` with the
        recipe's parameters, `config` replacing some; its protocol events go to
        `<run_dir>/<task>/events.jsonl`."""
        if self._stack is None:
            raise RuntimeError("the runner is not started: use `async with runner:`")
        found = load_protocol(protocol) if isinstance(protocol, str) else protocol
        values = {**self.recipe.protocols.get(found.name, {}), **(config or {})}
        parsed = found.Config.model_validate(values)
        ctx = self.context(task)
        trace = new_trace()
        async with self._locks[task]:
            with use_trace(trace), self.events.route(trace, ctx.directory / "events.jsonl"):
                ctx.emit("start", protocol=found.name, config=parsed.model_dump(mode="json"))
                result = await found.run(ctx, parsed)
                ctx.emit("end", protocol=found.name, result=result.model_dump(mode="json"))
        return result

    async def run_tasks(
        self, plan: Mapping[TaskName, TaskPlan]
    ) -> dict[TaskName, list[BaseModel] | Exception]:
        """Every task's protocols in order, the tasks in parallel: each task's results, or the
        error that stopped it (the other tasks go on)."""
        done: dict[TaskName, list[BaseModel] | Exception] = {}

        async def one(task: TaskName, steps: TaskPlan) -> None:
            results: list[BaseModel] = []
            try:
                for protocol, config in steps:
                    results.append(await self.run(protocol, task, config))
            except Exception as error:
                log.exception("task %s stopped", task)
                done[task] = error
                return
            done[task] = results

        async with anyio.create_task_group() as group:
            for task, steps in plan.items():
                group.start_soon(one, task, steps)
        return done

    def _manifest(self) -> None:
        """`run.json`, once per run directory: what the run is made of: the code, the config, the
        recipe, the method (its name and, per phase, each role's tier), the toolsets with the
        skillsets mounted on each (name and version), the catalog lock's hash (every mounted
        tool's descriptor and tier and the phases' visibility: `hub.lock`), the tasks."""
        path = self.run_dir / "run.json"
        if path.exists():
            return
        data = {
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "roboscript": roboscript.__version__,
            "source": require_source_sha(),
            "config": self.config.model_dump(mode="json"),
            "recipe": {"prompts": self.prompts.digest, "protocols": self.recipe.model_dump()},
            "method": {
                "name": self.method.name,
                "tiers": {
                    phase.name: {role: phase.tier(role) for role in phase.visibility}
                    for phase in self.method.phases
                },
            },
            "toolsets": [provider.name for provider in self.providers],
            "skillsets": skillsets_of(self.providers),
            "catalog": lock_hash(catalog_lock(self.hub.mounted, self.hub.phases.specs)),
            "tasks": sorted(self.controls),
            "embodiment": self.embodiment,
        }
        write_atomic(path, (json.dumps(data, indent=2, default=str) + "\n").encode())


def skillsets_of(providers: Sequence[ToolProvider]) -> dict[str, dict[str, str]]:
    """By toolset, the skillsets mounted on it (`tools.skillsets`): each one's manifest name by
    the ref it was mounted with, and its version."""
    found: dict[str, dict[str, str]] = {}
    for provider in providers:
        refs: Sequence[str] = getattr(provider, "skillsets", ())
        manifests = [load_skillset(ref).manifest for ref in refs]
        found[provider.name] = {m.name: m.version for m in manifests}
    return found


def _rpc_agent_env(http: httpx.AsyncClient, timeout: float) -> Callable[[EnvBinding], EnvAgent]:
    def client(binding: EnvBinding) -> EnvAgent:
        return rpc.client(
            EnvAgent,  # type: ignore[type-abstract]  # rpc interfaces are Protocols
            binding.url,
            token=binding.token,
            http=http,
            timeout=timeout,
        )

    return client
