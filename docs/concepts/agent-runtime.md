# Agent runtime

How a model's request becomes a physical action: tools, the Hub, brains, the gateway and the kernel.

## Tools

A tool is defined once with `@tool`: name, description, pydantic parameter and result models, handler, side-effect kind (`pure`, `perception`, `execution`, `memory`, `control`), tier, timeout and version. Physical actions take a required `rationale`. A `system_only` tool is called by the runtime, invisible to the model.

A toolset groups tools with one resource profile. Each validates arguments, calls a library function and returns a structured result with images as blob references.

| Toolset | Tools |
|---|---|
| `sensing` | `observe`, `get_image`, `show_frames`, `get_camera`, `get_robot_state` |
| `perception` | `identify`, `ground`, `segment`, `locate`, `locate_part`, `remeasure`, `refine`, `look_pose` |
| `geometry` | `lift`, `fit`, `table`, `project`, `measure_pixel` |
| `worldstate` | `register`, `register_part`, `refresh`, `orient`, `relation`, `check` |
| `manipulation` | `move_to_pose`, `move_by`, `open_gripper`, `close_gripper`, `return_home`, `grasp_object`, `place_object`, `follow_path`, `together`, `nudge`, `check_reach`, `plan_arms` |
| `memory` | `memory_read`, `memory_edit` |
| `privileged` | `true_objects` (tier P only) |
| `done` (wicl) | `check_done`, `read_relations`, `read_end_targets` |
| `library` | `library_index`, `library_search`, `library_read`, `library_use` |
| `demo` (wicl) | `demo_relation`, `fit_skill` |

## Skillsets

A skillset is a benchmark's operation API: a manifest (`SkillSet` with bindings like `PlannerProfile` and `ExecutorConfig`) plus `ToolsetExtension` classes whose `@tool` methods a core toolset runs as its own. A skillset also carries a `guide` (Markdown for the model, templated over the robot) and declares what it needs of the robot's arms and action kinds. Register in `roboscript.skillsets`.

## The Hub

The Hub (`roboscript.hub`) dispatches between brains and toolsets. For each episode it tracks visible tools, budgets, aliases, the action lock and stage state. Every tool call passes through a guard pipeline: visibility, validation, alias resolution, action lock, budget, stage rules, deadline, the actual call, output validation and finish handling. Guards register in the `roboscript.guards` entry-point group.

## Phases and roles

A method declares its phases and roles. Each phase gives a kind (`learn` or `eval`), default rules, tier per role, and an ordered visibility table (`toolset.tool`, `toolset.*`, `toolset.+`). A learn phase marked `d_only` is held to tier D as eval phases are. The `catalog.lock` written at run start pins tool schemas.

## Brains

A brain drives a model through an episode. Three are registered: `responses` (the reference Responses API loop with failover), `scripted` (plays scripted replies for testing), and `replay` (answers from a recorded run's tape).

## The model gateway

The gateway (`roboscript.gateway`) is the only exit for LLM calls. It reads `.env.api` keys and routes calls over endpoints. Configuration example:

```yaml
schema: rs-gateway/v1
listen: 127.0.0.1:18790
endpoints:
  primary: {url_env: RS_GW_EP1_URL, key_env: RS_GW_EP1_KEY, protocols: [responses]}
  backup:  {url_env: RS_GW_EP2_URL, key_env: RS_GW_EP2_KEY, protocols: [responses, chat]}
models:
  my-model:
    deployments:
      - {endpoint: primary, upstream: my-model, max_inflight: 8}
      - {endpoint: backup,  upstream: my-model, tier: fallback, max_inflight: 8}
aliases: {brain-default: my-model, grounder: my-model}
```

Routing: session pin first, then tier, then least loaded, then round robin. A session stays on its deployment because encrypted reasoning is bound to it. Failover happens only before the first byte. Concurrency per pool adapts between `min_inflight` and `max_inflight`.

## External brains

Any MCP client can reach the Hub through its MCP endpoint. Visibility rides on the episode's JWT token (episode, role, phase). A Hub is served with `python -m roboscript.integrations.hub_server`, and a toolset with `python -m roboscript.integrations.mcp.serve`.

## The kernel

`roboscript.kernel` is a few hundred lines of async standard library. `Context` holds typed services. A `Plugin` declares its name, required/provided services and a config. `EventBus` dispatches events. `compose_run` loads plugins, validates configs and yields the Hub pipeline and prechecks. Each extension point is an entry-point group with an interface and a conformance suite.
