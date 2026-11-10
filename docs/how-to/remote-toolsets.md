# Remote toolsets

A toolset can run as its own MCP server, and a Hub mounts it as a remote provider.

## Serving a toolset

```bash
python -m roboscript.integrations.mcp.serve --toolset <name> --config toolset-server
```

The config names the toolset, its embodiment (for fit checks) and a bearer token. The server
lists the same tools as a local provider would.

## Pointing a run at it

In the run's config, give the toolset a `url` instead of mounting it locally. The Hub connects
to it over MCP with an episode token. Tool calls go over the network; images travel as blob
references through a shared blob directory.

## Limits

- Only `roboscript.integrations.mcp` imports the MCP SDK (pinned to one version).
- A remote toolset's per-episode state lives in the server. The Hub still enforces visibility,
  budgets and the action lock.
- Latency adds to the tool-call deadline. A timeout fails the call as infrastructure.
- A served Hub (`python -m roboscript.integrations.hub_server`) can itself be an MCP endpoint
  for an external agent.
