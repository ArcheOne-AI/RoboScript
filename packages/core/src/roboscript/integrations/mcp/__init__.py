"""MCP hosting (docs/concepts/agent-runtime.md), the only user of the MCP SDK (extra `mcp`: the
official `mcp` package, pinned exactly).

- `hub.hub_app`: the Hub's endpoint for brains (tokens, per-token tool lists, the pipeline);
- `serve` (`python -m roboscript.integrations.mcp.serve`): a registered toolset as a remote
  MCP server (`ToolsetServer`); `remote.RemoteProvider` mounts one in a Hub;
- `client.connect`: an MCP client of these servers, in either protocol generation.
"""

from roboscript.integrations.mcp.client import Era, connect
from roboscript.integrations.mcp.hub import SERVER_NAME, hub_app
from roboscript.integrations.mcp.remote import RemoteProvider

__all__ = ["SERVER_NAME", "Era", "RemoteProvider", "connect", "hub_app"]
