# Tools

## Overview
The tools layer defines tool schemas, validates inputs, dispatches tool calls, and integrates MCP servers and resources.

## Files
- [execution.py](execution.py): ToolCall/ToolPayload models and execution metadata.
- [registry.py](registry.py): ToolRegistry for function tools, shell tools, and MCP tools.
- [specs.py](specs.py): ToolSpec dataclasses for tool schemas and timeouts.
- [validation.py](validation.py): Minimal JSON schema validation for tool inputs.
- [mcp.py](mcp.py): MCP manager, server config, filters, persistence, and in-memory client.
- [mcp_stdio.py](mcp_stdio.py): JSON-RPC stdio bridge for MCP server management.
- [__init__.py](__init__.py): Exports tools and MCP types.

## How It Fits Together
- [core/engine.py](../core/engine.py) calls [registry.py](registry.py) to dispatch tools.
- [runtime/prompt_router.py](../runtime/prompt_router.py) builds ToolCall objects for the registry.
- [web/server.py](../web/server.py) exposes endpoints to register tools and manage MCP servers.
