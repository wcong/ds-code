# Runtime

## Overview
The runtime layer exposes a high-level API for prompts, threads, approvals, and tool invocation. It adapts web requests into core operations.

## Files
- [api.py](api.py): RuntimeApi entry point used by the web server and UI shells.
- [models.py](models.py): Request/response dataclasses and runtime configuration.
- [prompt_router.py](prompt_router.py): Parses tool/shell/mcp prefixes into ToolCall objects.
- [approvals.py](approvals.py): ApprovalStore for tracking approval state in memory.
- [events.py](events.py): Event frame helpers for tool execution and errors.
- [__init__.py](__init__.py): Runtime package exports.

## How It Fits Together
- [web/server.py](../web/server.py) builds a RuntimeApi and calls [api.py](api.py) for all endpoints.
- [api.py](api.py) calls [core/engine.py](../core/engine.py) for model resolution, tool execution, and thread updates.
- [prompt_router.py](prompt_router.py) converts tool prefixes into [tools/execution.py](../tools/execution.py) payloads.
