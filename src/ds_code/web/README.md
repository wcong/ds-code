# Web

## Overview
FastAPI server that hosts the HTML UI and the JSON API surface for prompts, tools, jobs, threads, MCP, approvals, and config.

## Files
- [server.py](server.py): Builds the runtime stack, defines all endpoints, and serves the HTML UI.

## How It Fits Together
- [server.py](server.py) constructs the stack: LlmClient -> ToolRegistry/McpManager -> CoreEngine -> TaskManager -> RuntimeApi.
- The HTML page makes fetch calls to the JSON API endpoints on the same server.
- The server is launched from [app.py](../app.py).
