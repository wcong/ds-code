# DS Code (Python Package)

## Overview
This package is a Python port of the DS Code runtime. The main entry point launches the web server, which hosts a simple HTML UI and JSON APIs for prompts, tools, jobs, threads, MCP, and config.

## How It Fits Together
- [app.py](app.py) calls the web server entry point.
- [web/server.py](web/server.py) builds the runtime stack and exposes the HTTP API + HTML UI.
- [runtime/api.py](runtime/api.py) coordinates prompt handling, threads, approvals, and tool invocation.
- [core/engine.py](core/engine.py) orchestrates tools, jobs, threads, and exec-policy decisions.
- [llm/client.py](llm/client.py) resolves models and returns placeholder responses.
- [tools/registry.py](tools/registry.py) dispatches tool calls (including MCP tools).
- [tasks/manager.py](tasks/manager.py) wraps job lifecycle operations.
- [config.py](config.py) loads/saves configuration and resolves provider defaults.

State and configuration are stored under ~/.ds_code (config.json, state.json, mcp_state.json).

## Files and Folders
- [__init__.py](__init__.py): Package marker.
- [app.py](app.py): CLI entry point that starts the web server.
- [config.py](config.py): Config model, defaults, env overrides, and persistence.
- [execpolicy.py](execpolicy.py): Exec policy rules and approval decisions.
- [core/](core/README.md): Core engine, jobs, threads, state store.
- [llm/](llm/README.md): Model registry and LLM client.
- [runtime/](runtime/README.md): Runtime API, prompt routing, approvals, events.
- [tasks/](tasks/README.md): Job/task management wrapper.
- [tools/](tools/README.md): Tool specs, dispatch, MCP management, stdio bridge.
- [ui/](ui/README.md): Optional TUI/Tk UI shells.
- [web/](web/README.md): FastAPI server and HTML UI.
