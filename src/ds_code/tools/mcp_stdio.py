from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ds_code.tools.mcp import (
    InMemoryMcpClient,
    McpManager,
    McpServerDefinition,
    ToolFilter,
)


@dataclass
class StdioState:
    manager: McpManager
    definitions: Dict[str, McpServerDefinition]
    running: Dict[str, bool]
    lifecycle_state: str


def run_stdio_server(initial_definitions: List[McpServerDefinition]) -> List[McpServerDefinition]:
    state = _build_state(initial_definitions)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_json(_jsonrpc_error(None, -32700, f"invalid json: {exc}"))
            continue

        if request.get("jsonrpc") not in (None, "2.0"):
            _write_json(_jsonrpc_error(request.get("id"), -32600, "jsonrpc version must be 2.0"))
            continue

        method = request.get("method", "")
        params = request.get("params", {})
        try:
            result, should_exit = _dispatch(state, method, params)
        except ValueError as exc:
            _write_json(_jsonrpc_error(request.get("id"), -32602, str(exc)))
            continue
        except RuntimeError as exc:
            _write_json(_jsonrpc_error(request.get("id"), -32603, str(exc)))
            continue

        _write_json(_jsonrpc_result(request.get("id"), result))
        if should_exit:
            break

    state.lifecycle_state = "stopped"
    definitions = list(state.definitions.values())
    definitions.sort(key=lambda d: d.config.name)
    return definitions


def _build_state(initial_definitions: List[McpServerDefinition]) -> StdioState:
    manager = McpManager()
    definitions: Dict[str, McpServerDefinition] = {}
    running: Dict[str, bool] = {}

    for definition in initial_definitions:
        name = definition.config.name
        should_start = definition.config.enabled
        definitions[name] = definition
        if should_start:
            manager.register_server(
                definition.config,
                definition.filter,
                _default_stdio_client(name),
            )
            running[name] = True
        else:
            running[name] = False

    return StdioState(
        manager=manager,
        definitions=definitions,
        running=running,
        lifecycle_state="running",
    )


def _default_stdio_client(server_name: str) -> InMemoryMcpClient:
    health_uri = f"mcp://{server_name}/health"
    capabilities_uri = f"mcp://{server_name}/capabilities"
    return (
        InMemoryMcpClient()
        .with_tool("health", {"status": "ok", "server_name": server_name})
        .with_tool(
            "capabilities",
            {
                "tools": ["health", "capabilities"],
                "resources": [health_uri, capabilities_uri],
            },
        )
        .with_resource(health_uri, {"status": "ok", "server_name": server_name})
        .with_resource(
            capabilities_uri,
            {
                "server_name": server_name,
                "methods": _default_rpc_methods(),
            },
        )
    )


def _default_rpc_methods() -> List[str]:
    return [
        "initialize",
        "healthz",
        "capabilities",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/read",
        "server/list",
        "server/register",
        "server/start",
        "server/stop",
        "server/unregister",
        "shutdown",
    ]


def _dispatch(state: StdioState, method: str, params: dict) -> Tuple[dict, bool]:
    if method in ("initialize", "capabilities"):
        return (
            {
                "server": "ds-code-mcp",
                "transport": "stdio",
                "methods": _default_rpc_methods(),
                "lifecycle": _lifecycle_snapshot(state),
            },
            False,
        )
    if method == "healthz":
        return (
            {
                "status": "ok",
                "service": "ds-code-mcp",
                "transport": "stdio",
                "lifecycle": _lifecycle_snapshot(state),
            },
            False,
        )
    if method == "tools/list":
        server = params.get("server")
        tools = state.manager.list_tools()
        if server:
            tools = [tool for tool in tools if tool.server_name == server]
        return ({"tools": [tool.__dict__ for tool in tools]}, False)
    if method == "tools/call":
        name = params.get("name") or params.get("tool")
        server = params.get("server")
        arguments = params.get("arguments") or {}
        if not name:
            raise ValueError("missing tool name")
        if name.startswith("mcp__"):
            result = state.manager.call_qualified_tool(name, arguments)
            return ({"result": result}, False)
        if not server:
            raise ValueError("missing server for unqualified tool")
        result = state.manager.call_tool(server, name, arguments)
        return ({"result": result}, False)
    if method == "resources/list":
        server = params.get("server")
        resources = state.manager.list_resources()
        if server:
            resources = [r for r in resources if r.server_name == server]
        return ({"resources": [resource.__dict__ for resource in resources]}, False)
    if method == "resources/read":
        uri = params.get("uri")
        server = params.get("server") or _parse_server_from_uri(uri)
        if not uri or not server:
            raise ValueError("missing server for resource read")
        resource = state.manager.read_resource(server, uri)
        return ({"resource": resource}, False)
    if method in ("server/list", "servers/list"):
        return ({"lifecycle": _lifecycle_snapshot(state)}, False)
    if method in ("server/register", "servers/register"):
        server_def = params.get("server")
        if not server_def or not server_def.get("name"):
            raise ValueError("server.name must not be empty")
        name = server_def["name"]
        config = _server_config_from_dict(server_def)
        tool_filter = _tool_filter_from_dict(params.get("filter") or {})
        definition = McpServerDefinition(config=config, filter=tool_filter)
        state.definitions[name] = definition
        should_run = bool(params.get("start", True)) and config.enabled
        if should_run:
            state.manager.register_server(config, tool_filter, _default_stdio_client(name))
        state.running[name] = should_run
        return ({"lifecycle": _lifecycle_snapshot(state)}, False)
    if method in ("server/start", "servers/start"):
        name = params.get("name")
        if not name:
            raise ValueError("missing server name")
        definition = state.definitions.get(name)
        if not definition:
            raise ValueError(f"server '{name}' is not defined")
        if not definition.config.enabled:
            raise ValueError(f"server '{name}' is disabled")
        if not state.running.get(name):
            state.manager.register_server(
                definition.config,
                definition.filter,
                _default_stdio_client(name),
            )
            state.running[name] = True
        return ({"lifecycle": _lifecycle_snapshot(state)}, False)
    if method in ("server/stop", "servers/stop"):
        name = params.get("name")
        if not name:
            raise ValueError("missing server name")
        if state.running.get(name):
            state.manager.unregister_server(name)
            state.running[name] = False
        return ({"lifecycle": _lifecycle_snapshot(state)}, False)
    if method in ("server/unregister", "servers/unregister"):
        name = params.get("name")
        if not name:
            raise ValueError("missing server name")
        state.manager.unregister_server(name)
        state.definitions.pop(name, None)
        state.running.pop(name, None)
        return ({"lifecycle": _lifecycle_snapshot(state)}, False)
    if method == "shutdown":
        return ({"status": "stopping"}, True)
    raise ValueError(f"unsupported method: {method}")


def _parse_server_from_uri(uri: Optional[str]) -> Optional[str]:
    if not uri:
        return None
    if not uri.startswith("mcp://"):
        return None
    stripped = uri[len("mcp://") :]
    server = stripped.split("/", 1)[0]
    return server or None


def _lifecycle_snapshot(state: StdioState) -> dict:
    servers = []
    for name, definition in state.definitions.items():
        servers.append(
            {
                "name": name,
                "enabled": definition.config.enabled,
                "running": bool(state.running.get(name)),
                "command": definition.config.command,
                "args": list(definition.config.args),
            }
        )
    servers.sort(key=lambda item: item["name"])
    running_count = sum(1 for value in state.running.values() if value)
    return {
        "status": state.lifecycle_state,
        "servers": servers,
        "counts": {"defined": len(state.definitions), "running": running_count},
    }


def _server_config_from_dict(data: dict) -> "McpServerConfig":
    from ds_code.tools.mcp import McpServerConfig

    return McpServerConfig(
        name=data.get("name", ""),
        command=data.get("command", ""),
        args=list(data.get("args") or []),
        env=dict(data.get("env") or {}),
        enabled=bool(data.get("enabled", True)),
    )


def _tool_filter_from_dict(data: dict) -> ToolFilter:
    return ToolFilter(
        allow=list(data.get("allow") or []),
        deny=list(data.get("deny") or []),
    )


def _jsonrpc_result(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _write_json(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()
