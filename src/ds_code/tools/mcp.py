from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple
import hashlib
import json
import os


@dataclass
class McpServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ToolFilter:
    allow: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)


@dataclass
class McpServerDefinition:
    config: McpServerConfig
    filter: ToolFilter = field(default_factory=ToolFilter)


@dataclass
class McpToolDescriptor:
    server_name: str
    tool_name: str
    qualified_name: str
    description: Optional[str] = None


@dataclass
class McpResourceDescriptor:
    server_name: str
    uri: str
    description: Optional[str] = None


class McpManagedClient(Protocol):
    def list_tools(self) -> List[McpToolDescriptor]:
        ...

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        ...

    def list_resources(self) -> List[McpResourceDescriptor]:
        ...

    def read_resource(self, uri: str) -> dict:
        ...


class InMemoryMcpClient:
    def __init__(self) -> None:
        self._tools: Dict[str, dict] = {}
        self._resources: Dict[str, dict] = {}

    def with_tool(self, name: str, sample_result: dict) -> "InMemoryMcpClient":
        self._tools[name] = sample_result
        return self

    def with_resource(self, uri: str, data: dict) -> "InMemoryMcpClient":
        self._resources[uri] = data
        return self

    def list_tools(self) -> List[McpToolDescriptor]:
        return [
            McpToolDescriptor(
                server_name="in-memory",
                tool_name=name,
                qualified_name=name,
                description=None,
            )
            for name in self._tools.keys()
        ]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        if tool_name not in self._tools:
            raise KeyError(f"tool '{tool_name}' not found")
        return dict(self._tools[tool_name])

    def list_resources(self) -> List[McpResourceDescriptor]:
        return [
            McpResourceDescriptor(
                server_name="in-memory",
                uri=uri,
                description=None,
            )
            for uri in self._resources.keys()
        ]

    def read_resource(self, uri: str) -> dict:
        if uri not in self._resources:
            raise KeyError(f"resource '{uri}' not found")
        return dict(self._resources[uri])


class McpManager:
    def __init__(self) -> None:
        self._configs: Dict[str, Tuple[McpServerConfig, ToolFilter]] = {}
        self._clients: Dict[str, McpManagedClient] = {}

    def register_server(
        self,
        config: McpServerConfig,
        tool_filter: ToolFilter,
        client: McpManagedClient,
    ) -> None:
        self._clients[config.name] = client
        self._configs[config.name] = (config, tool_filter)
        self._save_state()

    def unregister_server(self, server_name: str) -> None:
        had_config = server_name in self._configs
        self._configs.pop(server_name, None)
        self._clients.pop(server_name, None)
        if not had_config:
            raise KeyError(f"server '{server_name}' is not registered")
        self._save_state()

    def start_server(self, server_name: str, client: McpManagedClient) -> None:
        if server_name not in self._configs:
            raise KeyError(f"server '{server_name}' is not registered")
        self._clients[server_name] = client
        self._save_state()

    def stop_server(self, server_name: str) -> None:
        if server_name not in self._configs:
            raise KeyError(f"server '{server_name}' is not registered")
        if server_name in self._clients:
            self._clients.pop(server_name, None)
        self._save_state()

    def load_state(self) -> None:
        path = _default_state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        for item in payload.get("servers", []):
            config = McpServerConfig(
                name=item.get("name", ""),
                command=item.get("command", ""),
                args=item.get("args", []),
                env=item.get("env", {}),
                enabled=item.get("enabled", True),
            )
            tool_filter = ToolFilter(
                allow=item.get("allow", []),
                deny=item.get("deny", []),
            )
            self._configs[config.name] = (config, tool_filter)
            if item.get("running"):
                self._clients[config.name] = InMemoryMcpClient()

    def _save_state(self) -> None:
        path = _default_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "servers": [
                {
                    "name": config.name,
                    "command": config.command,
                    "args": list(config.args),
                    "env": dict(config.env),
                    "enabled": config.enabled,
                    "allow": list(tool_filter.allow),
                    "deny": list(tool_filter.deny),
                    "running": name in self._clients,
                }
                for name, (config, tool_filter) in self._configs.items()
            ]
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def list_tools(self) -> List[McpToolDescriptor]:
        out: List[McpToolDescriptor] = []
        for server_name, (_, tool_filter) in self._configs.items():
            client = self._clients.get(server_name)
            if client is None:
                continue
            for tool in client.list_tools():
                if not _allowed_by_filter(tool.tool_name, tool_filter):
                    continue
                qualified_name = qualify_tool_name(server_name, tool.tool_name)
                out.append(
                    McpToolDescriptor(
                        server_name=server_name,
                        tool_name=tool.tool_name,
                        qualified_name=qualified_name,
                        description=tool.description,
                    )
                )
        return out

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        client = self._clients.get(server_name)
        if client is None:
            raise KeyError(f"MCP server '{server_name}' not available")
        return client.call_tool(tool_name, arguments)

    def call_qualified_tool(self, qualified_tool_name: str, arguments: dict) -> dict:
        server_name, tool_name = parse_qualified_tool_name(qualified_tool_name)
        return self.call_tool(server_name, tool_name, arguments)

    def list_resources(self) -> List[McpResourceDescriptor]:
        out: List[McpResourceDescriptor] = []
        for server_name in self._configs.keys():
            client = self._clients.get(server_name)
            if client is None:
                continue
            for resource in client.list_resources():
                out.append(
                    McpResourceDescriptor(
                        server_name=server_name,
                        uri=resource.uri,
                        description=resource.description,
                    )
                )
        return out

    def read_resource(self, server_name: str, uri: str) -> dict:
        client = self._clients.get(server_name)
        if client is None:
            raise KeyError(f"MCP server '{server_name}' not available")
        return client.read_resource(uri)

    def update_sandbox_state(self, sandbox_mode: str, cwd: str) -> List[dict]:
        notices: List[dict] = []
        for server_name in self._configs.keys():
            notices.append(
                {
                    "server_name": server_name,
                    "method": "codex/sandbox-state/update",
                    "params": {"sandbox_mode": sandbox_mode, "cwd": cwd},
                }
            )
        return notices


def _allowed_by_filter(name: str, tool_filter: ToolFilter) -> bool:
    if any(pattern == name for pattern in tool_filter.deny):
        return False
    if not tool_filter.allow:
        return True
    return any(pattern == name for pattern in tool_filter.allow)


def _sanitize_component(value: str) -> str:
    return "".join(
        ch.lower() if (ch.isalnum() or ch == "_") else "_" for ch in value
    )


def qualify_tool_name(server: str, tool: str) -> str:
    name = f"mcp__{_sanitize_component(server)}__{_sanitize_component(tool)}"
    if len(name) > 64:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        name = f"{name[:48]}_{digest[:12]}"
    return name


def parse_qualified_tool_name(value: str) -> Tuple[str, str]:
    if not value.startswith("mcp__"):
        raise ValueError("missing mcp__ prefix")
    stripped = value[len("mcp__") :]
    parts = stripped.split("__", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("missing server or tool segment")
    return parts[0], parts[1]


def _default_state_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".ds_code", "mcp_state.json")


def parse_json_request(payload: str) -> dict:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json: {exc}") from exc
