from typing import Callable, Dict, Optional
import subprocess

from ds_code.tools.mcp import McpManager, McpToolDescriptor
from ds_code.tools.execution import ToolCall, ToolOutput
from ds_code.tools.specs import ConfiguredToolSpec, ToolSpec
from ds_code.tools.validation import validate_input


class ToolRegistry:
    def __init__(self, mcp: Optional[McpManager] = None) -> None:
        self._tools: Dict[str, Callable[[str], str]] = {}
        self._mcp = mcp
        self._specs: Dict[str, ConfiguredToolSpec] = {}

    def register(self, name: str, tool: Callable[[str], str]) -> None:
        self._tools[name] = tool

    def register_spec(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = ConfiguredToolSpec(
            spec=spec, supports_parallel_tool_calls=spec.supports_parallel_tool_calls
        )

    def list_specs(self) -> list[ConfiguredToolSpec]:
        return list(self._specs.values())

    def list_tools(self) -> Dict[str, Callable[[str], str]]:
        return dict(self._tools)

    def list_mcp_tools(self) -> list[McpToolDescriptor]:
        if self._mcp is None:
            return []
        return self._mcp.list_tools()

    def call_mcp_tool(self, qualified_name: str, arguments: dict) -> dict:
        if self._mcp is None:
            raise KeyError("MCP manager not configured")
        return self._mcp.call_qualified_tool(qualified_name, arguments)

    def dispatch(self, call: ToolCall) -> ToolOutput:
        if call.payload.type == "local_shell":
            command = call.payload.command or ""
            if not command.strip():
                return ToolOutput(ok=False, content="Missing shell command")
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=call.payload.cwd or ".",
                    capture_output=True,
                    text=True,
                    timeout=(call.payload.timeout_ms or 0) / 1000 or None,
                )
            except subprocess.TimeoutExpired:
                return ToolOutput(ok=False, content="Shell command timed out")
            output = (result.stdout or "") + (result.stderr or "")
            return ToolOutput(ok=result.returncode == 0, content=output.strip())
        if call.name.startswith("mcp__"):
            if self._mcp is None:
                return ToolOutput(ok=False, content="MCP manager not configured")
            result = self._mcp.call_qualified_tool(
                call.name,
                call.payload.raw_arguments or {},
            )
            return ToolOutput(ok=True, content=str(result))

        handler = self._tools.get(call.name)
        if handler is None:
            return ToolOutput(ok=False, content=f"Tool not found: {call.name}")
        spec = self._specs.get(call.name)
        if spec is not None:
            payload = {"arguments": call.payload.arguments or ""}
            ok, error = validate_input(spec.spec.input_schema, payload)
            if not ok:
                return ToolOutput(ok=False, content=error or "invalid input")
        arguments = call.payload.arguments or ""
        return ToolOutput(ok=True, content=str(handler(arguments)))
