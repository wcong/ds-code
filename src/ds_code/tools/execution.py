from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ToolKind(str, Enum):
    Function = "function"
    Mcp = "mcp"


@dataclass
class ToolPayload:
    type: str
    arguments: Optional[str] = None
    server: Optional[str] = None
    tool: Optional[str] = None
    raw_arguments: Optional[dict] = None
    command: Optional[str] = None
    cwd: Optional[str] = None
    timeout_ms: Optional[int] = None


@dataclass
class ToolCall:
    name: str
    payload: ToolPayload
    raw_tool_call_id: Optional[str] = None

    def execution_subject(self, fallback_cwd: str) -> tuple[str, str, str]:
        if self.payload.type == "local_shell":
            return (
                self.payload.command or self.name,
                self.payload.cwd or fallback_cwd,
                "shell",
            )
        return (self.name, fallback_cwd, "tool")


@dataclass
class ToolOutput:
    ok: bool
    content: str
