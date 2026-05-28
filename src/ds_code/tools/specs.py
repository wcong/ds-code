from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolSpec:
    name: str
    input_schema: dict
    output_schema: dict
    supports_parallel_tool_calls: bool = False
    timeout_ms: Optional[int] = None


@dataclass
class ConfiguredToolSpec:
    spec: ToolSpec
    supports_parallel_tool_calls: bool
