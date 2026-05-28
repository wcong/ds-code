from __future__ import annotations

import json
from typing import Optional, Tuple

from ds_code.tools.execution import ToolCall, ToolPayload


class PromptRoute:
    def __init__(self, call: Optional[ToolCall] = None) -> None:
        self.call = call


def parse_prompt(prompt: str) -> PromptRoute:
    text = prompt.strip()
    if text.startswith("tool:"):
        return _parse_tool(text[len("tool:") :].strip())
    if text.startswith("shell:"):
        command = text[len("shell:") :].strip()
        payload = ToolPayload(type="local_shell", command=command)
        return PromptRoute(ToolCall(name="local_shell", payload=payload))
    if text.startswith("mcp:"):
        return _parse_mcp(text[len("mcp:") :].strip())
    return PromptRoute()


def _parse_tool(rest: str) -> PromptRoute:
    name, args = _split_name_args(rest)
    if not name:
        return PromptRoute()
    payload = ToolPayload(type="function", arguments=args)
    return PromptRoute(ToolCall(name=name, payload=payload))


def _parse_mcp(rest: str) -> PromptRoute:
    parts = rest.split(" ", 1)
    head = parts[0]
    raw_args = parts[1] if len(parts) > 1 else "{}"
    server_tool = head.split("::")
    if len(server_tool) != 2:
        return PromptRoute()
    server, tool = server_tool
    try:
        arguments = json.loads(raw_args)
    except json.JSONDecodeError:
        arguments = {"input": raw_args}
    payload = ToolPayload(type="mcp", raw_arguments=arguments, server=server, tool=tool)
    qualified = f"mcp__{server}__{tool}"
    return PromptRoute(ToolCall(name=qualified, payload=payload))


def _split_name_args(value: str) -> Tuple[str, str]:
    if not value:
        return "", ""
    parts = value.split(" ", 1)
    name = parts[0].strip()
    args = parts[1].strip() if len(parts) > 1 else ""
    return name, args
