"""
API request/response models for DeepSeek and OpenAI-compatible endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union


LEGACY_DEEPSEEK_CONTEXT_WINDOW_TOKENS = 128_000
DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_COMPACTION_TOKEN_THRESHOLD = 102_400
COMPACTION_THRESHOLD_PERCENT = 80


# === Core Message Types ===

@dataclass
class MessageRequest:
    model: str
    messages: List["Message"]
    max_tokens: int
    system: Optional["SystemPrompt"] = None
    tools: Optional[List["Tool"]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[Any] = None
    thinking: Optional[Any] = None
    # "off" | "low" | "medium" | "high" | "max"
    reasoning_effort: Optional[str] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


SystemPrompt = Union["SystemPromptText", "SystemPromptBlocks"]


@dataclass
class SystemPromptText:
    text: str


@dataclass
class SystemPromptBlocks:
    blocks: List["SystemBlock"]


@dataclass
class SystemBlock:
    block_type: str
    text: str
    cache_control: Optional["CacheControl"] = None


@dataclass
class Message:
    role: str
    content: List["ContentBlock"]


ContentBlock = Union[
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ServerToolUseBlock",
    "ToolSearchToolResultBlock",
    "CodeExecutionToolResultBlock",
]


@dataclass
class TextBlock:
    text: str
    cache_control: Optional["CacheControl"] = None


@dataclass
class ThinkingBlock:
    thinking: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: Any
    caller: Optional["ToolCaller"] = None


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: Optional[bool] = None
    content_blocks: Optional[List[Any]] = None


@dataclass
class ServerToolUseBlock:
    id: str
    name: str
    input: Any


@dataclass
class ToolSearchToolResultBlock:
    tool_use_id: str
    content: Any


@dataclass
class CodeExecutionToolResultBlock:
    tool_use_id: str
    content: Any


@dataclass
class CacheControl:
    cache_type: str


@dataclass
class ToolCaller:
    caller_type: str
    tool_id: Optional[str] = None


@dataclass
class Tool:
    tool_type: Optional[str]
    name: str
    description: str
    input_schema: Any
    allowed_callers: Optional[List[str]] = None
    defer_loading: Optional[bool] = None
    input_examples: Optional[List[Any]] = None
    strict: Optional[bool] = None
    cache_control: Optional[CacheControl] = None


@dataclass
class ContainerInfo:
    id: str
    expires_at: Optional[str] = None


@dataclass
class ServerToolUsage:
    code_execution_requests: Optional[int] = None
    tool_search_requests: Optional[int] = None


@dataclass
class MessageResponse:
    id: str
    type: str
    role: str
    content: List[ContentBlock]
    model: str
    stop_reason: Optional[str]
    stop_sequence: Optional[str]
    container: Optional[ContainerInfo]
    usage: "Usage"


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    prompt_cache_hit_tokens: Optional[int] = None
    prompt_cache_miss_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    reasoning_replay_tokens: Optional[int] = None
    server_tool_use: Optional[ServerToolUsage] = None


def context_window_for_model(model: str) -> Optional[int]:
    lower = model.lower()
    if "deepseek" in lower:
        explicit_window = deepseek_context_window_hint(lower)
        if explicit_window is not None:
            return explicit_window
        if "v4" in lower:
            return DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS
        return LEGACY_DEEPSEEK_CONTEXT_WINDOW_TOKENS
    if "claude" in lower:
        return 200_000
    return None


def deepseek_context_window_hint(model_lower: str) -> Optional[int]:
    bytes_list = model_lower.encode("utf-8")
    i = 0
    while i < len(bytes_list):
        if chr(bytes_list[i]).isdigit():
            start = i
            while i < len(bytes_list) and chr(bytes_list[i]).isdigit():
                i += 1
            if i >= len(bytes_list) or bytes_list[i] != ord("k"):
                continue

            before_ok = start == 0 or not chr(bytes_list[start - 1]).isalnum()
            after_ok = i + 1 >= len(bytes_list) or not chr(bytes_list[i + 1]).isalnum()
            if not before_ok or not after_ok:
                continue

            try:
                kilo_tokens = int(model_lower[start:i])
            except ValueError:
                continue

            if 8 <= kilo_tokens <= 1024:
                return kilo_tokens * 1000
        else:
            i += 1
    return None


def compaction_threshold_for_model(model: str) -> int:
    window = context_window_for_model(model)
    if window is None:
        return DEFAULT_COMPACTION_TOKEN_THRESHOLD
    threshold = (window * COMPACTION_THRESHOLD_PERCENT) // 100
    return int(threshold) if threshold > 0 else DEFAULT_COMPACTION_TOKEN_THRESHOLD


def compaction_threshold_for_model_and_effort(
    model: str,
    reasoning_effort: Optional[str],
) -> int:
    _ = reasoning_effort
    return compaction_threshold_for_model(model)


# === Streaming Structures ===

class StreamEventType(str, Enum):
    MessageStart = "message_start"
    ContentBlockStart = "content_block_start"
    ContentBlockDelta = "content_block_delta"
    ContentBlockStop = "content_block_stop"
    MessageDelta = "message_delta"
    MessageStop = "message_stop"
    Ping = "ping"


@dataclass
class StreamEvent:
    type: StreamEventType
    message: Optional[MessageResponse] = None
    index: Optional[int] = None
    content_block: Optional["ContentBlockStart"] = None
    delta: Optional["Delta"] = None
    usage: Optional[Usage] = None


ContentBlockStart = Union[
    "ContentBlockStartText",
    "ContentBlockStartThinking",
    "ContentBlockStartToolUse",
    "ContentBlockStartServerToolUse",
]


@dataclass
class ContentBlockStartText:
    text: str


@dataclass
class ContentBlockStartThinking:
    thinking: str


@dataclass
class ContentBlockStartToolUse:
    id: str
    name: str
    input: Any
    caller: Optional[ToolCaller] = None


@dataclass
class ContentBlockStartServerToolUse:
    id: str
    name: str
    input: Any


class DeltaType(str, Enum):
    TextDelta = "text_delta"
    ThinkingDelta = "thinking_delta"
    InputJsonDelta = "input_json_delta"


@dataclass
class Delta:
    type: DeltaType
    text: Optional[str] = None
    thinking: Optional[str] = None
    partial_json: Optional[str] = None


@dataclass
class MessageDelta:
    stop_reason: Optional[str]
    stop_sequence: Optional[str]


# === Tests (pytest style) ===

def test_v4_snapshots_preserve_context_window() -> None:
    assert context_window_for_model("deepseek-v4-flash-20260423") == DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS
    assert context_window_for_model("deepseek-v4-pro-20260423") == DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS


def test_unknown_legacy_deepseek_models_map_to_128k_context_window() -> None:
    assert context_window_for_model("deepseek-coder") == LEGACY_DEEPSEEK_CONTEXT_WINDOW_TOKENS
    assert context_window_for_model("deepseek-v3.2-0324") == LEGACY_DEEPSEEK_CONTEXT_WINDOW_TOKENS


def test_deepseek_v4_models_map_to_1m_context_window() -> None:
    assert context_window_for_model("deepseek-v4-pro") == DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS
    assert context_window_for_model("deepseek-v4-flash") == DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS
    assert context_window_for_model("deepseek-ai/deepseek-v4-pro") == DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS


def test_deepseek_models_with_k_suffix_use_hint() -> None:
    assert context_window_for_model("deepseek-v3.2-32k") == 32_000
    assert context_window_for_model("deepseek-v3.2-256k-preview") == 256_000
    assert context_window_for_model("deepseek-v3.2-2k-preview") == LEGACY_DEEPSEEK_CONTEXT_WINDOW_TOKENS


def test_compaction_threshold_scales_with_context_window() -> None:
    assert compaction_threshold_for_model("deepseek-v3.2-128k") == 102_400
    assert compaction_threshold_for_model("unknown-model") == 102_400


def test_compaction_scales_for_deepseek_v4_1m_context() -> None:
    assert compaction_threshold_for_model("deepseek-v4-pro") == 800_000


def test_v4_replacement_compaction_ignores_reasoning_effort() -> None:
    assert compaction_threshold_for_model_and_effort("deepseek-v4-pro", "off") == 800_000
    assert compaction_threshold_for_model_and_effort("deepseek-v4-pro", "high") == 800_000
    assert compaction_threshold_for_model_and_effort("deepseek-v4-pro", "max") == 800_000


def test_v4_soft_caps_only_apply_to_v4_models() -> None:
    assert compaction_threshold_for_model_and_effort("deepseek-v3.2-128k", "max") == 102_400
    assert compaction_threshold_for_model_and_effort("unknown-model", "max") == 102_400


def test_v4_replacement_compaction_defaults_to_late_guard_when_effort_unknown() -> None:
    assert compaction_threshold_for_model_and_effort("deepseek-v4-pro", None) == 800_000
    assert compaction_threshold_for_model_and_effort("deepseek-v4-pro", "unknown") == 800_000