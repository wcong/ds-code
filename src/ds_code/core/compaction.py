"""
Context compaction for long conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from ds_code.core.models import SystemPrompt, CacheControl,SystemPromptText, SystemPromptBlocks, SystemBlock, ThinkingBlock , Usage, Message, MessageRequest, TextBlock, ToolResultBlock, ToolUseBlock
from ds_code.core.client import DeepSeekClient
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, Union
import asyncio
import json
import logging
import re
import time

logger = logging.getLogger(__name__)


# -----------------------------
# Constants and configuration
# -----------------------------

DEFAULT_TEXT_MODEL = "deepseek-v4-flash"

MINIMUM_AUTO_COMPACTION_TOKENS = 500_000

KEEP_RECENT_MESSAGES = 4
RECENT_WORKING_SET_WINDOW = 12
MAX_WORKING_SET_PATHS = 24
MIN_SUMMARIZE_MESSAGES = 6
SUMMARY_TEXT_SNIPPET_CHARS = 800
SUMMARY_TOOL_RESULT_SNIPPET_CHARS = 240
SUMMARY_INPUT_MAX_CHARS = 24_000
SUMMARY_INPUT_HEAD_CHARS = 14_000
SUMMARY_INPUT_TAIL_CHARS = 6_000
LARGE_CONTEXT_SUMMARY_TEXT_SNIPPET_CHARS = 2_000
LARGE_CONTEXT_SUMMARY_TOOL_RESULT_SNIPPET_CHARS = 4_000
LARGE_CONTEXT_SUMMARY_INPUT_MAX_CHARS = 120_000
LARGE_CONTEXT_SUMMARY_INPUT_HEAD_CHARS = 72_000
LARGE_CONTEXT_SUMMARY_INPUT_TAIL_CHARS = 36_000
TOOL_PRUNE_STOP_CHECK_BYTES = 16 * 1024
LARGE_CONTEXT_SUMMARY_MAX_TOKENS = 2_048
LARGE_CONTEXT_WINDOW_TOKENS = 500_000
CACHE_ALIGNED_SUMMARY_CONTEXT_BUDGET_PERCENT = 85


@dataclass
class CompactionConfig:
    enabled: bool = True
    token_threshold: int = 800_000
    model: str = DEFAULT_TEXT_MODEL
    cache_summary: bool = True
    auto_floor_tokens: int = MINIMUM_AUTO_COMPACTION_TOKENS


@dataclass
class SummaryInputLimits:
    text_snippet_chars: int
    tool_result_snippet_chars: int
    input_max_chars: int
    input_head_chars: int
    input_tail_chars: int
    max_tokens: int
    word_limit: int


@dataclass
class CompactionPlan:
    pinned_indices: Set[int] = field(default_factory=set)
    summarize_indices: List[int] = field(default_factory=list)


@dataclass
class CompactionResult:
    messages: List[Message]
    summary_prompt: Optional[SystemPrompt]
    removed_messages: List[Message]
    retries_used: int


# -----------------------------
# Error classification helpers
# -----------------------------

class ErrorCategory(str, Enum):
    Network = "network"
    RateLimit = "rate_limit"
    Timeout = "timeout"
    InvalidInput = "invalid_input"
    Other = "other"


def classify_error_message(text: str) -> ErrorCategory:
    lower = text.lower()
    if "timeout" in lower:
        return ErrorCategory.Timeout
    if "rate limit" in lower or "too many requests" in lower or "429" in lower:
        return ErrorCategory.RateLimit
    if "network error" in lower or "connection refused" in lower or "connection" in lower:
        return ErrorCategory.Network
    if (
        "invalid request" in lower
        or "bad request" in lower
        or "prompt is too long" in lower
        or "maximum context length" in lower
        or "context window" in lower
        or "missing required field" in lower
        or "requested" in lower
    ):
        return ErrorCategory.InvalidInput
    return ErrorCategory.Other


# -----------------------------
# External-like helpers
# -----------------------------

def context_window_for_model(model: str) -> Optional[int]:
    model_lower = model.lower()
    if "deepseek-v4" in model_lower:
        return 1_000_000
    if "128k" in model_lower:
        return 128_000
    return None


def report_cost(model: str, usage: Usage) -> None:
    # Placeholder for cost reporting side-channel
    _ = (model, usage)


# -----------------------------
# Core logic
# -----------------------------

_PATH_RE: Optional[re.Pattern] = None


def path_regex() -> re.Pattern:
    global _PATH_RE
    if _PATH_RE is None:
        _PATH_RE = re.compile(
            r"""
            (?:
                (?P<root>
                    pyproject\.toml|
                    README\.md|
                    CHANGELOG\.md|
                    AGENTS\.md|
                    config\.example\.toml
                )
            )
            |
            (?P<path>
                (?:[A-Za-z0-9._-]+/)+
                [A-Za-z0-9._-]+
                \.(?:py|toml|md|json|ya?ml|txt)
            )
            """,
            re.VERBOSE,
        )
    return _PATH_RE


def normalize_path_candidate(candidate: str, workspace: Optional[Path]) -> Optional[str]:
    if not candidate:
        return None

    cleaned = candidate.replace("\\", "/")
    path = Path(cleaned)

    if path.is_absolute():
        if workspace is None:
            return None
        try:
            path = path.relative_to(workspace)
        except ValueError:
            return None

    rel = str(path).lstrip("./")
    if not rel or ".." in rel:
        return None

    if workspace is not None:
        repo_path = workspace / rel
        if repo_path.exists() or looks_repo_relative(rel):
            return rel
        return None

    if looks_repo_relative(rel):
        return rel

    return None


def looks_repo_relative(path: str) -> bool:
    return (
        path in {
            "pyproject.toml",
            "README.md",
            "CHANGELOG.md",
            "AGENTS.md",
            "config.example.toml",
        }
        or path.startswith("src/")
        or path.startswith("tests/")
        or path.startswith("docs/")
        or path.startswith("examples/")
        or path.startswith("benches/")
        or path.startswith(".github/")
        or ("/" in path and "." in path.rsplit("/", 1)[-1])
    )


def extract_paths_from_text(text: str, workspace: Optional[Path]) -> List[str]:
    out: List[str] = []
    for match in path_regex().finditer(text):
        candidate = match.group("path") or match.group("root")
        if not candidate:
            continue
        normalized = normalize_path_candidate(candidate, workspace)
        if normalized:
            out.append(normalized)
    return out


def extract_paths_from_tool_input(input_value: Any, workspace: Optional[Path]) -> List[str]:
    out: List[str] = []
    if not isinstance(input_value, dict):
        return out

    for key in ("path", "file", "target", "cwd"):
        val = input_value.get(key)
        if isinstance(val, str):
            normalized = normalize_path_candidate(val, workspace)
            if normalized:
                out.append(normalized)

    for key in ("paths", "files", "targets"):
        vals = input_value.get(key)
        if isinstance(vals, list):
            for val in vals:
                if isinstance(val, str):
                    normalized = normalize_path_candidate(val, workspace)
                    if normalized:
                        out.append(normalized)

    return out


def message_text(msg: Message) -> str:
    parts: List[str] = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(f"[tool_use:{block.name}] {block.input}")
        elif isinstance(block, ToolResultBlock):
            parts.append(block.content)
    return "\n".join(parts) + ("\n" if parts else "")


def is_user_text_query(msg: Message) -> bool:
    return msg.role == "user" and any(isinstance(b, TextBlock) for b in msg.content)


def extract_paths_from_message(message: Message, workspace: Optional[Path]) -> List[str]:
    paths: List[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            paths.extend(extract_paths_from_text(block.text, workspace))
        elif isinstance(block, ToolResultBlock):
            paths.extend(extract_paths_from_text(block.content, workspace))
        elif isinstance(block, ToolUseBlock):
            paths.extend(extract_paths_from_tool_input(block.input, workspace))
    return paths


def derive_working_set_paths(
    messages: Sequence[Message],
    workspace: Optional[Path],
    seed_indices: Sequence[int],
) -> Set[str]:
    paths: List[str] = []
    seen: Set[str] = set()

    seeds = [idx for idx in seed_indices if idx < len(messages)]
    seeds.sort(reverse=True)

    for idx in seeds:
        for candidate in extract_paths_from_message(messages[idx], workspace):
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
                if len(paths) >= MAX_WORKING_SET_PATHS:
                    return set(paths)

    for msg in list(messages)[-RECENT_WORKING_SET_WINDOW:][::-1]:
        for candidate in extract_paths_from_message(msg, workspace):
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
                if len(paths) >= MAX_WORKING_SET_PATHS:
                    return set(paths)

    return set(paths)


def should_pin_message(text: str, working_set_paths: Set[str]) -> bool:
    lower = text.lower()

    if any(p in text for p in working_set_paths):
        return True

    error_markers = [
        "error:",
        "error ",
        "failed",
        "panic",
        "traceback",
        "stack trace",
        "assertion failed",
        "test failed",
    ]
    if any(m in lower for m in error_markers):
        return True

    patch_markers = [
        "diff --git",
        "+++ b/",
        "--- a/",
        "*** begin patch",
        "*** update file:",
        "*** add file:",
        "*** delete file:",
        "```diff",
        "apply_patch",
    ]
    return any(m in lower for m in patch_markers)


def plan_compaction(
    messages: Sequence[Message],
    workspace: Optional[Path],
    keep_recent: int,
    external_pins: Optional[Sequence[int]],
    external_working_set_paths: Optional[Sequence[str]],
) -> CompactionPlan:
    pinned_indices: Set[int] = set()
    length = len(messages)
    if length == 0:
        return CompactionPlan()

    recent_start = max(0, length - keep_recent)
    pinned_indices.update(range(recent_start, length))

    seed_indices = list(external_pins or [])
    working_set_paths = derive_working_set_paths(messages, workspace, seed_indices)
    if external_working_set_paths:
        for path in external_working_set_paths:
            normalized = normalize_path_candidate(path, workspace)
            if normalized:
                working_set_paths.add(normalized)

    for idx, msg in enumerate(messages):
        if idx in pinned_indices:
            continue
        text = message_text(msg)
        if should_pin_message(text, working_set_paths):
            pinned_indices.add(idx)

    if external_pins:
        pinned_indices.update(idx for idx in external_pins if idx < length)

    enforce_tool_call_pairs(messages, pinned_indices)

    if not any(is_user_text_query(messages[idx]) for idx in pinned_indices):
        for idx in range(length - 1, -1, -1):
            if is_user_text_query(messages[idx]):
                pinned_indices.add(idx)
                break

    summarize_indices = [i for i in range(length) if i not in pinned_indices]
    return CompactionPlan(pinned_indices=pinned_indices, summarize_indices=summarize_indices)


def enforce_tool_call_pairs(messages: Sequence[Message], pinned_indices: Set[int]) -> None:
    if not pinned_indices:
        return

    call_id_to_idx: Dict[str, int] = {}
    result_id_to_idx: Dict[str, int] = {}

    for idx, msg in enumerate(messages):
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                call_id_to_idx[block.id] = idx
            elif isinstance(block, ToolResultBlock):
                result_id_to_idx[block.tool_use_id] = idx

    permanently_removed: Set[int] = set()

    max_iters = max(len(messages), 10)
    converged = False
    for _ in range(max_iters):
        to_add: List[int] = []
        to_remove: List[int] = []

        snapshot = list(pinned_indices)
        for idx in snapshot:
            msg = messages[idx]
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    call_idx = call_id_to_idx.get(block.tool_use_id)
                    if call_idx is not None and call_idx not in permanently_removed:
                        to_add.append(call_idx)
                    else:
                        to_remove.append(idx)
                elif isinstance(block, ToolUseBlock):
                    result_idx = result_id_to_idx.get(block.id)
                    if result_idx is not None and result_idx not in permanently_removed:
                        to_add.append(result_idx)
                    else:
                        to_remove.append(idx)

        remove_set = set(to_remove)
        changed = False
        for idx in to_add:
            if idx not in remove_set and idx not in pinned_indices:
                pinned_indices.add(idx)
                changed = True
        for idx in to_remove:
            if idx in pinned_indices:
                pinned_indices.remove(idx)
                permanently_removed.add(idx)
                changed = True

        if not changed:
            converged = True
            break

    if not converged:
        logger.warning(
            "enforce_tool_call_pairs did not converge after %d iterations (%d messages, %d pinned)",
            max_iters,
            len(messages),
            len(pinned_indices),
        )


def estimate_tokens_for_message(message: Message, include_thinking: bool) -> int:
    total = 0
    for block in message.content:
        if isinstance(block, TextBlock):
            total += len(block.text) // 4
        elif isinstance(block, ThinkingBlock):
            if include_thinking:
                total += len(block.thinking) // 4
        elif isinstance(block, ToolUseBlock):
            try:
                raw = json.dumps(block.input)
                total += len(raw) // 4
            except Exception:
                total += 100
        elif isinstance(block, ToolResultBlock):
            total += len(block.content) // 4
    return total


def message_has_tool_use(message: Message) -> bool:
    return any(isinstance(block, ToolUseBlock) for block in message.content)


def estimate_tokens(messages: Sequence[Message]) -> int:
    return sum(
        estimate_tokens_for_message(msg, message_has_tool_use(msg))
        for msg in messages
    )


def estimate_text_tokens_conservative(text: str) -> int:
    return (len(text) + 2) // 3


def estimate_system_tokens_conservative(system: Optional[SystemPrompt]) -> int:
    if isinstance(system, SystemPromptText):
        return estimate_text_tokens_conservative(system.text)
    if isinstance(system, SystemPromptBlocks):
        return sum(estimate_text_tokens_conservative(b.text) for b in system.blocks)
    return 0


def estimate_input_tokens_conservative(
    messages: Sequence[Message],
    system: Optional[SystemPrompt],
) -> int:
    message_tokens = (estimate_tokens(messages) * 3 + 1) // 2
    system_tokens = estimate_system_tokens_conservative(system)
    framing_overhead = len(messages) * 12 + 48
    return message_tokens + system_tokens + framing_overhead


def should_compact(
    messages: Sequence[Message],
    config: CompactionConfig,
    workspace: Optional[Path],
    external_pins: Optional[Sequence[int]],
    external_working_set_paths: Optional[Sequence[str]],
) -> bool:
    if not config.enabled:
        return False

    if config.auto_floor_tokens > 0:
        total_session_tokens = sum(estimate_tokens_for_message(m, False) for m in messages)
        if total_session_tokens < config.auto_floor_tokens:
            return False

    plan = plan_compaction(
        messages,
        workspace,
        KEEP_RECENT_MESSAGES,
        external_pins,
        external_working_set_paths,
    )
    pinned_tokens = sum(
        estimate_tokens_for_message(messages[idx], False) for idx in plan.pinned_indices
    )
    token_estimate = sum(
        estimate_tokens_for_message(messages[idx], False) for idx in plan.summarize_indices
    )
    message_count = len(plan.summarize_indices)

    effective_token_threshold = max(0, config.token_threshold - pinned_tokens)

    if effective_token_threshold == 0:
        return message_count >= MIN_SUMMARIZE_MESSAGES
    if message_count < MIN_SUMMARIZE_MESSAGES:
        return False
    return token_estimate > effective_token_threshold


def truncate_chars(text: str, max_chars: int) -> str:
    if max_chars == 0:
        return ""
    return text[:max_chars]


def tail_chars(text: str, max_chars: int) -> str:
    if max_chars == 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


@dataclass
class ToolUseInfo:
    name: str
    key: str
    args_preview: str


def tool_use_key(name: str, input_value: Any) -> str:
    try:
        raw = json.dumps(input_value)
    except Exception:
        raw = str(input_value)
    return f"{name}:{raw}"


def tool_args_preview(input_value: Any) -> str:
    try:
        raw = json.dumps(input_value)
    except Exception:
        raw = str(input_value)
    return truncate_chars(raw, 120)


def collect_tool_uses(messages: Sequence[Message]) -> Dict[str, ToolUseInfo]:
    tool_uses: Dict[str, ToolUseInfo] = {}
    for message in messages:
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                tool_uses[block.id] = ToolUseInfo(
                    name=block.name,
                    key=tool_use_key(block.name, block.input),
                    args_preview=tool_args_preview(block.input),
                )
    return tool_uses


@dataclass
class ToolResultPruneCandidate:
    message_idx: int
    block_idx: int
    key: str
    tool_name: str
    args_preview: str
    original_len: int


def prune_tool_results(messages: List[Message], protected_window: int) -> int:
    return prune_tool_results_until(messages, protected_window, lambda _m, _b: False)


def prune_tool_results_until(
    messages: List[Message],
    protected_window: int,
    should_stop,
) -> int:
    cutoff = max(0, len(messages) - protected_window)
    if cutoff == 0:
        return 0

    tool_uses = collect_tool_uses(messages)
    candidates: List[ToolResultPruneCandidate] = []
    latest_by_key: Dict[str, int] = {}
    count_by_key: Dict[str, int] = {}

    for message_idx, message in enumerate(messages[:cutoff]):
        for block_idx, block in enumerate(message.content):
            if not isinstance(block, ToolResultBlock):
                continue
            info = tool_uses.get(block.tool_use_id)
            if not info:
                continue
            latest_by_key[info.key] = message_idx
            count_by_key[info.key] = count_by_key.get(info.key, 0) + 1
            candidates.append(
                ToolResultPruneCandidate(
                    message_idx=message_idx,
                    block_idx=block_idx,
                    key=info.key,
                    tool_name=info.name,
                    args_preview=info.args_preview,
                    original_len=len(block.content),
                )
            )

    candidates.reverse()

    bytes_saved = 0
    for candidate in candidates:
        duplicate_count = count_by_key.get(candidate.key, 0)
        is_latest_duplicate = (
            duplicate_count > 1
            and latest_by_key.get(candidate.key) == candidate.message_idx
        )
        if is_latest_duplicate:
            continue
        if duplicate_count <= 1 and candidate.original_len <= SUMMARY_TOOL_RESULT_SNIPPET_CHARS:
            continue

        summary = (
            f"[{candidate.tool_name}] tool result pruned "
            f"({candidate.original_len} bytes; args: {candidate.args_preview})"
        )
        if len(summary) >= candidate.original_len:
            continue

        block = messages[candidate.message_idx].content[candidate.block_idx]
        if isinstance(block, ToolResultBlock):
            bytes_saved += max(0, len(block.content) - len(summary))
            block.content = summary
            block.content_blocks = None

            if should_stop(messages, bytes_saved):
                break

    return bytes_saved


def is_transient_error(err: Exception) -> bool:
    category = classify_error_message(str(err))
    return category in {ErrorCategory.Network, ErrorCategory.RateLimit, ErrorCategory.Timeout}


async def compact_messages_safe(
    client: DeepSeekClient,
    messages: Sequence[Message],
    config: CompactionConfig,
    workspace: Optional[Path],
    external_pins: Optional[Sequence[int]],
    external_working_set_paths: Optional[Sequence[str]],
) -> CompactionResult:
    max_retries = 3
    base_delay_ms = 1000

    was_over_threshold = should_compact(
        messages,
        config,
        workspace,
        external_pins,
        external_working_set_paths,
    )
    pruned_messages = list(messages)
    now_under_threshold = False
    next_stop_check_bytes = 0

    pruned_bytes = prune_tool_results_until(
        pruned_messages,
        KEEP_RECENT_MESSAGES,
        lambda candidate_messages, bytes_saved: _should_stop_after_prune(
            candidate_messages,
            bytes_saved,
            was_over_threshold,
            config,
            workspace,
            external_pins,
            external_working_set_paths,
            next_stop_check_bytes,
        ),
    )

    if pruned_bytes > 0:
        logger.info("Local tool-result prune saved %d bytes before LLM compaction", pruned_bytes)
        now_under_threshold = not should_compact(
            pruned_messages,
            config,
            workspace,
            external_pins,
            external_working_set_paths,
        )
        if was_over_threshold and now_under_threshold:
            return CompactionResult(
                messages=pruned_messages,
                summary_prompt=None,
                removed_messages=[],
                retries_used=0,
            )

    compaction_input = pruned_messages if pruned_bytes > 0 else list(messages)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        if attempt > 0:
            delay = base_delay_ms * (1 << (attempt - 1))
            await asyncio.sleep(delay / 1000.0)

        try:
            msgs, prompt, removed = await compact_messages(
                client,
                compaction_input,
                config,
                workspace,
                external_pins,
                external_working_set_paths,
            )
            return CompactionResult(
                messages=msgs,
                summary_prompt=prompt,
                removed_messages=removed,
                retries_used=attempt,
            )
        except Exception as err:
            if not is_transient_error(err):
                raise
            last_error = err

    raise last_error or RuntimeError(f"Compaction failed after {max_retries} retries")


def _should_stop_after_prune(
    candidate_messages: Sequence[Message],
    bytes_saved: int,
    was_over_threshold: bool,
    config: CompactionConfig,
    workspace: Optional[Path],
    external_pins: Optional[Sequence[int]],
    external_working_set_paths: Optional[Sequence[str]],
    next_stop_check_bytes: int,
) -> bool:
    if not was_over_threshold or bytes_saved < next_stop_check_bytes:
        return False
    # This helper mirrors the Rust behavior but is intentionally conservative.
    return not should_compact(
        candidate_messages, config, workspace, external_pins, external_working_set_paths
    )


def read_workspace_anchors(workspace: Optional[Path]) -> List[str]:
    if workspace is None:
        return []

    anchors_path = workspace / ".deepseek" / "anchors.md"
    try:
        content = anchors_path.read_text()
    except OSError:
        return []

    return [
        chunk.strip()
        for chunk in content.split("\n---\n")
        if chunk.strip()
    ]


def anchor_summary_section(workspace: Optional[Path]) -> str:
    anchors = read_workspace_anchors(workspace)
    if not anchors:
        return ""

    section = (
        "## Pinned Facts (User Anchors)\n\n"
        "The following facts were explicitly anchored by the user with `/anchor`. "
        "Preserve them across compaction cycles.\n\n"
    )
    for anchor in anchors:
        section += f"- {anchor}\n"
    section += "\n---\n\n"
    return section


async def compact_messages(
    client: DeepSeekClient,
    messages: Sequence[Message],
    config: CompactionConfig,
    workspace: Optional[Path],
    external_pins: Optional[Sequence[int]],
    external_working_set_paths: Optional[Sequence[str]],
) -> Tuple[List[Message], Optional[SystemPrompt], List[Message]]:
    if not messages:
        return [], None, []

    plan = plan_compaction(
        messages,
        workspace,
        KEEP_RECENT_MESSAGES,
        external_pins,
        external_working_set_paths,
    )
    if not plan.summarize_indices:
        return list(messages), None, []

    to_summarize = [messages[idx] for idx in plan.summarize_indices]

    summary = await create_summary(client, to_summarize, config.model)
    workflow_context = extract_workflow_context(to_summarize, workspace)
    anchors_section = anchor_summary_section(workspace)

    summary_block = SystemBlock(
        block_type="text",
        text=(
            f"{anchors_section}"
            "## Conversation Summary (Auto-Generated)\n\n"
            f"{summary}\n\n"
            "---\n\n"
            "## Workflow Context\n\n"
            f"{workflow_context}\n\n"
            "---\n\n"
            "## What to Do Next\n\n"
            "You have just resumed from a context compaction. The conversation above was summarized "
            "to save space. Review the summary and workflow context, then continue helping the user "
            "with their task. If you need more details about the summarized portion, ask the user to clarify.\n\n"
            "---\n\n"
            "Pinned messages follow:"
        ),
        cache_control=CacheControl(cache_type="ephemeral") if config.cache_summary else None,
    )

    pinned_messages = [
        msg for idx, msg in enumerate(messages) if idx in plan.pinned_indices
    ]

    return pinned_messages, SystemPromptBlocks([summary_block]), to_summarize


async def create_summary(
    client: DeepSeekClient,
    messages: Sequence[Message],
    model: str,
) -> str:
    limits = summary_input_limits_for_model(model)
    used_cache_aligned = should_use_cache_aligned_summary(model, messages)
    request = (
        build_cache_aligned_summary_request(model, messages, limits)
        if used_cache_aligned
        else build_formatted_summary_request(model, messages, limits)
    )

    telemetry_cache_aligned = used_cache_aligned
    try:
        response = await client.create_message(request)
    except Exception as err:
        if used_cache_aligned and is_context_window_error(err):
            logger.warning(
                "Cache-aligned compaction summary exceeded the model context window (%s); "
                "retrying with bounded formatted summary input",
                err,
            )
            telemetry_cache_aligned = False
            fallback_request = build_formatted_summary_request(model, messages, limits)
            response = await client.create_message(fallback_request)
        else:
            raise

    report_cost(response.model, response.usage)
    log_summary_cache_telemetry(telemetry_cache_aligned, response.usage)

    summary_chunks = [
        block.text for block in response.content if isinstance(block, TextBlock)
    ]
    return "\n".join(summary_chunks)


def is_context_window_error(err: Exception) -> bool:
    text = str(err)
    if classify_error_message(text) != ErrorCategory.InvalidInput:
        return False
    lower = text.lower()
    return (
        "context" in lower
        or "token" in lower
        or "prompt is too long" in lower
        or "requested" in lower
        or "maximum" in lower
    )


def summary_cache_hit_percent(cache_hit: int, input_tokens: int) -> float:
    if input_tokens > 0:
        return (float(cache_hit) * 100.0) / float(input_tokens)
    return 0.0


def log_summary_cache_telemetry(used_cache_aligned: bool, usage: Usage) -> None:
    path = "cache_aligned" if used_cache_aligned else "fallback"
    cache_hit = usage.prompt_cache_hit_tokens or 0
    cache_miss = usage.prompt_cache_miss_tokens or 0
    cache_hit_pct = summary_cache_hit_percent(cache_hit, usage.input_tokens)
    logger.debug(
        "compaction summary call: path=%s prompt_tokens=%d cache_hit_tokens=%d cache_miss_tokens=%d cache_hit_pct=%.1f",
        path,
        usage.input_tokens,
        cache_hit,
        cache_miss,
        cache_hit_pct,
    )


def should_use_cache_aligned_summary(model: str, messages: Sequence[Message]) -> bool:
    window = context_window_for_model(model)
    if window is None or window < LARGE_CONTEXT_WINDOW_TOKENS:
        return False

    budget = (window * CACHE_ALIGNED_SUMMARY_CONTEXT_BUDGET_PERCENT) // 100
    summary_prompt_tokens = 512
    return estimate_tokens(messages) + summary_prompt_tokens <= budget


def summary_instruction(word_limit: int) -> str:
    return (
        "Summarize the conversation above in a concise but comprehensive way. "
        "Preserve key information, decisions made, exact file paths, commands, "
        "errors, and tool-result facts needed to continue the work. "
        "Tool outputs may be abbreviated only when they are repetitive. "
        f"Keep it under {word_limit} words."
    )


def summary_input_limits_for_model(model: str) -> SummaryInputLimits:
    is_large_context = (
        context_window_for_model(model) is not None
        and context_window_for_model(model) >= LARGE_CONTEXT_WINDOW_TOKENS
    )
    if is_large_context:
        return SummaryInputLimits(
            text_snippet_chars=LARGE_CONTEXT_SUMMARY_TEXT_SNIPPET_CHARS,
            tool_result_snippet_chars=LARGE_CONTEXT_SUMMARY_TOOL_RESULT_SNIPPET_CHARS,
            input_max_chars=LARGE_CONTEXT_SUMMARY_INPUT_MAX_CHARS,
            input_head_chars=LARGE_CONTEXT_SUMMARY_INPUT_HEAD_CHARS,
            input_tail_chars=LARGE_CONTEXT_SUMMARY_INPUT_TAIL_CHARS,
            max_tokens=LARGE_CONTEXT_SUMMARY_MAX_TOKENS,
            word_limit=900,
        )
    return SummaryInputLimits(
        text_snippet_chars=SUMMARY_TEXT_SNIPPET_CHARS,
        tool_result_snippet_chars=SUMMARY_TOOL_RESULT_SNIPPET_CHARS,
        input_max_chars=SUMMARY_INPUT_MAX_CHARS,
        input_head_chars=SUMMARY_INPUT_HEAD_CHARS,
        input_tail_chars=SUMMARY_INPUT_TAIL_CHARS,
        max_tokens=1_024,
        word_limit=500,
    )


def build_cache_aligned_summary_request(
    model: str,
    messages: Sequence[Message],
    limits: SummaryInputLimits,
) -> MessageRequest:
    request_messages = list(messages)
    request_messages.append(
        Message(
            role="user",
            content=[TextBlock(text=summary_instruction(limits.word_limit))],
        )
    )
    return MessageRequest(
        model=model,
        messages=request_messages,
        max_tokens=limits.max_tokens,
        system=None,
        tools=None,
        tool_choice=None,
        metadata=None,
        thinking=None,
        reasoning_effort=None,
        stream=False,
        temperature=0.3,
        top_p=None,
    )


def build_formatted_summary_request(
    model: str,
    messages: Sequence[Message],
    limits: SummaryInputLimits,
) -> MessageRequest:
    conversation_text = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        for block in msg.content:
            if isinstance(block, TextBlock):
                snippet = truncate_chars(block.text, limits.text_snippet_chars)
                conversation_text.append(f"{role}: {snippet}\n")
            elif isinstance(block, ToolUseBlock):
                conversation_text.append(f"{role}: [Used tool: {block.name}]\n")
            elif isinstance(block, ToolResultBlock):
                snippet = truncate_chars(block.content, limits.tool_result_snippet_chars)
                conversation_text.append(f"Tool result: {snippet}\n")

    conversation_text = "\n".join(conversation_text)
    conversation_chars = len(conversation_text)
    if conversation_chars > limits.input_max_chars:
        head = truncate_chars(conversation_text, limits.input_head_chars)
        tail = tail_chars(conversation_text, limits.input_tail_chars)
        omitted = max(0, conversation_chars - len(head) - len(tail))
        conversation_text = (
            f"{head}\n\n[... {omitted} characters omitted before summary ...]\n\n{tail}"
        )

    return MessageRequest(
        model=model,
        messages=[
            Message(
                role="user",
                content=[
                    TextBlock(
                        text=f"{summary_instruction(limits.word_limit)}\n\n---\n\n{conversation_text}"
                    )
                ],
            )
        ],
        max_tokens=limits.max_tokens,
        system=SystemPromptText(
            text="You are a helpful assistant that creates concise conversation summaries."
        ),
        tools=None,
        tool_choice=None,
        metadata=None,
        thinking=None,
        reasoning_effort=None,
        stream=False,
        temperature=0.3,
        top_p=None,
    )


def extract_workflow_context(messages: Sequence[Message], workspace: Optional[Path]) -> str:
    files_touched: List[str] = []
    tools_used: List[str] = []
    tasks_identified: List[str] = []

    for msg in messages:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                tools_used.append(block.name)
                path = extract_path_from_input(block.input)
                if path and path not in files_touched:
                    files_touched.append(path)
            elif isinstance(block, TextBlock):
                if "TODO" in block.text or "task" in block.text or "need to" in block.text:
                    task = truncate_chars(block.text, 200)
                    if task not in tasks_identified:
                        tasks_identified.append(task)

    context = []
    if files_touched:
        context.append("**Files Modified/Read:**")
        for file in files_touched:
            if workspace is not None:
                try:
                    relative = str(Path(file).relative_to(workspace))
                except ValueError:
                    relative = file
                context.append(f"- `{relative}`")
            else:
                context.append(f"- `{file}`")
        context.append("")

    if tools_used:
        context.append("**Tools Used:** " + ", ".join(tools_used))
        context.append("")

    if tasks_identified:
        context.append("**Tasks/TODOs Identified:**")
        for task in tasks_identified:
            context.append(f"- {task}")
        context.append("")

    if not context:
        return "No specific workflow context detected. Continue assisting the user with their current task.\n"

    return "\n".join(context).rstrip() + "\n"


def extract_path_from_input(input_value: Any) -> Optional[str]:
    for key in ("path", "file", "file_path", "filename"):
        val = input_value.get(key) if isinstance(input_value, dict) else None
        if isinstance(val, str):
            return val

    if isinstance(input_value, dict):
        for value in input_value.values():
            if isinstance(value, str) and ("/" in value or "\\" in value or "." in value):
                return value
    return None


def merge_system_prompts(
    original: Optional[SystemPrompt],
    summary: Optional[SystemPrompt],
) -> Optional[SystemPrompt]:
    if original is None and summary is None:
        return None
    if original is not None and summary is None:
        return original
    if original is None and summary is not None:
        return summary

    if isinstance(original, SystemPromptText) and isinstance(summary, SystemPromptBlocks):
        blocks = [SystemBlock(block_type="text", text=original.text)]
        blocks.extend(summary.blocks)
        return SystemPromptBlocks(blocks)

    if isinstance(original, SystemPromptBlocks) and isinstance(summary, SystemPromptBlocks):
        blocks = list(original.blocks) + list(summary.blocks)
        return SystemPromptBlocks(blocks)

    if isinstance(original, SystemPromptText) and isinstance(summary, SystemPromptText):
        blocks = [
            SystemBlock(block_type="text", text=original.text),
            SystemBlock(block_type="text", text=summary.text),
        ]
        return SystemPromptBlocks(blocks)

    if isinstance(original, SystemPromptBlocks) and isinstance(summary, SystemPromptText):
        blocks = list(original.blocks) + [SystemBlock(block_type="text", text=summary.text)]
        return SystemPromptBlocks(blocks)

    return summary


# -----------------------------
# Tests (pytest style)
# -----------------------------

def _msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def _tool_use(id_: str, name: str, input_value: Any) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseBlock(id=id_, name=name, input=input_value)],
    )


def _tool_result(id_: str, content: str) -> Message:
    return Message(
        role="user",
        content=[ToolResultBlock(tool_use_id=id_, content=content)],
    )


def test_anchor_summary_section_is_empty_without_workspace_or_file(tmp_path: Path) -> None:
    assert anchor_summary_section(None) == ""
    assert anchor_summary_section(tmp_path) == ""


def test_anchor_summary_section_parses_anchor_file_into_bullets(tmp_path: Path) -> None:
    deepseek_dir = tmp_path / ".deepseek"
    deepseek_dir.mkdir(parents=True, exist_ok=True)
    (deepseek_dir / "anchors.md").write_text(
        "\n---\nDo not touch .ssh\n---\nStatus field is unreliable\n"
    )

    section = anchor_summary_section(tmp_path)

    assert "## Pinned Facts (User Anchors)" in section
    assert "- Do not touch .ssh\n" in section
    assert "- Status field is unreliable\n" in section
    assert "\n---\nDo not touch" not in section


def test_truncate_chars_respects_unicode_boundaries() -> None:
    text = "abc😀é"
    assert truncate_chars(text, 0) == ""
    assert truncate_chars(text, 1) == "a"
    assert truncate_chars(text, 3) == "abc"
    assert truncate_chars(text, 4) == "abc😀"
    assert truncate_chars(text, 5) == "abc😀é"


def test_prune_tool_results_summarizes_old_verbose_outputs() -> None:
    verbose = "x" * (SUMMARY_TOOL_RESULT_SNIPPET_CHARS + 80)
    messages = [
        _tool_use("call-1", "read_file", {"path": "Cargo.toml"}),
        _tool_result("call-1", verbose),
        _msg("user", "recent question"),
        _msg("assistant", "recent answer"),
    ]

    saved = prune_tool_results(messages, 2)

    assert saved > 0
    block = messages[1].content[0]
    assert isinstance(block, ToolResultBlock)
    assert "[read_file] tool result pruned" in block.content
    assert "Cargo.toml" in block.content
    assert len(block.content) < len(verbose)


def test_prune_tool_results_preserves_protected_tail() -> None:
    verbose = "x" * (SUMMARY_TOOL_RESULT_SNIPPET_CHARS + 80)
    messages = [
        _msg("user", "older context"),
        _tool_use("call-1", "read_file", {"path": "Cargo.toml"}),
        _tool_result("call-1", verbose),
    ]

    saved = prune_tool_results(messages, 2)

    assert saved == 0
    block = messages[2].content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.content == verbose


def test_prune_tool_results_preserves_prefix_bytes_when_reverse_prune_is_enough() -> None:
    older_verbose = ("old " * (SUMMARY_TOOL_RESULT_SNIPPET_CHARS + 40)).strip()
    newer_verbose = ("new " * (SUMMARY_TOOL_RESULT_SNIPPET_CHARS + 40)).strip()
    messages = [
        _tool_use("call-old", "read_file", {"path": "old.txt"}),
        _tool_result("call-old", older_verbose),
        _tool_use("call-new", "read_file", {"path": "new.txt"}),
        _tool_result("call-new", newer_verbose),
        _msg("user", "protected tail"),
    ]
    original = list(messages)

    saved = prune_tool_results_until(messages, 1, lambda _m, s: s > 0)

    assert saved > 0
    assert messages[:3] == original[:3]
    assert messages[4:] == original[4:]
    block = messages[3].content[0]
    assert isinstance(block, ToolResultBlock)
    assert "[read_file] tool result pruned" in block.content
    assert "new.txt" in block.content
    assert len(block.content) < len(newer_verbose)


def test_prune_tool_results_stops_after_newest_duplicate_prune() -> None:
    oldest = "oldest " * 80
    middle = "middle " * 80
    latest = "latest " * 80
    messages = [
        _tool_use("call-1", "read_file", {"path": "Cargo.toml"}),
        _tool_result("call-1", oldest),
        _tool_use("call-2", "read_file", {"path": "Cargo.toml"}),
        _tool_result("call-2", middle),
        _tool_use("call-3", "read_file", {"path": "Cargo.toml"}),
        _tool_result("call-3", latest),
        _msg("user", "protected tail"),
    ]
    original = list(messages)

    saved = prune_tool_results_until(messages, 1, lambda _m, s: s > 0)

    assert saved > 0
    assert messages[:3] == original[:3]
    assert messages[4:] == original[4:]
    block = messages[3].content[0]
    assert isinstance(block, ToolResultBlock)
    assert "tool result pruned" in block.content


def test_prune_tool_results_dedupes_identical_reads_but_keeps_latest_full_body() -> None:
    first = "first " * 80
    second = "second " * 80
    messages = [
        _tool_use("call-1", "read_file", {"path": "Cargo.toml"}),
        _tool_result("call-1", first),
        _tool_use("call-2", "read_file", {"path": "Cargo.toml"}),
        _tool_result("call-2", second),
        _msg("user", "tail"),
    ]

    saved = prune_tool_results(messages, 1)

    assert saved > 0
    older = messages[1].content[0]
    assert isinstance(older, ToolResultBlock)
    assert "tool result pruned" in older.content
    latest = messages[3].content[0]
    assert isinstance(latest, ToolResultBlock)
    assert latest.content == second


def test_is_transient_error_detects_network_issues() -> None:
    assert is_transient_error(RuntimeError("Connection timeout"))
    assert is_transient_error(RuntimeError("429 Too Many Requests"))
    assert is_transient_error(RuntimeError("network error: connection refused"))


def test_is_transient_error_rejects_permanent_errors() -> None:
    assert not is_transient_error(RuntimeError("401 Unauthorized: Invalid API key"))
    assert not is_transient_error(RuntimeError("Failed to parse JSON response"))
    assert not is_transient_error(RuntimeError("Invalid request: missing required field"))


def test_summary_limits_expand_for_v4_context() -> None:
    legacy = summary_input_limits_for_model("deepseek-v3.2-128k")
    v4 = summary_input_limits_for_model("deepseek-v4-pro")

    assert v4.input_max_chars > legacy.input_max_chars
    assert v4.tool_result_snippet_chars > legacy.tool_result_snippet_chars
    assert v4.max_tokens > legacy.max_tokens


def test_cache_aligned_summary_is_used_for_v4_scale_contexts() -> None:
    messages = [_msg("user", "Please edit crates/tui/src/compaction.rs")]

    assert should_use_cache_aligned_summary("deepseek-v4-flash", messages)
    assert not should_use_cache_aligned_summary("deepseek-v3.2-128k", messages)


def test_summary_cache_hit_percent_uses_input_tokens_as_denominator() -> None:
    assert abs(summary_cache_hit_percent(800, 1000) - 80.0) < 1e-9
    assert abs(summary_cache_hit_percent(0, 1000) - 0.0) < 1e-9
    assert abs(summary_cache_hit_percent(1000, 1000) - 100.0) < 1e-9
    assert abs(summary_cache_hit_percent(200, 1000) - 20.0) < 1e-9
    assert abs(summary_cache_hit_percent(0, 0) - 0.0) < 1e-9
    assert abs(summary_cache_hit_percent(50, 0) - 0.0) < 1e-9


def test_context_window_errors_are_detected_for_summary_fallback() -> None:
    msgs = [
        "HTTP 400 Bad Request: maximum context length is 1000000 tokens",
        "invalid_request_error: prompt is too long for the current model",
        "You requested 1000001 tokens but the maximum is 1000000",
        "request exceeds context window",
    ]
    for msg in msgs:
        assert is_context_window_error(RuntimeError(msg))

    assert not is_context_window_error(RuntimeError("Invalid request: missing required field"))
    assert not is_context_window_error(RuntimeError("503 Service Unavailable"))


def test_formatted_summary_request_bounds_large_input() -> None:
    messages = [
        _msg("user", f"turn {idx}: " + "中文上下文 " * 1000) for idx in range(90)
    ]
    limits = summary_input_limits_for_model("deepseek-v4-pro")
    request = build_formatted_summary_request("deepseek-v4-pro", messages, limits)

    assert len(request.messages) == 1
    block = request.messages[0].content[0]
    assert isinstance(block, TextBlock)
    assert "characters omitted before summary" in block.text
    assert len(block.text) <= limits.input_max_chars + 2000


def test_cache_aligned_summary_request_preserves_message_prefix() -> None:
    messages = [
        _msg("user", "Please edit crates/tui/src/compaction.rs"),
        _msg("assistant", "I will inspect the file."),
    ]
    limits = summary_input_limits_for_model("deepseek-v4-pro")
    request = build_cache_aligned_summary_request("deepseek-v4-pro", messages, limits)

    assert request.system is None
    assert request.messages[: len(messages)] == messages
    assert len(request.messages) == len(messages) + 1
    last = request.messages[-1]
    assert last.role == "user"
    assert isinstance(last.content[0], TextBlock)
    assert "conversation above" in last.content[0].text


def test_estimate_tokens_empty_messages() -> None:
    assert estimate_tokens([]) == 0


def test_estimate_tokens_with_text() -> None:
    messages = [
        Message(
            role="user",
            content=[TextBlock(text="Hello, world!")],
        )
    ]
    tokens = estimate_tokens(messages)
    assert 0 < tokens < 10


def test_estimate_tokens_counts_tool_round_thinking_across_turns() -> None:
    thinking = "reasoning " * 800
    current_messages = [
        Message(
            role="user",
            content=[TextBlock(text="Use a tool")],
        ),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking=thinking),
                ToolUseBlock(id="tool-1", name="read_file", input={"path": "Cargo.toml"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="tool-1", content="manifest")],
        ),
    ]
    historical_messages = current_messages + [
        Message(role="assistant", content=[TextBlock(text="Done.")]),
        Message(role="user", content=[TextBlock(text="Next question.")]),
    ]
    completed_messages = current_messages + [
        Message(role="assistant", content=[TextBlock(text="Done.")]),
    ]

    lower_bound = len(thinking) // 5
    assert estimate_tokens(current_messages) > lower_bound
    assert estimate_tokens(completed_messages) > lower_bound
    assert estimate_tokens(historical_messages) > lower_bound