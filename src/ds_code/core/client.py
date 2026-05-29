"""
HTTP client for DeepSeek's OpenAI-compatible Chat Completions API.
"""

from __future__ import annotations

from dataclasses import dataclass
from ds_code.core.models import SystemPrompt, MessageResponse, ServerToolUsage, CacheControl,SystemPromptText, SystemPromptBlocks, SystemBlock, ThinkingBlock , Usage, Message, MessageRequest, TextBlock, ToolResultBlock, ToolUseBlock
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union
import asyncio
import json
import logging
import os
import re
import time
from threading import Lock
from time import monotonic
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# -----------------------------
# External-like types (stubs)
# -----------------------------

class ApiProvider(str, Enum):
    Deepseek = "deepseek"
    DeepseekCN = "deepseek-cn"
    Openrouter = "openrouter"
    Novita = "novita"
    Sglang = "sglang"
    Fireworks = "fireworks"
    Vllm = "vllm"
    Openai = "openai"
    Atlascloud = "atlascloud"
    WanjieArk = "wanjie-ark"
    Ollama = "ollama"
    NvidiaNim = "nvidia-nim"

    def as_str(self) -> str:
        return self.value


@dataclass
class RetryPolicy:
    enabled: bool
    max_retries: int
    initial_delay: float
    max_delay: float


class Config(Protocol):
    def deepseek_api_key(self) -> str: ...
    def deepseek_base_url(self) -> str: ...
    def api_provider(self) -> ApiProvider: ...
    def retry_policy(self) -> RetryPolicy: ...
    def default_model(self) -> str: ...
    def http_headers(self) -> Dict[str, str]: ...


class LlmError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    @staticmethod
    def from_http_response_with_retry_after(status: int, body: str, retry_after: Optional[int]) -> "LlmError":
        return LlmError(f"HTTP {status}: {body}")

    @staticmethod
    def from_reqwest(err: Exception) -> "LlmError":
        return LlmError(str(err))


@dataclass
class LlmRetryConfig:
    enabled: bool
    max_retries: int
    initial_delay: float
    max_delay: float


def extract_retry_after(headers: Dict[str, str]) -> Optional[int]:
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


async def with_retry(
    retry_cfg: LlmRetryConfig,
    func: Callable[[], Any],
    on_retry: Optional[Callable[[LlmError, int, float], None]] = None,
) -> Any:
    attempts = 0
    delay = retry_cfg.initial_delay
    last_error: Optional[LlmError] = None

    while True:
        attempts += 1
        try:
            return await func()
        except LlmError as err:
            last_error = err
            if not retry_cfg.enabled or attempts >= retry_cfg.max_retries:
                break
            if on_retry:
                on_retry(err, attempts - 1, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, retry_cfg.max_delay)

    raise LlmError(str(last_error) if last_error else "retry failed")



# -----------------------------
# Local helpers
# -----------------------------

def to_api_tool_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch == "-":
            out.append("--")
        else:
            out.append("-x")
            out.append(f"{ord(ch):06X}")
            out.append("-")
    return "".join(out)


def from_api_tool_name(name: str) -> str:
    out: List[str] = []
    it = iter(range(len(name)))
    i = 0
    while i < len(name):
        ch = name[i]
        if ch != "-":
            out.append(ch)
            i += 1
            continue
        if i + 1 < len(name) and name[i + 1] == "-":
            out.append("-")
            i += 2
            continue
        if i + 1 < len(name) and name[i + 1] == "x":
            i += 2
            hex_part = name[i:i + 6]
            if len(hex_part) == 6:
                try:
                    code = int(hex_part, 16)
                    decoded = chr(code)
                    if i + 6 < len(name) and name[i + 6] == "-":
                        i += 7
                    else:
                        i += 6
                    out.append(decoded)
                    continue
                except ValueError:
                    pass
            out.append("-x")
            out.append(hex_part)
            i += 6
            continue
        out.append("-")
        i += 1

    return decode_bare_hex_escapes("".join(out))


_BARE_HEX_RE = re.compile(r"x([0-9A-Fa-f]{6})-?")


def decode_bare_hex_escapes(input_text: str) -> str:
    def repl(match: re.Match) -> str:
        hex_part = match.group(1)
        try:
            code = int(hex_part, 16)
            decoded = chr(code)
            if not decoded.isalnum() and decoded not in "_-":
                return decoded
        except ValueError:
            pass
        return match.group(0)

    return _BARE_HEX_RE.sub(repl, input_text)


@dataclass
class AvailableModel:
    id: str
    owned_by: Optional[str]
    created: Optional[int]


# -----------------------------
# Client internals
# -----------------------------

CONNECTION_FAILURE_THRESHOLD = 2
RECOVERY_PROBE_COOLDOWN = 15.0

DEFAULT_CLIENT_RATE_LIMIT_RPS = 8.0
DEFAULT_CLIENT_RATE_LIMIT_BURST = 16.0
ALLOW_INSECURE_HTTP_ENV = "DEEPSEEK_ALLOW_INSECURE_HTTP"

SSE_BACKPRESSURE_HIGH_WATERMARK = 8 * 1024 * 1024
SSE_BACKPRESSURE_SLEEP_MS = 10
SSE_MAX_LINES_PER_CHUNK = 256


class ConnectionState(Enum):
    Healthy = "healthy"
    Degraded = "degraded"
    Recovering = "recovering"


@dataclass
class ConnectionHealth:
    state: ConnectionState = ConnectionState.Healthy
    consecutive_failures: int = 0
    last_failure: Optional[float] = None
    last_success: Optional[float] = None
    last_probe: Optional[float] = None


@dataclass
class TokenBucket:
    enabled: bool
    capacity: float
    tokens: float
    refill_per_sec: float
    last_refill: float

    @staticmethod
    def from_env() -> "TokenBucket":
        rps = float(os.getenv("DEEPSEEK_RATE_LIMIT_RPS", DEFAULT_CLIENT_RATE_LIMIT_RPS))
        burst = float(os.getenv("DEEPSEEK_RATE_LIMIT_BURST", DEFAULT_CLIENT_RATE_LIMIT_BURST))
        rps = max(0.0, rps)
        burst = max(1.0, burst)
        enabled = rps > 0.0
        now = monotonic()
        return TokenBucket(
            enabled=enabled,
            capacity=burst,
            tokens=burst,
            refill_per_sec=rps,
            last_refill=now,
        )

    def refill(self, now: float) -> None:
        if not self.enabled:
            return
        elapsed = now - self.last_refill
        self.last_refill = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)

    def delay_until_available(self, tokens: float) -> Optional[float]:
        if not self.enabled:
            return None
        now = monotonic()
        self.refill(now)
        if self.tokens >= tokens:
            self.tokens -= tokens
            return None
        needed = tokens - self.tokens
        self.tokens = 0.0
        if self.refill_per_sec <= 0.0:
            return 1.0
        return needed / self.refill_per_sec


def apply_request_success(health: ConnectionHealth, now: float) -> bool:
    recovered = health.state != ConnectionState.Healthy
    health.state = ConnectionState.Healthy
    health.consecutive_failures = 0
    health.last_success = now
    return recovered


def apply_request_failure(health: ConnectionHealth, now: float) -> None:
    health.consecutive_failures = min(health.consecutive_failures + 1, 1_000_000)
    health.last_failure = now
    if health.consecutive_failures >= CONNECTION_FAILURE_THRESHOLD:
        health.state = ConnectionState.Degraded


def mark_recovery_probe_if_due(health: ConnectionHealth, now: float) -> bool:
    if health.state == ConnectionState.Healthy:
        return False
    if health.last_probe is not None and (now - health.last_probe) < RECOVERY_PROBE_COOLDOWN:
        return False
    health.last_probe = now
    health.state = ConnectionState.Recovering
    return True


_buffer_pool: List[bytearray] = []
_buffer_pool_lock = Lock()


def acquire_stream_buffer() -> bytearray:
    with _buffer_pool_lock:
        if _buffer_pool:
            return _buffer_pool.pop()
    return bytearray(8192)


def release_stream_buffer(buf: bytearray) -> None:
    buf.clear()
    if len(buf) > 256 * 1024:
        del buf[256 * 1024 :]
    with _buffer_pool_lock:
        if len(_buffer_pool) < 8:
            _buffer_pool.append(buf)


ERROR_BODY_MAX_BYTES = 64 * 1024


async def bounded_error_text(response: "HttpResponse", max_bytes: int) -> str:
    data = await response.read(max_bytes=max_bytes)
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def validate_base_url_security(base_url: str) -> None:
    if base_url.startswith("https://") or base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1") or base_url.startswith("http://[::1]"):
        return

    if base_url.startswith("http://"):
        allow = os.getenv(ALLOW_INSECURE_HTTP_ENV, "")
        if allow.lower() in ("1", "true"):
            logger.warning("Using insecure HTTP base URL because %s is set", ALLOW_INSECURE_HTTP_ENV)
            return
        raise ValueError(
            f"Refusing insecure base URL '{base_url}'.\n\n"
            "Loopback hosts (localhost, 127.0.0.1, [::1]) are auto-allowed.\n"
            "For other trusted local hosts (LAN, llama.cpp on a private IP, etc.)\n"
            f"set the env var {ALLOW_INSECURE_HTTP_ENV}=1 and re-run.\n"
        )

    raise ValueError(
        f"Refusing base URL '{base_url}': only HTTPS (or explicitly allowed HTTP) URLs are supported."
    )


def versioned_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if base_url_has_version_suffix(trimmed):
        return trimmed
    return f"{trimmed}/v1"


def unversioned_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    parts = trimmed.rsplit("/", 1)
    if len(parts) == 2 and is_version_segment(parts[1]):
        return parts[0]
    return trimmed


def base_url_has_version_suffix(trimmed: str) -> bool:
    last = trimmed.rsplit("/", 1)[-1]
    return is_version_segment(last)


def is_version_segment(segment: str) -> bool:
    if segment.lower() == "beta":
        return True
    if segment.startswith(("v", "V")) and len(segment) > 1:
        return segment[1:].isdigit()
    return False


def api_url(base_url: str, path: str) -> str:
    path = path.lstrip("/")
    if path.startswith("beta/"):
        return f"{unversioned_base_url(base_url)}/{path}"
    versioned = versioned_base_url(base_url)
    if versioned.endswith("beta"):
        versioned = f"{unversioned_base_url(base_url)}/v1"
    return f"{versioned.rstrip('/')}/{path}"


def force_http1_from_env() -> bool:
    v = os.getenv("DEEPSEEK_FORCE_HTTP1", "").strip().lower()
    return v in ("1", "true", "yes", "on")


# -----------------------------
# HTTP client wrapper
# -----------------------------

class HttpResponse(Protocol):
    status_code: int
    headers: Dict[str, str]
    async def read(self, max_bytes: Optional[int] = None) -> bytes: ...
    async def text(self) -> str: ...
    async def json(self) -> Any: ...


class HttpClient(Protocol):
    async def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> HttpResponse: ...
    async def post(self, url: str, json_body: Any, headers: Optional[Dict[str, str]] = None) -> HttpResponse: ...


def _default_headers(api_key: str, extra_headers: Dict[str, str]) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key}"

    for name, value in extra_headers.items():
        name = name.strip()
        value = value.strip()
        if not name or not value:
            continue
        if name.lower() in ("authorization", "content-type"):
            continue
        headers[name] = value
    return headers


# -----------------------------
# DeepSeekClient
# -----------------------------

@dataclass
class DeepSeekClient:
    http_client: HttpClient
    api_key: str
    base_url: str
    api_provider: ApiProvider
    retry: RetryPolicy
    default_model: str
    http_headers: Dict[str, str]
    connection_health: ConnectionHealth
    rate_limiter: TokenBucket

    @staticmethod
    def new(config: Config, http_client: HttpClient) -> "DeepSeekClient":
        api_key = config.deepseek_api_key()
        base_url = config.deepseek_base_url()
        api_provider = config.api_provider()
        validate_base_url_security(base_url)
        retry = config.retry_policy()
        default_model = config.default_model()
        http_headers = config.http_headers()

        logger.info("API provider: %s", api_provider.as_str())
        logger.info("API base URL: %s", base_url)
        if http_headers:
            logger.info("%d custom HTTP header(s) configured", len(http_headers))
        logger.info(
            "Retry policy: enabled=%s, max_retries=%s, initial_delay=%ss, max_delay=%ss",
            retry.enabled, retry.max_retries, retry.initial_delay, retry.max_delay
        )

        return DeepSeekClient(
            http_client=http_client,
            api_key=api_key,
            base_url=base_url,
            api_provider=api_provider,
            retry=retry,
            default_model=default_model,
            http_headers=http_headers,
            connection_health=ConnectionHealth(),
            rate_limiter=TokenBucket.from_env(),
        )

    def _headers(self) -> Dict[str, str]:
        return _default_headers(self.api_key, self.http_headers)

    async def translate(self, text: str, model: str, target_language: str) -> str:
        url = api_url(self.base_url, "chat/completions")
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translator. Your ONLY task is to translate text to {target_language}. "
                        "Rules:\n"
                        "1. Output ONLY the translation, nothing else — no explanations, no notes, no quotes.\n"
                        "2. Preserve all code blocks (```...```), URLs, file paths, command names, "
                        "and technical terms like API names, function names, and library names untranslated.\n"
                        "3. Keep Markdown formatting (headings, lists, bold, italics, links) intact.\n"
                        "4. Translate all natural-language prose naturally and professionally.\n"
                        "5. Do NOT add any prefix, suffix, or commentary.\n"
                        f"6. If the input is already in {target_language} or contains no prose to translate, "
                        "return it as-is."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "stream": False,
        }
        apply_reasoning_effort(body, "off", self.api_provider)

        response = await self.send_with_retry(lambda: self.http_client.post(url, body, headers=self._headers()))
        value = await response.json()
        try:
            translated = value["choices"][0]["message"]["content"]
        except Exception as err:
            raise RuntimeError("translate: unexpected API response shape") from err
        return str(translated).strip()

    async def list_models(self) -> List[AvailableModel]:
        url = api_url(self.base_url, "models")
        response = await self.send_with_retry(lambda: self.http_client.get(url, headers=self._headers()))

        if response.status_code < 200 or response.status_code >= 300:
            error_text = await bounded_error_text(response, ERROR_BODY_MAX_BYTES)
            raise RuntimeError(f"Failed to list models: HTTP {response.status_code}: {error_text}")
        response_text = await response.text()
        return parse_models_response(response_text)

    async def wait_for_rate_limit(self) -> None:
        delay = self.rate_limiter.delay_until_available(1.0)
        if delay:
            await asyncio.sleep(delay)

    async def mark_request_success(self) -> None:
        now = monotonic()
        if apply_request_success(self.connection_health, now):
            logger.info("Connection recovered")

    async def mark_request_failure(self, reason: str) -> None:
        now = monotonic()
        apply_request_failure(self.connection_health, now)
        logger.warning(
            "Connection degraded (failures=%d): %s",
            self.connection_health.consecutive_failures,
            reason,
        )

    async def maybe_probe_recovery(self) -> None:
        now = monotonic()
        if not mark_recovery_probe_if_due(self.connection_health, now):
            return
        health_url = api_url(self.base_url, "models")
        try:
            resp = await self.http_client.get(health_url, headers=self._headers())
            if 200 <= resp.status_code < 300:
                await self.mark_request_success()
                logger.info("Recovery probe succeeded")
            else:
                await self.mark_request_failure(f"probe status={resp.status_code}")
        except Exception as err:
            await self.mark_request_failure(f"probe error={err}")

    async def send_with_retry(self, build: Callable[[], Any]) -> HttpResponse:
        retry_cfg = LlmRetryConfig(
            enabled=self.retry.enabled,
            max_retries=self.retry.max_retries,
            initial_delay=self.retry.initial_delay,
            max_delay=self.retry.max_delay,
        )

        async def attempt():
            await self.wait_for_rate_limit()
            try:
                response = await build()
            except Exception as err:
                raise LlmError.from_reqwest(err)
            status = response.status_code
            if 200 <= status < 300:
                return response
            retry_after = extract_retry_after(response.headers)
            body = await bounded_error_text(response, ERROR_BODY_MAX_BYTES)
            raise LlmError.from_http_response_with_retry_after(status, body, retry_after)

        def on_retry(err: LlmError, attempt_idx: int, delay: float) -> None:
            label, human = retry_reason_label_and_human(err)
            logger.warning(
                "HTTP retry reason=%s attempt=%d delay=%.2fs",
                label,
                attempt_idx + 1,
                delay,
            )

        try:
            response = await with_retry(retry_cfg, attempt, on_retry)
            await self.mark_request_success()
            return response
        except LlmError as err:
            await self.mark_request_failure(str(err))
            await self.maybe_probe_recovery()
            raise RuntimeError(str(err))

    async def health_check(self) -> bool:
        health_url = api_url(self.base_url, "models")
        await self.wait_for_rate_limit()
        try:
            resp = await self.http_client.get(health_url, headers=self._headers())
            if 200 <= resp.status_code < 300:
                await self.mark_request_success()
                return True
            await self.mark_request_failure(f"health status={resp.status_code}")
            return False
        except Exception as err:
            await self.mark_request_failure(f"health error={err}")
            return False

    async def create_message(self, request: MessageRequest) -> MessageResponse:
        return await self.create_message_chat(request)

    async def create_message_stream(self, request: MessageRequest) -> Any:
        return await self.handle_chat_completion_stream(request)

    async def create_message_chat(self, request: MessageRequest) -> MessageResponse:
        # Placeholder for chat integration
        raise NotImplementedError

    async def handle_chat_completion_stream(self, request: MessageRequest) -> Any:
        # Placeholder for chat streaming integration
        raise NotImplementedError

    async def fim_completion(self, model: str, prompt: str, suffix: str, max_tokens: int) -> str:
        url = api_url(self.base_url, "beta/completions")
        body = {
            "model": model,
            "prompt": prompt,
            "suffix": suffix,
            "max_tokens": max_tokens,
        }
        response = await self.send_with_retry(lambda: self.http_client.post(url, body, headers=self._headers()))
        if response.status_code < 200 or response.status_code >= 300:
            error_text = await bounded_error_text(response, ERROR_BODY_MAX_BYTES)
            raise RuntimeError(f"FIM API error: HTTP {response.status_code}: {error_text}")
        response_text = await response.text()
        try:
            value = json.loads(response_text)
        except json.JSONDecodeError as err:
            raise RuntimeError("Failed to parse FIM API response") from err
        try:
            text = value["choices"][0]["text"]
        except Exception as err:
            raise RuntimeError("FIM response missing choices[0].text") from err
        return str(text)


def retry_reason_label_and_human(err: LlmError) -> Tuple[str, str]:
    text = str(err).lower()
    if "rate" in text and "limit" in text:
        return ("rate_limited", "rate limited")
    if "timeout" in text:
        return ("timeout", "timeout")
    if "network" in text:
        return ("network_error", "network error")
    if "http" in text:
        return ("server_error", "upstream error")
    return ("other", "other")


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_models_response(payload: str) -> List[AvailableModel]:
    data = json.loads(payload)
    items = data.get("data", [])
    models = [
        AvailableModel(
            id=item.get("id", ""),
            owned_by=item.get("owned_by"),
            created=item.get("created"),
        )
        for item in items
    ]
    models.sort(key=lambda m: m.id)
    deduped: List[AvailableModel] = []
    seen: set[str] = set()
    for model in models:
        if model.id in seen:
            continue
        seen.add(model.id)
        deduped.append(model)
    return deduped


def system_to_instructions(system: Optional[SystemPrompt]) -> Optional[str]:
    if isinstance(system, SystemPromptText):
        return system.text
    if isinstance(system, SystemPromptBlocks):
        joined = "\n\n---\n\n".join(block.text for block in system.blocks)
        return joined if joined.strip() else None
    return None


def apply_reasoning_effort(body: Dict[str, Any], effort: Optional[str], provider: ApiProvider) -> None:
    if effort is None:
        return
    normalized = effort.strip().lower()
    if normalized in ("off", "disabled", "none", "false"):
        if provider in (ApiProvider.Deepseek, ApiProvider.DeepseekCN, ApiProvider.Openrouter, ApiProvider.Novita, ApiProvider.Sglang):
            body["thinking"] = {"type": "disabled"}
        elif provider == ApiProvider.Vllm:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif provider == ApiProvider.NvidiaNim:
            body["chat_template_kwargs"] = {"thinking": False}
        return

    if normalized in ("low", "minimal", "medium", "mid", "high", ""):
        if provider in (ApiProvider.Deepseek, ApiProvider.DeepseekCN, ApiProvider.Openrouter, ApiProvider.Novita, ApiProvider.Sglang):
            body["reasoning_effort"] = "high"
            body["thinking"] = {"type": "enabled"}
        elif provider == ApiProvider.Fireworks:
            body["reasoning_effort"] = "high"
        elif provider == ApiProvider.Vllm:
            body["chat_template_kwargs"] = {"enable_thinking": True}
            body["reasoning_effort"] = "high"
        elif provider == ApiProvider.NvidiaNim:
            body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": "high"}
        return

    if normalized in ("xhigh", "max", "highest"):
        if provider in (ApiProvider.Deepseek, ApiProvider.DeepseekCN, ApiProvider.Openrouter, ApiProvider.Novita, ApiProvider.Sglang):
            body["reasoning_effort"] = "max"
            body["thinking"] = {"type": "enabled"}
        elif provider == ApiProvider.Fireworks:
            body["reasoning_effort"] = "max"
        elif provider == ApiProvider.Vllm:
            body["chat_template_kwargs"] = {"enable_thinking": True}
            body["reasoning_effort"] = "max"
        elif provider == ApiProvider.NvidiaNim:
            body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": "max"}


def parse_usage(usage: Optional[Dict[str, Any]]) -> Usage:
    if not usage:
        return Usage(input_tokens=0, output_tokens=0)

    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    total_tokens = usage.get("total_tokens")
    reasoning_tokens_raw = (
        usage.get("completion_tokens_details", {}) or {}
    ).get("reasoning_tokens")

    if output_tokens == 0 and reasoning_tokens_raw is not None:
        output_tokens = reasoning_tokens_raw
    elif output_tokens == 0 and total_tokens is not None:
        output_tokens = max(0, total_tokens - input_tokens)

    cached_tokens = (usage.get("prompt_tokens_details", {}) or {}).get("cached_tokens")
    prompt_cache_hit_tokens = usage.get("prompt_cache_hit_tokens") or cached_tokens
    prompt_cache_miss_tokens = usage.get("prompt_cache_miss_tokens")
    if prompt_cache_miss_tokens is None and cached_tokens is not None:
        prompt_cache_miss_tokens = max(0, input_tokens - cached_tokens)

    reasoning_tokens = reasoning_tokens_raw if reasoning_tokens_raw is not None else None

    server = usage.get("server_tool_use")
    server_tool_use = None
    if isinstance(server, dict):
        server_tool_use = ServerToolUsage(
            code_execution_requests=server.get("code_execution_requests"),
            tool_search_requests=server.get("tool_search_requests"),
        )

    return Usage(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        prompt_cache_hit_tokens=int(prompt_cache_hit_tokens) if prompt_cache_hit_tokens is not None else None,
        prompt_cache_miss_tokens=int(prompt_cache_miss_tokens) if prompt_cache_miss_tokens is not None else None,
        reasoning_tokens=int(reasoning_tokens) if reasoning_tokens is not None else None,
        reasoning_replay_tokens=None,
        server_tool_use=server_tool_use,
    )


# -----------------------------
# chat module hooks (placeholders)
# -----------------------------

class PromptInspection:
    pass


def inspect_prompt_for_request(request: MessageRequest) -> PromptInspection:
    raise NotImplementedError


def build_cache_warmup_request(request: MessageRequest) -> MessageRequest:
    raise NotImplementedError


# -----------------------------
# Tests (pytest style)
# -----------------------------

def test_tool_name_round_trip_basic() -> None:
    name = "my_tool"
    encoded = to_api_tool_name(name)
    assert encoded == name
    assert from_api_tool_name(encoded) == name


def test_tool_name_round_trip_with_special_chars() -> None:
    name = "tool-name/with:chars"
    encoded = to_api_tool_name(name)
    assert encoded != name
    assert from_api_tool_name(encoded) == name


def test_decode_bare_hex_escapes_replaces_valid_codes() -> None:
    text = "x00002d-abc"
    assert decode_bare_hex_escapes(text) == "-abc"


def test_decode_bare_hex_escapes_preserves_invalid_codes() -> None:
    text = "xZZZZZZ-abc"
    assert decode_bare_hex_escapes(text) == text


def test_extract_retry_after_reads_header() -> None:
    assert extract_retry_after({"Retry-After": "120"}) == 120
    assert extract_retry_after({"retry-after": "10"}) == 10
    assert extract_retry_after({"retry-after": "bad"}) is None


def test_parse_models_response_dedupes_and_sorts() -> None:
    payload = json.dumps(
        {
            "data": [
                {"id": "zeta", "owned_by": "a", "created": 1},
                {"id": "alpha", "owned_by": "b", "created": 2},
                {"id": "alpha", "owned_by": "b", "created": 2},
            ]
        }
    )
    models = parse_models_response(payload)
    assert [m.id for m in models] == ["alpha", "zeta"]