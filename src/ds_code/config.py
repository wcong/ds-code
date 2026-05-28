from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import os

from ds_code.llm.models import ProviderKind


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_NVIDIA_NIM_MODEL = "deepseek-ai/deepseek-v4-pro"
DEFAULT_NVIDIA_NIM_FLASH_MODEL = "deepseek-ai/deepseek-v4-flash"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"
DEFAULT_NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ATLASCLOUD_MODEL = "deepseek-ai/deepseek-v4-flash"
DEFAULT_ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
DEFAULT_WANJIE_ARK_MODEL = "deepseek-reasoner"
DEFAULT_WANJIE_ARK_BASE_URL = "https://maas-openapi.wanjiedata.com/api/v1"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_OPENROUTER_FLASH_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_NOVITA_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_NOVITA_FLASH_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
DEFAULT_SGLANG_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
DEFAULT_SGLANG_FLASH_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_NOVITA_BASE_URL = "https://api.novita.ai/v1"
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_SGLANG_BASE_URL = "http://localhost:30000/v1"
DEFAULT_VLLM_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
DEFAULT_VLLM_FLASH_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OLLAMA_MODEL = "deepseek-coder:1.3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass
class ProviderConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    http_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProvidersConfig:
    deepseek: ProviderConfig = field(default_factory=ProviderConfig)
    nvidia_nim: ProviderConfig = field(default_factory=ProviderConfig)
    openai: ProviderConfig = field(default_factory=ProviderConfig)
    atlascloud: ProviderConfig = field(default_factory=ProviderConfig)
    wanjie_ark: ProviderConfig = field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = field(default_factory=ProviderConfig)
    novita: ProviderConfig = field(default_factory=ProviderConfig)
    fireworks: ProviderConfig = field(default_factory=ProviderConfig)
    sglang: ProviderConfig = field(default_factory=ProviderConfig)
    vllm: ProviderConfig = field(default_factory=ProviderConfig)
    ollama: ProviderConfig = field(default_factory=ProviderConfig)

    def for_provider(self, provider: ProviderKind) -> ProviderConfig:
        return {
            ProviderKind.Deepseek: self.deepseek,
            ProviderKind.NvidiaNim: self.nvidia_nim,
            ProviderKind.Openai: self.openai,
            ProviderKind.Atlascloud: self.atlascloud,
            ProviderKind.WanjieArk: self.wanjie_ark,
            ProviderKind.Openrouter: self.openrouter,
            ProviderKind.Novita: self.novita,
            ProviderKind.Fireworks: self.fireworks,
            ProviderKind.Sglang: self.sglang,
            ProviderKind.Vllm: self.vllm,
            ProviderKind.Ollama: self.ollama,
        }[provider]


@dataclass
class CliRuntimeOverrides:
    provider: Optional[ProviderKind] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    auth_mode: Optional[str] = None
    output_mode: Optional[str] = None
    log_level: Optional[str] = None
    telemetry: Optional[bool] = None
    approval_policy: Optional[str] = None
    sandbox_mode: Optional[str] = None


@dataclass
class ResolvedRuntimeOptions:
    provider: ProviderKind
    model: str
    api_key: Optional[str]
    base_url: str
    auth_mode: Optional[str]
    output_mode: Optional[str]
    log_level: Optional[str]
    telemetry: bool
    approval_policy: Optional[str]
    sandbox_mode: Optional[str]
    http_headers: Dict[str, str]


@dataclass
class Config:
    provider: ProviderKind = ProviderKind.Deepseek
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    http_headers: Dict[str, str] = field(default_factory=dict)
    default_text_model: Optional[str] = None
    model: Optional[str] = None
    auth_mode: Optional[str] = None
    output_mode: Optional[str] = None
    log_level: Optional[str] = None
    telemetry: Optional[bool] = None
    approval_policy: Optional[str] = None
    sandbox_mode: Optional[str] = None
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)

    def save(self, path: Optional[str] = None) -> None:
        payload = {
            "provider": self.provider.as_str(),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "default_text_model": self.default_text_model,
            "model": self.model,
            "approval_policy": self.approval_policy,
            "sandbox_mode": self.sandbox_mode,
        }
        dest = path or _default_config_path()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def load(path: Optional[str] = None) -> "Config":
        dest = path or _default_config_path()
        if not os.path.exists(dest):
            return Config()
        try:
            with open(dest, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return Config()
        config = Config()
        provider = payload.get("provider")
        if provider:
            parsed = ProviderKind.parse(provider)
            if parsed is not None:
                config.provider = parsed
        config.api_key = payload.get("api_key")
        config.base_url = payload.get("base_url")
        config.default_text_model = payload.get("default_text_model")
        config.model = payload.get("model")
        config.approval_policy = payload.get("approval_policy")
        config.sandbox_mode = payload.get("sandbox_mode")
        return config

    def resolve_runtime_options(self, cli: CliRuntimeOverrides) -> ResolvedRuntimeOptions:
        env = EnvRuntimeOverrides.load()
        provider = cli.provider or env.provider or self.provider
        provider_cfg = self.providers.for_provider(provider)

        base_url = (
            cli.base_url
            or env.base_url_for(provider)
            or provider_cfg.base_url
            or (self.base_url if provider == ProviderKind.Deepseek else None)
            or default_base_url_for_provider(provider)
        )
        model = (
            cli.model
            or env.model
            or env.model_for(provider)
            or provider_cfg.model
            or (self.default_text_model if provider == ProviderKind.Deepseek else None)
            or self.model
            or default_model_for_provider(provider)
        )
        model = normalize_model_for_provider(provider, model)

        http_headers = dict(self.http_headers)
        http_headers.update(provider_cfg.http_headers)
        if env.http_headers:
            http_headers.update(env.http_headers)
        http_headers = {k: v for k, v in http_headers.items() if k and v}

        auth_mode = cli.auth_mode or env.auth_mode or self.auth_mode
        output_mode = cli.output_mode or env.output_mode or self.output_mode
        log_level = cli.log_level or env.log_level or self.log_level
        telemetry = cli.telemetry or env.telemetry or self.telemetry or False
        approval_policy = cli.approval_policy or env.approval_policy or self.approval_policy
        sandbox_mode = cli.sandbox_mode or env.sandbox_mode or self.sandbox_mode

        api_key = (
            cli.api_key
            or provider_cfg.api_key
            or (self.api_key if provider == ProviderKind.Deepseek else None)
            or env.api_key_for(provider)
        )

        return ResolvedRuntimeOptions(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            auth_mode=auth_mode,
            output_mode=output_mode,
            log_level=log_level,
            telemetry=telemetry,
            approval_policy=approval_policy,
            sandbox_mode=sandbox_mode,
            http_headers=http_headers,
        )

    def get_value(self, key: str) -> Optional[str]:
        if key == "provider":
            return self.provider.as_str()
        if key == "api_key":
            return self.api_key
        if key == "base_url":
            return self.base_url
        if key == "model":
            return self.model
        if key == "default_text_model":
            return self.default_text_model
        if key == "approval_policy":
            return self.approval_policy
        if key == "sandbox_mode":
            return self.sandbox_mode
        return None

    def set_value(self, key: str, value: str) -> None:
        if key == "provider":
            parsed = ProviderKind.parse(value)
            if parsed is None:
                raise ValueError(f"unknown provider '{value}'")
            self.provider = parsed
        elif key == "api_key":
            self.api_key = value
        elif key == "base_url":
            self.base_url = value
        elif key == "model":
            self.model = value
        elif key == "default_text_model":
            self.default_text_model = value
        elif key == "approval_policy":
            self.approval_policy = value
        elif key == "sandbox_mode":
            self.sandbox_mode = value
        else:
            raise ValueError(f"unknown config key '{key}'")

    def list_values(self) -> Dict[str, str]:
        items: Dict[str, str] = {"provider": self.provider.as_str()}
        if self.api_key:
            items["api_key"] = "********"
        if self.base_url:
            items["base_url"] = self.base_url
        if self.model:
            items["model"] = self.model
        if self.default_text_model:
            items["default_text_model"] = self.default_text_model
        if self.approval_policy:
            items["approval_policy"] = self.approval_policy
        if self.sandbox_mode:
            items["sandbox_mode"] = self.sandbox_mode
        return items


@dataclass
class EnvRuntimeOverrides:
    provider: Optional[ProviderKind] = None
    model: Optional[str] = None
    output_mode: Optional[str] = None
    auth_mode: Optional[str] = None
    log_level: Optional[str] = None
    telemetry: Optional[bool] = None
    approval_policy: Optional[str] = None
    sandbox_mode: Optional[str] = None
    http_headers: Optional[Dict[str, str]] = None
    deepseek_base_url: Optional[str] = None
    nvidia_base_url: Optional[str] = None
    openai_base_url: Optional[str] = None
    atlascloud_base_url: Optional[str] = None
    wanjie_ark_base_url: Optional[str] = None
    openrouter_base_url: Optional[str] = None
    novita_base_url: Optional[str] = None
    fireworks_base_url: Optional[str] = None
    sglang_base_url: Optional[str] = None
    vllm_base_url: Optional[str] = None
    ollama_base_url: Optional[str] = None
    wanjie_ark_model: Optional[str] = None

    @staticmethod
    def load() -> "EnvRuntimeOverrides":
        return EnvRuntimeOverrides(
            provider=ProviderKind.parse(os.getenv("DEEPSEEK_PROVIDER", "")),
            model=os.getenv("DEEPSEEK_MODEL"),
            output_mode=os.getenv("DEEPSEEK_OUTPUT_MODE"),
            auth_mode=os.getenv("DEEPSEEK_AUTH_MODE"),
            log_level=os.getenv("DEEPSEEK_LOG_LEVEL"),
            telemetry=_parse_bool(os.getenv("DEEPSEEK_TELEMETRY")),
            approval_policy=os.getenv("DEEPSEEK_APPROVAL_POLICY"),
            sandbox_mode=os.getenv("DEEPSEEK_SANDBOX_MODE"),
            http_headers=_parse_http_headers(os.getenv("DEEPSEEK_HTTP_HEADERS")),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL"),
            nvidia_base_url=os.getenv("NVIDIA_NIM_BASE_URL") or os.getenv("NIM_BASE_URL") or os.getenv("NVIDIA_BASE_URL"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            atlascloud_base_url=os.getenv("ATLASCLOUD_BASE_URL"),
            wanjie_ark_base_url=os.getenv("WANJIE_ARK_BASE_URL") or os.getenv("WANJIE_BASE_URL") or os.getenv("WANJIE_MAAS_BASE_URL"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL"),
            novita_base_url=os.getenv("NOVITA_BASE_URL"),
            fireworks_base_url=os.getenv("FIREWORKS_BASE_URL"),
            sglang_base_url=os.getenv("SGLANG_BASE_URL"),
            vllm_base_url=os.getenv("VLLM_BASE_URL"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
            wanjie_ark_model=os.getenv("WANJIE_ARK_MODEL") or os.getenv("WANJIE_MODEL") or os.getenv("WANJIE_MAAS_MODEL"),
        )

    def base_url_for(self, provider: ProviderKind) -> Optional[str]:
        return {
            ProviderKind.Deepseek: self.deepseek_base_url,
            ProviderKind.NvidiaNim: self.nvidia_base_url,
            ProviderKind.Openai: self.openai_base_url,
            ProviderKind.Atlascloud: self.atlascloud_base_url,
            ProviderKind.WanjieArk: self.wanjie_ark_base_url,
            ProviderKind.Openrouter: self.openrouter_base_url,
            ProviderKind.Novita: self.novita_base_url,
            ProviderKind.Fireworks: self.fireworks_base_url,
            ProviderKind.Sglang: self.sglang_base_url,
            ProviderKind.Vllm: self.vllm_base_url,
            ProviderKind.Ollama: self.ollama_base_url,
        }.get(provider)

    def model_for(self, provider: ProviderKind) -> Optional[str]:
        if provider == ProviderKind.WanjieArk:
            return self.wanjie_ark_model
        return None

    def api_key_for(self, provider: ProviderKind) -> Optional[str]:
        key_map = {
            ProviderKind.Deepseek: os.getenv("DEEPSEEK_API_KEY"),
            ProviderKind.NvidiaNim: os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY"),
            ProviderKind.Openai: os.getenv("OPENAI_API_KEY"),
            ProviderKind.Atlascloud: os.getenv("ATLASCLOUD_API_KEY"),
            ProviderKind.WanjieArk: os.getenv("WANJIE_ARK_API_KEY"),
            ProviderKind.Openrouter: os.getenv("OPENROUTER_API_KEY"),
            ProviderKind.Novita: os.getenv("NOVITA_API_KEY"),
            ProviderKind.Fireworks: os.getenv("FIREWORKS_API_KEY"),
            ProviderKind.Sglang: os.getenv("SGLANG_API_KEY"),
            ProviderKind.Vllm: os.getenv("VLLM_API_KEY"),
            ProviderKind.Ollama: os.getenv("OLLAMA_API_KEY"),
        }
        return key_map.get(provider)


def normalize_model_for_provider(provider: ProviderKind, model: str) -> str:
    normalized = model.strip().lower()
    if provider in (ProviderKind.Atlascloud, ProviderKind.WanjieArk, ProviderKind.Ollama):
        return model
    if provider == ProviderKind.NvidiaNim:
        if normalized in ("deepseek-v4-pro", "deepseek-v4pro"):
            return DEFAULT_NVIDIA_NIM_MODEL
        if normalized in (
            "deepseek-v4-flash",
            "deepseek-v4flash",
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-v3.2",
        ):
            return DEFAULT_NVIDIA_NIM_FLASH_MODEL
    if provider == ProviderKind.Openrouter:
        if normalized in ("deepseek-v4-pro", "deepseek-v4pro"):
            return DEFAULT_OPENROUTER_MODEL
        if normalized in (
            "deepseek-v4-flash",
            "deepseek-v4flash",
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-v3.2",
        ):
            return DEFAULT_OPENROUTER_FLASH_MODEL
    if provider == ProviderKind.Novita:
        if normalized in ("deepseek-v4-pro", "deepseek-v4pro"):
            return DEFAULT_NOVITA_MODEL
        if normalized in (
            "deepseek-v4-flash",
            "deepseek-v4flash",
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-v3.2",
        ):
            return DEFAULT_NOVITA_FLASH_MODEL
    if provider == ProviderKind.Fireworks:
        if normalized in ("deepseek-v4-pro", "deepseek-v4pro"):
            return DEFAULT_FIREWORKS_MODEL
    if provider == ProviderKind.Sglang:
        if normalized in ("deepseek-v4-pro", "deepseek-v4pro"):
            return DEFAULT_SGLANG_MODEL
        if normalized in (
            "deepseek-v4-flash",
            "deepseek-v4flash",
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-v3.2",
        ):
            return DEFAULT_SGLANG_FLASH_MODEL
    if provider == ProviderKind.Vllm:
        if normalized in ("deepseek-v4-pro", "deepseek-v4pro"):
            return DEFAULT_VLLM_MODEL
        if normalized in (
            "deepseek-v4-flash",
            "deepseek-v4flash",
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-r1",
            "deepseek-v3",
            "deepseek-v3.2",
        ):
            return DEFAULT_VLLM_FLASH_MODEL
    return model


def default_model_for_provider(provider: ProviderKind) -> str:
    return {
        ProviderKind.Deepseek: DEFAULT_DEEPSEEK_MODEL,
        ProviderKind.NvidiaNim: DEFAULT_NVIDIA_NIM_MODEL,
        ProviderKind.Openai: DEFAULT_OPENAI_MODEL,
        ProviderKind.Atlascloud: DEFAULT_ATLASCLOUD_MODEL,
        ProviderKind.WanjieArk: DEFAULT_WANJIE_ARK_MODEL,
        ProviderKind.Openrouter: DEFAULT_OPENROUTER_MODEL,
        ProviderKind.Novita: DEFAULT_NOVITA_MODEL,
        ProviderKind.Fireworks: DEFAULT_FIREWORKS_MODEL,
        ProviderKind.Sglang: DEFAULT_SGLANG_MODEL,
        ProviderKind.Vllm: DEFAULT_VLLM_MODEL,
        ProviderKind.Ollama: DEFAULT_OLLAMA_MODEL,
    }[provider]


def default_base_url_for_provider(provider: ProviderKind) -> str:
    return {
        ProviderKind.Deepseek: DEFAULT_DEEPSEEK_BASE_URL,
        ProviderKind.NvidiaNim: DEFAULT_NVIDIA_NIM_BASE_URL,
        ProviderKind.Openai: DEFAULT_OPENAI_BASE_URL,
        ProviderKind.Atlascloud: DEFAULT_ATLASCLOUD_BASE_URL,
        ProviderKind.WanjieArk: DEFAULT_WANJIE_ARK_BASE_URL,
        ProviderKind.Openrouter: DEFAULT_OPENROUTER_BASE_URL,
        ProviderKind.Novita: DEFAULT_NOVITA_BASE_URL,
        ProviderKind.Fireworks: DEFAULT_FIREWORKS_BASE_URL,
        ProviderKind.Sglang: DEFAULT_SGLANG_BASE_URL,
        ProviderKind.Vllm: DEFAULT_VLLM_BASE_URL,
        ProviderKind.Ollama: DEFAULT_OLLAMA_BASE_URL,
    }[provider]


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on", "enabled"):
        return True
    if value in ("0", "false", "no", "off", "disabled"):
        return False
    return None


def _parse_http_headers(raw: Optional[str]) -> Optional[Dict[str, str]]:
    if not raw:
        return None
    headers: Dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            continue
        name, value = pair.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            continue
        headers[name] = value
    return headers or None


def _default_config_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".ds_code", "config.json")
