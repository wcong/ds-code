from dataclasses import dataclass, field
from typing import List, Optional

from ds_code.llm.models import ProviderKind
from ds_code.config import CliRuntimeOverrides, Config


@dataclass
class RuntimeConfig:
    provider: ProviderKind = ProviderKind.Deepseek
    model: str = "deepseek-v4-pro"
    telemetry: bool = False
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    approval_policy: Optional[str] = None
    sandbox_mode: Optional[str] = None

    @classmethod
    def from_config(
        cls, config: Config, cli: Optional[CliRuntimeOverrides] = None
    ) -> "RuntimeConfig":
        resolved = config.resolve_runtime_options(cli or CliRuntimeOverrides())
        return cls(
            provider=resolved.provider,
            model=resolved.model,
            telemetry=resolved.telemetry,
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            approval_policy=resolved.approval_policy,
            sandbox_mode=resolved.sandbox_mode,
        )


@dataclass
class PromptRequest:
    prompt: str
    model: Optional[str] = None
    provider: Optional[ProviderKind] = None
    thread_id: Optional[str] = None


@dataclass
class PromptResponse:
    output: str
    model: str
    events: List[dict] = field(default_factory=list)


@dataclass
class ThreadResponse:
    status: str
    thread_id: str
    model: Optional[str] = None
    model_provider: Optional[str] = None
    data: dict = field(default_factory=dict)
