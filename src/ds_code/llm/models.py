from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ProviderKind(str, Enum):
    Deepseek = "deepseek"
    NvidiaNim = "nvidia-nim"
    Openai = "openai"
    Atlascloud = "atlascloud"
    WanjieArk = "wanjie-ark"
    Openrouter = "openrouter"
    Novita = "novita"
    Fireworks = "fireworks"
    Sglang = "sglang"
    Vllm = "vllm"
    Ollama = "ollama"

    def as_str(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: str) -> Optional["ProviderKind"]:
        normalized = value.strip().lower()
        aliases = {
            "deepseek": cls.Deepseek,
            "deep-seek": cls.Deepseek,
            "deepseek-cn": cls.Deepseek,
            "deepseek_china": cls.Deepseek,
            "deepseekcn": cls.Deepseek,
            "deepseek-china": cls.Deepseek,
            "nvidia": cls.NvidiaNim,
            "nvidia-nim": cls.NvidiaNim,
            "nvidia_nim": cls.NvidiaNim,
            "nim": cls.NvidiaNim,
            "openai": cls.Openai,
            "open-ai": cls.Openai,
            "atlascloud": cls.Atlascloud,
            "atlas-cloud": cls.Atlascloud,
            "atlas_cloud": cls.Atlascloud,
            "atlas": cls.Atlascloud,
            "wanjie": cls.WanjieArk,
            "wanjie-ark": cls.WanjieArk,
            "wanjie_ark": cls.WanjieArk,
            "ark-wanjie": cls.WanjieArk,
            "ark_wanjie": cls.WanjieArk,
            "wanjieark": cls.WanjieArk,
            "wanjie-maas": cls.WanjieArk,
            "wanjie_maas": cls.WanjieArk,
            "wanjiemaas": cls.WanjieArk,
            "openrouter": cls.Openrouter,
            "open_router": cls.Openrouter,
            "novita": cls.Novita,
            "fireworks": cls.Fireworks,
            "fireworks-ai": cls.Fireworks,
            "sglang": cls.Sglang,
            "sg-lang": cls.Sglang,
            "vllm": cls.Vllm,
            "v-llm": cls.Vllm,
            "ollama": cls.Ollama,
            "ollama-local": cls.Ollama,
        }
        return aliases.get(normalized)


@dataclass
class ModelInfo:
    id: str
    provider: ProviderKind
    aliases: List[str]
    supports_tools: bool
    supports_reasoning: bool


@dataclass
class ModelResolution:
    requested: Optional[str]
    resolved: ModelInfo
    used_fallback: bool
    fallback_chain: List[str]


class ModelRegistry:
    def __init__(self, models: List[ModelInfo]) -> None:
        self._models = models
        self._alias_map: Dict[str, int] = {}
        for idx, model in enumerate(models):
            self._alias_map.setdefault(_normalize(model.id), idx)
            for alias in model.aliases:
                self._alias_map.setdefault(_normalize(alias), idx)

    @classmethod
    def default(cls) -> "ModelRegistry":
        models = [
            ModelInfo(
                id="deepseek-v4-pro",
                provider=ProviderKind.Deepseek,
                aliases=[],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek-v4-flash",
                provider=ProviderKind.Deepseek,
                aliases=[
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "deepseek-r1",
                    "deepseek-v3",
                    "deepseek-v3.2",
                ],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek-ai/deepseek-v4-pro",
                provider=ProviderKind.NvidiaNim,
                aliases=[
                    "deepseek-v4-pro",
                    "nvidia-deepseek-v4-pro",
                    "nim-deepseek-v4-pro",
                ],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek-ai/deepseek-v4-flash",
                provider=ProviderKind.NvidiaNim,
                aliases=[
                    "deepseek-v4-flash",
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "nvidia-deepseek-v4-flash",
                    "nim-deepseek-v4-flash",
                ],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="gpt-4.1",
                provider=ProviderKind.Openai,
                aliases=["gpt4.1", "gpt-4o"],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="gpt-4.1-mini",
                provider=ProviderKind.Openai,
                aliases=["gpt-4o-mini"],
                supports_tools=True,
                supports_reasoning=False,
            ),
            ModelInfo(
                id="deepseek-reasoner",
                provider=ProviderKind.WanjieArk,
                aliases=[
                    "wanjie-deepseek-reasoner",
                    "ark-wanjie-deepseek-reasoner",
                ],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek/deepseek-v4-pro",
                provider=ProviderKind.Openrouter,
                aliases=["deepseek-v4-pro", "openrouter-deepseek-v4-pro"],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek/deepseek-v4-flash",
                provider=ProviderKind.Openrouter,
                aliases=[
                    "deepseek-v4-flash",
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "openrouter-deepseek-v4-flash",
                ],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek/deepseek-v4-pro",
                provider=ProviderKind.Novita,
                aliases=["deepseek-v4-pro", "novita-deepseek-v4-pro"],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek/deepseek-v4-flash",
                provider=ProviderKind.Novita,
                aliases=[
                    "deepseek-v4-flash",
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "novita-deepseek-v4-flash",
                ],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="accounts/fireworks/models/deepseek-v4-pro",
                provider=ProviderKind.Fireworks,
                aliases=["deepseek-v4-pro", "fireworks-deepseek-v4-pro"],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek-ai/DeepSeek-V4-Pro",
                provider=ProviderKind.Sglang,
                aliases=["deepseek-v4-pro", "sglang-deepseek-v4-pro"],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek-ai/DeepSeek-V4-Flash",
                provider=ProviderKind.Sglang,
                aliases=[
                    "deepseek-v4-flash",
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "sglang-deepseek-v4-flash",
                ],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek-ai/DeepSeek-V4-Pro",
                provider=ProviderKind.Vllm,
                aliases=["deepseek-v4-pro", "vllm-deepseek-v4-pro"],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek-ai/DeepSeek-V4-Flash",
                provider=ProviderKind.Vllm,
                aliases=[
                    "deepseek-v4-flash",
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "vllm-deepseek-v4-flash",
                ],
                supports_tools=True,
                supports_reasoning=True,
            ),
            ModelInfo(
                id="deepseek-coder:1.3b",
                provider=ProviderKind.Ollama,
                aliases=[],
                supports_tools=True,
                supports_reasoning=False,
            ),
        ]
        return cls(models)

    def list(self) -> List[ModelInfo]:
        return list(self._models)

    def resolve(
        self, requested: Optional[str], provider_hint: Optional[ProviderKind]
    ) -> ModelResolution:
        fallback_chain: List[str] = []
        if requested:
            fallback_chain.append(f"requested:{requested}")
            if provider_hint == ProviderKind.Ollama:
                return ModelResolution(
                    requested=requested,
                    resolved=ModelInfo(
                        id=requested.strip(),
                        provider=ProviderKind.Ollama,
                        aliases=[],
                        supports_tools=True,
                        supports_reasoning=False,
                    ),
                    used_fallback=False,
                    fallback_chain=fallback_chain,
                )
            if provider_hint is not None:
                match = next(
                    (
                        model
                        for model in self._models
                        if model.provider == provider_hint
                        and _model_matches(model, requested)
                    ),
                    None,
                )
                if match is not None:
                    return ModelResolution(
                        requested=requested,
                        resolved=_preserve_requested_model_id_case(match, requested),
                        used_fallback=False,
                        fallback_chain=fallback_chain,
                    )
            alias_idx = self._alias_map.get(_normalize(requested))
            if alias_idx is not None:
                return ModelResolution(
                    requested=requested,
                    resolved=_preserve_requested_model_id_case(
                        self._models[alias_idx], requested
                    ),
                    used_fallback=False,
                    fallback_chain=fallback_chain,
                )

        provider = provider_hint or ProviderKind.Deepseek
        fallback_chain.append(f"provider_default:{provider.as_str()}")
        fallback_match = next(
            (model for model in self._models if model.provider == provider), None
        )
        if fallback_match is not None:
            return ModelResolution(
                requested=requested,
                resolved=fallback_match,
                used_fallback=True,
                fallback_chain=fallback_chain,
            )

        fallback_chain.append("global_default:deepseek-v4-pro")
        return ModelResolution(
            requested=requested,
            resolved=ModelInfo(
                id="deepseek-v4-pro",
                provider=ProviderKind.Deepseek,
                aliases=[],
                supports_tools=True,
                supports_reasoning=True,
            ),
            used_fallback=True,
            fallback_chain=fallback_chain,
        )


def _normalize(value: str) -> str:
    return value.strip().lower()


def _model_matches(model: ModelInfo, requested: str) -> bool:
    normalized = _normalize(requested)
    return _normalize(model.id) == normalized or any(
        _normalize(alias) == normalized for alias in model.aliases
    )


def _preserve_requested_model_id_case(model: ModelInfo, requested: str) -> ModelInfo:
    if model.id.lower() == requested.strip().lower():
        return ModelInfo(
            id=requested.strip(),
            provider=model.provider,
            aliases=list(model.aliases),
            supports_tools=model.supports_tools,
            supports_reasoning=model.supports_reasoning,
        )
    return model
