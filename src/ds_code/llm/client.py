from typing import Optional

from ds_code.llm.models import ModelRegistry, ModelResolution, ProviderKind


class LlmClient:
    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self._registry = registry or ModelRegistry.default()

    def resolve_model(
        self, requested: Optional[str], provider_hint: Optional[ProviderKind]
    ) -> ModelResolution:
        return self._registry.resolve(requested, provider_hint)

    def generate(
        self,
        prompt: str,
        requested_model: Optional[str] = None,
        provider_hint: Optional[ProviderKind] = None,
    ) -> str:
        resolution = self.resolve_model(requested_model, provider_hint)
        return (
            f"LLM placeholder response for: {prompt} "
            f"(model={resolution.resolved.id}, provider={resolution.resolved.provider.as_str()})"
        )
