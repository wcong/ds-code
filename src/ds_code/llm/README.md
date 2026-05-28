# LLM

## Overview
The LLM layer resolves model selections and returns placeholder responses. It is intentionally lightweight to keep the runtime deterministic for now.

## Files
- [client.py](client.py): LlmClient that resolves models and returns placeholder text.
- [models.py](models.py): ProviderKind, ModelInfo, ModelRegistry, and resolution logic.
- [__init__.py](__init__.py): Re-exports LLM symbols.

## How It Fits Together
- [core/engine.py](../core/engine.py) calls [client.py](client.py) to resolve models and generate responses.
- [config.py](../config.py) and [runtime/models.py](../runtime/models.py) influence provider/model selection.
