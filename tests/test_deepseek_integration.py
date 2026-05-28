import os

import httpx
import pytest

from ds_code.core.client import ApiProvider, DeepSeekClient, RetryPolicy
from ds_code.core.models import Message, MessageRequest, TextBlock


class EnvConfig:
    def deepseek_api_key(self) -> str:
        return os.getenv("DEEPSEEK_API_KEY", "")

    def deepseek_base_url(self) -> str:
        return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    def api_provider(self) -> ApiProvider:
        return ApiProvider.Deepseek

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(enabled=False, max_retries=0, initial_delay=0.1, max_delay=0.2)

    def default_model(self) -> str:
        return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def http_headers(self) -> dict:
        return {}


class HttpxClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    async def get(self, url: str):
        return await self._client.get(url)

    async def post(self, url: str, json_body):
        return await self._client.post(url, json=json_body)

    async def aclose(self) -> None:
        await self._client.aclose()


def _require_env() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models_live() -> None:
    _require_env()
    http_client = HttpxClient()
    try:
        client = DeepSeekClient.new(EnvConfig(), http_client)
        models = await client.list_models()
        assert models
        assert all(model.id for model in models)
    finally:
        await http_client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_check_live() -> None:
    _require_env()
    http_client = HttpxClient()
    try:
        client = DeepSeekClient.new(EnvConfig(), http_client)
        ok = await client.health_check()
        assert ok is True
    finally:
        await http_client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_translate_live() -> None:
    _require_env()
    http_client = HttpxClient()
    try:
        client = DeepSeekClient.new(EnvConfig(), http_client)
        model = os.getenv("DEEPSEEK_TRANSLATE_MODEL", client.default_model)
        text = "Hello world"
        translated = await client.translate(text, model, "Chinese")
        assert translated
    finally:
        await http_client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fim_completion_live() -> None:
    _require_env()
    model = os.getenv("DEEPSEEK_FIM_MODEL")
    if not model:
        pytest.skip("DEEPSEEK_FIM_MODEL not set")
    http_client = HttpxClient()
    try:
        client = DeepSeekClient.new(EnvConfig(), http_client)
        text = await client.fim_completion(model, "def hello():\n    ", "\n", 32)
        assert text
    finally:
        await http_client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_message_chat_live() -> None:
    _require_env()
    http_client = HttpxClient()
    try:
        client = DeepSeekClient.new(EnvConfig(), http_client)
        request = MessageRequest(
            model=client.default_model,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
            max_tokens=16,
        )
        try:
            await client.create_message_chat(request)
        except NotImplementedError:
            pytest.skip("create_message_chat is not implemented in this client")
    finally:
        await http_client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_message_stream_live() -> None:
    _require_env()
    http_client = HttpxClient()
    try:
        client = DeepSeekClient.new(EnvConfig(), http_client)
        request = MessageRequest(
            model=client.default_model,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
            max_tokens=16,
            stream=True,
        )
        try:
            await client.create_message_stream(request)
        except NotImplementedError:
            pytest.skip("create_message_stream is not implemented in this client")
    finally:
        await http_client.aclose()
