import os

import pytest

from ds_code.core.client import DeepSeekClient
from ds_code.core.models import Message, MessageRequest, TextBlock
from tests.utils import EnvConfig, HttpxClient, require_env


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models_live() -> None:
    require_env()
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
    require_env()
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
    require_env()
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
    require_env()
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
    require_env()
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
    require_env()
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
