import pytest

from ds_code.core.client import DeepSeekClient
from ds_code.core.models import Message as CoreMessage
from ds_code.core.models import MessageRequest as CoreMessageRequest
from ds_code.core.models import TextBlock as CoreTextBlock
from ds_code.core.models import Message, MessageRequest, MessageResponse, TextBlock
from ds_code.core.compaction import Usage as CompactionUsage
from ds_code.core import compaction
from tests.utils import EnvConfig, HttpxClient, require_env


class CompactionClientAdapter:
    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    async def create_message(self, request: MessageRequest) -> MessageResponse:
        core_request = CoreMessageRequest(
            model=request.model,
            messages=[
                CoreMessage(role=msg.role, content=[CoreTextBlock(text=msg.content[0].text)])
                for msg in request.messages
            ],
            max_tokens=request.max_tokens,
            system=None,
            tools=None,
            tool_choice=None,
            metadata=request.metadata,
            thinking=request.thinking,
            reasoning_effort=request.reasoning_effort,
            stream=False,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        core_response = await self._client.create_message_chat(core_request)
        text_parts = []
        for block in core_response.content:
            if isinstance(block, CoreTextBlock):
                text_parts.append(block.text)
        usage = CompactionUsage(
            input_tokens=core_response.usage.input_tokens,
            output_tokens=core_response.usage.output_tokens,
            prompt_cache_hit_tokens=core_response.usage.prompt_cache_hit_tokens,
            prompt_cache_miss_tokens=core_response.usage.prompt_cache_miss_tokens,
        )
        return MessageResponse(
            id=core_response.id,
            type=core_response.type,
            role=core_response.role,
            stop_reason=core_response.stop_reason,
            stop_sequence=core_response.stop_sequence,
            container=core_response.container,
            model=core_response.model,
            content=[TextBlock(text="\n".join(text_parts))],
            usage=usage,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compaction_summary_live() -> None:
    require_env()
    http_client = HttpxClient()
    try:
        client = DeepSeekClient.new(EnvConfig(), http_client)
        adapter = CompactionClientAdapter(client)
        messages = [
            Message(role="user", content=[TextBlock(text="Summarize this conversation.")]),
            Message(role="assistant", content=[TextBlock(text="Sure, I can do that.")]),
            Message(role="user", content=[TextBlock(text="We discussed tests and config.")]),
        ]
        summary = await compaction.create_summary(adapter, messages, client.default_model)
        assert summary
    finally:
        await http_client.aclose()
