import json
import os

import httpx
import pytest

from ds_code.core.client import ApiProvider, RetryPolicy


class EnvConfig:
    def deepseek_api_key(self) -> str:
        return os.getenv("DEEPSEEK_API_KEY", "")

    def deepseek_base_url(self) -> str:
        return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    def api_provider(self) -> ApiProvider:
        return ApiProvider.Deepseek

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(enabled=False, max_retries=0, initial_delay=0.1, max_delay=0.2)

    def default_model(self) -> str:
        return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def http_headers(self) -> dict:
        return {}


class HttpxResponseAdapter:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def headers(self) -> dict:
        return dict(self._response.headers)

    async def read(self, max_bytes: int | None = None) -> bytes:
        data = await self._response.aread()
        if max_bytes is not None:
            return data[:max_bytes]
        return data

    async def text(self) -> str:
        data = await self.read()
        return data.decode("utf-8", errors="replace")

    async def json(self):
        return json.loads(await self.text())


class HttpxClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    async def get(self, url: str, headers: dict | None = None):
        response = await self._client.get(url, headers=headers)
        return HttpxResponseAdapter(response)

    async def post(self, url: str, json_body, headers: dict | None = None):
        response = await self._client.post(url, json=json_body, headers=headers)
        return HttpxResponseAdapter(response)

    async def aclose(self) -> None:
        await self._client.aclose()


def require_env() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
