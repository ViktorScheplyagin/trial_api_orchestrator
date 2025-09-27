import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.providers.base import ChatCompletionRequest
from app.providers.openrouter import OpenRouterProvider


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400


def _stub_async_client(response, recorder):
    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None):
            recorder["url"] = url
            recorder["json"] = json
            recorder["headers"] = headers
            return response

    return _DummyAsyncClient


@pytest.fixture
def provider_model() -> ProviderModel:
    return ProviderModel(
        id="openrouter",
        name="OpenRouter",
        priority=30,
        base_url="https://openrouter.ai/api",
        chat_completions_path="/v1/chat/completions",
        availability={},
        credentials={},
        models={},
    )


@pytest.mark.asyncio
async def test_openrouter_adapter_success(monkeypatch, provider_model):
    adapter = OpenRouterProvider(provider_model)
    recorder: dict = {}

    response_payload = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "openrouter/auto",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    fake_http = FakeResponse(200, response_payload)

    monkeypatch.setattr("app.providers.openrouter.credentials.get_api_key", lambda _: "router-key")
    monkeypatch.setattr("app.providers.openrouter.credentials.record_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.openrouter.credentials.clear_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.openrouter.httpx.AsyncClient", _stub_async_client(fake_http, recorder))

    request = ChatCompletionRequest(
        model="openrouter/auto",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        user="abc123",
    )

    response = await adapter.chat_completions(request)

    assert recorder["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert recorder["headers"]["Authorization"] == "Bearer router-key"
    assert recorder["json"]["temperature"] == 0.7
    assert recorder["json"]["presence_penalty"] == 0.1
    assert response.choices[0]["message"]["content"] == "Hello"


@pytest.mark.asyncio
async def test_openrouter_adapter_missing_credentials(monkeypatch, provider_model):
    adapter = OpenRouterProvider(provider_model)
    monkeypatch.setattr("app.providers.openrouter.credentials.get_api_key", lambda _: None)

    request = ChatCompletionRequest(model="openrouter/auto", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(AuthenticationRequiredError):
        await adapter.chat_completions(request)


@pytest.mark.asyncio
async def test_openrouter_adapter_rate_limited(monkeypatch, provider_model):
    adapter = OpenRouterProvider(provider_model)
    recorder: dict = {}
    fake_http = FakeResponse(429)

    monkeypatch.setattr("app.providers.openrouter.credentials.get_api_key", lambda _: "router-key")
    monkeypatch.setattr("app.providers.openrouter.credentials.record_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.openrouter.httpx.AsyncClient", _stub_async_client(fake_http, recorder))

    request = ChatCompletionRequest(model="openrouter/auto", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(ProviderUnavailableError):
        await adapter.chat_completions(request)
