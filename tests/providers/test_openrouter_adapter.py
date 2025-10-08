import pathlib
import sys
from http import HTTPStatus

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.providers.base import ChatCompletionRequest
from app.providers.openrouter import OpenRouterProvider

TEMPERATURE = 0.7
MAX_TOKENS = 256
TOP_P = 0.9
PRESENCE_PENALTY = 0.1
FREQUENCY_PENALTY = 0.2


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self._text = text

    def json(self) -> dict:
        return self._payload

    @property
    def is_error(self) -> bool:
        return self.status_code >= HTTPStatus.BAD_REQUEST

    @property
    def text(self) -> str:
        if self._text is not None:
            return self._text
        if not self._payload:
            return ""
        try:
            import json as _json

            return _json.dumps(self._payload)
        except Exception:
            return ""


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
    logs: list = []

    response_payload = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "deepseek/deepseek-chat-v3.1:free",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    fake_http = FakeResponse(HTTPStatus.OK, response_payload)

    monkeypatch.setattr("app.providers.openrouter.credentials.get_api_key", lambda _: "router-key")
    monkeypatch.setattr(
        "app.providers.openrouter.credentials.record_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.openrouter.credentials.clear_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.openrouter.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )
    monkeypatch.setattr(
        "app.providers.openrouter.record_provider_log",
        lambda provider_id, *, request_body, response_body, request_id=None: logs.append(
            {
                "provider_id": provider_id,
                "request_body": request_body,
                "response_body": response_body,
                "request_id": request_id,
            }
        ),
    )

    request = ChatCompletionRequest(
        model="deepseek/deepseek-chat-v3.1:free",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        presence_penalty=PRESENCE_PENALTY,
        frequency_penalty=FREQUENCY_PENALTY,
        user="abc123",
    )

    response = await adapter.chat_completions(request)

    assert recorder["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert recorder["headers"]["Authorization"] == "Bearer router-key"
    assert recorder["json"]["temperature"] == TEMPERATURE
    assert recorder["json"]["presence_penalty"] == PRESENCE_PENALTY
    assert response.choices[0]["message"]["content"] == "Hello"
    assert len(logs) == 1
    assert "error" not in logs[0]["response_body"]


@pytest.mark.asyncio
async def test_openrouter_adapter_missing_credentials(monkeypatch, provider_model):
    adapter = OpenRouterProvider(provider_model)
    monkeypatch.setattr("app.providers.openrouter.credentials.get_api_key", lambda _: None)

    request = ChatCompletionRequest(
        model="deepseek/deepseek-chat-v3.1:free", messages=[{"role": "user", "content": "Hi"}]
    )

    with pytest.raises(AuthenticationRequiredError):
        await adapter.chat_completions(request)


@pytest.mark.asyncio
async def test_openrouter_adapter_rate_limited(monkeypatch, provider_model):
    adapter = OpenRouterProvider(provider_model)
    recorder: dict = {}
    logs: list = []
    fake_http = FakeResponse(HTTPStatus.TOO_MANY_REQUESTS)

    monkeypatch.setattr("app.providers.openrouter.credentials.get_api_key", lambda _: "router-key")
    monkeypatch.setattr(
        "app.providers.openrouter.credentials.record_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.openrouter.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )
    monkeypatch.setattr(
        "app.providers.openrouter.record_provider_log",
        lambda provider_id, *, request_body, response_body, request_id=None: logs.append(
            {
                "provider_id": provider_id,
                "request_body": request_body,
                "response_body": response_body,
                "request_id": request_id,
            }
        ),
    )

    request = ChatCompletionRequest(
        model="deepseek/deepseek-chat-v3.1:free", messages=[{"role": "user", "content": "Hi"}]
    )

    with pytest.raises(ProviderUnavailableError):
        await adapter.chat_completions(request)

    assert len(logs) == 1
    error_payload = logs[0]["response_body"]
    assert error_payload["error"]["type"] == "rate_limit"
    assert error_payload["error"]["status_code"] == HTTPStatus.TOO_MANY_REQUESTS
