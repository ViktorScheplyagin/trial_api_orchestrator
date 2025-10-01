import pathlib
import sys
from http import HTTPStatus

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.providers.base import ChatCompletionRequest
from app.providers.cohere import CohereProvider


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    @property
    def is_error(self) -> bool:
        return self.status_code >= HTTPStatus.BAD_REQUEST


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
        id="cohere",
        name="Cohere",
        priority=20,
        base_url="https://api.cohere.com",
        chat_completions_path="/v2/chat",
        availability={},
        credentials={},
        models={},
    )


@pytest.mark.asyncio
async def test_cohere_adapter_normalizes_response(monkeypatch, provider_model):
    adapter = CohereProvider(provider_model)

    recorder: dict = {}
    response_payload = {
        "id": "chat-123",
        "message": {
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": {"item": "value"},
                            },
                        }
                    ],
                },
                {"type": "citation", "citations": [{"source": "doc-1"}]},
            ]
        },
        "usage": {"tokens": {"input": 5, "output": 7}},
        "finish_reason": "COMPLETE",
    }
    fake_http = FakeResponse(HTTPStatus.OK, response_payload)

    monkeypatch.setattr("app.providers.cohere.credentials.get_api_key", lambda _: "test-key")
    monkeypatch.setattr(
        "app.providers.cohere.credentials.record_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.cohere.credentials.clear_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.cohere.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )

    request = ChatCompletionRequest(
        model="command-r-plus", messages=[{"role": "user", "content": "Hi"}]
    )
    response = await adapter.chat_completions(request)

    assert recorder["url"] == "https://api.cohere.com/v2/chat"
    assert recorder["headers"]["Authorization"] == "Bearer test-key"
    assert "presence_penalty" not in recorder["json"]

    choice = response.choices[0]
    assert choice["message"]["content"] == "Hello"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert choice["message"]["metadata"]["cohere"]["citations"] == [{"source": "doc-1"}]
    assert response.usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


@pytest.mark.asyncio
async def test_cohere_adapter_handles_missing_credentials(monkeypatch, provider_model):
    adapter = CohereProvider(provider_model)
    monkeypatch.setattr("app.providers.cohere.credentials.get_api_key", lambda _: None)

    request = ChatCompletionRequest(
        model="command-r-plus", messages=[{"role": "user", "content": "Hi"}]
    )

    with pytest.raises(AuthenticationRequiredError):
        await adapter.chat_completions(request)


@pytest.mark.asyncio
async def test_cohere_adapter_raises_on_rate_limit(monkeypatch, provider_model):
    adapter = CohereProvider(provider_model)

    recorder: dict = {}
    fake_http = FakeResponse(HTTPStatus.TOO_MANY_REQUESTS)

    monkeypatch.setattr("app.providers.cohere.credentials.get_api_key", lambda _: "test-key")
    monkeypatch.setattr(
        "app.providers.cohere.credentials.record_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.cohere.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )

    request = ChatCompletionRequest(
        model="command-r-plus", messages=[{"role": "user", "content": "Hi"}]
    )

    with pytest.raises(ProviderUnavailableError):
        await adapter.chat_completions(request)
