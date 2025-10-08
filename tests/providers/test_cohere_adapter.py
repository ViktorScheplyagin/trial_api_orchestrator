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
    logs: list = []
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
    monkeypatch.setattr(
        "app.providers.cohere.record_provider_log",
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
        model="command-r-plus", messages=[{"role": "user", "content": "Hi"}]
    )
    response = await adapter.chat_completions(request)

    assert recorder["url"] == "https://api.cohere.com/v2/chat"
    assert recorder["headers"]["Authorization"] == "Bearer test-key"
    assert "presence_penalty" not in recorder["json"]
    assert recorder["json"]["messages"][0]["content"] == [
        {"type": "text", "text": "Hi"}
    ]

    choice = response.choices[0]
    assert choice["message"]["content"] == "Hello"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert choice["message"]["metadata"]["cohere"]["citations"] == [{"source": "doc-1"}]
    assert response.usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    assert len(logs) == 1
    assert "error" not in logs[0]["response_body"]


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
    logs: list = []
    fake_http = FakeResponse(HTTPStatus.TOO_MANY_REQUESTS)

    monkeypatch.setattr("app.providers.cohere.credentials.get_api_key", lambda _: "test-key")
    monkeypatch.setattr(
        "app.providers.cohere.credentials.record_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.cohere.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )
    monkeypatch.setattr(
        "app.providers.cohere.record_provider_log",
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
        model="command-r-plus", messages=[{"role": "user", "content": "Hi"}]
    )

    with pytest.raises(ProviderUnavailableError):
        await adapter.chat_completions(request)

    assert len(logs) == 1
    error_payload = logs[0]["response_body"]
    assert error_payload["error"]["type"] == "rate_limit"
    assert error_payload["error"]["status_code"] == HTTPStatus.TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_cohere_adapter_builds_multimodal_payload(monkeypatch, provider_model):
    adapter = CohereProvider(provider_model)

    recorder: dict = {}
    logs: list = []
    response_payload = {
        "id": "chat-456",
        "message": {
            "content": [
                {"type": "text", "text": "Describing image"},
            ]
        },
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
    monkeypatch.setattr(
        "app.providers.cohere.record_provider_log",
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
        model="command-a-vision-07-2025",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image": {
                            "url": "https://example.com/image.png",
                            "media_type": "image/png",
                        },
                    },
                ],
            }
        ],
    )

    response = await adapter.chat_completions(request)

    payload_message = recorder["json"]["messages"][0]
    assert payload_message["content"][0] == {"type": "text", "text": "What is in this image?"}
    image_part = payload_message["content"][1]
    assert image_part["type"] == "image"
    assert image_part["source"] == {
        "type": "url",
        "url": "https://example.com/image.png",
        "media_type": "image/png",
    }

    message_content = response.choices[0]["message"]["content"]
    assert message_content == "Describing image"


@pytest.mark.asyncio
async def test_cohere_adapter_preserves_image_content(monkeypatch, provider_model):
    adapter = CohereProvider(provider_model)

    recorder: dict = {}
    logs: list = []
    response_payload = {
        "id": "chat-789",
        "message": {
            "content": [
                {"type": "text", "text": "Here is the chart:"},
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/chart.png",
                        "media_type": "image/png",
                    },
                },
            ]
        },
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
    monkeypatch.setattr(
        "app.providers.cohere.record_provider_log",
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
        model="command-a-vision-07-2025",
        messages=[{"role": "user", "content": "Describe"}],
    )

    response = await adapter.chat_completions(request)

    content = response.choices[0]["message"]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Here is the chart:"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/chart.png",
            "media_type": "image/png",
        },
    }
