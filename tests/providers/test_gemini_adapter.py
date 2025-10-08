import pathlib
import sys
from http import HTTPStatus

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.providers.base import ChatCompletionRequest
from app.providers.gemini import GeminiProvider

TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 256
TOP_P = 0.95
FREQUENCY_PENALTY = 0.1
PRESENCE_PENALTY = 0.2
HEALTHCHECK_MAX_TOKENS = 16


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
        id="gemini",
        name="Gemini API",
        priority=25,
        base_url="https://generativelanguage.googleapis.com",
        chat_completions_path="/v1beta/models/{model}:generateContent",
        availability={},
        credentials={},
        models={"default": "models/gemini-2.5-flash"},
    )


@pytest.mark.asyncio
async def test_gemini_adapter_success(monkeypatch, provider_model):
    adapter = GeminiProvider(provider_model)
    recorder: dict = {}
    logs: list = []

    response_payload = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello from Gemini"}]},
                "finishReason": "STOP",
                "safetyRatings": [{"category": "HARM_CATEGORY_HATE", "probability": "NEGLIGIBLE"}],
                "citationMetadata": {"citations": [{"source": "doc-1"}]},
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 15,
            "totalTokenCount": 25,
        },
    }
    fake_http = FakeResponse(HTTPStatus.OK, response_payload)

    monkeypatch.setattr("app.providers.gemini.credentials.get_api_key", lambda _: "gemini-key")
    monkeypatch.setattr(
        "app.providers.gemini.credentials.record_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.gemini.credentials.clear_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.gemini.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )
    monkeypatch.setattr(
        "app.providers.gemini.record_provider_log",
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
        model="models/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "Hi there"},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )

    response = await adapter.chat_completions(request)

    assert (
        recorder["url"]
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert recorder["headers"]["x-goog-api-key"] == "gemini-key"

    payload = recorder["json"]
    assert payload["systemInstruction"]["parts"][0]["text"] == "You are helpful."
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][1]["role"] == "model"
    config = payload["generationConfig"]
    assert config["temperature"] == TEMPERATURE
    assert config["maxOutputTokens"] == MAX_OUTPUT_TOKENS
    assert config["topP"] == TOP_P
    assert config["frequencyPenalty"] == FREQUENCY_PENALTY
    assert config["presencePenalty"] == PRESENCE_PENALTY

    choice = response.choices[0]
    assert choice["message"]["content"] == "Hello from Gemini"
    assert choice["finish_reason"] == "stop"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
    assert len(logs) == 1
    assert "error" not in logs[0]["response_body"]


@pytest.mark.asyncio
async def test_gemini_adapter_missing_credentials(monkeypatch, provider_model):
    adapter = GeminiProvider(provider_model)
    monkeypatch.setattr("app.providers.gemini.credentials.get_api_key", lambda _: None)

    request = ChatCompletionRequest(
        model="models/gemini-2.5-flash", messages=[{"role": "user", "content": "Ping"}]
    )

    with pytest.raises(AuthenticationRequiredError):
        await adapter.chat_completions(request)


@pytest.mark.asyncio
async def test_gemini_adapter_includes_provider_error_detail(monkeypatch, provider_model):
    adapter = GeminiProvider(provider_model)
    recorder: dict = {}
    logs: list = []
    payload = {"error": {"status": "PERMISSION_DENIED", "message": "API key invalid."}}
    fake_http = FakeResponse(HTTPStatus.FORBIDDEN, payload)

    monkeypatch.setattr("app.providers.gemini.credentials.get_api_key", lambda _: "gemini-key")
    monkeypatch.setattr(
        "app.providers.gemini.credentials.record_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.gemini.credentials.clear_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.gemini.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )
    monkeypatch.setattr(
        "app.providers.gemini.record_provider_log",
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
        model="models/gemini-2.5-flash", messages=[{"role": "user", "content": "Ping"}]
    )

    with pytest.raises(ProviderUnavailableError) as excinfo:
        await adapter.chat_completions(request)

    message = excinfo.value.message
    assert "PERMISSION_DENIED" in message
    assert "API key invalid." in message
    assert len(logs) == 1
    error_payload = logs[0]["response_body"]
    assert error_payload["error"]["type"] == "rate_limit"
    assert error_payload["error"]["status_code"] == HTTPStatus.FORBIDDEN


@pytest.mark.asyncio
async def test_gemini_adapter_rate_limited(monkeypatch, provider_model):
    adapter = GeminiProvider(provider_model)
    recorder: dict = {}
    logs: list = []
    fake_http = FakeResponse(HTTPStatus.TOO_MANY_REQUESTS)

    monkeypatch.setattr("app.providers.gemini.credentials.get_api_key", lambda _: "gemini-key")
    monkeypatch.setattr(
        "app.providers.gemini.credentials.record_error", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "app.providers.gemini.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )
    monkeypatch.setattr(
        "app.providers.gemini.record_provider_log",
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
        model="models/gemini-2.5-flash", messages=[{"role": "user", "content": "Ping"}]
    )

    with pytest.raises(ProviderUnavailableError):
        await adapter.chat_completions(request)
    assert len(logs) == 1
    error_payload = logs[0]["response_body"]
    assert error_payload["error"]["type"] == "rate_limit"
    assert error_payload["error"]["status_code"] == HTTPStatus.TOO_MANY_REQUESTS


@pytest.mark.asyncio
async def test_validate_api_key_removes_models_prefix(monkeypatch, provider_model):
    adapter = GeminiProvider(provider_model)
    recorder: dict = {}
    fake_http = FakeResponse(HTTPStatus.OK, {"candidates": []})

    monkeypatch.setattr(
        "app.providers.gemini.httpx.AsyncClient", _stub_async_client(fake_http, recorder)
    )

    await adapter.validate_api_key("test-key")

    assert (
        recorder["url"]
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert recorder["headers"]["x-goog-api-key"] == "test-key"
    payload = recorder["json"]
    assert payload["contents"][0]["parts"][0]["text"] == "healthcheck"
    assert payload["generationConfig"]["maxOutputTokens"] == HEALTHCHECK_MAX_TOKENS
