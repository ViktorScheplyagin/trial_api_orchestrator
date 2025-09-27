import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.providers.base import ChatCompletionRequest
from app.providers.gemini import GeminiProvider


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
        id="gemini",
        name="Gemini API",
        priority=25,
        base_url="https://generativelanguage.googleapis.com",
        chat_completions_path="/v1beta/models/{model}:generateContent",
        availability={},
        credentials={},
        models={},
    )


@pytest.mark.asyncio
async def test_gemini_adapter_success(monkeypatch, provider_model):
    adapter = GeminiProvider(provider_model)
    recorder: dict = {}

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
    fake_http = FakeResponse(200, response_payload)

    monkeypatch.setattr("app.providers.gemini.credentials.get_api_key", lambda _: "gemini-key")
    monkeypatch.setattr("app.providers.gemini.credentials.record_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.gemini.credentials.clear_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.gemini.httpx.AsyncClient", _stub_async_client(fake_http, recorder))

    request = ChatCompletionRequest(
        model="models/gemini-1.5-flash-latest",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "Hi there"},
        ],
        temperature=0.3,
        max_tokens=256,
        top_p=0.95,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )

    response = await adapter.chat_completions(request)

    assert (
        recorder["url"]
        == "https://generativelanguage.googleapis.com/v1beta/models/models/gemini-1.5-flash-latest:generateContent"
    )
    assert recorder["headers"]["x-goog-api-key"] == "gemini-key"

    payload = recorder["json"]
    assert payload["systemInstruction"]["parts"][0]["text"] == "You are helpful."
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][1]["role"] == "model"
    config = payload["generationConfig"]
    assert config["temperature"] == 0.3
    assert config["maxOutputTokens"] == 256
    assert config["topP"] == 0.95
    assert config["frequencyPenalty"] == 0.1
    assert config["presencePenalty"] == 0.2

    choice = response.choices[0]
    assert choice["message"]["content"] == "Hello from Gemini"
    assert choice["finish_reason"] == "stop"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}


@pytest.mark.asyncio
async def test_gemini_adapter_missing_credentials(monkeypatch, provider_model):
    adapter = GeminiProvider(provider_model)
    monkeypatch.setattr("app.providers.gemini.credentials.get_api_key", lambda _: None)

    request = ChatCompletionRequest(model="models/gemini-1.5-flash-latest", messages=[{"role": "user", "content": "Ping"}])

    with pytest.raises(AuthenticationRequiredError):
        await adapter.chat_completions(request)


@pytest.mark.asyncio
async def test_gemini_adapter_rate_limited(monkeypatch, provider_model):
    adapter = GeminiProvider(provider_model)
    recorder: dict = {}
    fake_http = FakeResponse(429)

    monkeypatch.setattr("app.providers.gemini.credentials.get_api_key", lambda _: "gemini-key")
    monkeypatch.setattr("app.providers.gemini.credentials.record_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.gemini.httpx.AsyncClient", _stub_async_client(fake_http, recorder))

    request = ChatCompletionRequest(model="models/gemini-1.5-flash-latest", messages=[{"role": "user", "content": "Ping"}])

    with pytest.raises(ProviderUnavailableError):
        await adapter.chat_completions(request)
