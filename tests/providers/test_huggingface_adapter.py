import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.providers.base import ChatCompletionRequest
from app.providers.huggingface import HuggingFaceProvider


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
        id="huggingface",
        name="Hugging Face Inference API",
        priority=40,
        base_url="https://api-inference.huggingface.co",
        chat_completions_path="/models/{model_id}/chat/completions",
        availability={},
        credentials={},
        models={},
    )


@pytest.mark.asyncio
async def test_huggingface_adapter_success(monkeypatch, provider_model):
    adapter = HuggingFaceProvider(provider_model)
    recorder: dict = {}

    response_payload = {
        "id": "chatcmpl-hf-xyz",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from HF"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
    }
    fake_http = FakeResponse(200, response_payload)

    monkeypatch.setattr("app.providers.huggingface.credentials.get_api_key", lambda _: "hf-token")
    monkeypatch.setattr("app.providers.huggingface.credentials.record_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.huggingface.credentials.clear_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.huggingface.httpx.AsyncClient", _stub_async_client(fake_http, recorder))

    request = ChatCompletionRequest(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.5,
        max_tokens=128,
    )

    response = await adapter.chat_completions(request)

    assert (
        recorder["url"]
        == "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta/chat/completions"
    )
    assert recorder["headers"]["Authorization"] == "Bearer hf-token"
    assert recorder["json"]["temperature"] == 0.5
    assert response.choices[0]["message"]["content"] == "Hello from HF"


@pytest.mark.asyncio
async def test_huggingface_adapter_generates_choice(monkeypatch, provider_model):
    adapter = HuggingFaceProvider(provider_model)
    recorder: dict = {}

    response_payload = {
        "generated_text": "Fallback text",
        "finish_reason": "length",
    }
    fake_http = FakeResponse(200, response_payload)

    monkeypatch.setattr("app.providers.huggingface.credentials.get_api_key", lambda _: "hf-token")
    monkeypatch.setattr("app.providers.huggingface.credentials.record_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.huggingface.credentials.clear_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.huggingface.httpx.AsyncClient", _stub_async_client(fake_http, recorder))

    request = ChatCompletionRequest(model="meta-llama/Llama-3", messages=[{"role": "user", "content": "Ping"}])
    response = await adapter.chat_completions(request)

    choice = response.choices[0]
    assert choice["message"]["content"] == "Fallback text"
    assert choice["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_huggingface_adapter_missing_credentials(monkeypatch, provider_model):
    adapter = HuggingFaceProvider(provider_model)
    monkeypatch.setattr("app.providers.huggingface.credentials.get_api_key", lambda _: None)

    request = ChatCompletionRequest(model="meta-llama/Llama-3", messages=[{"role": "user", "content": "Ping"}])

    with pytest.raises(AuthenticationRequiredError):
        await adapter.chat_completions(request)


@pytest.mark.asyncio
async def test_huggingface_adapter_rate_limited(monkeypatch, provider_model):
    adapter = HuggingFaceProvider(provider_model)
    recorder: dict = {}
    fake_http = FakeResponse(429)

    monkeypatch.setattr("app.providers.huggingface.credentials.get_api_key", lambda _: "hf-token")
    monkeypatch.setattr("app.providers.huggingface.credentials.record_error", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.providers.huggingface.httpx.AsyncClient", _stub_async_client(fake_http, recorder))

    request = ChatCompletionRequest(model="meta-llama/Llama-3", messages=[{"role": "user", "content": "Ping"}])

    with pytest.raises(ProviderUnavailableError):
        await adapter.chat_completions(request)
