from http import HTTPStatus

import app.main as app_main
import pytest
from app.api import openai
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.main import app
from app.providers.base import ChatCompletionResponse
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    async def default_select_provider(payload, provider_id):
        return _mock_response(model=payload.model)

    monkeypatch.setattr(openai, "select_provider", default_select_provider)
    monkeypatch.setattr(app_main, "init_db", lambda: None)

    with TestClient(app) as test_client:
        yield test_client


def _mock_response(model: str = "test-model") -> ChatCompletionResponse:
    return ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 123,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def test_create_chat_completion_returns_success(monkeypatch, client):
    async def fake_select_provider(payload, provider_id):
        return _mock_response(model=payload.model)

    monkeypatch.setattr(openai, "select_provider", fake_select_provider)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "demo-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["model"] == "demo-model"
    assert body["choices"][0]["message"]["content"] == "ok"


def test_create_chat_completion_handles_missing_credentials(monkeypatch, client):
    async def fake_select_provider(payload, provider_id):
        raise AuthenticationRequiredError("cohere")

    monkeypatch.setattr(openai, "select_provider", fake_select_provider)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "cohere/command", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == HTTPStatus.UNAUTHORIZED
    body = response.json()
    assert body["error"]["code"] == "provider_auth_required"
    assert "credentials missing" in body["error"]["message"].lower()


def test_create_chat_completion_handles_provider_unavailable(monkeypatch, client):
    async def fake_select_provider(payload, provider_id):
        raise ProviderUnavailableError("openrouter", message="rate limit")

    monkeypatch.setattr(openai, "select_provider", fake_select_provider)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openrouter/test", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    body = response.json()
    assert body["error"]["code"] == "provider_unavailable"
    assert "rate limit" in body["error"]["message"].lower()
