import pytest
from app.core.config import AppConfig, ProviderModel
from app.core.exceptions import ProviderUnavailableError
from app.providers.base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter
from app.router import selector


class _FailingAdapter(ProviderAdapter):
    provider_id = "fail"

    def __init__(self, config: ProviderModel) -> None:  # pragma: no cover - simple init
        self.config = config

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise ProviderUnavailableError(self.provider_id, message="boom")

    async def validate_api_key(self, api_key: str) -> None:  # pragma: no cover - unused
        return None


class _SuccessfulAdapter(ProviderAdapter):
    provider_id = "ok"

    def __init__(self, config: ProviderModel) -> None:  # pragma: no cover - simple init
        self.config = config
        self.calls = 0

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.calls += 1
        return ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 123,
                "model": request.model,
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

    async def validate_api_key(self, api_key: str) -> None:  # pragma: no cover - unused
        return None


@pytest.mark.asyncio
async def test_select_provider_fails_over_to_next_adapter(monkeypatch):
    events: list[tuple[str, str, dict]] = []

    def capture_event(kind: str, level: str, **fields) -> None:
        events.append((kind, level, fields))

    monkeypatch.setattr(selector, "record_event", capture_event)

    config = AppConfig(
        providers=[
            ProviderModel(
                id="fail",
                name="Fail Provider",
                priority=10,
                base_url="https://fail.example",
                chat_completions_path="/chat",
            ),
            ProviderModel(
                id="ok",
                name="OK Provider",
                priority=20,
                base_url="https://ok.example",
                chat_completions_path="/chat",
            ),
        ]
    )

    monkeypatch.setattr(
        selector.ProviderRegistry,
        "_adapter_map",
        {"fail": _FailingAdapter, "ok": _SuccessfulAdapter},
    )
    registry = selector.ProviderRegistry(config=config)
    monkeypatch.setattr(selector, "registry", registry)

    request = ChatCompletionRequest(
        model="ok-model", messages=[{"role": "user", "content": "ping"}]
    )
    response = await selector.select_provider(request)

    assert response.choices[0]["message"]["content"] == "ok"

    kinds = [kind for kind, _, _ in events]
    assert "provider_fail" in kinds
    assert "provider_switched" in kinds
