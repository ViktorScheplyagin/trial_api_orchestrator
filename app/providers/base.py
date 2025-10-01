"""Provider adapter interfaces."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.core.config import ProviderModel


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    user: str | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    top_p: float | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list[dict[str, Any]]
    created: int
    model: str
    object: str
    usage: dict[str, Any] | None = None


class ProviderAdapter:
    """Abstract provider adapter."""

    provider_id: str

    def __init__(self, config: ProviderModel) -> None:
        self._config = config

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError

    async def validate_api_key(self, api_key: str) -> None:
        raise NotImplementedError
