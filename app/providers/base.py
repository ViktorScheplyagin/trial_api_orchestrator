"""Provider adapter interfaces."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel


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
    choices: list[Dict[str, Any]]
    created: int
    model: str
    object: str
    usage: Dict[str, Any] | None = None


class ProviderAdapter:
    """Abstract provider adapter."""

    provider_id: str

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError
