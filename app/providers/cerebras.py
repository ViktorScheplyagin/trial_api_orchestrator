"""Cerebras provider adapter."""

from __future__ import annotations

import time
from typing import Any, Dict

import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials
from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter


class CerebrasProvider(ProviderAdapter):
    provider_id = "cerebras"

    def __init__(self, config: ProviderModel) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._path = config.chat_completions_path

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        api_key = credentials.get_api_key(self.provider_id)
        if not api_key:
            raise AuthenticationRequiredError(self.provider_id)

        url = f"{self._base_url}{self._path}"
        payload = self._build_payload(request)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code == 401:
            credentials.record_error(self.provider_id, "auth")
            raise AuthenticationRequiredError(self.provider_id)

        if response.status_code in {402, 403, 429}:
            credentials.record_error(self.provider_id, "rate_limit")
            raise ProviderUnavailableError(self.provider_id, message="Provider quota exhausted")

        if response.is_error:
            credentials.record_error(self.provider_id, f"http_{response.status_code}")
            raise ProviderUnavailableError(self.provider_id, message="Provider error")

        data: Dict[str, Any] = response.json()
        credentials.clear_error(self.provider_id)

        # Ensure OpenAI-compatible keys exist
        data.setdefault("object", "chat.completion")
        data.setdefault("created", int(time.time()))
        data.setdefault("model", request.model)

        return ChatCompletionResponse.model_validate(data)

    def _build_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
        }
        optional_fields = (
            "temperature",
            "max_tokens",
            "stream",
            "user",
            "presence_penalty",
            "frequency_penalty",
            "top_p",
        )
        for field in optional_fields:
            value = getattr(request, field)
            if value is not None:
                payload[field] = value
        return payload
