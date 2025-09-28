"""OpenRouter provider adapter."""

from __future__ import annotations

import time
from typing import Any, Dict

import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials
from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter


class OpenRouterProvider(ProviderAdapter):
    provider_id = "openrouter"

    def __init__(self, config: ProviderModel) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._path = config.chat_completions_path

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        api_key = credentials.get_api_key(self.provider_id)
        if not api_key:
            raise AuthenticationRequiredError(self.provider_id)

        payload = self._build_payload(request)
        data = await self._post_chat(payload, api_key, track_errors=True)

        data.setdefault("object", "chat.completion")
        data.setdefault("created", int(time.time()))
        data.setdefault("model", request.model)

        return ChatCompletionResponse.model_validate(data)

    async def validate_api_key(self, api_key: str) -> None:
        model = self._config.models.get("default")
        if not model:
            raise ProviderUnavailableError(self.provider_id, message="Health check model not configured")

        request = ChatCompletionRequest(
            model=model,
            messages=[{"role": "user", "content": "healthcheck"}],
            max_tokens=1,
        )
        payload = self._build_payload(request)
        await self._post_chat(payload, api_key, track_errors=False)

    def _build_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
        }

        for field in (
            "temperature",
            "max_tokens",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "stream",
            "user",
        ):
            value = getattr(request, field)
            if value is not None:
                payload[field] = value
        return payload

    async def _post_chat(self, payload: Dict[str, Any], api_key: str, track_errors: bool) -> Dict[str, Any]:
        url = f"{self._base_url}{self._path}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            if track_errors:
                credentials.record_error(self.provider_id, "network")
            raise ProviderUnavailableError(self.provider_id, message="Provider request failed") from exc

        if response.status_code == 401:
            if track_errors:
                credentials.record_error(self.provider_id, "auth")
            raise AuthenticationRequiredError(self.provider_id)
        if response.status_code in {402, 403, 429}:
            if track_errors:
                credentials.record_error(self.provider_id, "rate_limit")
            raise ProviderUnavailableError(self.provider_id, message="Provider quota exhausted")
        if response.is_error:
            if track_errors:
                credentials.record_error(self.provider_id, f"http_{response.status_code}")
            raise ProviderUnavailableError(self.provider_id, message="Provider error")

        if track_errors:
            credentials.clear_error(self.provider_id)

        return response.json()
