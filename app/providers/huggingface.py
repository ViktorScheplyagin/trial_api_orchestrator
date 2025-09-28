"""Hugging Face Inference API provider adapter."""

from __future__ import annotations

import time
from typing import Any, Dict

import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials
from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter


class HuggingFaceProvider(ProviderAdapter):
    provider_id = "huggingface"

    def __init__(self, config: ProviderModel) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._path = config.chat_completions_path

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        api_key = credentials.get_api_key(self.provider_id)
        if not api_key:
            raise AuthenticationRequiredError(self.provider_id)

        payload = self._build_payload(request)
        data = await self._post_chat(request.model, payload, api_key, track_errors=True)

        normalized = self._normalize_response(data, request)
        return ChatCompletionResponse.model_validate(normalized)

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
        await self._post_chat(request.model, payload, api_key, track_errors=False)

    def _build_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": request.messages,
        }
        for field in ("temperature", "top_p", "max_tokens", "stream"):
            value = getattr(request, field)
            if value is not None:
                payload[field] = value
        return payload

    def _normalize_response(self, data: Dict[str, Any], request: ChatCompletionRequest) -> Dict[str, Any]:
        if "choices" in data and data.get("choices"):
            choice = data["choices"][0]
        else:
            content = data.get("generated_text", "")
            choice = {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": data.get("finish_reason") or "stop",
            }
            data["choices"] = [choice]

        normalized_response: Dict[str, Any] = {
            "id": data.get("id") or f"chatcmpl-hf-{int(time.time()*1000)}",
            "object": data.get("object") or "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": request.model,
            "choices": data.get("choices", []),
        }

        usage = data.get("usage")
        if usage:
            normalized_response["usage"] = usage

        return normalized_response

    async def _post_chat(
        self,
        model_id: str,
        payload: Dict[str, Any],
        api_key: str,
        track_errors: bool,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}{self._path.format(model_id=model_id)}"
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
