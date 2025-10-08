"""Cerebras provider adapter."""

from __future__ import annotations

import time
from http import HTTPStatus
from typing import Any

import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials
from app.storage.provider_logs import record_provider_log

from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter
from .utils import build_error_log, extract_error_body


class CerebrasProvider(ProviderAdapter):
    provider_id = "cerebras"

    def __init__(self, config: ProviderModel) -> None:
        super().__init__(config)
        self._base_url = config.base_url.rstrip("/")
        self._path = config.chat_completions_path

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        api_key = credentials.get_api_key(self.provider_id)
        if not api_key:
            raise AuthenticationRequiredError(self.provider_id)

        payload = self._build_payload(request)
        data = await self._post_chat(payload, api_key, track_errors=True)

        # Ensure OpenAI-compatible keys exist
        data.setdefault("object", "chat.completion")
        data.setdefault("created", int(time.time()))
        data.setdefault("model", request.model)

        return ChatCompletionResponse.model_validate(data)

    async def validate_api_key(self, api_key: str) -> None:
        model = self._config.models.get("default")
        if not model:
            raise ProviderUnavailableError(
                self.provider_id, message="Health check model not configured"
            )

        request = ChatCompletionRequest(
            model=model,
            messages=[{"role": "user", "content": "healthcheck"}],
            max_tokens=16,
        )
        payload = self._build_payload(request)
        await self._post_chat(payload, api_key, track_errors=False)

    def _build_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
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

    async def _post_chat(
        self, payload: dict[str, Any], api_key: str, track_errors: bool
    ) -> dict[str, Any]:
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
                record_provider_log(
                    self.provider_id,
                    request_body=payload,
                    response_body=build_error_log(
                        error_type="network",
                        message=str(exc),
                    ),
                )
            raise ProviderUnavailableError(
                self.provider_id, message="Provider request failed"
            ) from exc

        if response.status_code == HTTPStatus.UNAUTHORIZED:
            if track_errors:
                credentials.record_error(self.provider_id, "auth")
                record_provider_log(
                    self.provider_id,
                    request_body=payload,
                    response_body=build_error_log(
                        error_type="unauthorized",
                        message="HTTP 401",
                        status_code=response.status_code,
                        response_body=extract_error_body(response),
                    ),
                )
            raise AuthenticationRequiredError(self.provider_id)

        if response.status_code in {
            HTTPStatus.PAYMENT_REQUIRED,
            HTTPStatus.FORBIDDEN,
            HTTPStatus.TOO_MANY_REQUESTS,
        }:
            if track_errors:
                credentials.record_error(self.provider_id, "rate_limit")
                record_provider_log(
                    self.provider_id,
                    request_body=payload,
                    response_body=build_error_log(
                        error_type="rate_limit",
                        message=f"HTTP {response.status_code}",
                        status_code=response.status_code,
                        response_body=extract_error_body(response),
                    ),
                )
            raise ProviderUnavailableError(self.provider_id, message="Provider quota exhausted")

        if response.is_error:
            if track_errors:
                credentials.record_error(self.provider_id, f"http_{response.status_code}")
                record_provider_log(
                    self.provider_id,
                    request_body=payload,
                    response_body=build_error_log(
                        error_type="http_error",
                        message=f"HTTP {response.status_code}",
                        status_code=response.status_code,
                        response_body=extract_error_body(response),
                    ),
                )
            raise ProviderUnavailableError(self.provider_id, message="Provider error")

        if track_errors:
            credentials.clear_error(self.provider_id)

        data = response.json()
        if not isinstance(data, dict):
            if track_errors:
                record_provider_log(
                    self.provider_id,
                    request_body=payload,
                    response_body=build_error_log(
                        error_type="unexpected_response",
                        message="Response was not a JSON object",
                        response_body=extract_error_body(response),
                    ),
                )
            raise ProviderUnavailableError(self.provider_id, message="Unexpected response format")
        if track_errors:
            record_provider_log(
                self.provider_id,
                request_body=payload,
                response_body=data,
            )
        return data
