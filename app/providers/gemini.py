"""Gemini provider adapter."""

from __future__ import annotations

import time
from http import HTTPStatus
from typing import Any

import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials

from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter

MAX_ERROR_DETAIL_LENGTH = 300


class GeminiProvider(ProviderAdapter):
    provider_id = "gemini"

    def __init__(self, config: ProviderModel) -> None:
        super().__init__(config)
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
            raise ProviderUnavailableError(
                self.provider_id, message="Health check model not configured"
            )

        request = ChatCompletionRequest(
            model=model,
            messages=[{"role": "user", "content": "healthcheck"}],
            max_tokens=16,
        )
        payload = self._build_payload(request)
        await self._post_chat(request.model, payload, api_key, track_errors=False)

    def _build_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, str]] = []

        for message in request.messages:
            text = self._extract_text(message.get("content"))
            if not text:
                continue

            role = message.get("role")
            if role == "system":
                system_parts.append({"text": text})
                continue

            mapped_role = "model" if role == "assistant" else "user"
            contents.append({"role": mapped_role, "parts": [{"text": text}]})

        payload: dict[str, Any] = {}
        if contents:
            payload["contents"] = contents
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        optional_generation_fields = {
            "temperature": "temperature",
            "max_tokens": "maxOutputTokens",
            "top_p": "topP",
            "frequency_penalty": "frequencyPenalty",
            "presence_penalty": "presencePenalty",
        }

        generation_config = {
            target: getattr(request, attr)
            for attr, target in optional_generation_fields.items()
            if getattr(request, attr) is not None
        }

        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        pieces.append(item["text"])
                    elif "content" in item and isinstance(item["content"], str):
                        pieces.append(item["content"])
                elif isinstance(item, str):
                    pieces.append(item)
            return "".join(pieces)
        if isinstance(content, dict):
            value = content.get("text") or content.get("content")
            return value if isinstance(value, str) else ""
        return str(content)

    def _normalize_response(
        self, data: dict[str, Any], request: ChatCompletionRequest
    ) -> dict[str, Any]:
        candidate = self._select_candidate(data.get("candidates", []))
        text = ""
        metadata: dict[str, Any] = {}
        finish_reason = "stop"

        if candidate:
            parts = candidate.get("content", {}).get("parts", [])
            text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
            text = "".join(text_parts)
            finish_reason = candidate.get("finishReason") or finish_reason
            safety = candidate.get("safetyRatings")
            if safety:
                metadata["safetyRatings"] = safety
            citations = (
                candidate.get("citationMetadata", {}).get("citations")
                if candidate.get("citationMetadata")
                else None
            )
            if citations:
                metadata.setdefault("gemini", {})["citations"] = citations

        message: dict[str, Any] = {"role": "assistant", "content": text}
        if metadata:
            message["metadata"] = metadata

        choice = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason.lower()
            if isinstance(finish_reason, str)
            else finish_reason,
        }

        usage = self._normalize_usage(data.get("usageMetadata"))

        normalized_response: dict[str, Any] = {
            "id": data.get("id") or f"chatcmpl-gemini-{int(time.time() * 1000)}",
            "object": data.get("object") or "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": request.model,
            "choices": [choice],
        }
        if usage:
            normalized_response["usage"] = usage

        return normalized_response

    def _select_candidate(self, candidates: Any) -> dict[str, Any] | None:
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if isinstance(candidate, dict):
                return candidate
        return None

    def _normalize_usage(self, usage: dict[str, Any] | None) -> dict[str, Any] | None:
        if not usage or not isinstance(usage, dict):
            return None
        prompt = usage.get("promptTokenCount")
        completion = usage.get("candidatesTokenCount")
        total = usage.get("totalTokenCount")

        result: dict[str, Any] = {}
        if isinstance(prompt, int):
            result["prompt_tokens"] = prompt
        if isinstance(completion, int):
            result["completion_tokens"] = completion
        if isinstance(total, int):
            result["total_tokens"] = total
        elif isinstance(prompt, int) and isinstance(completion, int):
            result["total_tokens"] = prompt + completion

        return result or None

    async def _post_chat(
        self,
        model: str,
        payload: dict[str, Any],
        api_key: str,
        track_errors: bool,
    ) -> dict[str, Any]:
        model_slug = model.removeprefix("models/")
        url = f"{self._base_url}{self._path.format(model=model_slug)}"
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            if track_errors:
                credentials.record_error(self.provider_id, "network")
            raise ProviderUnavailableError(
                self.provider_id, message="Provider request failed"
            ) from exc

        if response.status_code == HTTPStatus.UNAUTHORIZED:
            if track_errors:
                credentials.record_error(self.provider_id, "auth")
            raise AuthenticationRequiredError(self.provider_id)
        if response.status_code in {
            HTTPStatus.PAYMENT_REQUIRED,
            HTTPStatus.FORBIDDEN,
            HTTPStatus.TOO_MANY_REQUESTS,
        }:
            if track_errors:
                credentials.record_error(self.provider_id, "rate_limit")
            detail = self._extract_error_detail(response)
            message = "Provider quota exhausted"
            if detail:
                message = f"{message}: {detail}"
            raise ProviderUnavailableError(self.provider_id, message=message)
        if response.is_error:
            if track_errors:
                credentials.record_error(self.provider_id, f"http_{response.status_code}")
            detail = self._extract_error_detail(response)
            message = "Provider error"
            if detail:
                message = f"{message}: {detail}"
            raise ProviderUnavailableError(self.provider_id, message=message)

        if track_errors:
            credentials.clear_error(self.provider_id)

        data = response.json()
        if not isinstance(data, dict):
            raise ProviderUnavailableError(self.provider_id, message="Unexpected response format")
        return data

    def _extract_error_detail(self, response: httpx.Response) -> str | None:
        """Return a trimmed provider error detail, if available."""

        detail: str | None = None
        try:
            data = response.json()
        except ValueError:
            text_summary = (response.text or "").strip()
            if text_summary:
                detail = text_summary
        else:
            if isinstance(data, dict):
                error_obj = data.get("error")
                if isinstance(error_obj, dict):
                    status = error_obj.get("status")
                    message = error_obj.get("message")
                    parts = [
                        part.strip()
                        for part in (status, message)
                        if isinstance(part, str) and part.strip()
                    ]
                    if parts:
                        detail = " - ".join(parts)
                    elif error_obj:
                        detail = str(error_obj)
                elif isinstance(data.get("message"), str):
                    detail = data["message"]
                elif data:
                    detail = str(data)
            elif data:
                detail = str(data)

        if detail:
            compact = " ".join(detail.split())
            if len(compact) > MAX_ERROR_DETAIL_LENGTH:
                compact = f"{compact[: MAX_ERROR_DETAIL_LENGTH - 3]}..."
            return compact
        return None
