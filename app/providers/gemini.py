"""Gemini provider adapter."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials
from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter


class GeminiProvider(ProviderAdapter):
    provider_id = "gemini"

    def __init__(self, config: ProviderModel) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._path = config.chat_completions_path

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        api_key = credentials.get_api_key(self.provider_id)
        if not api_key:
            raise AuthenticationRequiredError(self.provider_id)

        url = f"{self._base_url}{self._path.format(model=request.model)}"
        payload = self._build_payload(request)
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            credentials.record_error(self.provider_id, "network")
            raise ProviderUnavailableError(self.provider_id, message="Provider request failed") from exc

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

        normalized = self._normalize_response(data, request)
        return ChatCompletionResponse.model_validate(normalized)

    def _build_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        contents: List[Dict[str, Any]] = []
        system_parts: List[Dict[str, str]] = []

        for message in request.messages:
            role = message.get("role")
            text = self._extract_text(message.get("content"))
            if not text:
                continue

            if role == "system":
                system_parts.append({"text": text})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        payload: Dict[str, Any] = {}
        if contents:
            payload["contents"] = contents
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        generation_config: Dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.frequency_penalty is not None:
            generation_config["frequencyPenalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            generation_config["presencePenalty"] = request.presence_penalty

        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: List[str] = []
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

    def _normalize_response(self, data: Dict[str, Any], request: ChatCompletionRequest) -> Dict[str, Any]:
        candidate = self._select_candidate(data.get("candidates", []))
        text = ""
        metadata: Dict[str, Any] = {}
        finish_reason = "stop"

        if candidate:
            parts = candidate.get("content", {}).get("parts", [])
            text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
            text = "".join(text_parts)
            finish_reason = candidate.get("finishReason") or finish_reason
            safety = candidate.get("safetyRatings")
            if safety:
                metadata["safetyRatings"] = safety
            citations = candidate.get("citationMetadata", {}).get("citations") if candidate.get("citationMetadata") else None
            if citations:
                metadata.setdefault("gemini", {})["citations"] = citations

        message: Dict[str, Any] = {"role": "assistant", "content": text}
        if metadata:
            message["metadata"] = metadata

        choice = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason.lower() if isinstance(finish_reason, str) else finish_reason,
        }

        usage = self._normalize_usage(data.get("usageMetadata"))

        normalized_response: Dict[str, Any] = {
            "id": data.get("id") or f"chatcmpl-gemini-{int(time.time()*1000)}",
            "object": data.get("object") or "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": request.model,
            "choices": [choice],
        }
        if usage:
            normalized_response["usage"] = usage

        return normalized_response

    def _select_candidate(self, candidates: Any) -> Dict[str, Any] | None:
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if isinstance(candidate, dict):
                return candidate
        return None

    def _normalize_usage(self, usage: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not usage or not isinstance(usage, dict):
            return None
        prompt = usage.get("promptTokenCount")
        completion = usage.get("candidatesTokenCount")
        total = usage.get("totalTokenCount")

        result: Dict[str, Any] = {}
        if isinstance(prompt, int):
            result["prompt_tokens"] = prompt
        if isinstance(completion, int):
            result["completion_tokens"] = completion
        if isinstance(total, int):
            result["total_tokens"] = total
        elif isinstance(prompt, int) and isinstance(completion, int):
            result["total_tokens"] = prompt + completion

        return result or None
