"""Cohere provider adapter."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials
from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter


class CohereProvider(ProviderAdapter):
    provider_id = "cohere"

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

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:  # Network or transport issues
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
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stream is not None:
            payload["stream"] = request.stream

        return payload

    def _normalize_response(self, data: Dict[str, Any], request: ChatCompletionRequest) -> Dict[str, Any]:
        message = data.get("message") or {}
        content_items: List[Dict[str, Any]] = message.get("content") or []

        text_segments: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        citations: List[Any] = []

        for item in content_items:
            item_type = item.get("type")
            if item_type == "text":
                text_segments.append(item.get("text", ""))
            elif item_type == "tool_calls":
                for tool in item.get("tool_calls", []):
                    normalized_tool = self._normalize_tool_call(tool)
                    if normalized_tool:
                        tool_calls.append(normalized_tool)
            elif item_type == "citation":
                citations.extend(item.get("citations", []))

        if not text_segments and data.get("text"):
            text_segments.append(data.get("text", ""))

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_segments),
        }

        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if citations:
            assistant_message["metadata"] = {"cohere": {"citations": citations}}
        if not assistant_message["content"] and not tool_calls:
            assistant_message["content"] = ""

        choice: Dict[str, Any] = {
            "index": 0,
            "message": assistant_message,
            "finish_reason": data.get("finish_reason")
            or message.get("finish_reason")
            or data.get("stop_reason")
            or "stop",
        }

        usage = self._normalize_usage(data.get("usage"))

        normalized_response: Dict[str, Any] = {
            "id": data.get("id") or f"chatcmpl-cohere-{int(time.time()*1000)}",
            "object": data.get("object") or "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": data.get("model") or request.model,
            "choices": [choice],
        }
        if usage:
            normalized_response["usage"] = usage

        return normalized_response

    def _normalize_tool_call(self, tool: Dict[str, Any]) -> Dict[str, Any] | None:
        function = tool.get("function")
        normalized: Dict[str, Any] = {
            "id": tool.get("id") or "",
            "type": tool.get("type") or "function",
        }
        if function:
            arguments = function.get("arguments")
            if isinstance(arguments, (dict, list)):
                arguments_str = json.dumps(arguments)
            else:
                arguments_str = arguments or "{}"
            normalized["function"] = {
                "name": function.get("name", ""),
                "arguments": arguments_str,
            }
        return normalized

    def _normalize_usage(self, usage: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not usage:
            return None

        tokens = usage.get("tokens") if isinstance(usage, dict) else None
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        if isinstance(tokens, dict):
            prompt_tokens = tokens.get("input") or tokens.get("prompt")
            completion_tokens = tokens.get("output") or tokens.get("generation")
            total_tokens = tokens.get("total")
        else:
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None

        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        result: Dict[str, Any] = {}
        if prompt_tokens is not None:
            result["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            result["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            result["total_tokens"] = total_tokens

        return result or None
