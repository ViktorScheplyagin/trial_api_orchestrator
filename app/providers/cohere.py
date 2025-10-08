"""Cohere provider adapter."""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from typing import Any

from urllib.parse import urlparse

import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials
from app.storage.provider_logs import record_provider_log

from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter
from .utils import build_error_log, extract_error_body


class CohereProvider(ProviderAdapter):
    provider_id = "cohere"

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
            max_tokens=1,
        )
        payload = self._build_payload(request)
        await self._post_chat(payload, api_key, track_errors=False)

    def _build_payload(self, request: ChatCompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._build_message(message) for message in request.messages],
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

    def _normalize_response(
        self, data: dict[str, Any], request: ChatCompletionRequest
    ) -> dict[str, Any]:
        message = data.get("message") or {}
        content_items: list[dict[str, Any]] = message.get("content") or []

        ordered_content: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        citations: list[Any] = []
        has_non_text = False

        for item in content_items:
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    ordered_content.append({"type": "text", "text": text})
            elif item_type == "tool_calls":
                for tool in item.get("tool_calls", []):
                    normalized_tool = self._normalize_tool_call(tool)
                    if normalized_tool:
                        tool_calls.append(normalized_tool)
            elif item_type == "citation":
                citations.extend(item.get("citations", []))
            else:
                converted = self._convert_response_content_item(item)
                if converted:
                    ordered_content.append(converted)
                    has_non_text = True

        if not ordered_content and data.get("text"):
            ordered_content.append({"type": "text", "text": data.get("text", "")})

        assistant_message: dict[str, Any] = {
            "role": "assistant",
        }

        if ordered_content:
            if not has_non_text and all(part.get("type") == "text" for part in ordered_content):
                assistant_message["content"] = "".join(
                    part.get("text", "") for part in ordered_content
                )
            else:
                assistant_message["content"] = ordered_content
        else:
            assistant_message["content"] = ""

        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if citations:
            assistant_message["metadata"] = {"cohere": {"citations": citations}}
        if not assistant_message["content"] and not tool_calls:
            assistant_message["content"] = ""

        choice: dict[str, Any] = {
            "index": 0,
            "message": assistant_message,
            "finish_reason": data.get("finish_reason")
            or message.get("finish_reason")
            or data.get("stop_reason")
            or "stop",
        }

        usage = self._normalize_usage(data.get("usage"))

        normalized_response: dict[str, Any] = {
            "id": data.get("id") or f"chatcmpl-cohere-{int(time.time() * 1000)}",
            "object": data.get("object") or "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": data.get("model") or request.model,
            "choices": [choice],
        }
        if usage:
            normalized_response["usage"] = usage

        return normalized_response

    def _build_message(self, message: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(message)
        content = message.get("content")
        normalized_content = self._normalize_request_content(content)
        if normalized_content is not None:
            normalized["content"] = normalized_content
        elif isinstance(content, str):
            normalized["content"] = [{"type": "text", "text": content}]
        elif content is None:
            normalized["content"] = []
        else:
            normalized["content"] = [{"type": "text", "text": str(content)}]
        return normalized

    def _normalize_request_content(self, content: Any) -> list[dict[str, Any]] | None:
        if content is None:
            return None
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            return [{"type": "text", "text": str(content)}]

        parts: list[dict[str, Any]] = []
        for item in content:
            converted = self._convert_request_content_item(item)
            if converted:
                parts.append(converted)
        return parts or None

    def _convert_request_content_item(self, item: Any) -> dict[str, Any] | None:
        if isinstance(item, str):
            return {"type": "text", "text": item}
        if not isinstance(item, dict):
            return None

        item_type = item.get("type")
        if item_type in {"text", "input_text"}:
            text = item.get("text")
            if isinstance(text, str):
                return {"type": "text", "text": text}
        elif item_type in {"image", "image_url", "input_image"}:
            image_part = self._build_image_part(item)
            if image_part:
                return image_part
        else:
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                return {"type": "text", "text": text}
        return None

    def _build_image_part(self, item: dict[str, Any]) -> dict[str, Any] | None:
        if "source" in item and isinstance(item["source"], dict):
            return {"type": "image", "source": item["source"]}

        if "image" in item and isinstance(item["image"], dict):
            image_payload = item["image"]
            b64_data = image_payload.get("b64_json") or image_payload.get("base64")
            url_ref = image_payload.get("url")
            media_type = image_payload.get("media_type") or image_payload.get("mime_type")

            if isinstance(b64_data, str):
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type or "image/png",
                        "data": b64_data,
                    },
                }

            if isinstance(url_ref, str):
                source = self._build_image_source_from_url(url_ref, media_type)
                if source:
                    return {"type": "image", "source": source}

        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            media_type = image_url.get("media_type")
        else:
            url = image_url
            media_type = None

        if isinstance(url, str):
            source = self._build_image_source_from_url(url, media_type)
            if source:
                return {"type": "image", "source": source}

        return None

    def _build_image_source_from_url(
        self, url: str, media_type: str | None
    ) -> dict[str, Any] | None:
        if url.startswith("data:"):
            parsed_media_type, data = self._parse_data_url(url)
            if data:
                return {
                    "type": "base64",
                    "media_type": parsed_media_type or media_type or "image/png",
                    "data": data,
                }
            return None

        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            source: dict[str, Any] = {"type": "url", "url": url}
            if media_type:
                source["media_type"] = media_type
            return source
        return None

    def _parse_data_url(self, url: str) -> tuple[str | None, str | None]:
        try:
            header, data = url.split(",", 1)
        except ValueError:
            return (None, None)

        if not header.startswith("data:"):
            return (None, None)

        meta = header[len("data:") :]
        if ";base64" not in meta:
            return (None, None)

        media_type = meta.split(";")[0] or None
        return (media_type, data)

    def _convert_response_content_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        item_type = item.get("type")
        if item_type == "image":
            source = item.get("source")
            if isinstance(source, dict):
                source_type = source.get("type")
                if source_type == "url" and isinstance(source.get("url"), str):
                    image_payload: dict[str, Any] = {
                        "type": "image_url",
                        "image_url": {"url": source["url"]},
                    }
                    if source.get("media_type"):
                        image_payload["image_url"]["media_type"] = source["media_type"]
                    return image_payload
                if source_type == "base64" and isinstance(source.get("data"), str):
                    media_type = source.get("media_type") or "image/png"
                    data_url = f"data:{media_type};base64,{source['data']}"
                    return {"type": "image_url", "image_url": {"url": data_url}}
        return None

    def _normalize_tool_call(self, tool: dict[str, Any]) -> dict[str, Any] | None:
        function = tool.get("function")
        normalized: dict[str, Any] = {
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

    def _normalize_usage(self, usage: dict[str, Any] | None) -> dict[str, Any] | None:
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

        result: dict[str, Any] = {}
        if prompt_tokens is not None:
            result["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            result["completion_tokens"] = completion_tokens
        if total_tokens is not None:
            result["total_tokens"] = total_tokens

        return result or None

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
