"""OpenAI-compatible API routes."""

from __future__ import annotations

from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.providers.base import ChatCompletionRequest, ChatCompletionResponse
from app.router.selector import select_provider

router = APIRouter(prefix="/v1")


CHAT_COMPLETION_EXAMPLES = {
    "cerebras": {
        "summary": "Cerebras adapter",
        "value": {
            "model": "cerebras/llama3.1-8b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Summarize the provider failover strategy in two sentences.",
                },
            ],
            "temperature": 0.2,
        },
    },
    "cohere": {
        "summary": "Cohere adapter",
        "value": {
            "model": "cohere/command-r-plus",
            "messages": [
                {"role": "system", "content": "You are an orchestrator QA assistant."},
                {
                    "role": "user",
                    "content": "List two Cohere-specific response fields we map into OpenAI format.",
                },
            ],
        },
    },
    "openrouter": {
        "summary": "OpenRouter adapter",
        "value": {
            "model": "openrouter/anthropic/claude-3-haiku",
            "messages": [
                {"role": "system", "content": "You normalize OpenRouter responses."},
                {
                    "role": "user",
                    "content": "Respond with a JSON object showing the keys you return.",
                },
            ],
            "temperature": 0.5,
        },
    },
    "gemini": {
        "summary": "Gemini adapter",
        "value": {
            "model": "gemini-2.5-flash",
            "messages": [
                {
                    "role": "system",
                    "content": "Translate OpenAI chat requests into Gemini contents and back.",
                },
                {
                    "role": "user",
                    "content": "Explain how safety attributes are normalized in one sentence.",
                },
            ],
            "top_p": 0.9,
        },
    },
}


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": CHAT_COMPLETION_EXAMPLES,
                }
            }
        }
    },
)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    provider_id: Annotated[
        Optional[
            Literal["cerebras", "cohere", "gemini", "openrouter"]
        ],
        Header(
            alias="x-provider-id",
            description="Force routing to a specific provider adapter",
        ),
    ] = None,
) -> ChatCompletionResponse:
    try:
        return await select_provider(payload, provider_id)
    except AuthenticationRequiredError as exc:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": f"Provider '{exc.provider_id}' credentials missing",
                    "type": "invalid_request_error",
                    "code": "provider_auth_required",
                }
            },
        )
    except ProviderUnavailableError as exc:
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": f"Provider '{exc.provider_id}' is unavailable: {exc.message}",
                    "type": "rate_limit_exceeded",
                    "code": "provider_unavailable",
                }
            },
        )
