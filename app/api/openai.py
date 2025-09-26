"""OpenAI-compatible API routes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.providers.base import ChatCompletionRequest, ChatCompletionResponse
from app.router.selector import select_provider

router = APIRouter(prefix="/v1")


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(payload: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        return await select_provider(payload)
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
