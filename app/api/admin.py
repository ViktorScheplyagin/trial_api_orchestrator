"""Admin/status endpoints for provider management."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from fastapi import APIRouter, Body, Form, HTTPException, Response
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.config import load_config
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.router.selector import registry
from app.storage.credentials import (
    clear_error,
    delete_api_key,
    get_api_key,
    list_credentials,
    record_error,
    upsert_api_key,
)
from app.storage.models import ProviderCredential

router = APIRouter(prefix="/admin")


@router.get("/providers")
def list_providers() -> dict:
    config = load_config()
    stored: dict[str, ProviderCredential] = {}
    for item in list_credentials():
        provider_id_value = cast(str, item.provider_id)
        stored[provider_id_value] = item
    data = []
    for provider in sorted(config.providers, key=lambda p: p.priority):
        cred = stored.get(provider.id)
        has_api_key = False
        last_error: str | None = None
        last_error_at_iso: str | None = None
        if cred:
            api_key_value = cast(str | None, cred.api_key)
            has_api_key = bool(api_key_value)
            last_error = cast(str | None, cred.last_error)
            last_error_dt = cast(datetime | None, cred.last_error_at)
            if last_error_dt is not None:
                last_error_at_iso = last_error_dt.isoformat()

        data.append(
            {
                "id": provider.id,
                "name": provider.name,
                "priority": provider.priority,
                "has_api_key": has_api_key,
                "last_error": last_error,
                "last_error_at": last_error_at_iso,
            }
        )
    return {"providers": data}


@router.post("/providers/{provider_id}/credentials")
async def set_provider_key(
    provider_id: str,
    api_key: str | None = Form(default=None),
    api_key_body: dict | None = Body(default=None),
) -> Response:
    api_key_value = api_key
    if api_key_value is None and api_key_body:
        api_key_value = api_key_body.get("api_key")
    if not api_key_value:
        raise HTTPException(status_code=400, detail="Missing api_key")

    config = load_config()
    provider_model = next((p for p in config.providers if p.id == provider_id), None)
    if provider_model is None:
        raise HTTPException(status_code=404, detail="Provider not configured")

    adapter = registry.get_adapter(provider_model)

    try:
        await adapter.validate_api_key(api_key_value)
    except AuthenticationRequiredError as exc:
        record_error(provider_id, "auth")
        raise HTTPException(status_code=400, detail="Invalid API key") from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Provider health check failed: {exc.message}") from exc

    upsert_api_key(provider_id, api_key_value)
    if api_key is not None:
        return RedirectResponse(url="/", status_code=303)
    return JSONResponse({"status": "ok"})


@router.post("/providers/{provider_id}/healthcheck")
async def healthcheck_provider(provider_id: str) -> Response:
    config = load_config()
    provider_model = next((p for p in config.providers if p.id == provider_id), None)
    if provider_model is None:
        raise HTTPException(status_code=404, detail="Provider not configured")

    api_key_value = get_api_key(provider_id)
    if not api_key_value:
        raise HTTPException(status_code=400, detail="Provider API key not configured")

    adapter = registry.get_adapter(provider_model)
    try:
        await adapter.validate_api_key(api_key_value)
    except AuthenticationRequiredError as exc:
        record_error(provider_id, "auth")
        raise HTTPException(status_code=400, detail="Invalid API key") from exc
    except ProviderUnavailableError as exc:
        record_error(provider_id, exc.message or "provider_unavailable")
        raise HTTPException(status_code=503, detail=f"Provider health check failed: {exc.message}") from exc

    clear_error(provider_id)
    return RedirectResponse(url="/", status_code=303)


@router.delete("/providers/{provider_id}/credentials")
def delete_provider_key(provider_id: str) -> dict:
    removed = delete_api_key(provider_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Provider credential not found")
    return {"status": "ok"}


@router.post("/providers/{provider_id}/credentials/delete")
def delete_provider_key_post(provider_id: str) -> RedirectResponse:
    removed = delete_api_key(provider_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Provider credential not found")
    return RedirectResponse(url="/", status_code=303)
