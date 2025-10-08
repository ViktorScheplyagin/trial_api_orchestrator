"""Admin/status endpoints for provider management."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, cast

from fastapi import APIRouter, Body, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

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
from app.storage.provider_logs import list_provider_logs
from app.telemetry.events import list_recent_events, record_event

router = APIRouter(prefix="/admin")
templates = Jinja2Templates(directory="app/templates")


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


@router.get("/events")
def list_events(limit: int = 25) -> dict:
    """Return recent orchestrator events for the admin dashboard."""
    limit_value = max(1, min(limit, 100))
    events = list_recent_events(limit=limit_value)
    return {"events": events}


@router.post("/providers/{provider_id}/credentials")
async def set_provider_key(
    provider_id: str,
    api_key: Annotated[str | None, Form()] = None,
    api_key_body: Annotated[dict | None, Body()] = None,
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
        record_event(
            "provider_credentials_invalid",
            "WARNING",
            provider_from=provider_id,
            message="Credential validation failed: invalid API key",
            meta={"source": "admin_credentials"},
        )
        raise HTTPException(status_code=400, detail="Invalid API key") from exc
    except ProviderUnavailableError as exc:
        record_event(
            "provider_health_fail",
            "WARNING",
            provider_from=provider_id,
            message=exc.message,
            meta={"source": "admin_credentials"},
        )
        raise HTTPException(
            status_code=503, detail=f"Provider health check failed: {exc.message}"
        ) from exc

    upsert_api_key(provider_id, api_key_value)
    record_event(
        "provider_credentials_updated",
        "INFO",
        provider_from=provider_id,
        message="API key saved via admin",
        meta={"source": "admin_credentials"},
    )
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
        record_event(
            "provider_health_fail",
            "WARNING",
            provider_from=provider_id,
            message="Health check failed: invalid API key",
            meta={"source": "admin_healthcheck"},
        )
        raise HTTPException(status_code=400, detail="Invalid API key") from exc
    except ProviderUnavailableError as exc:
        record_error(provider_id, exc.message or "provider_unavailable")
        record_event(
            "provider_health_fail",
            "WARNING",
            provider_from=provider_id,
            message=exc.message,
            meta={"source": "admin_healthcheck"},
        )
        raise HTTPException(
            status_code=503, detail=f"Provider health check failed: {exc.message}"
        ) from exc

    clear_error(provider_id)
    record_event(
        "provider_health_ok",
        "INFO",
        provider_from=provider_id,
        message="Health check succeeded",
        meta={"source": "admin_healthcheck"},
    )
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


@router.get("/{provider_id}/logs", response_class=HTMLResponse)
def provider_logs_page(provider_id: str, request: Request) -> Response:
    config = load_config()
    provider_model = next((p for p in config.providers if p.id == provider_id), None)
    if provider_model is None:
        raise HTTPException(status_code=404, detail="Provider not configured")

    logs = list_provider_logs(provider_id)
    return templates.TemplateResponse(
        "provider_logs.html",
        {
            "request": request,
            "provider": provider_model,
            "logs": logs,
        },
    )
