"""Admin/status endpoints for provider management."""

from __future__ import annotations

from fastapi import APIRouter, Body, Form, HTTPException
from fastapi.responses import RedirectResponse

from app.core.config import load_config
from app.storage.credentials import delete_api_key, list_credentials, upsert_api_key

router = APIRouter(prefix="/admin")


@router.get("/providers")
def list_providers() -> dict:
    config = load_config()
    stored = {item.provider_id: item for item in list_credentials()}
    data = []
    for provider in sorted(config.providers, key=lambda p: p.priority):
        cred = stored.get(provider.id)
        data.append(
            {
                "id": provider.id,
                "name": provider.name,
                "priority": provider.priority,
                "has_api_key": bool(cred and cred.api_key),
                "last_error": cred.last_error if cred else None,
                "last_error_at": cred.last_error_at.isoformat() if cred and cred.last_error_at else None,
            }
        )
    return {"providers": data}


@router.post("/providers/{provider_id}/credentials")
def set_provider_key(
    provider_id: str,
    api_key_form: str | None = Form(default=None),
    api_key_body: dict | None = Body(default=None),
) -> dict:
    api_key = api_key_form
    if api_key is None and api_key_body:
        api_key = api_key_body.get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing api_key")
    upsert_api_key(provider_id, api_key)
    if api_key_form is not None:
        return RedirectResponse(url="/", status_code=303)
    return {"status": "ok"}


@router.delete("/providers/{provider_id}/credentials")
def delete_provider_key(provider_id: str) -> dict:
    removed = delete_api_key(provider_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Provider credential not found")
    return {"status": "ok"}


@router.post("/providers/{provider_id}/credentials/delete")
def delete_provider_key_post(provider_id: str) -> dict:
    removed = delete_api_key(provider_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Provider credential not found")
    return RedirectResponse(url="/", status_code=303)
