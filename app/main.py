"""FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html

from app.api import admin, openai
from app.core.config import load_config
from app.storage.credentials import init_db
from app.router.selector import registry
from app.logging import configure_logging, get_request_id
from app.middleware.request_context import RequestContextMiddleware
from app.telemetry.events import list_recent_events, record_event

configure_logging()

logger = logging.getLogger("orchestrator.app")

app = FastAPI(
    title="Trial API Orchestrator",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
app.include_router(openai.router)
app.include_router(admin.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.add_middleware(RequestContextMiddleware)

templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    load_config()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    states = registry.get_states()
    events = list_recent_events(limit=25)
    return templates.TemplateResponse(
        "providers.html",
        {
            "request": request,
            "states": states,
            "events": events,
        },
    )


@app.get("/api/docs", response_class=HTMLResponse)
def swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="Trial API Orchestrator API",
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled error",
        extra={
            "event": "request_error",
            "path": request.url.path,
            "request_id": get_request_id(),
        },
    )
    record_event(
        "request_error",
        "ERROR",
        message=str(exc),
        meta={
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_server_error",
                "code": "internal_error",
            }
        },
    )
