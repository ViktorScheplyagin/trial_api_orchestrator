"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html

from app.api import admin, openai
from app.core.config import load_config
from app.storage.credentials import init_db
from app.router.selector import registry

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

templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    load_config()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    states = registry.get_states()
    return templates.TemplateResponse(
        "providers.html",
        {
            "request": request,
            "states": states,
        },
    )


@app.get("/api/docs", response_class=HTMLResponse)
def swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="Trial API Orchestrator API",
    )
