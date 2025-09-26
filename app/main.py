"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import admin, openai
from app.core.config import load_config
from app.storage.credentials import init_db
from app.router.selector import registry

app = FastAPI(title="Trial API Orchestrator", version="0.1.0")
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
