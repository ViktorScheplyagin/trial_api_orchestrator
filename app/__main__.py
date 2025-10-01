"""CLI entry point for running the FastAPI app with the default dev settings."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the FastAPI application with default host/port overrides."""
    host = os.getenv("UVICORN_HOST", os.getenv("HOST", "127.0.0.1"))
    port = int(os.getenv("UVICORN_PORT", os.getenv("PORT", "3001")))
    reload_enabled = os.getenv("UVICORN_RELOAD", os.getenv("RELOAD", "true")).lower() == "true"

    uvicorn.run("app.main:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    main()
