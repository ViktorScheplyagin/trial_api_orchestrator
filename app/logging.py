"""Application logging configuration utilities."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

_LOG_CONFIGURED = False
_REQUEST_ID_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent


class RequestContextFilter(logging.Filter):
    """Inject request-scoped values into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _REQUEST_ID_CTX.get()
        return True


class JsonFormatter(logging.Formatter):
    """Serialize log records as JSON lines."""

    _RESERVED = {
        "args",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "created",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


def set_request_id(request_id: str | None) -> contextvars.Token[str | None]:
    """Bind the current request ID to the logging context."""
    return _REQUEST_ID_CTX.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    """Reset the request ID context."""
    _REQUEST_ID_CTX.reset(token)


def get_request_id() -> str | None:
    """Return the request ID associated with the current context, if any."""
    return _REQUEST_ID_CTX.get()


def _log_file_path() -> pathlib.Path:
    configured = os.getenv("LOG_FILE", "logs/app.jsonl")
    path = pathlib.Path(configured)
    if not path.is_absolute():
        path = BASE_DIR.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging() -> None:
    """Configure global logging for the application."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    console_level_name = os.getenv("LOG_LEVEL", "WARNING").upper()
    console_level = getattr(logging, console_level_name, logging.WARNING)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove default handlers that may exist in certain execution environments.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    context_filter = RequestContextFilter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.addFilter(context_filter)
    console_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s (request_id=%(request_id)s)"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        _log_file_path(),
        maxBytes=10_000_000,
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(context_filter)
    file_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(file_handler)

    # Suppress verbose third-party loggers.
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _LOG_CONFIGURED = True


__all__ = [
    "configure_logging",
    "get_request_id",
    "reset_request_id",
    "set_request_id",
]
