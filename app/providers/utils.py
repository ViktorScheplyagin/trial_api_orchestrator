"""Helper utilities for provider adapters."""

from __future__ import annotations

from typing import Any

import httpx


def extract_error_body(response: httpx.Response) -> Any:
    """Return structured error details if available, else a trimmed text body."""

    try:
        return response.json()
    except ValueError:
        text = getattr(response, "text", None)
        if text:
            stripped = text.strip()
            if stripped:
                return stripped
        return None


def build_error_log(
    *,
    error_type: str,
    message: str,
    status_code: int | None = None,
    response_body: Any | None = None,
) -> dict[str, Any]:
    """Assemble a consistent provider error log payload."""

    payload: dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": message,
        }
    }
    if status_code is not None:
        payload["error"]["status_code"] = status_code
    if response_body is not None:
        payload["response"] = response_body
    return payload


__all__ = ["extract_error_body", "build_error_log"]
