"""Request context middleware for logging correlation."""

from __future__ import annotations

import time
import uuid
from typing import Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.logging import reset_request_id, set_request_id


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to each inbound request."""

    async def dispatch(  # type: ignore[override]
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = set_request_id(request_id)
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            request.state.request_duration_ms = duration_ms
            reset_request_id(token)
        response.headers.setdefault("x-request-id", request_id)
        return response
