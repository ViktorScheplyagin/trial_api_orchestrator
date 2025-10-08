"""Provider request/response log storage helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select

from app.logging import get_request_id

from .database import session_scope
from .models import ProviderLog

logger = logging.getLogger("orchestrator.provider_logs")


def _start_of_today() -> datetime:
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)


def _encode(payload: Any) -> str | None:
    if payload is None:
        return None
    try:
        return json.dumps(payload, ensure_ascii=True)
    except (TypeError, ValueError):
        try:
            return json.dumps(str(payload), ensure_ascii=True)
        except (TypeError, ValueError):
            logger.debug("Unable to encode payload for provider log", exc_info=True)
            return None


def _decode(serialized: str | None) -> Any:
    if not serialized:
        return None
    try:
        return json.loads(serialized)
    except json.JSONDecodeError:
        return serialized


def _pretty(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)


def record_provider_log(
    provider_id: str,
    *,
    request_body: Any,
    response_body: Any,
    request_id: str | None = None,
) -> None:
    """Persist a provider interaction, keeping only entries from the current day."""

    entry = ProviderLog(
        provider_id=provider_id,
        request_id=request_id or get_request_id(),
        request_body=_encode(request_body),
        response_body=_encode(response_body),
    )

    cutoff = _start_of_today()

    try:
        with session_scope() as session:
            session.execute(delete(ProviderLog).where(ProviderLog.created_at < cutoff))
            session.add(entry)
    except Exception:
        logger.exception(
            "Failed to persist provider log", extra={"provider_id": provider_id}
        )


def list_provider_logs(provider_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent logs for a provider, limited to entries from the current day."""

    cutoff = _start_of_today()

    with session_scope() as session:
        stmt = (
            select(ProviderLog)
            .where(ProviderLog.provider_id == provider_id)
            .where(ProviderLog.created_at >= cutoff)
            .order_by(ProviderLog.created_at.desc())
            .limit(limit)
        )
        rows = session.scalars(stmt).all()

    logs: list[dict[str, Any]] = []
    for row in rows:
        decoded_request = _decode(row.request_body)
        decoded_response = _decode(row.response_body)
        logs.append(
            {
                "id": row.id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "request_id": row.request_id,
                "request_body": decoded_request,
                "response_body": decoded_response,
                "request_body_pretty": _pretty(decoded_request),
                "response_body_pretty": _pretty(decoded_response),
            }
        )
    return logs


__all__ = ["list_provider_logs", "record_provider_log"]
