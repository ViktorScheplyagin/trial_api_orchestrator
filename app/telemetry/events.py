"""Event recording helpers for provider telemetry."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select

from app.logging import get_request_id
from app.storage.database import session_scope
from app.storage.models import OrchestratorEvent

logger = logging.getLogger("orchestrator.events")

_EVENTS_ENABLED = os.getenv("EVENTS_ENABLED", "true").lower() in {"1", "true", "yes"}


_RETENTION_DAYS = 2  # keep today + yesterday


def _current_retention_cutoff() -> datetime:
    """Return the UTC timestamp cutoff for events to retain."""
    now_utc = datetime.now(timezone.utc)
    start_of_today = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
    return start_of_today - timedelta(days=_RETENTION_DAYS - 1)


def _prune_old_events(session) -> None:
    """Remove events older than the retention window."""
    cutoff = _current_retention_cutoff()
    session.execute(
        delete(OrchestratorEvent).where(OrchestratorEvent.ts < cutoff)
    )


def record_event(
    kind: str,
    level: str,
    *,
    message: str | None = None,
    request_id: str | None = None,
    meta: Dict[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Persist a high-value event for later inspection."""
    if not _EVENTS_ENABLED:
        return

    event = OrchestratorEvent(
        ts=datetime.now(timezone.utc),
        level=level.upper(),
        kind=kind,
        request_id=request_id or get_request_id(),
        message=message,
        provider_from=fields.get("provider_from"),
        provider_to=fields.get("provider_to"),
        model=fields.get("model"),
        error_code=fields.get("error_code"),
        meta=json.dumps(meta, ensure_ascii=True) if meta else None,
    )

    try:
        with session_scope() as session:
            session.add(event)
            _prune_old_events(session)
    except Exception:
        logger.exception(
            "Failed to record event", extra={"event": "event_persist_error", "kind": kind}
        )


def list_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent events ordered newest first."""
    if not _EVENTS_ENABLED:
        return []

    cutoff = _current_retention_cutoff()

    with session_scope() as session:
        _prune_old_events(session)

        stmt = (
            select(OrchestratorEvent)
            .where(OrchestratorEvent.ts >= cutoff)
            .order_by(OrchestratorEvent.ts.desc())
            .limit(limit)
        )
        rows = session.scalars(stmt).all()

    events: List[Dict[str, Any]] = []
    for row in rows:
        meta_value: Optional[Dict[str, Any] | str | None]
        if row.meta:
            try:
                meta_value = json.loads(row.meta)
            except json.JSONDecodeError:
                meta_value = row.meta
        else:
            meta_value = None

        events.append(
            {
                "id": row.id,
                "timestamp": row.ts.isoformat() if row.ts else None,
                "level": row.level,
                "kind": row.kind,
                "request_id": row.request_id,
                "provider_from": row.provider_from,
                "provider_to": row.provider_to,
                "model": row.model,
                "error_code": row.error_code,
                "message": row.message,
                "meta": meta_value,
            }
        )
    return events


__all__ = ["list_recent_events", "record_event"]
