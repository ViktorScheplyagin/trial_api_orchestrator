from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from app.storage.database import Base
from app.storage.models import OrchestratorEvent
from app.telemetry import events
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

EXPECTED_EVENT_COUNT = 2


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    """Provide an isolated in-memory database for telemetry tests."""

    engine = create_engine("sqlite:///:memory:", future=True)
    TestingSession = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    Base.metadata.create_all(engine)

    @contextmanager
    def session_scope():
        session = TestingSession()
        try:
            yield session
            session.commit()
        except Exception:  # pragma: no cover - defensive
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(events, "session_scope", session_scope)


def _add_event(ts: datetime, level: str = "INFO", kind: str = "test_event") -> None:
    with events.session_scope() as session:
        session.add(
            OrchestratorEvent(
                ts=ts,
                level=level,
                kind=kind,
            )
        )


def test_record_event_prunes_events_older_than_yesterday():
    threshold = datetime.now(timezone.utc)

    _add_event(threshold - timedelta(days=3))
    _add_event(threshold - timedelta(days=1))

    events.record_event("provider_test", "INFO", message="kept")

    with events.session_scope() as session:
        rows = session.scalars(select(OrchestratorEvent).order_by(OrchestratorEvent.ts)).all()

    assert len(rows) == EXPECTED_EVENT_COUNT
    assert all(row.ts is not None for row in rows)
    first_ts = rows[0].ts
    assert first_ts is not None
    if first_ts.tzinfo is None:
        first_ts = first_ts.replace(tzinfo=timezone.utc)
    assert first_ts >= events._current_retention_cutoff()


def test_list_recent_events_returns_only_retained_records():
    now = datetime.now(timezone.utc)

    _add_event(now - timedelta(days=4))
    _add_event(now - timedelta(days=2))
    recent = now - timedelta(hours=6)
    _add_event(recent)

    # Trigger pruning and retrieval.
    data = events.list_recent_events(limit=10)

    assert len(data) == EXPECTED_EVENT_COUNT
    for item in data:
        timestamp = datetime.fromisoformat(item["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        assert timestamp >= events._current_retention_cutoff()
