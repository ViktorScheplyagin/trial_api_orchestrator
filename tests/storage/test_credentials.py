from __future__ import annotations

from contextlib import contextmanager

import pytest
from app.storage import credentials
from app.storage.database import Base
from app.storage.models import ProviderCredential
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
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

    monkeypatch.setattr(credentials, "session_scope", session_scope)

    yield

    engine.dispose()


def _fetch_credentials(provider_id: str) -> ProviderCredential | None:
    with credentials.session_scope() as session:  # type: ignore[attr-defined]
        stmt = select(ProviderCredential).where(ProviderCredential.provider_id == provider_id)
        return session.scalar(stmt)


def test_upsert_and_get_api_key_round_trip():
    assert credentials.get_api_key("demo") is None

    credentials.upsert_api_key("demo", "secret")
    assert credentials.get_api_key("demo") == "secret"

    credentials.upsert_api_key("demo", "updated")
    assert credentials.get_api_key("demo") == "updated"


def test_record_and_clear_error_state():
    credentials.upsert_api_key("demo", "secret")
    credentials.record_error("demo", "rate_limit")

    stored = _fetch_credentials("demo")
    assert stored is not None
    assert stored.last_error == "rate_limit"
    assert stored.last_error_at is not None

    credentials.clear_error("demo")
    stored = _fetch_credentials("demo")
    assert stored is not None
    assert stored.last_error is None
    assert stored.last_error_at is None


def test_delete_api_key_removes_record():
    credentials.upsert_api_key("demo", "secret")
    assert credentials.delete_api_key("demo") is True
    assert credentials.get_api_key("demo") is None
    assert credentials.delete_api_key("demo") is False
