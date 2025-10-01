"""Storage helpers for provider credentials."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

from sqlalchemy import select, update

from .database import Base, engine, session_scope
from .models import ProviderCredential


def init_db() -> None:
    """Create tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def upsert_api_key(provider_id: str, api_key: str) -> None:
    """Insert or update the API key for a provider."""
    with session_scope() as session:
        existing = session.scalar(
            select(ProviderCredential).where(ProviderCredential.provider_id == provider_id)
        )
        if existing:
            session.execute(
                update(ProviderCredential)
                .where(ProviderCredential.id == existing.id)
                .values(api_key=api_key, last_error=None, last_error_at=None)
            )
        else:
            credential = ProviderCredential(provider_id=provider_id, api_key=api_key)
            session.add(credential)


def get_api_key(provider_id: str) -> str | None:
    """Return the stored API key for the given provider, if any."""
    with session_scope() as session:
        result = session.scalar(
            select(ProviderCredential.api_key).where(ProviderCredential.provider_id == provider_id)
        )
        return cast(str | None, result)


def record_error(provider_id: str, error_code: str) -> None:
    """Record the latest error for a provider."""
    with session_scope() as session:
        existing = session.scalar(
            select(ProviderCredential).where(ProviderCredential.provider_id == provider_id)
        )
        if existing:
            session.execute(
                update(ProviderCredential)
                .where(ProviderCredential.id == existing.id)
                .values(last_error=error_code, last_error_at=datetime.now(timezone.utc))
            )
        else:
            credential = ProviderCredential(
                provider_id=provider_id,
                api_key="",
                last_error=error_code,
                last_error_at=datetime.now(timezone.utc),
            )
            session.add(credential)


def clear_error(provider_id: str) -> None:
    """Clear the last error for a provider."""
    with session_scope() as session:
        existing = session.scalar(
            select(ProviderCredential).where(ProviderCredential.provider_id == provider_id)
        )
        if existing and existing.last_error:
            session.execute(
                update(ProviderCredential)
                .where(ProviderCredential.id == existing.id)
                .values(last_error=None, last_error_at=None)
            )


def list_credentials() -> list[ProviderCredential]:
    """Return all provider credentials with their error state."""
    with session_scope() as session:
        rows = session.scalars(select(ProviderCredential)).all()
        return cast(list[ProviderCredential], rows)


def delete_api_key(provider_id: str) -> bool:
    """Delete a stored API key if present."""
    with session_scope() as session:
        credential = session.scalar(
            select(ProviderCredential).where(ProviderCredential.provider_id == provider_id)
        )
        if not credential:
            return False
        session.delete(credential)
        return True
