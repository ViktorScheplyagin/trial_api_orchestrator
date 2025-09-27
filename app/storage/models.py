"""ORM models for provider credential storage and telemetry events."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, UniqueConstraint, func

from .database import Base


class ProviderCredential(Base):
    __tablename__ = "provider_credentials"
    __table_args__ = (
        UniqueConstraint("provider_id", name="uq_provider_credentials_provider_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(100), nullable=False)
    api_key = Column(String(512), nullable=False)
    last_error = Column(String(255))
    last_error_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class OrchestratorEvent(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    level = Column(String(16), nullable=False)
    kind = Column(String(64), nullable=False)
    request_id = Column(String(64))
    provider_from = Column(String(100))
    provider_to = Column(String(100))
    model = Column(String(200))
    error_code = Column(String(128))
    message = Column(String(512))
    meta = Column(Text)

    __table_args__ = (
        Index("ix_events_ts", "ts"),
        Index("ix_events_kind_ts", "kind", "ts"),
    )
