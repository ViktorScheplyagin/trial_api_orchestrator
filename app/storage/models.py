"""ORM models for provider credential storage."""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint, func

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
