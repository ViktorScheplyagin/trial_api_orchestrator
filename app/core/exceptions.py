"""Custom exception types."""

from __future__ import annotations


class ProviderUnavailableError(Exception):
    """Raised when a provider cannot satisfy the request."""

    def __init__(self, provider_id: str, message: str = "Provider unavailable") -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.message = message


class AuthenticationRequiredError(ProviderUnavailableError):
    """Raised when a provider requires credentials that are missing."""

    def __init__(self, provider_id: str) -> None:
        super().__init__(provider_id, message="Provider credentials missing")
