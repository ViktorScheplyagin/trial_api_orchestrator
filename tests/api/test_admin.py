import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import admin
from app.core.config import AppConfig, ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage.models import ProviderCredential


def _config() -> AppConfig:
    return AppConfig(
        providers=[
            ProviderModel(
                id="demo",
                name="Demo",
                priority=10,
                base_url="https://demo.example",
                chat_completions_path="/chat",
            )
        ]
    )


@pytest.mark.asyncio
async def test_set_provider_key_success(monkeypatch):
    saved_keys: dict[str, str] = {}

    class DummyAdapter:
        async def validate_api_key(self, api_key: str) -> None:
            saved_keys["validated"] = api_key

    adapter = DummyAdapter()

    monkeypatch.setattr(admin, "load_config", _config)
    monkeypatch.setattr(admin, "registry", SimpleNamespace(get_adapter=lambda _: adapter))
    monkeypatch.setattr(admin, "upsert_api_key", lambda provider_id, api_key: saved_keys.update({provider_id: api_key}))
    monkeypatch.setattr(admin, "record_event", lambda *args, **kwargs: None)

    response = await admin.set_provider_key("demo", api_key=None, api_key_body={"api_key": "secret"})

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload == {"status": "ok"}
    assert saved_keys["demo"] == "secret"
    assert saved_keys["validated"] == "secret"


@pytest.mark.asyncio
async def test_set_provider_key_invalid(monkeypatch):
    monkeypatch.setattr(admin, "load_config", _config)
    monkeypatch.setattr(admin, "record_event", lambda *args, **kwargs: None)

    class FailingAdapter:
        async def validate_api_key(self, api_key: str) -> None:
            raise AuthenticationRequiredError("demo")

    adapter = FailingAdapter()
    recorded_errors: list[tuple[str, str]] = []

    monkeypatch.setattr(admin, "registry", SimpleNamespace(get_adapter=lambda _: adapter))
    monkeypatch.setattr(admin, "record_error", lambda provider_id, error_code: recorded_errors.append((provider_id, error_code)))

    with pytest.raises(HTTPException) as excinfo:
        await admin.set_provider_key("demo", api_key_body={"api_key": "secret"}, api_key=None)

    assert excinfo.value.status_code == 400
    assert recorded_errors == [("demo", "auth")]


def test_list_providers_includes_credentials(monkeypatch):
    monkeypatch.setattr(admin, "load_config", _config)

    record = ProviderCredential(provider_id="demo", api_key="secret", last_error="rate_limit")
    monkeypatch.setattr(admin, "list_credentials", lambda: [record])

    result = admin.list_providers()

    assert result["providers"][0]["id"] == "demo"
    assert result["providers"][0]["has_api_key"] is True
    assert result["providers"][0]["last_error"] == "rate_limit"
