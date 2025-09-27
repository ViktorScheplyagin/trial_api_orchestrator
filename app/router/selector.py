"""Provider selection and management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Type

from app.core.config import AppConfig, ProviderModel, load_config
from app.core.exceptions import ProviderUnavailableError
from app.providers.base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter
from app.providers.cerebras import CerebrasProvider
from app.providers.cohere import CohereProvider
from app.providers.gemini import GeminiProvider
from app.providers.huggingface import HuggingFaceProvider
from app.providers.openrouter import OpenRouterProvider
from app.storage import credentials
from app.storage.models import ProviderCredential
from app.telemetry.events import record_event

logger = logging.getLogger("orchestrator.router")


@dataclass
class ProviderState:
    provider: ProviderModel
    credential: Optional[ProviderCredential]

    @property
    def has_api_key(self) -> bool:
        return bool(self.credential and self.credential.api_key)

    @property
    def is_available(self) -> bool:
        if not self.has_api_key:
            return False
        if self.credential.last_error:
            return False
        return True


class ProviderRegistry:
    """Registry handling provider adapters and configuration."""

    _adapter_map: Dict[str, Type[ProviderAdapter]] = {
        "cerebras": CerebrasProvider,
        "cohere": CohereProvider,
        "gemini": GeminiProvider,
        "huggingface": HuggingFaceProvider,
        "openrouter": OpenRouterProvider,
    }

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or load_config()
        self._instances: Dict[str, ProviderAdapter] = {}

    def providers(self) -> Iterable[ProviderModel]:
        return sorted(self._config.providers, key=lambda p: p.priority)

    def get_adapter(self, provider: ProviderModel) -> ProviderAdapter:
        if provider.id not in self._instances:
            adapter_cls = self._adapter_map.get(provider.id)
            if not adapter_cls:
                raise ProviderUnavailableError(provider.id, message="No adapter configured")
            self._instances[provider.id] = adapter_cls(provider)
        return self._instances[provider.id]

    def get_states(self) -> List[ProviderState]:
        stored = {item.provider_id: item for item in credentials.list_credentials()}
        return [
            ProviderState(provider=provider, credential=stored.get(provider.id))
            for provider in self.providers()
        ]


registry = ProviderRegistry()


async def select_provider(
    request: ChatCompletionRequest,
    provider_id: str | None = None,
) -> ChatCompletionResponse:
    """Select a provider for the request, honoring explicit overrides."""

    available_providers = list(registry.providers())
    if provider_id:
        provider_lookup = {provider.id: provider for provider in available_providers}
        provider_model = provider_lookup.get(provider_id)
        if not provider_model:
            raise ProviderUnavailableError(provider_id, message="Provider not configured")
        providers_to_try = [provider_model]
    else:
        providers_to_try = available_providers

    last_failed_provider_id: str | None = None
    last_failure_message: str | None = None
    final_failure: ProviderUnavailableError | None = None

    for attempt_index, provider_model in enumerate(providers_to_try, start=1):
        if last_failed_provider_id:
            logger.info(
                "Provider switched",
                extra={
                    "event": "provider_switched",
                    "provider_from": last_failed_provider_id,
                    "provider_to": provider_model.id,
                    "model": request.model,
                    "reason": last_failure_message,
                    "attempt": attempt_index,
                },
            )
            record_event(
                "provider_switched",
                "INFO",
                provider_from=last_failed_provider_id,
                provider_to=provider_model.id,
                model=request.model,
                message=last_failure_message,
                meta={"attempt": attempt_index},
            )
            last_failed_provider_id = None
            last_failure_message = None

        adapter = registry.get_adapter(provider_model)
        try:
            response = await adapter.chat_completions(request)
            final_failure = None
            return response
        except ProviderUnavailableError as exc:
            logger.warning(
                "Provider failed",
                extra={
                    "event": "provider_fail",
                    "provider_from": provider_model.id,
                    "model": request.model,
                    "error_message": exc.message,
                    "attempt": attempt_index,
                },
            )
            record_event(
                "provider_fail",
                "WARNING",
                provider_from=provider_model.id,
                model=request.model,
                message=exc.message,
                meta={"attempt": attempt_index},
            )
            last_failed_provider_id = provider_model.id
            last_failure_message = exc.message
            final_failure = exc
            continue

    if final_failure:
        logger.error(
            "All providers exhausted",
            extra={
                "event": "request_error",
                "provider_from": final_failure.provider_id,
                "model": request.model,
                "error_message": final_failure.message,
            },
        )
        record_event(
            "request_error",
            "ERROR",
            provider_from=final_failure.provider_id,
            model=request.model,
            message=final_failure.message,
        )
        raise final_failure
    raise ProviderUnavailableError("unknown", message="No providers configured")
