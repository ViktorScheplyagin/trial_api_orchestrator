"""Provider selection and management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Type

from app.core.config import AppConfig, ProviderModel, load_config
from app.core.exceptions import ProviderUnavailableError
from app.providers.base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter
from app.providers.cerebras import CerebrasProvider
from app.providers.cohere import CohereProvider
from app.storage import credentials
from app.storage.models import ProviderCredential


@dataclass
class ProviderState:
    provider: ProviderModel
    credential: Optional[ProviderCredential]

    @property
    def is_available(self) -> bool:
        if not self.credential or not self.credential.api_key:
            return False
        if self.credential.last_error:
            return False
        return True


class ProviderRegistry:
    """Registry handling provider adapters and configuration."""

    _adapter_map: Dict[str, Type[ProviderAdapter]] = {
        "cerebras": CerebrasProvider,
        "cohere": CohereProvider,
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


async def select_provider(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Select the first available provider to satisfy the request."""
    last_error: Dict[str, ProviderUnavailableError] = {}

    for provider_model in registry.providers():
        adapter = registry.get_adapter(provider_model)
        try:
            return await adapter.chat_completions(request)
        except ProviderUnavailableError as exc:
            last_error[provider_model.id] = exc
            continue

    if last_error:
        # Re-raise the last encountered error for context.
        raise last_error[list(last_error.keys())[-1]]
    raise ProviderUnavailableError("unknown", message="No providers configured")
