# Adapters — Detailed Guide

This document explains how provider adapters work in the Trial API Orchestrator and gives a step‑by‑step guide to add, test, and maintain new adapters. Read together with:

- Architecture overview: [./architecture.md](./architecture.md)
- Provider API notes: [./providers_api_docs.md](./providers_api_docs.md)
- Supported providers reference: [./supported_providers.md](./supported_providers.md)


## Overview

Adapters translate between the orchestrator’s OpenAI‑style interface and each external provider’s API. They:

- Accept a normalized `ChatCompletionRequest`.
- Build the provider‑specific payload and call the external HTTP API.
- Map provider errors to orchestrator exceptions and record transient state for availability.
- Normalize the provider response into `ChatCompletionResponse`.

In v1, only non‑streaming chat completions are supported.


## Core Interfaces

- `app/providers/base.py`
  - `ChatCompletionRequest`: Pydantic model matching OpenAI chat input (model, messages, temperature, max_tokens, etc.).
  - `ChatCompletionResponse`: Pydantic model for the normalized response.
  - `ProviderAdapter`: abstract base; adapters implement:
    - `chat_completions(self, request)` (async): perform the actual chat call using the stored key.
    - `validate_api_key(self, api_key)` (async): perform a lightweight healthcheck using the provided key without persisting it. Must raise:
      - `AuthenticationRequiredError(provider_id)` on 401/invalid key.
      - `ProviderUnavailableError(provider_id, message=...)` on network/HTTP/ratelimit issues.
      - Return `None` on success. Must not write to storage.

- `app/router/selector.py`
  - `ProviderRegistry` holds the mapping from provider id → adapter class (`_adapter_map`) and selects providers by priority.
  - `select_provider()` tries providers in order and returns the first success; maps failures to OpenAI‑style errors upstream.

- `app/core/exceptions.py`
  - `AuthenticationRequiredError`: raise when credentials are missing/invalid.
  - `ProviderUnavailableError`: raise for rate limits/quotas or generic provider errors.

- `app/storage/credentials.py`
  - Credential CRUD and error markers (`record_error`, `clear_error`), used by adapters and UI availability badges.

- `config/providers.yaml`
  - Declares provider metadata: `id`, `name`, `priority`, `base_url`, `chat_completions_path`, `availability`, `credentials`, `models`.

See ./architecture.md for the full module graph and request flow.


## Adding A New Adapter

High‑level steps:

1) Create an adapter module `app/providers/<provider>.py` implementing `ProviderAdapter`.
2) Register the adapter in `app/router/selector.py` `_adapter_map`.
3) Add the provider to `config/providers.yaml` with URL/path, priority, and model defaults.
4) Restart the app; set API key via the dashboard (`GET /`) or Admin API.
5) Implement `validate_api_key()` mirroring existing adapters (Cerebras/OpenRouter/HF/Gemini/Cohere) with a minimal `"healthcheck"` message and `max_tokens=1`.
6) Smoke test `POST /v1/chat/completions` with a minimal payload.


### Adapter Skeleton

Use `app/providers/cerebras.py` as a reference. A minimal pattern:

```python
# app/providers/openrouter.py
from __future__ import annotations
import time
from typing import Any, Dict
import httpx

from app.core.config import ProviderModel
from app.core.exceptions import AuthenticationRequiredError, ProviderUnavailableError
from app.storage import credentials
from .base import ChatCompletionRequest, ChatCompletionResponse, ProviderAdapter


class OpenRouterProvider(ProviderAdapter):
    provider_id = "openrouter"

    def __init__(self, config: ProviderModel) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._path = config.chat_completions_path

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        api_key = credentials.get_api_key(self.provider_id)
        if not api_key:
            raise AuthenticationRequiredError(self.provider_id)

        url = f"{self._base_url}{self._path}"
        payload = self._build_payload(request)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code == 401:
            credentials.record_error(self.provider_id, "auth")
            raise AuthenticationRequiredError(self.provider_id)
        if resp.status_code in {402, 403, 429}:
            credentials.record_error(self.provider_id, "rate_limit")
            raise ProviderUnavailableError(self.provider_id, message="Provider quota exhausted")
        if resp.is_error:
            credentials.record_error(self.provider_id, f"http_{resp.status_code}")
            raise ProviderUnavailableError(self.provider_id, message="Provider error")

        data: Dict[str, Any] = resp.json()
        credentials.clear_error(self.provider_id)

        # Ensure OpenAI-compatible keys
        data.setdefault("object", "chat.completion")
        data.setdefault("created", int(time.time()))
        data.setdefault("model", request.model)

        return ChatCompletionResponse.model_validate(data)

    async def validate_api_key(self, api_key: str) -> None:
        # Use default model from config and a minimal payload
        model = self._config.models.get("default")
        if not model:
            raise ProviderUnavailableError(self.provider_id, message="Health check model not configured")
        probe = ChatCompletionRequest(
            model=model,
            messages=[{"role": "user", "content": "healthcheck"}],
            max_tokens=1,
        )
        payload = self._build_payload(probe)
        url = f"{self._base_url}{self._path}"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 401:
            raise AuthenticationRequiredError(self.provider_id)
        if resp.status_code in {402, 403, 429} or resp.is_error:
            raise ProviderUnavailableError(self.provider_id, message=f"http_{resp.status_code}")
        return None

    def _build_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
        }
        for field in (
            "temperature",
            "max_tokens",
            "stream",  # v1 orchestrator does not stream, but field is passed through
            "user",
            "presence_penalty",
            "frequency_penalty",
            "top_p",
        ):
            value = getattr(request, field)
            if value is not None:
                payload[field] = value
        return payload
```


### Register In The Registry

Add your adapter class to the map in `app/router/selector.py`:

```python
_adapter_map = {
    "cerebras": CerebrasProvider,
    "openrouter": OpenRouterProvider,
}
```


### Update Provider Configuration

Append a provider entry to `config/providers.yaml`:

```yaml
providers:
  - id: openrouter
    name: OpenRouter
    priority: 20
    base_url: "https://openrouter.ai/api"
    chat_completions_path: "/v1/chat/completions"
    availability:
      cooldown_seconds: 300
    credentials:
      type: sqlite
      key_ref: openrouter_api_key
    models:
      default: "deepseek/deepseek-chat-v3.1:free"
```

Notes:

- `priority` controls selection order (lower is earlier/primary).
- `base_url` and `chat_completions_path` are concatenated; ensure exactly one slash between them.
- `models.default` is used by UI examples and smoke tests.


### Cohere Adapter Notes

- Vision-capable models (for example `command-a-vision-07-2025`) accept OpenAI-style multimodal
  requests. The adapter rewrites `messages[*].content` entries so that `image_url` or
  `input_image` parts (including those using `image: {url: ...}` or `image: {b64_json: ...}`)
  become Cohere `{"type": "image", "source": ...}` objects before calling
  `/v2/chat`.
- Cohere responses that include non-text content are surfaced as structured OpenAI-style content
  lists instead of being flattened to a plain string. Pure text replies remain simple strings so
  existing clients continue to work without changes.


## Error Mapping & Availability Heuristic

Adapters must map provider responses to shared exceptions and record availability state:

- 401 Unauthorized → `AuthenticationRequiredError` and `credentials.record_error(provider_id, "auth")`.
- 402/403/429 Quota/Rate limits → `ProviderUnavailableError` and `credentials.record_error(provider_id, "rate_limit")`.
- Other 4xx/5xx → `ProviderUnavailableError` and `credentials.record_error(provider_id, f"http_{status}")`.
- On success → `credentials.clear_error(provider_id)`.

The dashboard “green/red” availability is computed from presence of an API key and absence of recent `last_error`. See ./architecture.md for more.


## Response Normalization

Ensure the result conforms to `ChatCompletionResponse` and includes at least:

- `object`: default to `"chat.completion"` if missing.
- `created`: Unix epoch seconds; set if missing.
- `model`: echo the request model if provider omits.
- `choices`: list; the provider’s top message mapped into OpenAI‑style `choices[0].message.content` or equivalent.
- `usage`: optional; preserve when provided (prompt_tokens, completion_tokens, total_tokens).

Where a provider’s schema diverges, adapt minimally to preserve OpenAI parity (see examples in ./providers_api_docs.md).


### Cohere-specific Notes

- Cohere’s `/v2/chat` response wraps assistant output in `message.content[]` items where `type` may be `text`, `tool_calls`, or `citation`. Concatenate `text` segments for `message.content` and surface `tool_calls` using OpenAI’s `function` schema (stringify `arguments`). Preserve citations in `message.metadata` to aid downstream UIs.
- Usage metrics appear under `usage.tokens` with `input` and `output` counts. Derive `total_tokens` where missing so the router’s quota heuristics remain consistent.
- Only forward parameters Cohere accepts (`temperature`, `max_tokens`, `top_p`, `stream`). Presence/frequency penalties are disallowed by Cohere; dropping unsupported controls prevents 400 responses while matching orchestrator defaults.


### OpenRouter-specific Notes

- OpenRouter mirrors OpenAI’s schema, so adapters can forward most optional tuning parameters (`temperature`, `top_p`, penalties, `user`) without translation. Keep payloads minimal: omit `None` values to avoid API validation noise.
- Responses generally include `object`, `created`, and `usage`; still default these fields to safe values so downstream consumers have consistent data even when OpenRouter omits them for specific providers.
- HTTP errors (including provider-level rate limits surfaced as 429) should be captured via `credentials.record_error` to ensure the registry deprioritizes OpenRouter while it recovers.


### Gemini-specific Notes

- Gemini’s REST API expects `contents[]` with roles `user`/`model` plus an optional `systemInstruction`. Convert OpenAI-style messages by merging system prompts into `systemInstruction` and mapping assistant messages to the `model` role.
- Generation controls live under `generationConfig` (`temperature`, `maxOutputTokens`, `topP`, `frequencyPenalty`, `presencePenalty`). Only include keys that are provided to keep requests minimal.
- Responses return `candidates` with `content.parts[].text`; concatenate text parts for the assistant message and surface safety/citation metadata inside `message.metadata`. Map `usageMetadata` (`promptTokenCount`, `candidatesTokenCount`, `totalTokenCount`) into the orchestrator’s usage fields.


## Credentials & Security

- Store and fetch keys via `app/storage/credentials.py`.
- Set or update keys through the dashboard (`GET /`) or Admin API.
- Do not log API keys; restrict logs to provider id and coarse error types.
- v1 stores keys in local SQLite (see ./architecture.md). For multi‑instance deployments, migrate to Postgres or a secrets manager.


## Testing & Debugging

Recommended approaches:

- Smoke test (manual):
  - Add/update key in the dashboard.
  - `POST /v1/chat/completions` with a tiny payload; confirm 200 and normalized fields.

- Unit test the adapter:
  - Use `httpx`'s `MockTransport` or a library like `respx` to stub HTTP calls.
  - Validate: payload shape, header auth, error mapping, and normalization.

- Common pitfalls:
  - `base_url` missing/extra slash → malformed URL.
  - Returning raw provider JSON without ensuring `object/created/model`.
  - Forgetting to clear `last_error` on success.
  - Setting `stream=true` while the orchestrator v1 does not send streaming responses.


## Provider‑Specific Notes (Examples)

- OpenRouter: Schema is very close to OpenAI; pass through tuning fields. See details in ./providers_api_docs.md.
- Gemini: Uses `...:generateContent` endpoints and different content shape; a Gemini adapter should translate `messages` to `contents` and map candidates → choices.
- Cohere: `POST /v2/chat`; message schema differs (e.g., tools, documents). Normalize to `choices[].message.content`.
Consult ./supported_providers.md for capability/limit nuances.


## Operational Concerns

- Timeouts: Use sensible HTTP timeouts (e.g., 30s) and consider per‑provider overrides.
- Backoff: The selector currently retries sequentially; for many providers, consider adding per‑provider cooldowns or circuit breakers.
- Metrics: Capture counts of successes, auth errors, and rate limits; in v1 this is manual via logs. Some providers return rate‑limit headers you can surface later.
- Streaming: Deferred for v1. Adapters may accept `stream` in payload, but the API route returns non‑streaming responses.


## Maintenance & Versioning

- Keep adapter behavior aligned with the provider’s latest schema.
- Update ./providers_api_docs.md with any provider‑specific quirks discovered.
- When changing adapter behavior, update `_adapter_map` registrations and the docs here.


## Checklist — Add A Provider

- Implement `app/providers/<provider>.py` with `ProviderAdapter`.
- Register it in `app/router/selector.py` `_adapter_map`.
- Add config in `config/providers.yaml` with `id`, URLs, and priority.
- Restart app; set API key in dashboard.
- POST to `/v1/chat/completions` for a smoke test.


## Appendix: Minimal Smoke Test

Example request (replace model and content appropriately):

```bash
curl -sS -X POST http://localhost:3001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "deepseek/deepseek-chat-v3.1:free",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 16
      }'
```

Expected: HTTP 200 with `object`, `created`, `model`, `choices[0]`, and optional `usage`. If misconfigured: HTTP 401 (missing credentials) or 429 (rate/availability).
