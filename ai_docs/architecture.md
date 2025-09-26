# Trial API Orchestrator — Architecture

This document describes the overall architecture, key modules, request lifecycles, and the file structure of the app.

## Overview

The orchestrator exposes an OpenAI‑compatible endpoint for chat completions and routes requests to the first available provider based on configured priority. In v1, only non‑streaming chat completions are supported and the Cerebras provider is implemented. Provider API keys are stored locally (SQLite) and can be managed through a minimal dashboard.

Key goals:
- Normalize requests/responses to an OpenAI‑like schema
- Simple provider selection with basic failover
- Lightweight local credential storage and availability badge

## High‑Level Flow

```
Client (OpenAI-style)                              Admin (Browser)
      |                                                   |
      |  POST /v1/chat/completions                        |  GET /
      v                                                   v
+----------------------+                        +----------------------+
| FastAPI (app/main.py)|                        | Jinja2 Template UI   |
|  - Routers mounted   |                        | providers.html       |
|  - Startup hooks     |                        +----------+-----------+
+----------+-----------+                                   |
           |                                               | POST /admin/providers/{id}/credentials
           v                                               v
  API Controller (app/api/openai.py)                 Admin API (app/api/admin.py)
           |                                               |
           v                                               v
  Provider Selector (app/router/selector.py)       Credentials Manager (app/storage/credentials.py)
           |                                               |
           v                                               v
    Provider Registry                              SQLite (data/orchestrator.db)
   (adapters, priority)                                     ^
           |                                                |
           v                                                |
   Cerebras Adapter (app/providers/cerebras.py)  <----------+
           |
           v
   External Provider HTTP API
           |
           v
  OpenAI-like Response -> Client
```

## Modules

- app/main.py
  - FastAPI app entry point. Mounts static files, templates, and includes API routers.
  - Startup: initializes SQLite tables and loads provider config.
  - Root route (`GET /`) renders the Providers dashboard with availability badges.

- app/api/openai.py
  - Exposes `POST /v1/chat/completions` with a Pydantic request/response schema compatible with OpenAI.
  - Delegates to the provider selector. Maps internal exceptions to OpenAI‑style JSON errors (401/429).

- app/api/admin.py
  - Minimal admin endpoints to list providers and manage credentials.
  - Supports form submissions (from the dashboard) and JSON body for API use.

- app/router/selector.py
  - ProviderRegistry holds configuration, lazily instantiates adapters, and exposes provider states.
  - `select_provider()` iterates providers by priority and returns the first successful response.
  - Availability heuristic (for UI) is based on key presence and lack of recent error marker.

- app/providers/base.py
  - Defines `ChatCompletionRequest` and `ChatCompletionResponse` Pydantic models.
  - Declares `ProviderAdapter` interface with `chat_completions()`.

- app/providers/cerebras.py
  - Implements `ProviderAdapter` for Cerebras. Builds an OpenAI‑like payload and calls the provider via `httpx`.
  - Handles 401/402/403/429 and generic HTTP errors, recording error state in SQLite and raising typed exceptions.
  - Normalizes response shape (ensures `object`, `created`, `model`).

- app/storage/database.py
  - SQLAlchemy engine/session setup for a local SQLite DB in `data/orchestrator.db`.
  - Provides `session_scope()` context manager and declarative base.

- app/storage/models.py
  - ORM model `ProviderCredential` storing: `provider_id`, `api_key`, `last_error`, and timestamps.

- app/storage/credentials.py
  - CRUD helpers for provider API keys and error markers.
  - `init_db()` bootstraps tables at app startup.

- app/core/config.py
  - Loads provider configuration from YAML (`config/providers.yaml`) into `AppConfig` and `ProviderModel`.
  - Cached via `lru_cache` for efficient reuse across requests.

- app/core/exceptions.py
  - Custom exception types: `ProviderUnavailableError` and `AuthenticationRequiredError` used across adapters and routing.

- app/templates/providers.html and app/static/
  - Simple dashboard to view availability and set/remove API keys.

## Request Lifecycle

Chat completion (non‑streaming):
1) Client sends OpenAI‑style JSON to `POST /v1/chat/completions`.
2) `app/api/openai.py` deserializes into `ChatCompletionRequest` and calls `select_provider()`.
3) `ProviderRegistry` enumerates providers by priority from `config/providers.yaml` and instantiates the adapter if needed.
4) Adapter (e.g., Cerebras) fetches API key from SQLite, calls the external API via `httpx`, updates error state on failures, and normalizes the response.
5) Response is returned as `ChatCompletionResponse` (OpenAI‑like) to the client.

Admin credential management:
1) Browser loads `GET /` to render providers and current availability.
2) Submitting the form calls `POST /admin/providers/{provider_id}/credentials` which upserts the API key.
3) Errors (auth or throttling) set `last_error`, toggling the red badge until cleared by a successful call or key update.

## Configuration

- config/providers.yaml
  - Declares providers with: `id`, `name`, `priority`, `base_url`, and `chat_completions_path`.
  - Example (current default): a single `cerebras` provider with high priority.

Environment and persistence:
- SQLite database file at `data/orchestrator.db` (created automatically).
- No external secrets manager; keys are stored locally for v1.

## Error Handling & Availability

- Authentication (401) raises `AuthenticationRequiredError` and records `last_error="auth"`.
- Rate limits/quotas (402/403/429) raise `ProviderUnavailableError` and record `last_error="rate_limit"`.
- Other HTTP errors mark `last_error` with `http_<status>`.
- UI green badge: API key present and `last_error` is empty. Red otherwise.

## File Structure

```
app/
  api/
    __init__.py
    openai.py                  # /v1/chat/completions route
    admin.py                   # /admin provider management routes
  core/
    __init__.py
    config.py                  # YAML config loader (providers)
    exceptions.py              # Shared exception types
  providers/
    __init__.py
    base.py                    # Adapter interface + I/O schemas
    cerebras.py                # Cerebras provider implementation
  router/
    __init__.py
    selector.py                # Provider registry + selection logic
  storage/
    __init__.py
    database.py                # SQLAlchemy engine/session setup
    models.py                  # ProviderCredential ORM model
    credentials.py             # Credential CRUD + error markers
  static/css/styles.css        # Dashboard styling
  templates/providers.html     # Providers dashboard
  main.py                      # FastAPI app entry

config/providers.yaml          # Provider list and settings
data/orchestrator.db           # SQLite database (generated)

ai_docs/architecture.md        # This document
ai_docs/providers_api_docs.md  # Provider API notes
ai_docs/supported_providers.md # Supported providers reference
README.md                      # Project overview & quickstart
```

## Extensibility (Adding a Provider)

1) Implement a new adapter in `app/providers/<provider>.py` that subclasses `ProviderAdapter` and translates to/from the provider’s API.
2) Register the adapter in `ProviderRegistry._adapter_map` (id → class).
3) Add the provider to `config/providers.yaml` with appropriate fields and priority.
4) Restart; the admin dashboard will show the provider. Add/update the API key via the UI or Admin API.

## Decisions Snapshot (v1)

- API surface: Only `POST /v1/chat/completions` (no streaming). `GET /v1/models` deferred.
- Providers: Start with Cerebras; OpenRouter and Gemini later.
- Auth: Provider API keys stored in SQLite via dashboard.
- Availability badge: Green if key present and no recent errors; red otherwise.
- Usage tracking: Deferred; default daily reset at 00:00 UTC if needed.
- Security: Local‑only; no external secrets manager.
- UI: Minimal Providers table with Add/Update key; no usage metrics.
