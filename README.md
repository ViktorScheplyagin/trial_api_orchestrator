# Trial API Orchestrator

FastAPI application that presents an OpenAI-compatible `POST /v1/chat/completions` endpoint while rotating through trial-friendly providers such as Cerebras, Gemini, OpenRouter, HuggingFace and Cohere. Here's possible to add any other provider support by registering your own [adapter](./ai_docs/adapter_docs.md). The MVP focuses on credential management and provider failover without detailed usage tracking.

## Prerequisites

- Python 3.10+
- Optional: virtual environment (`python -m venv venv`)

## Install

```bash
pip install -e .
```

## Run the app

```bash
python -m app
```

The dev server now listens at http://localhost:3001/ by default; override the host or port with `UVICORN_HOST` / `UVICORN_PORT` if needed.

The OpenAPI/Swagger documentation is available at http://localhost:3001/api/docs.


## Telemetry & Logging

- Logs are emitted in JSONL format to `logs/app.jsonl` (rotating at ~10MB × 5 files) and in a concise plain format to stdout.
- Each request receives an `x-request-id`; it is attached to log records and reflected in responses.
- High-signal events (`provider_fail`, `provider_switched`, `request_error`) are also written to the SQLite database and surfaced in the dashboard under **Recent Events**.
- Toggle verbosity and persistence via environment variables:
  - `LOG_LEVEL` (default `WARNING`) controls console output.
  - `LOG_FILE` overrides the JSONL path.
  - `EVENTS_ENABLED` (default `true`) disables/enables SQLite event recording.

To inspect logs locally:

```bash
tail -f logs/app.jsonl | jq
```


## Extensibility (Adding a Provider)

1) Implement a new adapter in `app/providers/<provider>.py` that subclasses `ProviderAdapter` and translates to/from the provider’s API.
2) Register the adapter in `ProviderRegistry._adapter_map` (id → class).
3) Add the provider to `config/providers.yaml` with appropriate fields and priority.
4) Restart; the admin dashboard will show the provider. Add/update the API key via the UI or Admin API.
To learn more, read the detailed [adapters guide](./ai_docs/adapter_docs.md)
