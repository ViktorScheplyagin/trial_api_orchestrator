# Trial API Orchestrator

FastAPI application that presents an OpenAI-compatible `POST /v1/chat/completions` endpoint while rotating through trial-friendly providers such as Cerebras. The MVP focuses on credential management and provider failover without detailed usage tracking.

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

## Extensibility (Adding a Provider)

1) Implement a new adapter in `app/providers/<provider>.py` that subclasses `ProviderAdapter` and translates to/from the provider’s API.
2) Register the adapter in `ProviderRegistry._adapter_map` (id → class).
3) Add the provider to `config/providers.yaml` with appropriate fields and priority.
4) Restart; the admin dashboard will show the provider. Add/update the API key via the UI or Admin API.
