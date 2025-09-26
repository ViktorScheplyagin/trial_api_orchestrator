# Client-Exposed API

## Overview
The Trial API Orchestrator presents a single OpenAI-compatible endpoint that downstream clients can call without being aware of the underlying provider fleet. The orchestrator handles provider selection, error normalization, and response shaping before returning an OpenAI-like payload to the caller. For a system-level view, see [architecture.md](architecture.md).

- **API surface (v1):** `POST /v1/chat/completions`
- **Transport:** HTTPS recommended (HTTP permitted for local development)
- **Media type:** `application/json`
- **Streaming:** Not supported in v1 (requests with `stream=true` will be treated as non-streaming)
- **Versioning:** Path-based (`/v1`). Breaking changes will trigger a new prefix.

## Base URL
- **Local development:** `http://localhost:3001`
- **Production / staging:** Configure per deployment environment; ensure TLS termination in front of the FastAPI app.

## Authentication
The orchestrator does **not** enforce client-side authentication in v1. Provider credentials are managed separately via the admin dashboard/API (see [admin.py](../app/api/admin.py)). If external authentication is required, terminate requests behind your chosen gateway or add a FastAPI dependency in `app/api/openai.py`.

## Endpoint Reference

### POST /v1/chat/completions
Initiate a chat completion request. The orchestrator accepts the subset of OpenAI parameters used by downstream providers. Additional, unknown fields are passed through when supported by adapters.

**Headers**
- `Content-Type: application/json`

**Request Body Schema** (extends `ChatCompletionRequest` in [app/providers/base.py](../app/providers/base.py))

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model` | `string` | ✅ | Identifier exposed by the orchestrator (matches provider `id` in `config/providers.yaml`). |
| `messages` | `array` | ✅ | Conversation history in OpenAI format (`[{"role": "user", "content": "..."}, ...]`). |
| `temperature` | `number` | ❌ | Sampling temperature forwarded to the provider. |
| `max_tokens` | `integer` | ❌ | Upper bound on generated tokens. |
| `stream` | `boolean` | ❌ | Currently ignored; response is always non-streaming. |
| `user` | `string` | ❌ | Client-provided identifier for observability. |
| `presence_penalty` | `number` | ❌ | Adjusts likelihood of introducing new topics. |
| `frequency_penalty` | `number` | ❌ | Penalizes repeated tokens. |
| `top_p` | `number` | ❌ | Nucleus sampling probability mass. |
| _Additional fields_ | `any` | ❌ | Preserved for future provider-specific extensions; unsupported keys are ignored.

**Sample Request**
```http
POST /v1/chat/completions HTTP/1.1
Host: localhost:3001
Content-Type: application/json

{
  "model": "cerebras",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain trial API orchestrator design."}
  ],
  "temperature": 0.3,
  "max_tokens": 512
}
```

**Success Response** (`200 OK`, matches `ChatCompletionResponse`)
```json
{
  "id": "chatcmpl-9dcf82ab",
  "object": "chat.completion",
  "created": 1717171717,
  "model": "cerebras",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 96,
    "total_tokens": 216
  }
}
```

## Error Handling
Errors are normalized to the OpenAI error envelope when possible.

| Status | Code | Description |
| --- | --- | --- |
| `400 Bad Request` | `invalid_request_error` | Pydantic validation failures (missing required fields, wrong data types). |
| `401 Unauthorized` | `provider_auth_required` | The selected provider lacks a configured API key (raised from `AuthenticationRequiredError`). |
| `429 Too Many Requests` | `provider_unavailable` | Provider throttling, quota exhaustion, or temporary downtime (raised from `ProviderUnavailableError`). |
| `5xx` | `server_error` | Unhandled server-side exceptions. Logs contain provider context. |

Example `401` payload:
```json
{
  "error": {
    "message": "Provider 'cerebras' credentials missing",
    "type": "invalid_request_error",
    "code": "provider_auth_required"
  }
}
```

## Rate Limiting & Quotas
The orchestrator does not impose its own rate limiting in v1. Limits are inherited from upstream providers. When a provider signals throttling, the orchestrator returns `429` with `provider_unavailable`. Rate-limit headers from providers are not yet surfaced; see [Decisions Snapshot](architecture.md#decisions-snapshot-v1) for roadmap notes.

## Usage Notes & Compatibility
- The orchestrator selects providers based on priority order defined in `config/providers.yaml`. If the top-choice provider fails, the request is retried with the next available one.
- Responses include provider `model` identifiers; clients should not rely on provider-specific fields unless coordinated with the orchestrator team.
- Streaming support is planned but disabled; clients should avoid enabling SDK streaming until `/v1/chat/completions` advertises Server-Sent Events.
- For provider-specific behaviors and capabilities, refer to [providers_api_docs.md](providers_api_docs.md).

## Change Log
- **v1.0:** Initial release with `POST /v1/chat/completions` only; non-streaming; OpenAI-compatible schema.
