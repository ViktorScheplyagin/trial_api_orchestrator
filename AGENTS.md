# Agents Directory

## Overview
Document the autonomous agents participating in the trial API orchestrator. Update this file as agents are added or modified.

## Agent Template
- **Name:** TODO
- **Description:** TODO
- **Primary Capabilities:**
  - TODO
- **Inputs:** TODO
- **Outputs:** TODO
- **Owner / Maintainer:** TODO
- **Source Location:** TODO

## Current Agents

- **Name:** Provider Router
  - **Description:** Routes OpenAI-compatible chat requests to the first available provider and handles failover on provider errors.
  - **Primary Capabilities:**
    - Provider selection and basic failover
    - OpenAI schema normalization for responses
  - **Inputs:** OpenAI-like `POST /v1/chat/completions` request
  - **Outputs:** OpenAI-like chat completion response (non-streaming in v1)
  - **Owner / Maintainer:** Orchestrator Team
  - **Source Location:** `app/router/selector.py`

- **Name:** Credentials Manager
  - **Description:** Stores and retrieves provider API keys from SQLite and supplies credentials to adapters.
  - **Primary Capabilities:**
    - Provider API key CRUD
    - Availability heuristic (key presence + recent error state)
  - **Inputs:** Admin UI actions (HTMX) and provider identifiers
  - **Outputs:** Provider credential records for adapters
  - **Owner / Maintainer:** Orchestrator Team
  - **Source Location:** `app/storage/credentials.py`

- **Name:** Cohere Adapter
  - **Description:** Calls Cohere's `/v2/chat` endpoint and normalizes responses into the orchestrator's OpenAI-style schema.
  - **Primary Capabilities:**
    - Cohere chat payload construction and HTTP invocation
    - Response normalization including tool-calls and citation metadata
  - **Inputs:** Normalized chat completion requests from the router and Cohere API credentials
  - **Outputs:** Chat completion responses compatible with `ChatCompletionResponse`
  - **Owner / Maintainer:** Orchestrator Team
  - **Source Location:** `app/providers/cohere.py`

- **Name:** OpenRouter Adapter
  - **Description:** Bridges OpenRouter's OpenAI-style gateway for multi-provider access and ensures responses align with orchestrator expectations.
  - **Primary Capabilities:**
    - Forwards normalized chat completion requests to OpenRouter's `/api/v1/chat/completions`
    - Maps OpenRouter errors to orchestrator exceptions and harmonizes response defaults
  - **Inputs:** Normalized chat completion requests and OpenRouter API credentials
  - **Outputs:** Chat completion responses compatible with `ChatCompletionResponse`
  - **Owner / Maintainer:** Orchestrator Team
  - **Source Location:** `app/providers/openrouter.py`

- **Name:** Gemini Adapter
  - **Description:** Bridges Google’s Gemini `generateContent` endpoint with the orchestrator by translating OpenAI-style messages into Gemini contents and back.
  - **Primary Capabilities:**
    - Converts system/user/assistant messages into Gemini `contents` and `systemInstruction`
    - Normalizes Gemini candidates, safety data, and usage metadata into `ChatCompletionResponse`
  - **Inputs:** Normalized chat completion requests and Gemini API key
  - **Outputs:** Chat completion responses compatible with `ChatCompletionResponse`
  - **Owner / Maintainer:** Orchestrator Team
  - **Source Location:** `app/providers/gemini.py`

- **Name:** Telemetry Recorder
  - **Description:** Captures provider failovers and errors, persisting high-value events for operators.
  - **Primary Capabilities:**
    - Structured logging with request correlation
    - SQLite-backed event persistence for admin dashboards
  - **Inputs:** Event payloads from router, middleware, and exception handlers
  - **Outputs:** Recent event records for UI consumption and log stream entries
  - **Owner / Maintainer:** Orchestrator Team
  - **Source Location:** `app/telemetry/events.py`

## Decisions Snapshot (v1)
- **API surface:** Only `POST /v1/chat/completions` (no streaming). `GET /v1/models` deferred.
- **Providers:** Start with an OpenAI-compatible provider (Cerebras), then OpenRouter; Gemini later.
- **Auth:** Provider API keys only, stored in SQLite via a minimal HTMX dashboard.
- **Availability badge:** Green if key present and no recent rate-limit/auth errors; red otherwise. No detailed usage tracking in v1.
- **Usage tracking:** Deferred. Future default reset at 00:00 UTC unless provider headers indicate otherwise.
- **Security:** Local-only; keys stored in SQLite. No external secrets manager in v1.
- **UI:** Minimal Providers table with “Add/Update API Key”; no usage metrics in the first iteration.

## Notes
- Keep this document in sync with any configuration files that instantiate agents.
- Include links to related documentation or dashboards when available.

## Architecture & File Structure
- For a complete overview of the project architecture, module responsibilities, request lifecycles, and file layout, see [ai_docs/architecture.md](ai_docs/architecture.md).
