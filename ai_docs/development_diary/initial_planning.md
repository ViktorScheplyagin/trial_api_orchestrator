#### Key Clarifications
- Supported providers and priority: The providers listed in this [table](../supported_providers.md). 
The priority for now is not set. Let's just iterate the providers in the order they are listed.
- OpenAI compatibility surface: Only `POST /v1/chat/completions` for the first iteration (no streaming). `GET /v1/models` is deferred.
- Auth model: Use provider API keys (per provider) managed via a simple [dashboard](./frontend.md). Store provider API keys locally in SQLite.
- Trial quota semantics: Defer usage tracking for the first iteration (no token/request counters yet). We will add tracking later; when added, default reset behavior will be daily at 00:00 UTC unless a provider supplies explicit reset headers.
- Multi-tenancy and quotas: Only global provider limits (no per‑client quotas) once tracking is introduced in a future iteration.
- Streaming and tool support: SSE streaming is optional and not included in the first iteration. No tool/function calling or JSON mode in MVP.
- Error handling and mapping: Errors should match OpenAI’s error schema and codes.
- Observability and admin: Minimal status only in the first iteration (no usage tracking). Health checks and richer admin controls to be discussed in later iterations.
- Deployment constraints: Minimalistic stack. FastAPI + SQLite for the MVP. HTMX + CSS for the tiny GUI (optional, minimal setup only).
- Security: Not planned for cloud deployment in MVP; keep provider keys in SQLite locally.

#### MVP Scope (Proposed)
- Endpoints: `POST /v1/chat/completions` (non‑streaming). `GET /v1/models` is deferred.
- Providers: Start with an OpenAI‑compatible provider adapter (Cerebras first). Add OpenRouter next; add Gemini afterwards.
- Routing: Basic provider selection with failover on provider errors (see Error Policy).
- Trial tracking: Not implemented in the first iteration.
- Storage: Single‑file SQLite instance. No in‑memory mode.
- Admin: Minimal — provider status view and credential (API key) management.

#### Architecture overview
All the options are agreed.

#### Data Model
All the options are agreed.

#### Routing & Failover Logic
All the options are agreed.
For the first iteration, availability is determined without usage counters: if the provider is configured with a valid API key and has not recently returned a rate‑limit error, show as available; on rate‑limit errors show unavailable until manual retry. When usage tracking is introduced, we will compute availability from remaining budget.

#### OpenAI-Compatible Details
- Streaming: Deferred to a future iteration.
- `GET /v1/models`: Deferred.
All the rest options are agreed.

#### Non-Functional
- Security: simplest viable implementation; local‑only, store provider keys in SQLite.
- Observability: minimal logs and a status view; no external tools or complex setup.
- Testing: skip automated tests in the first iteration to keep scope minimal.

#### Initial Files/Modules (Suggested)
All the options are agreed.

#### AGENTS.md Draft Entries (Optional)
All the options are agreed.
Write the file with information that is necessary for you. The file purpose is to help you to keep the context of the project development between the sessions. Write there all the information that you'll consider as important to remember.
