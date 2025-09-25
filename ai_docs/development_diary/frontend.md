#### Common description
It is a simple tiny dashboard. It should be minimalistic both in UI and in technical implementation. Use only HTMX and CSS for its implementation. If some extra JavaScript is strongly needed — use jQuery, otherwise avoid JS and keep the implementation as minimal as possible.
It communicates with the backend using a REST API.

#### GUI and userflow description
The user sees all configured [providers](../supported_providers.md) used in this application.
- For each provider, show:
  - Name
  - Availability badge: green if considered available; red if not available (e.g., recent rate‑limit or auth error). For the first iteration, availability is based on simple heuristics — presence of a valid API key and last known error state (no detailed usage tracking yet).
  - Actions: “Add/Update API Key” (modal or inline form) to store/update the provider API key in SQLite.

No usage counters (tokens/requests) are shown in the first iteration.
