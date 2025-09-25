## Cerebras Inference API

* **POST /api/chat/completions**  
  Used to generate chat-style conversational responses.  
  Request body includes: `model`, `messages` (array of objects with `role`: “user”/“assistant”/“system”), and optional control parameters: `temperature`, `max_tokens`, `stream`, etc.  
  **Notes / Features**:  
  • Authentication is via API key, passed in header `Authorization: Bearer <CEREBRAS_API_KEY>`.  
  • If `stream = true`, responses are delivered incrementally via SSE (Server-Sent Events).  
  • The response includes rate-limit headers, e.g. `x-ratelimit-limit-requests-day`, `x-ratelimit-remaining-requests-day`, `x-ratelimit-limit-tokens-minute`, etc. — useful for usage tracking.  
  • You can request **structured output** using JSON Schema constraints in requests.  

* **POST /api/completions**  
  Classic “completion” interface: you send a `prompt` (string or list), `model`, and parameters like `max_tokens`, `stream`, etc.  
  **Notes**:  
  • Streaming works similarly (if `stream=true`)  
  • You may use extra flags like `return_raw_tokens` to receive the raw token output (rather than textual).  

* **GET /api/models**  
  Returns the list of available models, each with metadata: model name, context window, capabilities, etc.  
  Useful for dynamic clients to adapt to what’s supported in your account.

---

## OpenRouter API

OpenRouter acts as a unified gateway to multiple LLM providers. Its API closely mirrors OpenAI’s style.

* **POST /api/v1/chat/completions**  
  Main method to generate chat responses. Request JSON includes:  
  - `model` — model name or qualified provider:model string (e.g. `"deepseek/deepseek-r1"` or `"openrouter/auto"`)  
  - `messages` — list of message objects (with `role` and `content`)  
  - Other tuning parameters: `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, etc.  
  **Notes / Features**:  
  • Use header `Authorization: Bearer <OpenRouter_API_Key>` for auth.  
  • Because the interface matches OpenAI closely, migrating from OpenAI client is straightforward.  
  • Supports **provider-specific parameters** (extra fields) that will be passed down to the underlying provider when supported.  
  • Response also includes a `usage` section (prompt_tokens, completion_tokens, total_tokens) and metadata fields (model, provider).  

* **GET /api/v1/models**  
  Returns a unified catalog of models accessible to your account, with metadata (provider, context length, etc.).  

* **Example / nuance**  
  - You can constrain which provider is used via parameters like `provider.only` or `provider.except` in the request JSON.  
  - Supports streaming (if the underlying provider and model support streaming).  

---

## Hugging Face Inference API (Chat / Models)

* **POST /models/{model_id}/chat/completions**  
  For conversational generation. You submit `messages` in the system/user/assistant format.  
  **Notes / Features**:  
  • Authentication: via header `Authorization: Bearer <HF_TOKEN>`.  
  • Streaming (`stream`) is supported when model/infrastructure allows.  
  • Optional parameters: you may override system prompts, pass additional controls or overrides.  
  • The model must be one supported by the inference provider API.  

* **GET /models**  
  Retrieves metadata about models available in the inference API, including context lengths and capabilities.

---

## Gemini API (Google)

Gemini primarily encourages using the client SDK, but supports REST-style endpoints as well.

* **POST `{model}:generateContent`**  
  Generates content (text, multimodal) for a given model.  
  Request body includes `contents`, optional `system_instruction`, `tools` (function calling), and generation parameters like `temperature`, `maxOutputTokens`.  
  **Notes / Features**:  
  • Authentication: via header `x-goog-api-key: <KEY>` or via `?key=` parameter in URL.  
  • Returns `candidates[]` with `content.parts[].text`, along with usage and safety metadata.  
  • Supports streaming via `POST {model}:streamGenerateContent` (SSE) — same shape but streaming.  
  • **GET /v1/models** and **GET /v1/models/{model}** endpoints exist to list models and detailed metadata (max context, modalities, etc.).  
  • **POST /v1beta/models/{model}:countTokens** allows token counting for prospective requests.  
  • **POST /v1beta/models/{embeddingModel}:embedContent** for embedding generation.  
  • Quotas: RPD resets at midnight PT; RPM/TPM reset every minute.  

---

## Cohere API

* **POST /v2/chat**  
  This is the main chat endpoint. You pass `model`, `messages` (system / user / assistant / tool roles), plus options like `stream`.  
  **Notes / Features**:  
  • Authentication: `Authorization: Bearer <COHERE_API_KEY>`.  
  • If `stream = true`, responses stream via SSE (token by token).  
  • Supports `tools` in requests — you may allow models to call tools.  
  • `documents` parameter: a list of documents (strings or objects) that the model may cite from, useful for RAG style.  
  • `citation_options` field to control how citations are generated (e.g. `"accurate"`, `"fast"`, `"off"`).  
  • Response includes `usage` object: how many input/output tokens were used, etc.  
  • There is an older **Chat API v1** variant too, which uses a `chat_history` parameter and `preamble` for system messages. (In v1 you also have `prompt_truncation` options: `"AUTO"`, `"OFF"`, `"AUTO_PRESERVE_ORDER"`)  

* **Other endpoints (embedded in docs)**  
  - Embeddings: generate vector embeddings  
  - Rerank: ranking options  
  - Tokenization / other utilities  

---

## DeepSeek API

* **POST /chat/completions** (or `/completions`)  
  DeepSeek’s chat/completion endpoint is OpenAI-compatible. You send JSON with `model`, `messages`, and optional `stream: true/false`.  
  **Notes / Features**:  
  • Authentication: `Authorization: Bearer <DeepSeek_API_Key>`. ([DeepSeek docs](https://api-docs.deepseek.com))  
  • If `stream = true`, SSE / streaming responses are supported.  
  • DeepSeek supports function calling, JSON output, reasoning mode (for `deepseek-reasoner` model).  
  • Clients can also reuse OpenAI SDKs by pointing `base_url = https://api.deepseek.com` instead of OpenAI.  
  • Example cURL and Python/Node snippets found in “Your First API Call” guide.  

* **GET /models**  
  Returns list of models with metadata (name, version, capabilities, context length).  

* **Other features**  
  - Context caching: DeepSeek includes a caching mechanism to reduce token costs for repeated prompts; you may see `providerMetadata` in responses reflecting cache hits/misses.  
  - Reasoning / tool modes: certain models (e.g. `deepseek-reasoner`) can output reasoning chains or tool calls as part of response.

---

Let me know if you want me to format this into JSON or a single reference sheet for your orchestration module (for all supported providers).
::contentReference[oaicite:0]{index=0}
