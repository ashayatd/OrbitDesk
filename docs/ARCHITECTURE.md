# Architecture

OrbitDesk is one FastAPI service (the Gradio UI is mounted inside it at `/ui`), one RQ worker, Redis, a
vector store (Chroma; FAISS behind the same interface), and SQLite as the mock enterprise systems.
Embeddings and reranking run locally on CPU; generation, speech-to-text and vision call Groq.

```mermaid
flowchart LR
    U[Employee<br/>text / voice / screenshot] --> API[FastAPI + Gradio /ui]
    API -->|enqueue ingest job| Q[(Redis<br/>queue · cache · locks · memory · rate limits)]
    Q --> W[RQ worker<br/>parse · chunk · embed · upsert]
    W --> V[(Chroma<br/>role + department metadata)]
    API -->|hybrid search, role pre-filter| V
    API --> BM[BM25 per role<br/>in memory]
    API --> CE[Cross-encoder<br/>rerank, CPU]
    API --> LLM[Groq: gpt-oss-120b / 20b<br/>Whisper · vision · Prompt Guard]
    LLM -. fallback .-> OR[OpenRouter<br/>optional]
    API --> DB[(SQLite<br/>employees · leave · tickets<br/>request log · feedback)]
    API -. traces .-> LF[Langfuse<br/>optional]
```

Source of each box: `app/api/*` (HTTP), `app/ui/*` (Gradio), `worker.py` + `app/ingest/*` (worker),
`app/retrieval/*` (search), `app/llm/client.py` (LLM + fallback), `app/db/*` (SQLite),
`app/observability/*` (logs, stats, tracing).

## Ingestion (`POST /ingest`, worker, startup)

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant R as Redis
    participant W as Worker
    participant V as Chroma
    participant S as SQLite
    C->>A: POST /ingest (file + department, allowed_roles, version)
    A->>R: enqueue job (file bytes in payload, 5 MB cap)
    A-->>C: 202 {job_id}
    W->>R: take job
    W->>S: content hash unchanged? -> skip
    W->>R: SET lock:ingest:{doc_id} NX PX (wait up to 30 s)
    W->>S: re-check hash (another worker may have finished)
    W->>W: parse (md / PDF headings) -> token-aware chunks with heading path
    W->>W: embed in batches of 8
    W->>V: upsert new chunks (ids contain the new hash), then delete the old ids
    W->>S: update registry; W->>R: INCR index:version (invalidates caches)
    W->>R: release lock (Lua compare-and-delete)
    C->>A: GET /ingest/{job_id}
```

- Files: `app/api/ingest.py`, `app/ingest/jobs.py`, `app/ingest/pipeline.py`, `app/ingest/chunker.py`,
  `app/ingest/parsers.py`, `app/infra/locks.py`, `app/infra/queue.py`.
- At startup the API ingests `data/docs/manifest.yaml` in a background thread; unchanged docs are
  skipped by hash, so restarts cost seconds.

## `/ask` (streamed RAG)

1. **Rate limit** per session (Redis fixed window), then **input guardrails**: length cap, PII masking
   (the masked text is the only version used afterwards: LLM, cache key, logs, memory), injection check
   (regex heuristics, then Prompt Guard 2 on Groq). Blocked input gets a refusal without retrieval.
2. **Memory + rewrite**: recent turns + rolling summary from Redis. A heuristic decides whether the question
   is a follow-up or too vague; only then the small model rewrites it into a standalone query.
3. **Cache**: exact (normalised question) then semantic (cosine >= 0.95 and same numbers), both namespaced by
   index version + role + prompt version (+ department). A hit skips retrieval and generation (0 tokens).
4. **Hybrid retrieval with the role filter inside the search**: Chroma `where={"role_<role>": True}` and a
   per-role BM25 index, 20 candidates each, merged with Reciprocal Rank Fusion. Chunks that look like
   instructions to the AI (indirect injection) are dropped.
5. **Rerank** 20 -> 5 with the cross-encoder. If the best score is below the threshold, the answer is
   "I couldn't find this in the policies." and no LLM call is made.
6. **Generate** with `prompts/answer_<PROMPT_VERSION>.yaml`: context inside `<context>` / `<chunk>` tags,
   answer streamed over SSE, followed by a `---META---` JSON block (follow-up suggestions) that is never
   streamed. The output guard runs on the accumulated text while streaming and can cut the stream.
7. **Verify citations** `[doc_id, section]` against the retrieved chunks; invalid ones are removed.
   Confidence = sigmoid(top rerank score) x share of valid citations.
8. Return `AskResponse` (answer, citations, confidence, follow-ups, per-stage timings, tokens, est. cost),
   then store it in the cache, append to memory, and write a `request_log` row.

Files: `app/api/ask.py`, `app/rag/pipeline.py`, `app/rag/*`, `app/retrieval/*`, `app/guardrails/*`,
`app/cache/*`, `app/agent/memory.py`.

## `/agent` (LangGraph)

```mermaid
flowchart LR
    S([message]) --> G[input guardrails] --> R{router<br/>intent + urgency}
    R -- policy_question --> P[RAG pipeline<br/>no tools]
    R -- action --> A[LLM with tools]
    A -- tool calls --> T[tools] --> A
    A -- final answer / max steps / deadline --> E([reply])
    R -- small_talk --> ST[canned reply, 0 tokens] --> E
    R -- out_of_scope --> RF[canned refusal, 0 tokens] --> E
    P --> E
```

- Router: `ROUTER_MODE=prompt` (Groq gpt-oss-20b, few-shot) or `finetuned` (local Qwen2.5-0.5B + LoRA,
  falls back to prompt mode if the adapter can't load). Unparseable output routes to the read-only RAG path.
- Tools are built per request as closures over the trusted identity: `get_leave_balance` takes **no
  arguments**, so the model cannot ask for someone else's balance. `get_ticket_status` hides other users'
  tickets (same "not found" message, so existence isn't leaked).
- `create_ticket` only stores a pending action in Redis (TTL 10 min) and returns a `confirmation_id`.
  `POST /agent/confirm` executes it once: `SET idem:{key} NX` in Redis, then a `UNIQUE` idempotency key
  in SQLite. A double click returns the same ticket with `status: duplicate`.
- Limits: `AGENT_MAX_STEPS`, an in-graph deadline checked before each LLM call, a hard wall-clock cap on
  the whole graph, and LangGraph's recursion limit. Every failure becomes a polite reply, never a 500.

Files: `app/agent/graph.py`, `router.py`, `tools.py`, `actions.py`, `memory.py`, `app/api/agent.py`.

## `/voice`

Audio file (<= 25 MB) -> Groq Whisper (`whisper-large-v3-turbo`, `verbose_json` for duration and cost) ->
the transcript goes through the full `/ask` pipeline -> response has the transcript, STT latency and cost,
and the answer. Files: `app/api/multimodal.py`, `app/multimodal/stt.py`.

## `/image`

Screenshot (png/jpg/webp, <= 4 MB) -> vision model extracts `{error_code, error_message, application,
search_query}` -> `/ask` pipeline with `department="IT"`, so only IT runbooks are searched (filter inside
the vector search and BM25, on top of the role filter). The vision model is a preview model, so it
degrades: Groq vision -> OpenRouter vision (if `FALLBACK_PROVIDER_KEY`) -> HTTP 503
`image_input_unavailable` listing what was tried. Other endpoints are unaffected.
Files: `app/api/multimodal.py`, `app/multimodal/vision.py`.

## Reliability and limits in one place

| Concern | Mechanism | Where |
|---|---|---|
| Provider hiccup | `.with_retry()` exponential backoff + jitter, transient errors only | `app/llm/client.py` |
| Model gone / quota hit | `.with_fallbacks()`: 120b -> 20b -> OpenRouter | `app/llm/client.py` |
| Slow provider | per-call timeout; agent deadline + hard cap | `app/config.py`, `app/agent/graph.py` |
| Abuse | per-session rate limit, input length cap | `app/infra/ratelimit.py`, `app/rag/pipeline.py` |
| Duplicate work | per-document Redis lock + content hash; idempotent confirm | `app/infra/locks.py`, `app/agent/actions.py` |
| Stale answers | cache keys include the index version, bumped by every changed ingest | `app/cache/*` |
| No API key | every LLM endpoint returns 503 with a clear message; ingestion + `/search` work | `app/deps.py` |
