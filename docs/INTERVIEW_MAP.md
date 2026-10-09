# Interview map

Every number below is copied from a file in `reports/` (named next to it). Anything not measured yet says so.
"Not measured" items need a Groq key; the scripts are ready (see README > Run the evals).

## (a) Requirement -> where it is -> the number that proves it

| Requirement | Where in the repo | Evidence |
|---|---|---|
| RAG for enterprise docs with citations | `app/rag/pipeline.py`, `app/rag/citations.py`, `prompts/answer_v1.yaml` | recall@5 **1.000**, MRR **0.962** on 40 golden questions (`reports/retrieval_eval.json`); every citation checked against retrieved chunks (`tests/test_rag_units.py`) |
| Role-based access control | `app/retrieval/vectorstore/*` (`where` / `IDSelectorBatch`), `app/retrieval/bm25.py` (index per role), `app/guardrails/output.py` | **0 leaks in 198** adversarial queries; 0 leaks in all **64** grid configs; adversarial test repeated on pgvector and LlamaIndex: 0 leaks in 198 each (`reports/retrieval_eval.json`, `reports/grid_results.json`) |
| Vector DBs (pgvector, Chroma, FAISS) behind one interface | `app/retrieval/vectorstore/base.py`, `pgvector_store.py`, `chroma_store.py`, `faiss_store.py` | Same recall and MRR for all three with rerank; raw store call medians in `reports/store_microbench.json` |
| Hybrid search + reranking | `app/retrieval/retriever.py`, `fusion.py`, `reranker.py` | Rerank raises MRR **0.925 -> 0.962** (baseline vs production) and refusal on unanswerable questions **0.14 -> 1.00** (`reports/grid_results.json`, `reports/retrieval_eval.json`) |
| Queues for multi-step work | `app/infra/queue.py`, `app/ingest/jobs.py`, `worker.py` | `POST /ingest` returns 202 + job id; tests: `tests/test_ingest_api.py` |
| Caching (exact + semantic) | `app/cache/answer_cache.py` | Semantic threshold **0.95**: **0/7** false hits, **3/6** paraphrase hits (`reports/semantic_cache_calibration.json`); live hit rate **not measured** |
| Redis distributed locks | `app/infra/locks.py`, `app/ingest/pipeline.py`, `app/agent/memory.py` | Two racing ingests -> exactly one indexes (`tests/test_ingest_pipeline.py::test_concurrent_ingest_indexes_once`) |
| Idempotent actions | `app/agent/actions.py`, `app/db/repo.py` | 5 concurrent confirms -> 1 ticket on SQLite and Postgres (`tests/test_agent.py`, `tests/test_postgres.py`) and over HTTP against the compose stack (`reports/postgres_compose_smoke.json`) |
| Session memory under a token budget | `app/agent/memory.py` | Budget enforced after 6 turns at a 200-token budget (`tests/test_agent.py::test_memory_compacts_into_summary_within_budget`) |
| Agent with tools, step limit, timeout | `app/agent/graph.py`, `tools.py` | Step-limit and confirm-before-write tests (`tests/test_agent.py`) |
| Guardrails: PII, injection, output | `app/guardrails/*` | Direct injection blocked by regex layer **8/15**, indirect **5/5**, benign false positives **0/10** (`reports/safety_eval.json`); Prompt Guard layer **not measured** |
| Bias check | `evals/safety_eval.py`, `evals/suites/bias_templates.jsonl` | Same correct top source for **8** names/genders x **5** templates: **1.00** (`reports/safety_eval.json`); answer-level **not measured** |
| Evaluation (retrieval, generation, safety, performance) | `evals/*` | Retrieval and safety measured; RAGAS **not measured** (`reports/generation_eval.json` not created yet) |
| Experiment tracking | `evals/grid.py`, `evals/mlflow_utils.py` | **64** configs logged to MLflow (`reports/grid_results.json`) |
| Fine-tuning vs prompt-only | `finetune/*` | CPU smoke test: strict JSON 4/4, CPU inference median 4,504 ms (`reports/finetune_smoke.json`); comparison **not measured** (needs generated data + Colab run) |
| Multimodal (voice, screenshot) | `app/multimodal/*`, `app/api/multimodal.py` | Fallback chain tested (`tests/test_multimodal_ui.py`); live latency **not measured** |
| Performance | `evals/perf_eval.py` | Retrieval p50 **152.8 ms** / p95 **183.4 ms**, of which rerank p50 **138.4 ms** (`reports/perf_eval.json`); LLM stages **not measured** |
| Batching | `app/ingest/pipeline.py`, `scripts/bench_embeddings.py` | Batch 8 = **1.59x** one-by-one (`reports/embedding_benchmark.json`) |
| CI/CD, Docker, rollback, monitoring | `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml`, `docs/DEPLOY.md`, `app/observability/*` | CI gate: recall@5 >= 0.95, MRR >= 0.9, leaks = 0 (`evals/gate.json`); image built and smoke-tested locally, compose 4/4 services healthy (`reports/docker_smoke.json`) |

## (b) Questions

### 1. Architecture for a multi-step AI feature using queues and caching for scalability

Two flows with different latency needs. **Ingestion** is slow and bursty, so it goes through a queue:
`POST /ingest` stores nothing heavy, enqueues an RQ job with the file bytes in the payload and returns 202 with
a job id. The worker parses, chunks, embeds in batches and upserts; because the payload carries the bytes,
workers need only Redis and the vector store, so they scale horizontally. **Questions** are synchronous and
streamed, and they go through two caches before any model call: an exact cache on the normalised question and a
semantic cache on the embedding. Both are keyed by index version + role + prompt version, so re-ingesting a
document invalidates everything at once, and an HR answer can never be served to an employee. The threshold
came from measurement, not a guess: the closest *wrong* pair scored 0.927, so 0.95 gives 0/7 false hits and
catches 3/6 paraphrases (`reports/semantic_cache_calibration.json`). Unanswerable questions skip the LLM
entirely after retrieval, and the agent sends small talk and out-of-scope messages to canned replies, so the
expensive path is reserved for real questions. Live cache hit rate isn't measured yet (needs the key).

### 2. Fine-tuning vs prompt-only approaches when deploying GenAI features

In this project I'd only fine-tune the **router**, not the answerer: answers must come from documents that
change, which is a retrieval problem, while routing is a narrow, stable classification (4 intents x 3
urgencies) where a 0.5B model could match a large prompted model at lower cost and latency. The setup is
deliberately fair: the fine-tuned model is trained on exactly the same instruction text as the prompt router,
and all three (Groq zero-shot, Groq few-shot, QLoRA Qwen2.5-0.5B) are scored on the same 60 held-out
examples, with strict-JSON rate reported separately from accuracy. The results are **not measured yet**: the
data needs the key and training needs Colab. What the CPU smoke test already showed (2 training steps on 16
hand-written examples, `reports/finetune_smoke.json`): the code path works end to end and the adapter emits
valid strict JSON 4/4, but CPU inference took a median of 4,504 ms per message (fp32, unmerged LoRA, busy
laptop). That's a deployment cost the accuracy table won't show: on a free CPU Space the fine-tuned router
would be slower than a network call to Groq, so to win on latency it needs a GPU, or a merged and quantized
model (e.g. ONNX/llama.cpp). One bias I'd flag up front: the labels were generated by
gpt-oss-120b, the same family as the prompt baseline, which may favour prompt-only; that's why the test set is
reviewed by hand first. If prompt-only wins, the report will say so.

### 3. Integrating vector databases like FAISS/Chroma into microservices-based backends

The service depends on a `VectorStore` interface (`upsert`, `delete`, `query(embedding, k, role, department)`,
`all_chunks`) with three implementations: pgvector, Chroma and FAISS. `VECTOR_BACKEND=auto` picks pgvector when
`DATABASE_URL` is set, so vectors, the full-text index and the business data share one Postgres (one backup,
one access-control model, hybrid search as one SQL query), and falls back to Chroma otherwise; in the Space,
Chroma runs as its own server so the API and the worker share one index instead of two processes writing an
embedded database. Access control is part of the interface contract: filters are applied *inside* the search
(SQL `WHERE`, Chroma `where`, FAISS `IDSelectorBatch`), and the eval gate fails CI on any leak: 0 leaks in 198
adversarial queries on each backend. With reranking all three give the same recall and MRR on this corpus; the
raw store call takes a few milliseconds on each (`reports/store_microbench.json`).

### 4. Replacing N+1 queries with aggregation pipelines, related to vector retrieval

_(The first half of this question is about past work outside this repo; only the GenAI parallel is below.)_

The same "one batched call instead of N small ones" pattern shows up in four places here:
- **Embedding**: one batched `encode` per document instead of one call per chunk: batch 8 runs 1.59x faster
  than one-by-one (`reports/embedding_benchmark.json`).
- **Reranking**: the 20 (query, chunk) pairs go to the cross-encoder in one `predict` call.
- **Filtering inside the search**: one filtered vector query and one per-role BM25 index, instead of fetching
  candidates and then checking each chunk's permissions one by one.
- **Admin stats**: questions per day, failure rate, top unanswered questions are SQL `GROUP BY` queries, one
  per panel, not rows loaded and counted in Python; the leave balance is one query for all leave types.

### 5. Cutting search API latency from 90s to 300ms

_(The first half is past work outside this repo; only the GenAI parallel is below.)_

What I measured here is where the time goes, which is the first step of any latency cut: retrieval p50 is
152.8 ms, and 138.4 ms of it is the cross-encoder rerank; embedding the query is ~11 ms and the vector search
~3 ms (`reports/perf_eval.json`, `reports/retrieval_eval.json`). So the levers are the reranker (100-token
chunks cut grid-average p50 from 96 to 82 ms because the cross-encoder scores shorter pairs; dropping rerank
gets ~17 ms but loses refusal accuracy, 0.95 -> 0.85), the caches (a hit skips retrieval and generation), and
skipping the LLM entirely when retrieval finds nothing relevant. End-to-end latency with generation is
**not measured yet**.

### 6. How Redis distributed locks inform safe GenAI workflow design

Three places where two workers or two requests could corrupt state:
- **Ingestion**: a per-document lock (`SET lock:ingest:{doc} NX PX`), so two workers never index the same
  document; the release is a Lua compare-and-delete, so a slow worker whose lock expired can't delete
  someone else's lock. After acquiring the lock the worker re-checks the content hash: the loser of a race
  skips instead of redoing the work. Tested with two racing threads: exactly one `indexed`, one `skipped`.
- **Agent actions**: here I moved from a Redis gate to the database, because the pending action is now a
  database row: `SELECT ... FOR UPDATE` serialises double clicks and `INSERT ... ON CONFLICT (idempotency_key)
  DO NOTHING` makes a second ticket impossible. Tested with 5 concurrent confirms on SQLite and Postgres and 5
  concurrent HTTP requests against the compose stack: 1 created, 4 duplicates. The choice of lock follows the
  data: Redis locks where the work spans processes and stores (ingestion), a row lock where the state is a row.
- **Memory**: summarising old turns runs under a per-session lock with a short wait; if another request holds
  it, this one skips compaction instead of blocking the user.
The general lesson: LLM calls are slow and retried, so any side effect they trigger must be idempotent and
guarded, and a lock needs a TTL plus an owner check.

### 7. Managing session memory and token limits for real-time chatbots

Memory lives in Redis per session: the last 6 user/assistant pairs verbatim plus a rolling summary. After each
turn, if there are too many turns or the estimated tokens exceed the budget (1,500 by default), the oldest
pairs are folded into the summary by the small model, or into an extractive summary if no LLM is available.
Reads also trim defensively, so the context never exceeds the budget even if compaction hasn't run. Two
design choices keep it cheap: the answer prompt doesn't carry chat history at all (a follow-up is first
rewritten into a standalone question, and only when a heuristic says it needs it), and summarisation runs
*after* the response is sent. Tested: after 6 turns at a 200-token budget the oldest question survives in the
summary and the context stays under budget.

### 8. Evaluating LLM output accuracy, safety and bias in production

Four layers, most of them runnable without an API key:
- **Retrieval** on a 40-question golden set (follow-ups, multi-document, unanswerable, role-restricted):
  recall@5 1.000, MRR 0.962, no-answer accuracy 0.95. It runs in CI as a gate.
- **Safety**: an adversarial access-control test (0 leaks in 198 queries), an injection suite (regex layer
  alone blocks 8/15 direct attacks and drops 5/5 poisoned documents with 0/10 false positives; the misses are
  role claims and "repeat your instructions" tricks, which is what the Prompt Guard layer is for; I didn't
  tune the regexes on the suite I wrote myself). These misses still can't leak data, because retrieval is
  role-filtered.
- **Bias**: the same questions with 8 names/genders: identical, correct top source in 5/5 templates.
- **Generation**: RAGAS faithfulness and answer relevancy plus ID-based context precision, run only on the
  baseline and best configs to fit the free tier, with a judge cache and resume. **Not measured yet.** Known
  weakness: the judge is from the same model family as the generator; the fix is a different-family judge
  plus human review of a sample (see `docs/DECISIONS.md`).
In production the same signals continue as thumbs up/down, unanswered-question counts and failure rates in the
admin view.

### 9. Building end-to-end GenAI apps with LLM APIs safely and reliably

- **Works without keys**: ingestion and retrieval use local models; generation endpoints return a clear 503.
- **Reliability**: per-model retries with backoff only on transient errors, a fallback chain across models and
  providers, timeouts, an agent step limit and a wall-clock cap, per-session rate limits.
- **Safety in code, not just in the prompt**: role filtering before ranking, citations verified against what
  was retrieved, tool identity taken from the session (the leave tool has no arguments), confirm-before-write,
  PII masked before the LLM, an output guard that can stop a stream mid-answer.
- **Degrading instead of failing**: the vision model is a preview model, so `/image` falls back to another
  provider and then returns a specific 503 while everything else keeps working.
- **Checked against reality**: during the build, the planned Llama models turned out to be Enterprise-only on
  Groq and the planned OpenRouter ids no longer existed; both were caught by checking live model lists, and
  the defaults were changed and documented.

### 10. Prompt engineering and optimization for context-heavy applications

Prompts are versioned YAML files (`prompts/answer_v1.yaml` few-shot, `answer_v2.yaml` zero-shot and
shorter); `PROMPT_VERSION` switches them without a code change, and the cache is namespaced by version. The
context is delimited (`<context>`, one `<chunk doc_id=... section=...>` per chunk, chunk text HTML-escaped so a
document can't close the tag), the model is told the context is data, not instructions, and it must cite
`[doc_id, section]` exactly as labelled, which the code then verifies. Output is answer-then-`---META---`-JSON,
so the answer streams and the JSON never does. Context size is controlled upstream: rerank to the top 5,
token-aware chunks with heading paths, and no chat history in the answer prompt. The v1 vs v2 comparison
(quality vs tokens) is part of the RAGAS runs: **not measured yet**.

### 11. RAG architecture for enterprise applications

Ingestion: parse (markdown and PDF headings rebuilt into a heading path) -> token-aware chunks sized with the
embedding model's own tokenizer -> batched local embeddings -> pgvector (or Chroma) with department, version and role
metadata; content hashes skip unchanged documents and the index version invalidates caches. Query: guardrails
-> rewrite if needed -> cache -> vector + BM25 with the role filter inside the search -> RRF -> cross-encoder
rerank -> refuse below the threshold (no LLM call) -> generate with delimited context -> verify citations ->
structured response with confidence, per-stage timings, tokens and cost. Measured: recall@5 1.000, MRR 0.962,
0 leaks; reranking is what makes refusal work (unanswerable refused 0.14 -> 1.00 vs the no-rerank baseline).

### 12. Deploying an AI feature: CI/CD, Docker, monitoring, rollback

CI runs lint, tests with no keys, a retrieval eval gate (fail if recall@5 < 0.95, MRR < 0.9, or any leak),
builds the image and smoke-tests the container, then deploys to a Hugging Face Space. The image is one
container (API + UI + RQ worker + Redis + Chroma server under supervisord) with CPU-only torch, the local
models baked in, and the documents ingested at build time so a fresh container works. Monitoring: JSON logs
with request ids, `/health`, an admin view (failure rate, unanswered questions, latency, cache hit rate,
feedback) and optional Langfuse traces. Rollback has two levers: redeploy an older git tag through the same
gates (the Space becomes an exact copy), or switch `PROMPT_VERSION` when the code is fine but answers got
worse. Verified locally: the image builds, the container passes the same smoke test CI runs, a real RQ
worker indexes an upload, and compose runs all four services healthy (`reports/docker_smoke.json`). One
finding from that: uv's download cache was being baked into the image (5.6GB); `--no-cache` brought it to
3.46GB. Not done yet: the Space doesn't exist yet (it needs your account and token).


---

# Second target role: agentic AI, RAG / MCP / LangGraph / LangChain / LlamaIndex, Python microservices, Unix, Postgres

Same rule: every number comes from `reports/`; anything not measured says so.

| Requirement | Where in the repo | Evidence |
|---|---|---|
| Agentic AI (tool-using agent with routing, limits, human-in-the-loop) | `app/agent/graph.py`, `tools.py`, `actions.py`, `router.py` | Router -> RAG / tool loop / canned replies; step limit + deadline + wall-clock cap; confirm-before-write with exactly-once execution (`tests/test_agent.py`, `tests/test_postgres.py`) |
| Agent tool that queries a database safely (text-to-SQL) | `app/analytics/text_to_sql.py`, `prompts/text_to_sql_v1.yaml` | Raw-SQL attacks blocked by the validator: 25/25 on SQLite and Postgres; with the validator bypassed, Postgres still refused 20/25 and the data was unchanged (`reports/sql_eval*.json`); accuracy on 10 questions **not measured** (needs the key) |
| RAG | `app/rag/*`, `app/retrieval/*` | recall@5 1.000, MRR 0.962 (`reports/retrieval_eval.json`) |
| MCP | `app/mcp_server/*`, `app/agent/mcp_tools.py` | stdio + streamable HTTP; identity from config or signed bearer token; 401 without a token; the full test suite passes with `TOOLS_VIA_MCP=true` (`tests/test_mcp.py`, CI); verified against the compose stack (`reports/postgres_compose_smoke.json`) |
| LangGraph | `app/agent/graph.py` | the agent graph (router, tool loop, limits) |
| LangChain | `app/llm/client.py`, `app/agent/tools.py` | chat models, retry + fallback chain, tool definitions, Langfuse callbacks |
| LlamaIndex | `app/retrieval/llamaindex_retriever.py` | Same chunks/embeddings/filter: same quality as the main path with rerank; 16 grid configs; 0 leaks in 198 (`reports/grid_results.json`, `reports/retrieval_eval.json`) |
| Python microservices | `app/main.py` (FastAPI), `worker.py` (RQ), `docker-compose.yml`, `k8s/` | api, worker, redis, postgres as separate services, all healthy in compose (`reports/postgres_compose_smoke.json`); stateless API (sessions in Redis, MCP stateless HTTP) so it can scale horizontally |
| Postgres | `app/db/*`, `app/retrieval/vectorstore/pgvector_store.py` | Constraints, indexes, `FOR UPDATE`, `ON CONFLICT`, read-only role, HNSW + GIN indexes, in-database hybrid search; 9 Postgres integration tests run in CI against a service container |
| Unix / Linux | `Dockerfile`, `docker/supervisord.conf`, `worker.py`, CI on ubuntu | Linux container with supervisord managing 4 processes; RQ's fork-per-job worker (why it runs on Linux, not Windows); shell-based CI smoke tests |
| Oracle | | **Not used in this project.** |

## Questions for this role

### Building agentic AI that is safe to give real tools

The agent can read (search, ticket status, analytics) and propose writes, but it can't write on its own:
`create_ticket` only creates a pending action, and a separate confirm call executes it exactly once in one
database transaction. Identity is never a tool argument: in-process tools close over the session's identity,
and over MCP the identity is in a signed token the server verifies before any tool runs. The riskiest tool,
text-to-SQL, is only offered to IT admins and is defended in three independent layers: the model only sees
two PII-free views; a validator parses the SQL and allows exactly one SELECT over those views; and the query
runs in a READ ONLY transaction as a database role that can only SELECT those views, with a statement
timeout. Measured: 25/25 raw attacks blocked by the validator; with the validator deliberately bypassed,
Postgres still refused 20/25 and nothing changed. I also report what the database layer alone *doesn't* stop:
system-catalog reads, which only the validator blocks.

### MCP: why and how

MCP turns the same three tools into a standard interface that any MCP client (Claude Desktop, MCP Inspector,
another agent) can use, instead of custom glue per client. The server is built with the official Python SDK
and runs two ways: stdio for a local client, with the user configured in the server's environment; and
streamable HTTP mounted inside the API at `/mcp/`, where every request carries a bearer token. The design rule is
that the protocol never lets the caller choose who they are: tool schemas have no identity fields (a test checks
this), and the same call returns different results for different tokens. Running the agent's own tools through
MCP (`TOOLS_VIA_MCP=true`) proves the protocol path is a drop-in: the whole suite passes in both modes.

### LangChain vs LlamaIndex vs LangGraph

LangChain is the component layer (models, tools, retries, fallbacks, callbacks); LlamaIndex is a retrieval
and data framework; LangGraph is the agent runtime. I use LangChain for every LLM call and LangGraph for the
agent, and I kept the main retrieval path hand-written because its security property (the role filter inside
the search) must be obvious. To check that this wasn't costing quality, I built a LlamaIndex path over the same
chunks, embeddings, filter and reranker and put it in the eval grid: with reranking it scores the same; its
hybrid variant without rerank scored lower on this corpus (`reports/grid_results.json`).

### Postgres in an AI backend

Postgres is the main database: business records with real constraints and indexes, `FOR UPDATE` +
`ON CONFLICT` for exactly-once actions, a read-only role for LLM-written SQL, and pgvector so the document
vectors (HNSW, cosine) and the full-text index (tsvector, GIN) live next to the data. That makes hybrid search
a single SQL statement (vector top-k + full-text top-k + reciprocal rank fusion) and the role filter a plain
`WHERE` clause; pgvector 0.8's iterative scan keeps a selective filter from returning fewer than k results. In
the grid pgvector matches Chroma and FAISS on quality with rerank; its full-text keyword side was weaker than
Python BM25 without rerank. Integration tests run against a real Postgres in CI, and the compose stack was run
end to end on Postgres. Without `DATABASE_URL` the app falls back to SQLite + Chroma and logs it.

### Python microservices and Unix

The system is split by workload: a stateless FastAPI service (API, UI and MCP in one process), an RQ worker for
slow ingestion, Redis for queue/cache/locks/session memory, and Postgres. Sessions live in Redis and the MCP
endpoint is stateless, so API replicas are interchangeable (k8s manifests include an HPA on CPU, since the
cross-encoder is the CPU-bound stage; those manifests are written but not tested). The Unix side shows up in
the deployment: a Linux container with supervisord managing four processes for the single-container demo, RQ's
fork-per-job worker model (and why it can't run on Windows), and CI smoke tests that start the container and
probe it with curl.

### Oracle

Not used in this project.
