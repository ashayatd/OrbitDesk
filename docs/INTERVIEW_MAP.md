# Interview map

Every number below is copied from a file in `reports/` (named next to it). Anything not measured yet says so.
"Not measured" items need a Groq key; the scripts are ready (see README > Run the evals).

## (a) Requirement -> where it is -> the number that proves it

| Requirement | Where in the repo | Evidence |
|---|---|---|
| RAG for enterprise docs with citations | `app/rag/pipeline.py`, `app/rag/citations.py`, `prompts/answer_v1.yaml` | recall@5 **1.000**, MRR **0.962** on 40 golden questions (`reports/retrieval_eval.json`); every citation checked against retrieved chunks (`tests/test_rag_units.py`) |
| Role-based access control | `app/retrieval/vectorstore/*` (`where` / `IDSelectorBatch`), `app/retrieval/bm25.py` (index per role), `app/guardrails/output.py` | **0 leaks in 198** adversarial queries; 0 leaks in all **32** grid configs (`reports/retrieval_eval.json`, `reports/grid_results.json`) |
| Vector DBs (Chroma + FAISS) behind one interface | `app/retrieval/vectorstore/base.py`, `chroma_store.py`, `faiss_store.py` | Identical recall and MRR for Chroma vs FAISS in the grid (`reports/grid_results.json`) |
| Hybrid search + reranking | `app/retrieval/retriever.py`, `fusion.py`, `reranker.py` | Rerank raises MRR **0.925 -> 0.962** (baseline vs production) and refusal on unanswerable questions **0.14 -> 1.00** (`reports/grid_results.json`, `reports/retrieval_eval.json`) |
| Queues for multi-step work | `app/infra/queue.py`, `app/ingest/jobs.py`, `worker.py` | `POST /ingest` returns 202 + job id; tests: `tests/test_ingest_api.py` |
| Caching (exact + semantic) | `app/cache/answer_cache.py` | Semantic threshold **0.95**: **0/7** false hits, **3/6** paraphrase hits (`reports/semantic_cache_calibration.json`); live hit rate **not measured** |
| Redis distributed locks | `app/infra/locks.py`, `app/ingest/pipeline.py`, `app/agent/memory.py` | Two racing ingests -> exactly one indexes (`tests/test_ingest_pipeline.py::test_concurrent_ingest_indexes_once`) |
| Idempotent actions | `app/agent/actions.py`, `app/db/repo.py` | 5 concurrent confirms -> 1 ticket (`tests/test_agent.py::test_concurrent_confirms_create_one_ticket`) |
| Session memory under a token budget | `app/agent/memory.py` | Budget enforced after 6 turns at a 200-token budget (`tests/test_agent.py::test_memory_compacts_into_summary_within_budget`) |
| Agent with tools, step limit, timeout | `app/agent/graph.py`, `tools.py` | Step-limit and confirm-before-write tests (`tests/test_agent.py`) |
| Guardrails: PII, injection, output | `app/guardrails/*` | Direct injection blocked by regex layer **8/15**, indirect **5/5**, benign false positives **0/10** (`reports/safety_eval.json`); Prompt Guard layer **not measured** |
| Bias check | `evals/safety_eval.py`, `evals/suites/bias_templates.jsonl` | Same correct top source for **8** names/genders x **5** templates: **1.00** (`reports/safety_eval.json`); answer-level **not measured** |
| Evaluation (retrieval, generation, safety, performance) | `evals/*` | Retrieval and safety measured; RAGAS **not measured** (`reports/generation_eval.json` not created yet) |
| Experiment tracking | `evals/grid.py`, `evals/mlflow_utils.py` | **32** configs logged to MLflow (`reports/grid_results.json`) |
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
`all_chunks`), with Chroma and FAISS implementations; `VECTOR_BACKEND` picks one. In docker-compose and in
the Space, Chroma runs as its own server, so the API and the worker share one index over HTTP instead of
two processes writing one embedded database. Access control is part of the interface contract: filters are
applied *inside* the search (Chroma `where`, FAISS `IDSelectorBatch`), and the eval gate fails CI on any leak:
0 leaks in 198 adversarial queries. On this corpus the two stores give identical recall and MRR; Chroma stays
primary because filtering, persistence and client/server mode come built in.

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
- **Agent actions**: confirming a ticket uses `SET idem:{key} NX` as a one-shot gate; a double click or retry
  returns the same ticket, and the SQLite `UNIQUE` key is the backstop. Tested with 5 concurrent confirms.
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
embedding model's own tokenizer -> batched local embeddings -> Chroma with department, version and role
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
