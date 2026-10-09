# OrbitDesk — Build Plan

> Status: **APPROVED 2026-10-09** with all four defaults and the amendments in section 8.
> Every number that later appears in the docs will come from a script's output file under `reports/`. Nothing gets typed in by hand.

## 0. Environment findings (checked 2026-10-09)

| Item | Found | Plan |
|---|---|---|
| Python | 3.13.7 only (no 3.11) | Install **uv** (`pip install uv` or `winget install astral-sh.uv`), then `uv python install 3.11`. The project pins `requires-python = ">=3.11,<3.12"`. The Docker image uses `python:3.11-slim`. |
| uv | not installed | see above |
| Docker | CLI present, engine not running | Start Docker Desktop before Phase 1 (Redis + worker). |
| git | present; folder is not a repo yet | `git init` in Phase 0, with `.gitignore` covering `.env`, `data/chroma`, `mlruns`, model caches |

**Groq models** (from console.groq.com/docs/models, today):

| Env var | Default | Why |
|---|---|---|
| `GROQ_CHAT_MODEL` | `llama-3.3-70b-versatile` | Production tier, supports tool calling |
| `GROQ_VISION_MODEL` | `qwen/qwen3.8-27b` | The only Groq model that takes images (preview tier, so it can change) |
| `GROQ_STT_MODEL` | `whisper-large-v3-turbo` | Production, fast, cheap |
| (internal) router/judge/small model | `llama-3.1-8b-instant` | Cheap and fast for classification, query rewrite and the injection classifier. Also the first fallback. |

Fallback chain: primary Groq model → `llama-3.1-8b-instant` on Groq → optional second provider (if `FALLBACK_PROVIDER_KEY` is set), any OpenAI-compatible API. I propose **OpenRouter** via `langchain-openai` with a `base_url` (see Open Question 1).

## 1. Folder structure

```
orbitdesk/
├── app/
│   ├── main.py                 # FastAPI app factory, mounts Gradio at /ui, lifespan (startup ingest)
│   ├── config.py               # pydantic-settings; reads .env; `llm_available` flag
│   ├── deps.py                 # FastAPI dependencies (redis, store, retriever, llm or 503)
│   ├── logging.py              # structured JSON logs (structlog)
│   ├── api/                    # one router per area: ingest, ask, agent, voice, image, feedback, admin, health
│   ├── schemas/                # Pydantic request/response models
│   ├── ingest/
│   │   ├── parsers.py          # md + pdf (pypdf) → text with heading structure
│   │   ├── chunker.py          # recursive splitter, token-aware, keeps heading_path
│   │   ├── pipeline.py         # hash check → lock → parse → chunk → embed(batch) → replace chunks
│   │   └── jobs.py             # RQ job entrypoints
│   ├── retrieval/
│   │   ├── embeddings.py       # sentence-transformers wrapper (bge-small-en-v1.5)
│   │   ├── vectorstore/        # base.py (VectorStore protocol), chroma_store.py, faiss_store.py
│   │   ├── bm25.py             # rank-bm25 index, role-filtered
│   │   ├── fusion.py           # Reciprocal Rank Fusion
│   │   ├── reranker.py         # cross-encoder (ms-marco-MiniLM-L-6-v2)
│   │   └── retriever.py        # HybridRetriever: filter → vector+bm25 → RRF → rerank → threshold
│   ├── rag/
│   │   ├── prompts.py          # loads prompts/answer_vN.yaml
│   │   ├── rewrite.py          # follow-up / vague question rewriting
│   │   ├── citations.py        # parse + verify [doc, section] against retrieved chunks
│   │   └── pipeline.py         # /ask orchestration with per-stage timings + token counts
│   ├── agent/
│   │   ├── graph.py            # LangGraph: router → (rag | tools agent | smalltalk | refuse)
│   │   ├── tools.py            # search_policies, get_leave_balance, create_ticket, get_ticket_status
│   │   ├── router.py           # prompt router (Groq) | finetuned router (HF adapter)
│   │   ├── actions.py          # pending actions + confirm + idempotency
│   │   └── memory.py           # Redis session memory: recent turns + rolling summary + token budget
│   ├── guardrails/             # pii.py, injection.py, limits.py, output.py
│   ├── llm/                    # client.py (retry/backoff/timeout/fallback), tokens.py, cost.py
│   ├── cache/                  # exact.py, semantic.py, index_version.py
│   ├── infra/                  # redis_client.py, locks.py, ratelimit.py, queue.py
│   ├── db/                     # sqlite schema, seed, repositories
│   ├── observability/          # langfuse.py (no-op when keys empty), request_log.py
│   └── ui/gradio_app.py        # Chat (text) / Voice / Screenshot / Agent / Admin tabs
├── worker.py                   # RQ worker entrypoint (runs on Linux/Docker)
├── data/
│   ├── docs/                   # 10–12 fictional docs (md) + 2 PDFs + manifest.yaml (metadata)
│   └── orbitdesk.db            # SQLite (gitignored, rebuilt by seed)
├── prompts/                    # answer_v1.yaml, answer_v2.yaml, router_v1.yaml, rewrite_v1.yaml, ...
├── scripts/                    # generate_docs.py, seed_db.py, bench_embeddings.py, ingest_all.py
├── evals/
│   ├── golden.jsonl            # 40 questions
│   ├── retrieval_eval.py       # recall@5, MRR, leak test, no key needed
│   ├── generation_eval.py      # RAGAS + no-answer accuracy
│   ├── safety_eval.py          # injection suite, bias check
│   ├── perf_eval.py            # p50/p95 per stage, tokens, cost, cache hit rate
│   ├── grid.py                 # MLflow experiment grid
│   └── suites/                 # injection.jsonl, bias_templates.jsonl
├── finetune/
│   ├── generate_router_data.py # ~500 labelled examples via LLM
│   ├── data/                   # train.jsonl, test.jsonl (60, for hand review)
│   ├── router_lora.ipynb       # Colab T4 QLoRA notebook
│   └── compare_routers.py      # zero-shot vs few-shot vs fine-tuned → MLflow
├── reports/                    # JSON/CSV outputs of every script; the only source of numbers in docs
├── tests/                      # pytest (unit + API with fakeredis, no keys needed)
├── docker/                     # Dockerfile (HF Space all-in-one), Dockerfile.api, supervisord.conf, start.sh
├── docker-compose.yml          # api, worker, redis
├── k8s/                        # optional: deployment, service, hpa, configmap
├── .github/workflows/ci.yml
├── docs/                       # PLAN, ARCHITECTURE, DECISIONS, EVAL_RESULTS, INTERVIEW_MAP
├── pyproject.toml, uv.lock, .env.example, .gitignore, README.md
```

## 2. Services

| Service | Local (compose) | HF Space (single container) |
|---|---|---|
| api (FastAPI + Gradio at `/ui`) | `api` container :8000 (or run natively on Windows with uv) | supervisord process, port **7860** |
| worker (RQ) | `worker` container | supervisord process |
| redis | `redis:7-alpine` | `redis-server` installed via apt in the image, also run by supervisord |
| SQLite, Chroma, MLflow | files under `data/` and `mlruns/` (volume) | files in the container (ephemeral; re-seeded at startup) |
| Langfuse | Langfuse Cloud if keys are set; otherwise a no-op | same |

**Queue choice: RQ, with the worker always on Linux.** RQ's default worker forks per job, and Windows can't fork. Options I looked at:
- *Dramatiq / Celery-solo / Huey*: these work on Windows, but they add another library and Celery is heavy for one job type.
- *RQ `SimpleWorker` on Windows*: unofficial. RQ's timeout handling uses `SIGALRM`, which Windows doesn't have.
- **Chosen:** RQ worker in Docker (compose locally, supervisord on the Space). On Windows the API can still run natively and enqueue to the Redis container. pytest and the "no-Docker" dev mode use `Queue(is_async=False)`, which runs the job in-process without forking. Tests need no worker, and the enqueue code path stays identical.

## 3. Endpoints

| Method | Path | Needs LLM key | Purpose |
|---|---|---|---|
| GET | `/health` | no | liveness, plus `{llm_available, index_version, redis_ok, doc_count}` |
| POST | `/ingest` (multipart file + metadata form fields) | no | save the file, enqueue a job, return `{job_id}` |
| GET | `/ingest/{job_id}` | no | job status and result (chunks written / skipped-unchanged) |
| POST | `/search` | no | debug retrieval only: hybrid + rerank results with scores (proves retrieval works without a key) |
| POST | `/ask` | yes → 503 | `{question, session_id, role}`, SSE stream: `token` events, then a final `result` event (structured answer + timings + tokens) |
| POST | `/agent` | yes → 503 | `{message, session_id, role, employee_id}` → reply, or a `pending_action{confirmation_id, summary}` |
| POST | `/agent/confirm` | no LLM | `{confirmation_id, idempotency_key}` → executes the pending action once |
| POST | `/voice` | yes → 503 | audio file → transcript + /ask result |
| POST | `/image` | yes → 503 | screenshot → extracted error text + /ask result (IT runbooks only) |
| POST | `/feedback` | no | `{session_id, request_id, rating: up/down, comment}` |
| GET | `/admin/stats` | no | questions/day, failure rate, avg latency, top unanswered, cache hit rate |
| GET | `/ui` | no (LLM tabs show the 503 message) | Gradio |

When no key is set, the 503 body is `{"error":"llm_unavailable","detail":"GROQ_API_KEY is not set. Retrieval (/search) and ingestion still work."}`

Auth is out of scope: `role` and `employee_id` arrive in the request body, which a real system would take from SSO. DECISIONS.md will say so. The key safety rule still holds: tools take `employee_id` **from the session**, never from model output.

## 4. Data model

**Document manifest** (`data/docs/manifest.yaml`): `doc_id, title, file, department, allowed_roles[], version, updated_at`.

**Chunk metadata** (Chroma + FAISS sidecar): `chunk_id (doc_id::n), doc_id, title, heading_path ("Leave Policy > Carry-over"), department, version, updated_at, content_hash, role_employee: bool, role_hr: bool, role_it_admin: bool, token_count`.
Roles are stored as boolean flags because Chroma's `where` filter works on scalar values. So `where={"role_hr": True}` filters *inside* the vector search. BM25 and FAISS use the same flags as a pre-filter, so the candidate set is restricted before any scoring.

**Role model:** each doc lists its allowed roles explicitly. Example: salary bands = `[hr]`; privileged-access runbook = `[it_admin]`; most policies = `[employee, hr, it_admin]`.

**SQLite tables**
- `employees(id, name, email, role, department, manager_id)`
- `leave_balances(employee_id, leave_type, entitled, used)`
- `tickets(id, employee_id, title, description, category, priority, status, created_at, idempotency_key UNIQUE)`
- `documents(doc_id PK, filename, content_hash, version, chunk_count, indexed_at)`: the registry used for hash-skip
- `request_log(id, ts, endpoint, session_id, role, question_masked, answered bool, cache_hit, latency_ms, stage_timings json, tokens_in, tokens_out, cost_usd, error)`
- `feedback(id, request_id, session_id, rating, comment, ts)`

**Redis keys**
- `lock:ingest:{doc_id}`: distributed lock (redis-py `Lock`, token-based release, TTL)
- `index:version`: integer, bumped by every ingest that changes chunks
- `cache:exact:{v}:{role}:{sha(normalised q)}`
- `cache:sem:{v}:{role}`: list of `{embedding, answer}` (linear cosine scan over a capped list; no RediSearch needed, explained in DECISIONS)
- `mem:{session_id}`: recent turns list + `mem:{session_id}:summary`
- `pending:{confirmation_id}` (TTL 10 min), `idem:{key}` (SET NX)
- `rl:{session_id}:{window}`: per-session rate limit

## 5. Models (local, CPU)

- Embeddings: `BAAI/bge-small-en-v1.5` (384-d, **512-token max input**). Chunks target about 300 tokens (and 500 in the grid), measured with the model's own tokenizer, with ~15% overlap.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Fine-tune base (Colab): `Qwen/Qwen2.5-0.5B-Instruct` (fallback `Qwen2.5-1.5B-Instruct`). QLoRA 4-bit, PEFT + TRL `SFTTrainer`.
- Docker: CPU-only torch from `https://download.pytorch.org/whl/cpu`. Model weights are downloaded at image build so the Space starts fast.

## 6. Phases (one commit each; `feat(phaseN): …`)

| # | Phase | Done when |
|---|---|---|
| 0 | Scaffold: git init, uv + pyproject, config, JSON logging, `/health`, 503 behaviour, `.env.example`, ruff + pytest skeleton | app starts with an empty `.env`; tests pass |
| 1 | Data + ingestion: generated docs/PDFs, manifest, SQLite seed, parsers, chunker, embeddings, Chroma + FAISS stores, RQ job, lock, hash-skip, chunk replacement, embedding benchmark | `reports/embedding_benchmark.json`; re-ingesting an unchanged doc is skipped; tests for lock + replacement |
| 2 | RAG: rewrite, hybrid + RRF with role pre-filter, rerank, threshold, prompt YAML v1 (+ v2 for the grid), citation verification, structured output, SSE, timings + tokens, `/search` | leak unit test: an employee never sees HR chunks |
| 3 | Agent + memory: LangGraph, 4 tools, pending/confirm + idempotency, step limit + timeout, Redis memory with summary + token budget, router (prompt mode) | double-confirm test creates exactly 1 ticket |
| 4 | Guardrails + cache + reliability: PII masking, injection (heuristics + LLM) on input and on retrieved text, output checks, exact + semantic cache with index version, retry/backoff/timeout/fallback, rate limit | tests for each |
| 5 | Multimodal: `/voice`, `/image`, UI tabs | sample audio + screenshot in `evals/samples/` |
| 6 | Evals + MLflow grid + `docs/EVAL_RESULTS.md` | retrieval eval runs with no key; grid logged |
| 7 | Router fine-tuning: data generation, 60-example held-out test set, Colab notebook, comparison script | zero-/few-shot numbers recorded; fine-tuned row marked **pending Colab run** until you run it |
| 8 | Observability: Langfuse wrapper, `/feedback`, admin stats + Gradio admin tab | |
| 9 | Ship: Space Dockerfile (supervisord), compose, startup ingest, GitHub Actions (ruff, pytest, eval gate, build, deploy), rollback doc, optional k8s/kind | image builds locally; CI file validated (it can only run once pushed) |
| 10 | Interview docs: README + Mermaid, ARCHITECTURE, DECISIONS, INTERVIEW_MAP | every number traces to a `reports/` file |

**Phases that need your Groq key to produce numbers:** 2 (generation), 3–5 (live tests), 6 (RAGAS, perf), 7 (data generation, Groq baselines). I'll build and unit-test everything with a stubbed LLM first. If no key is in `.env` when I reach a measurement step, I'll mark that number "not yet measured" and keep going rather than make one up. You can drop your key into `.env` at any point.

## 7. Open questions (defaults in bold; I'll use them unless you say otherwise)

1. **Fallback provider:** **OpenRouter (OpenAI-compatible) with a free-tier model**, plus the same-key Groq small-model fallback. Alternative: Gemini.
2. **Python tooling:** **I install uv via `pip install uv`** in your user Python, then uv fetches 3.11. Alternative: you install 3.11 yourself and I use venv + pip.
3. **Repo name / GitHub remote:** I'll commit locally only. You create the GitHub repo and HF Space (both listed as manual steps at the end).
4. **RAGAS judge LLM:** **Groq `llama-3.3-70b-versatile`** as the judge, plus local bge embeddings for answer relevancy. RAGAS calls are rate-limited, so the 40-question run will be throttled.

## 8. Approved amendments

1. **`/image` degrades gracefully.** Primary: Groq vision model. If that call fails, or the model has been removed, and `FALLBACK_PROVIDER_KEY` is set, use a vision-capable OpenRouter model. Otherwise return HTTP 503 `{"error":"image_input_unavailable", ...}`. Every other endpoint is unaffected.
2. **Groq free-tier budget.** The Phase 6 grid runs **retrieval metrics only** (no LLM calls). RAGAS runs only on the baseline and the 2 best grid configs. Judge results are cached on disk, keyed by (config, question, answer, metric), so reruns cost no tokens. If a rate or daily limit is hit, progress is saved and the run resumes from the checkpoint next time instead of failing.
3. **DECISIONS.md known limitation:** the same model family generates and judges answers. Fix: use a judge from a different family, and have humans review a sample.
4. `.gitignore` covers `.env`, Chroma data, MLflow runs, model caches and the SQLite file. The Docker image uses Python 3.11, matching local.

## 9. Changes made during the build (with reasons)

- **Model defaults (Phase 6):** Groq now lists `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` as
  Enterprise / contact-sales models; the free plan has `openai/gpt-oss-120b` and `openai/gpt-oss-20b`
  (tool calling, no parallel tool calls). Defaults are now gpt-oss-120b (main) and gpt-oss-20b (small,
  router, judge). Both are reasoning models: requests use `reasoning_effort=low` and extra max_tokens headroom.
- **Injection classifier:** Llama Prompt Guard 2 (86M, free plan) replaces the general LLM classifier as the
  default second layer (`SAFETY_CLASSIFIER=promptguard|llm|off`).
- **OpenRouter fallback:** the planned model ids no longer exist; defaults are `google/gemma-4-31b-it:free`
  (image input + tools), overridable by env.
- **MLflow:** MLflow 3.x put the `./mlruns` file store in maintenance mode, so tracking uses a local SQLite
  file `mlruns/mlflow.db`. MLflow + RAGAS live in the `eval` dependency group, not the runtime image.
- **RAGAS 0.4.3** imports a module langchain-community 0.4 removed; `evals/generation_eval.py` registers an
  empty stand-in for it (we never use VertexAI).
- **Chunk-size grid values:** 100 vs 300 tokens (not 300 vs 500): most sections are under 150 tokens, so
  300 and 500 produce identical chunks.
