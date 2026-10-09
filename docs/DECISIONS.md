# Decisions

Each entry: what I chose, why, and the alternative I rejected. Numbers point to files in `reports/`.

## Platform and serving

| Decision | Why | Rejected alternative |
|---|---|---|
| **FastAPI + Gradio mounted inside it** (`gr.mount_gradio_app`) | One process, one port, one URL: the HF Space and docker-compose both expose a single service; the UI calls the same in-process services as the API, so there's one code path to test. | Separate Gradio app calling the API over HTTP: two deployables and duplicated error handling for a demo UI. |
| **SSE for `/ask`** (sse-starlette) | One-way token stream over plain HTTP; works through proxies and from `curl`. A final `result` event carries the structured answer. | WebSockets: bidirectional, which we don't need, and harder to put behind a load balancer. |
| **Pipeline as a generator of events** | The same code serves SSE, JSON, the UI and evals; output guard can stop the stream mid-answer. | Separate streaming and non-streaming implementations that drift apart. |
| **SQLite** for the mock enterprise systems + request log | Zero setup, file-based, WAL mode lets the API and worker share it. | Postgres: right for production (multiple API replicas on different nodes), unnecessary here. Called out in `k8s/README.md`. |
| **No authentication** (role and employee id come in the request) | Out of scope for the demo. Safety-critical identity is still never taken from the model: tools close over the request identity, and the agent rejects a role that doesn't match the employee record. | Real SSO/OIDC; the first thing to add before real use. |

## LLM provider and models

| Decision | Why | Rejected alternative |
|---|---|---|
| **Groq `openai/gpt-oss-120b` (main) and `gpt-oss-20b` (small)** | On 2026-10-09 the Llama 3.x models on Groq are "Enterprise / contact sales"; gpt-oss-120b/20b are on the free plan, support tool calling, and are fast. Both are reasoning models, so calls use `reasoning_effort=low` and extra `max_tokens` headroom (hidden reasoning tokens count against it). | `llama-3.3-70b-versatile` (the original plan): not available on a free key anymore. |
| **Small model for router, rewrite, memory summary** | These are short classification/rewriting tasks; the 20b model is cheaper and faster and has its own daily quota. | Main model for everything: burns the main model's free daily quota. |
| **Retry only transient errors, fall back on any error** (`.with_retry` per model, `.with_fallbacks` across models) | Retrying a 400 or a decommissioned model is wasted time; a 429/timeout/5xx often succeeds on retry; anything else should move to the next model. Tools are bound *before* wrapping (wrappers hide `.bind_tools`). | LangChain's built-in client retries: provider-specific behaviour and they retry non-transient errors too. |
| **OpenRouter as the optional second provider**, default `google/gemma-4-31b-it:free` | OpenAI-compatible API, one key, many models, free tier; this model takes images and tools. The ids I first planned no longer existed when I checked the live model list, so both are env-overridable. | Gemini: a second SDK and a different message format for images. |
| **Vision degrades instead of failing** | `qwen/qwen3.8-27b` is a preview model and may be removed. `/image` tries Groq, then OpenRouter vision, then returns `503 image_input_unavailable` listing what was tried; nothing else is affected. | Hard dependency on one preview model. |
| **Prompt Guard 2 (86M) as the second injection layer** | Purpose-built classifier, on the free plan, tiny input (512 tokens), far cheaper than asking a chat model. | General chat model as classifier (kept as `SAFETY_CLASSIFIER=llm`): more tokens, slower. |

## Retrieval

| Decision | Why | Rejected alternative |
|---|---|---|
| **Local embeddings `BAAI/bge-small-en-v1.5`** | Runs on CPU, 384 dims, no API key, so ingestion and retrieval work with every key empty. | API embeddings: cost, network latency, and a key needed just to index. |
| **Token-aware chunks measured with the embedding model's tokenizer**, one section at a time, heading path kept | bge truncates at 512 tokens, so a chunk sized in characters can silently lose its tail. Chunks never span two headings, and the heading path makes citations and BM25 better. | Fixed character windows: the simplest, but they cut across sections and can exceed the model limit. |
| **Embedding batch size 8** | Fastest on this CPU: 1.59x one-by-one; batch 64 was no faster than one-by-one (`reports/embedding_benchmark.json`). | The library default of 32. |
| **Chroma as the main store, FAISS behind the same interface** | Chroma does metadata filtering inside the search (`where`), persistence, and has a client/server mode (used in compose and in the Space so the API and worker share one index). FAISS is the comparison: same quality on this corpus (`reports/grid_results.json`). | FAISS only: needs a hand-built sidecar for metadata and filtering. pgvector: an extra service. |
| **Role filter BEFORE ranking** (Chroma `where`, FAISS `IDSelectorBatch`, one BM25 index per role) | Filtering after ranking can return fewer than k results and lets restricted text influence ranking; separate BM25 indexes also keep HR-only words out of the IDF statistics. Adversarial test: 0 leaks in 198 queries (`reports/retrieval_eval.json`). | Post-filtering the top-k: the classic way to leak. |
| **Role flags as booleans** (`role_hr: true`) | Chroma's `where` filters work on scalar metadata. | A list field: not filterable the same way. |
| **Hybrid BM25 + vector with RRF (k=60)** | RRF combines ranks, so there's no need to normalise BM25 scores against cosine. Kept in production though the grid shows no gain on this corpus: it costs ~0.2 ms and protects identifier-heavy queries. | Weighted score sum: needs per-corpus score normalisation. |
| **Cross-encoder rerank 20 -> 5 + refusal threshold on its score** | Reranking is the decisive factor in the grid (MRR and no-answer accuracy both up) and its logit is a usable "is this relevant?" signal. | Threshold on cosine similarity: in the grid, configs without rerank refuse correctly much less often. |
| **Threshold kept at -4 instead of the sweep's best (-5)** | -4 was chosen before the golden set existed; moving it to the best value on the same 40 questions would inflate the reported score. | Tuning on the test set. |
| **Upsert-then-delete when a document changes** | New chunk ids contain the new content hash, so new chunks are written first and the old ones deleted after: readers see old+new briefly rather than nothing. | Delete-then-insert: a window where the document has no chunks. |

## Caching, queue, locks, memory

| Decision | Why | Rejected alternative |
|---|---|---|
| **RQ, with the worker on Linux (Docker)** | Simple Redis-backed queue we already need Redis for. RQ forks per job (isolation), and Windows has no `fork()`, so the worker runs in Docker; tests and no-Docker dev use RQ's `is_async=False`, which runs the same job in-process. Job payload carries the file bytes, so workers need no shared disk. | Celery: heavier; Dramatiq/Huey: Windows-friendly but another library for one job type; RQ `SimpleWorker` on Windows: unofficial (no SIGALRM timeouts). |
| **Per-document Redis lock** (`SET NX PX`, token-checked release) **+ content hash + double-check after acquiring** | Two workers can't index the same doc at once; a crashed worker can't hold the lock forever (TTL); a slow worker can't release someone else's lock (Lua compare-and-delete); the second worker skips instead of redoing the work. Tested with two racing threads. | Database row lock: not shared across Redis-only workers; no lock: duplicate chunks. |
| **Exact + semantic cache in plain Redis, namespaced by index version + role + prompt version** | Re-ingestion bumps the index version, so stale answers are never served; role in the key stops an HR answer reaching an employee; prompt version makes a prompt rollback clean. Semantic lookup is a numpy scan over <= 500 vectors, cheap at this size. | RediSearch vector index: an extra module for no gain at this scale. |
| **Semantic threshold 0.95 + "same numbers" rule** | The closest wrong pair scored 0.927, so 0.95 gives 0 false hits on the calibration set (and catches 3/6 paraphrases); "VPN error 809" vs "691" differ only in the number (`reports/semantic_cache_calibration.json`). | 0.90: would serve "domestic" answers to "international" questions. |
| **Session memory: recent turns + rolling summary under a token budget, compacted under a per-session lock** | Bounded prompt size for long chats without losing older facts; the lock prevents two concurrent requests from both rewriting the summary. Tokens approximated as chars/4 (several tokenizers in play; the budget only needs to be conservative). | Full history: unbounded tokens; sliding window only: forgets the start of the conversation. |
| **Rate limit: fixed 60 s window per session** | One atomic `INCR`; shared across replicas. Allows a burst of up to 2x at a window edge. | Sliding log in a sorted set: exact, but O(limit) memory per session. |

## Agent

| Decision | Why | Rejected alternative |
|---|---|---|
| **LangGraph with a router in front** | Policy questions go straight to the RAG pipeline (one retrieval + one generation, predictable); small talk and out-of-scope get canned replies (0 tokens); only actions enter the tool loop. | One ReAct agent for everything: more tokens and less predictable for the common case. |
| **Confirm before write** (`create_ticket` proposes; `/agent/confirm` executes) | The model can't create tickets on its own; the user sees exactly what will be created. | Letting the model call a write tool directly. |
| **Idempotency in two layers** (Redis `SET NX` + SQLite `UNIQUE`) | Redis stops concurrent double clicks; the DB constraint is the last line of defence even if Redis is flushed. Tested with 5 concurrent confirms. | App-level "check then insert": races. |

## Evals and experiments

| Decision | Why | Rejected alternative |
|---|---|---|
| **Sources matched by (doc_id, section), not chunk id** | Stable across chunk sizes, so the grid can compare 100- vs 300-token chunks on the same golden set. | Chunk ids: change whenever chunking changes. |
| **Grid is retrieval-only; RAGAS only on baseline + 2 best configs (+ prompt v2)** | Groq free-tier daily token limits. Retrieval metrics need no key and cover 32 configs. | RAGAS on all 32 configs: weeks of free-tier quota. |
| **RAGAS context precision = ID-based** | Compares retrieved vs reference (doc_id, section) ids: no tokens. | LLM-judged context precision: one judge call per retrieved chunk. |
| **Judge cache + checkpoints + resume on daily limit** | A re-run never pays twice; hitting the daily cap saves progress and exits with code 3. | Re-running from scratch. |
| **MLflow with a local SQLite file** (`mlruns/mlflow.db`) | MLflow 3.x put the plain `./mlruns` file store into maintenance mode; SQLite is still a single local file. | The deprecated file store. |
| **RAGAS 0.4.3 + a stand-in module** | RAGAS 0.4.3 imports `langchain_community.chat_models.vertexai`, which langchain-community 0.4 removed; we never use VertexAI, so `evals/generation_eval.py` registers an empty module before importing RAGAS. | Downgrading langchain-community: conflicts with langchain-core 1.x. |

### Known limitation: the same model family generates and judges

Answers are generated by `openai/gpt-oss-120b` and judged by `openai/gpt-oss-20b` (both on Groq's free plan,
with separate daily quotas). Models from the same family share training data and stylistic preferences, so
the judge is likely to be lenient toward answers that "sound like" its own: faithfulness and relevancy scores
may be inflated, and the comparison between configs is more trustworthy than the absolute numbers. The same
applies to the router experiment: its labels were generated by gpt-oss-120b, which may favour the gpt-oss
prompt router.

How I'd fix it:
1. **A judge from a different family**: set `JUDGE_MODEL` to a non-gpt-oss model (e.g. a Gemma or Claude
   model via OpenRouter or another provider) and re-run `evals/generation_eval.py`; the judge cache is keyed
   per config file, so only judging is re-paid.
2. **Human review of a sample**: take ~20 answers stratified by question type, have a person score
   faithfulness and relevancy on the same 0-1 rubric, and report the agreement (e.g. Cohen's kappa) between
   human and judge. If agreement is low, the LLM-judge numbers shouldn't be used to pick configs.
3. For the router: hand-review the 60-example test set (`finetune/data/test_review.csv`) before comparing.

## Fine-tuning

| Decision | Why | Rejected alternative |
|---|---|---|
| **Qwen2.5-0.5B-Instruct + QLoRA (4-bit NF4, r=16, all linear layers), TRL `SFTTrainer`, completion-only loss** | Fits a free T4 easily; 4-way intent + 3-way urgency is a narrow task where a small tuned model can compete; loss only on the JSON answer, not the long system prompt. | 1.5B: slower inference for a router that runs on every message; full fine-tune: doesn't fit the T4. |
| **TRL pinned to 1.14.2** | TRL 1.15 always uses a Triton-only fused LM head (Linux only), which broke the local Windows smoke test; pinning keeps the notebook and the smoke test on the same code. | Untested notebook code. |
| **Notebook generated from `router_train.py`** | The Colab code is the code that was smoke-tested locally. | Hand-maintained notebook drifting from the scripts. |
| **Strict JSON measured separately from accuracy** | The contract is "output only a JSON object"; production parses leniently, so both numbers matter. | Only lenient accuracy: hides format failures. |

## Observability

| Decision | Why | Rejected alternative |
|---|---|---|
| **Structured JSON logs always, Langfuse only when keys are set** | Logs work everywhere with no account; Langfuse adds per-request traces when available and is a no-op otherwise. | Langfuse required: the app wouldn't start with empty keys. |
| **Trace id derived from the request id, passed explicitly on every LLM call** | `/ask` is a generator that hops threadpool threads, so OpenTelemetry context can't be trusted to group calls; an explicit seed always groups them into one trace. | Context-manager spans around the request. |
| **Admin stats as SQL `GROUP BY` queries** | One query per panel; aggregation happens in the database. | Loading rows and aggregating in Python. |
| **Gradio analytics and HF telemetry disabled** | Internal tool: no usage data should leave the process. | Library defaults. |

## Packaging

| Decision | Why | Rejected alternative |
|---|---|---|
| **One container for the Space, supervisord running redis + chroma + api + worker** | Spaces run one container; a Chroma *server* (not embedded Chroma) lets the API and the worker share one index safely from two processes. | Embedded Chroma opened by two processes: unsupported. |
| **Docs ingested at image build time** | A fresh container works immediately; startup ingestion then skips all 12 docs by hash (verified: `reports/docker_smoke.json`). | Ingest only at startup: first requests race the indexing. |
| **`uv sync --no-cache` in the Dockerfile** | Without it uv's download cache (1.6G) stays in the image layer: 5.6GB -> 3.46GB (`reports/docker_smoke.json`). | Default uv behaviour. |
| **Runtime deps only in the image** (`--no-default-groups`; MLflow, RAGAS, TRL, PEFT in separate groups) | Keeps eval and training tooling out of the serving image. | One dependency list for everything. |
