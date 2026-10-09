# OrbitDesk: enterprise helpdesk copilot (GenAI portfolio project)

OrbitDesk is a helpdesk copilot for **Orbit Systems, a fictional IT company**. Employees ask by text, voice or
screenshot; it answers only from 12 fictional HR policies and IT runbooks, with citations that the code checks
against what was actually retrieved. Retrieval is filtered by role *before* ranking, so an employee can never
retrieve the HR-only salary bands. An agent (LangGraph) checks leave balances, raises IT tickets (only after the
user confirms, and exactly once) and lets IT admins ask ticket statistics in plain English through a guarded
text-to-SQL tool. The same tools are exposed as an **MCP server**. Postgres + pgvector hold the business data,
the vectors and the full-text index, so hybrid search can run inside one database. Every layer is measured:
retrieval (three vector stores and a LlamaIndex path compared), guardrails, caching, SQL safety, evals in MLflow,
a fine-tuned router compared honestly against prompting, and CI that blocks a deploy if retrieval quality or
access control regresses.

**Live demo:** _(placeholder: https://huggingface.co/spaces/&lt;your-name&gt;/orbitdesk)_

```mermaid
flowchart LR
    U[Employee<br/>text / voice / screenshot] --> API[FastAPI + Gradio /ui]
    C[MCP client<br/>Claude Desktop · Inspector] -->|stdio or HTTP + bearer token| MCP[MCP server /mcp]
    API -->|enqueue ingest| Q[(Redis<br/>queue · cache · locks · memory)]
    Q --> W[RQ worker<br/>parse · chunk · embed]
    W --> PG[(Postgres + pgvector<br/>employees · tickets · pending actions<br/>vectors · full-text · logs)]
    API -->|role filter in SQL WHERE · vector + full-text + RRF in one query| PG
    API --> CE[Cross-encoder rerank]
    API --> AG[LangGraph agent<br/>router · tools · confirm]
    AG -->|in-process or via MCP| MCP
    AG -->|text-to-SQL: validated SELECT, read-only role| PG
    API --> LLM[Groq gpt-oss · Whisper · vision · Prompt Guard]
    API -. optional .-> LF[Langfuse traces]
```

Without `DATABASE_URL` the same app runs on SQLite + Chroma (logged as a fallback at startup).

## Key numbers (from `reports/`, rendered in [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md))

| What | Result | Source |
|---|---|---|
| Retrieval recall@5 / MRR (40-question golden set) | 1.000 / 0.962 | `reports/retrieval_eval.json` |
| Access-control leaks, adversarial test, per backend | 0 in 198 queries each for Chroma, pgvector, LlamaIndex | `reports/retrieval_eval.json` |
| No-answer accuracy (refuse iff unanswerable) | 0.95 with rerank vs 0.85 without (grid average) | `reports/grid_factor_effects.json` |
| Text-to-SQL attacks blocked by the validator | 25/25 on SQLite and on Postgres | `reports/sql_eval.json`, `reports/sql_eval_postgres.json` |
| Same attacks with the validator bypassed (database layer alone) | Postgres 20/25 refused, data unchanged | `reports/sql_eval_postgres.json` |
| 5 concurrent "Confirm" clicks over HTTP on Postgres | 1 ticket created, 4 duplicates | `reports/postgres_compose_smoke.json` |
| Injection: direct blocked by regex layer / indirect dropped / benign false positives | 8/15 · 5/5 · 0/10 | `reports/safety_eval.json` |
| Bias: same top source across 8 names/genders x 5 templates | 1.00 | `reports/safety_eval.json` |
| Embedding batch 8 vs one-by-one | 1.59x | `reports/embedding_benchmark.json` |
| RAGAS, text-to-SQL accuracy, LLM latency, cost, cache hit rate, router comparison | **not measured yet: need a Groq key** | see [status](#status-whats-done-whats-stubbed) |

## Run locally

Requirements: Python 3.11 via [uv](https://docs.astral.sh/uv/) (`pip install uv`), optionally Docker Desktop.

```powershell
uv sync                                   # runtime + dev deps (CPU-only torch)
copy .env.example .env                    # every key may stay empty
uv run python scripts/ingest_all.py       # seed the DB + index the 12 docs (local models, no key)
```

**Option A, no Docker (Windows-friendly):** SQLite + Chroma, in-process Redis and ingestion jobs.

```powershell
$env:REDIS_URL="fakeredis://"; $env:QUEUE_MODE="inline"
uv run uvicorn app.main:app --port 8000   # http://localhost:8000/ui  (API docs: /docs, MCP: /mcp/)
```

**Option B, real services:** `docker compose up --build` -> http://localhost:8000/ui. Runs api, RQ worker,
Redis and **Postgres with pgvector**; `DATABASE_URL` is set, so data, vectors and full-text search are in Postgres.

**Hosted Postgres (e.g. Neon, free tier, has pgvector):** put its URL in `DATABASE_URL`
(`postgresql+psycopg://user:password@host/db?sslmode=require`); tables, indexes, views and the read-only
analytics role are created at startup.

With every LLM key empty: ingestion, `/search`, the UI, the admin view and the MCP ticket tools work;
`/ask`, `/agent`, `/voice` and `/image` return **503** with a message saying which key is missing.

```powershell
curl -X POST localhost:8000/search -H "content-type: application/json" -d '{\"query\":\"VPN-809\",\"role\":\"employee\"}'
curl -N -X POST localhost:8000/ask -H "content-type: application/json" -d '{\"question\":\"How many leave days can I carry over?\",\"session_id\":\"s1\",\"role\":\"employee\"}'
```

## MCP server

Tools: `search_policies` (role-filtered RAG answer with citations), `get_ticket_status` (own tickets only,
IT admins see all), `ticket_analytics` (IT admins only: plain-English ticket statistics; returns the SQL that ran).
**Who is calling is never a tool argument**: over stdio it comes from server config (`MCP_EMPLOYEE_ID` +
`MCP_ROLE`, checked against the employee record); over HTTP from a signed bearer token.

**Claude Desktop (stdio, local).** Add to `claude_desktop_config.json`
(Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "orbitdesk": {
      "command": "uv",
      "args": ["--directory", "C:\\projects\\AI_Project", "run", "python", "-m", "app.mcp_server", "stdio"],
      "env": {
        "MCP_EMPLOYEE_ID": "E1004",
        "MCP_ROLE": "it_admin",
        "REDIS_URL": "fakeredis://",
        "GROQ_API_KEY": "<needed for search_policies and ticket_analytics>"
      }
    }
  }
}
```

If `uv` is not on your PATH (e.g. installed with `pip install --user uv`), use
`"command": "C:\\projects\\AI_Project\\.venv\\Scripts\\python.exe"`,
`"args": ["-m", "app.mcp_server", "stdio"]`, and add `"PYTHONPATH": "C:\\projects\\AI_Project"` to `env`.

**MCP Inspector (streamable HTTP, remote).** The API serves MCP at `/mcp/`. Get a token, then connect:

```powershell
uv run python -m app.mcp_server token E1004 --ttl 3600     # prints a bearer token (role from the employee record)
npx @modelcontextprotocol/inspector                         # Transport: Streamable HTTP
                                                            # URL: http://localhost:8000/mcp/
                                                            # Header: Authorization = Bearer <token>
```

A standalone HTTP server is also available: `uv run python -m app.mcp_server http --port 8100`.
Set `MCP_AUTH_SECRET` to a long random value anywhere other than your laptop.

**Agent via MCP:** `TOOLS_VIA_MCP=true` makes the LangGraph agent load `search_policies`, `get_ticket_status`
and `ticket_analytics` from the MCP server (it mints a 5-minute token for the current user) instead of calling
them in-process. The whole test suite passes in both modes.

## Run the tests and evals

```powershell
uv run pytest -q                                            # 156 tests, no keys, SQLite
$env:TEST_DATABASE_URL="postgresql+psycopg://orbit:orbit@localhost:5432/orbitdesk"
uv run pytest -q                                            # + 9 Postgres integration tests (165)
$env:TOOLS_VIA_MCP="true"; uv run pytest -q                 # agent tools through the MCP server

$env:REDIS_URL="fakeredis://"; $env:EVAL_DATABASE_URL=$env:TEST_DATABASE_URL
uv run --group llamaindex python evals/retrieval_eval.py --gate   # recall@5, MRR, leak test on every backend
uv run --group eval python evals/grid.py                    # 64-config grid (Chroma/FAISS/pgvector/LlamaIndex) -> MLflow
uv run --group eval python evals/store_microbench.py        # raw store latency per backend
uv run python evals/sql_eval.py                             # text-to-SQL attack suites (+ accuracy with a key)
uv run python evals/safety_eval.py                          # injection suite + bias check (no key)
uv run python evals/safety_eval.py --llm                    # + Prompt Guard + answer-level checks (key)
uv run --group eval python evals/generation_eval.py         # RAGAS, resumable on the free tier (key)
uv run python evals/perf_eval.py                            # latency per stage, tokens, cost, cache hits (key for full)
uv run python evals/render_results.py                       # re-render docs/EVAL_RESULTS.md from reports/
uv run python scripts/check_doc_numbers.py                  # every number in the docs must exist in reports/
uv run --group eval mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Router fine-tuning (Phase 7): `finetune/generate_router_data.py` (key) -> hand-review
`finetune/data/test_review.csv` -> run `finetune/router_lora.ipynb` on a free Colab T4 -> put
`finetuned_predictions.json` into `finetune/results/` -> `finetune/compare_routers.py`.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): request flows (ingestion, `/ask`, `/agent`, text-to-SQL, MCP, `/voice`, `/image`)
- [docs/DECISIONS.md](docs/DECISIONS.md): every choice, why, and the rejected alternative (incl. LangChain vs LlamaIndex vs LangGraph)
- [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md): numbers, generated from `reports/`
- [docs/DEPLOY.md](docs/DEPLOY.md): CI/CD, Space setup, monitoring, rollback
- [docs/INTERVIEW_MAP.md](docs/INTERVIEW_MAP.md): requirement-to-evidence maps for both target roles and interview answers
- [docs/PLAN.md](docs/PLAN.md): the original plan, Phase 8b, and every change made during the build

## Status: what's done, what's stubbed

| Area | Status |
|---|---|
| Ingestion, hybrid retrieval, role pre-filter, rerank, caching, guardrails, agent, memory, multimodal endpoints, UI, admin view, tracing hooks | Built and tested with fakes (no key). End-to-end LLM paths **not yet run against Groq**: no key was available during the build. |
| Postgres + pgvector, text-to-SQL safety layers, MCP server (stdio + HTTP), TOOLS_VIA_MCP, LlamaIndex path | **Built and verified**: Postgres integration tests, compose run on Postgres, MCP over HTTP and stdio. Text-to-SQL *accuracy* needs the key. |
| Retrieval evals, 64-config grid, leak tests, SQL attack suites, safety (heuristic layer), bias (retrieval level), benchmarks | **Measured** |
| RAGAS, text-to-SQL accuracy, Prompt Guard layer, answer-level bias, LLM latency/cost, cache hit rate | Scripts ready; **not measured** (need `GROQ_API_KEY`) |
| Router fine-tuning | Data generator, notebook and comparison ready; training smoke-tested on CPU with a 16-example hand-written set; **real data + Colab run pending** |
| Docker image, compose | **Built and run locally** (`reports/docker_smoke.json`, `reports/postgres_compose_smoke.json`) |
| HF Space deploy | Workflow ready; **needs your Space + `HF_TOKEN`** (+ optional `DATABASE_URL` secret for Neon) |
| k8s manifests | Written, **not tested on kind** |

All documents, employees and data are fictional.
