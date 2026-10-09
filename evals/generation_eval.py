"""Generation eval with RAGAS, built for the Groq free tier. Needs GROQ_API_KEY.

    uv run --group eval python evals/generation_eval.py              # run / resume
    uv run --group eval python evals/generation_eval.py --limit 5    # smoke test on 5 questions

Budget rules (Groq free tier):
- Only 4 runs: baseline, the 2 best retrieval configs from the grid (reports/grid_results.json),
  and the best config with prompt v2 (the prompt v1 vs v2 comparison).
- Faithfulness + answer relevancy are LLM-judged; context precision uses RAGAS's ID-based
  metric (retrieved vs reference (doc_id, section) ids), so it costs no tokens.
- Each question's generation is checkpointed to evals/.checkpoints/<config>.jsonl and each
  judge score to evals/.judge_cache/<config>.jsonl (keyed by answer hash), and LangChain's
  SQLite LLM cache stores every raw judge call. Re-running never pays twice.
- Per-minute 429s are retried with backoff; a DAILY limit stops the run cleanly (exit code 3)
  with progress saved. Run the same command tomorrow to resume.
- LLM fallbacks are disabled so a quota hit can't silently swap in another model.

Judge: JUDGE_MODEL (default openai/gpt-oss-20b) judges answers from openai/gpt-oss-120b.
Same model family: a known limitation, see docs/DECISIONS.md.

Writes reports/generation_eval.json (status "complete" or "partial").
"""

import argparse
import hashlib
import json
import sys
import time
import uuid

from common import BASELINE, EVAL_DIR, PRODUCTION, REPORTS, EvalConfig, load_golden, make_retriever, write_report

from app.config import get_settings

CKPT = EVAL_DIR / ".checkpoints"
JUDGE = EVAL_DIR / ".judge_cache"
NOT_FOUND = "I couldn't find this in the policies."


class DailyLimitError(Exception):
    pass


def is_daily_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return "rate" in msg and any(s in msg for s in ("per day", "tpd", "rpd", "daily"))


def is_rate_limit(e: Exception) -> bool:
    return "429" in str(e) or "rate limit" in str(e).lower() or type(e).__name__ == "RateLimitError"


def with_rate_limit_handling(fn, *args, attempts: int = 4):
    """Retry per-minute rate limits with growing sleeps; escalate daily limits."""
    for i in range(attempts):
        try:
            return fn(*args)
        except Exception as e:
            if is_daily_limit(e):
                raise DailyLimitError(str(e)[:300]) from e
            if not is_rate_limit(e) or i == attempts - 1:
                raise
            wait = 20 * (i + 1)
            print(f"  per-minute rate limit, sleeping {wait}s")
            time.sleep(wait)


def read_jsonl(path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["key"]: r for r in rows}


def append_jsonl(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def configs_to_run() -> list[EvalConfig]:
    grid = json.loads((REPORTS / "grid_results.json").read_text())
    best = [EvalConfig(**r["config"]) for r in grid["best"][:2]]
    v2 = EvalConfig(**{**best[0].params(), "prompt_version": "v2"})
    out = [BASELINE, *best, v2]
    return list(dict.fromkeys(out))  # keep order, drop duplicates


# ---------------- generation ----------------
def generate_one(rag, q: dict) -> dict:
    from app.schemas.ask import AskRequest

    sid = uuid.uuid4().hex
    for i in range(0, len(q["history"]), 2):  # seed memory so the real rewrite step runs
        rag.memory.append(sid, q["history"][i]["content"], q["history"][i + 1]["content"], summarize=False)
    res = [
        d
        for n, d in rag.run(AskRequest(question=q["question"], session_id=sid, role=q["role"], stream=False))
        if n == "result"
    ][0]
    retriever = rag.retriever
    by_id = {c.chunk_id: c for c in retriever.store.all_chunks(q["role"])}
    return {
        "answer": res.answer,
        "found": res.found,
        "rewritten": res.rewritten_question,
        "contexts": [by_id[r["chunk_id"]].text for r in res.retrieved if r["chunk_id"] in by_id],
        "context_ids": [f"{r['doc_id']}|{r['section']}" for r in res.retrieved],
        "citations": [f"{c.doc_id}|{c.section}" for c in res.citations],
        "invalid_citations": res.invalid_citations,
        "timings": res.timings_ms,
        "tokens": res.tokens.model_dump(),
        "guardrails": res.guardrails,
    }


def run_generation(cfg: EvalConfig, golden: list[dict]) -> dict[str, dict]:
    import fakeredis

    from app.agent.memory import SessionMemory
    from app.rag.pipeline import RagPipeline

    path = CKPT / f"{cfg.name}.jsonl"
    done = read_jsonl(path)
    r = fakeredis.FakeRedis()
    rag = RagPipeline(make_retriever(cfg), SessionMemory(r), r, cfg.prompt_version, use_cache=False)
    for q in golden:
        if q["id"] in done:
            continue
        row = with_rate_limit_handling(generate_one, rag, q)
        row["key"] = q["id"]
        append_jsonl(path, row)
        done[q["id"]] = row
        print(f"  [{cfg.name}] {q['id']} found={row['found']} {row['timings'].get('total', 0):.0f}ms")
    return done


# ---------------- judging (RAGAS) ----------------
def ragas_setup():
    import sys as _s
    import types

    # RAGAS 0.4.3 imports langchain_community.chat_models.vertexai, removed in langchain-community 0.4.
    # We never use VertexAI, so register empty stand-ins before importing ragas.
    for name in ("langchain_community.chat_models.vertexai", "langchain_community.llms.vertexai"):
        mod = types.ModuleType(name)
        mod.ChatVertexAI = type("ChatVertexAI", (), {})
        mod.VertexAI = type("VertexAI", (), {})
        _s.modules.setdefault(name, mod)

    from langchain_community.cache import SQLiteCache
    from langchain_core.embeddings import Embeddings
    from langchain_core.globals import set_llm_cache
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, IDBasedContextPrecision, ResponseRelevancy

    from app.llm.client import groq_model
    from app.retrieval.embeddings import get_embedder

    JUDGE.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(JUDGE / "langchain_llm_cache.db")))

    class LocalEmbeddings(Embeddings):
        """Local bge embeddings for ResponseRelevancy (no embedding API calls)."""

        def __init__(self):
            self.m = get_embedder().model

        def embed_documents(self, texts):
            return self.m.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    judge = LangchainLLMWrapper(groq_model(get_settings().judge_model, temperature=0.0, max_tokens=2048))
    emb = LangchainEmbeddingsWrapper(LocalEmbeddings())
    metrics = [Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=emb, strictness=1)]
    return metrics, IDBasedContextPrecision(), judge, emb


def judge_one(setup, q: dict, gen: dict) -> dict:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.run_config import RunConfig

    llm_metrics, id_precision, judge, emb = setup
    sample = SingleTurnSample(
        user_input=q.get("standalone") or q["question"],
        response=gen["answer"],
        retrieved_contexts=gen["contexts"] or [""],
        retrieved_context_ids=gen["context_ids"],
        reference_context_ids=[f"{d}|{s}" for d, s in q["expected_sources"]],
        reference=q["reference_answer"],
    )
    result = evaluate(
        EvaluationDataset([sample]),
        metrics=[*llm_metrics, id_precision],
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=3, max_wait=60),
        raise_exceptions=True,
        show_progress=False,
    )
    scores = result.to_pandas().iloc[0].to_dict()
    pick = {
        "faithfulness": scores.get("faithfulness"),
        "answer_relevancy": scores.get("answer_relevancy"),
        "context_precision": scores.get("id_based_context_precision"),
    }
    return {k: (None if v is None or v != v else round(float(v), 4)) for k, v in pick.items()}


def run_judging(cfg: EvalConfig, golden: list[dict], gens: dict[str, dict], setup) -> dict[str, dict]:
    path = JUDGE / f"{cfg.name}.jsonl"
    cache = read_jsonl(path)
    out = {}
    for q in golden:
        g = gens.get(q["id"])
        if not g or not q["answerable"] or not g["found"]:
            continue  # refusals are scored by no-answer accuracy, not RAGAS
        key = f"{q['id']}:{hashlib.sha256(g['answer'].encode()).hexdigest()[:16]}"
        if key not in cache:
            scores = with_rate_limit_handling(judge_one, setup, q, g)
            cache[key] = {"key": key, **scores}
            append_jsonl(path, cache[key])
            print(f"  [judge {cfg.name}] {q['id']} {scores}")
        out[q["id"]] = cache[key]
    return out


# ---------------- aggregation ----------------
def summarise(cfg: EvalConfig, golden: list[dict], gens: dict, judged: dict) -> dict:
    from app.guardrails.output import RestrictedVocabulary, check_output

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    vocab = RestrictedVocabulary(make_retriever(cfg).store)
    rows = [(q, gens[q["id"]]) for q in golden if q["id"] in gens]
    no_answer_ok = [(g["found"] == q["answerable"]) for q, g in rows]
    leaks = [q["id"] for q, g in rows if check_output(g["answer"], vocab.for_role(q["role"], 0))]
    totals = [g["timings"].get("total", 0) for _, g in rows]
    tokens = [g["tokens"]["total_tokens"] for _, g in rows]
    costs = [g["tokens"]["cost_usd"] for _, g in rows if g["tokens"]["cost_usd"] is not None]
    answerable_found = [q["id"] for q, g in rows if q["answerable"] and g["found"]]
    return {
        "config": cfg.params(),
        "n_generated": len(rows),
        "n_judged": len(judged),
        "n_judgeable": len(answerable_found),
        "faithfulness": mean(j["faithfulness"] for j in judged.values()),
        "answer_relevancy": mean(j["answer_relevancy"] for j in judged.values()),
        "context_precision": mean(j["context_precision"] for j in judged.values()),
        "no_answer_accuracy": round(sum(no_answer_ok) / len(rows), 4) if rows else None,
        "answer_leaks": leaks,
        "invalid_citations_total": sum(len(g["invalid_citations"]) for _, g in rows),
        "mean_total_ms": mean(totals),
        "mean_tokens": mean(tokens),
        "mean_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
    }


def collect_results(configs: list[EvalConfig], golden: list[dict]) -> list[dict]:
    results = []
    for cfg in configs:
        gens = read_jsonl(CKPT / f"{cfg.name}.jsonl")
        if not gens:
            continue
        judged = {}
        for key, row in read_jsonl(JUDGE / f"{cfg.name}.jsonl").items():
            qid, h = key.split(":")
            if qid in gens and hashlib.sha256(gens[qid]["answer"].encode()).hexdigest()[:16] == h:
                judged[qid] = row
        results.append(summarise(cfg, golden, gens, judged))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N golden questions (smoke test)")
    args = ap.parse_args()
    s = get_settings()
    if not s.groq_api_key:
        sys.exit("GROQ_API_KEY is not set: generation eval needs it. Retrieval evals run without a key.")
    s.llm_fallbacks_enabled = False
    s.fallback_provider_key = ""

    golden = load_golden()[: args.limit or None]
    configs = configs_to_run()
    print("configs:", [c.name for c in configs], "| production:", PRODUCTION.name)
    status = "complete"
    try:
        setup = ragas_setup()
        for cfg in configs:
            gens = run_generation(cfg, golden)
            run_judging(cfg, golden, gens, setup)
    except DailyLimitError as e:
        status = "partial"
        print(
            f"\nDAILY RATE LIMIT reached: progress saved in {CKPT} and {JUDGE}.\n"
            f"Re-run the same command tomorrow to resume.\n{e}"
        )
    finally:
        results = collect_results(configs, golden)  # checkpoints are the source of truth (works for partial runs)
        complete = len(results) == len(configs) and all(
            r["n_generated"] == len(golden) and r["n_judged"] == r["n_judgeable"] for r in results
        )
        status = "complete" if complete else "partial"
        write_report(
            "generation_eval.json",
            {
                "status": status,
                "n_questions": len(golden),
                "generator": s.groq_chat_model,
                "judge": s.judge_model,
                "configs": results,
            },
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "configs": [
                        {
                            k: r[k]
                            for k in (
                                "n_generated",
                                "n_judged",
                                "faithfulness",
                                "answer_relevancy",
                                "context_precision",
                                "no_answer_accuracy",
                            )
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
    if status == "partial":
        sys.exit(3)


if __name__ == "__main__":
    main()
