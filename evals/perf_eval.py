"""Performance eval: p50/p95 latency per stage, tokens + estimated cost per query, cache hit rate.

    uv run python evals/perf_eval.py           # needs GROQ_API_KEY (generation)
    uv run python evals/perf_eval.py --n 20    # questions in the replay (default 20, to save free-tier quota)

Replay through the production pipeline WITH the answer cache on:
  pass 1: N golden questions (cache misses -> full pipeline)
  pass 2: the same N questions again (expected: exact-cache hits)
  pairs:  each paraphrase pair from reports/semantic_cache_calibration.json, original first,
          then the paraphrase (semantic hit only if cosine >= threshold)
Without a key, it reports retrieval-stage latency only (from a no-LLM retrieval run).

Writes reports/perf_eval.json.
"""

import argparse
import json
import uuid
from collections import defaultdict

import fakeredis
from common import PRODUCTION, REPORTS, load_golden, make_retriever, percentile, run_retrieval, write_report

from app.agent.memory import SessionMemory
from app.config import get_settings
from app.rag.pipeline import RagPipeline
from app.schemas.ask import AskRequest


def stage_stats(rows: list[dict]) -> dict:
    stages = defaultdict(list)
    for r in rows:
        for k, v in r["timings"].items():
            stages[k].append(v)
    return {
        k: {"p50": round(percentile(sorted(v), 50), 1), "p95": round(percentile(sorted(v), 95), 1), "n": len(v)}
        for k, v in stages.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    s = get_settings()

    if not s.groq_api_key:
        rows = run_retrieval(PRODUCTION, load_golden())
        write_report(
            "perf_eval.json",
            {
                "mode": "retrieval_only (no GROQ_API_KEY)",
                "config": PRODUCTION.params(),
                "stages_ms": stage_stats(rows),
                "total_retrieval_ms": {
                    "p50": round(percentile(sorted(r["latency_ms"] for r in rows), 50), 1),
                    "p95": round(percentile(sorted(r["latency_ms"] for r in rows), 95), 1),
                },
            },
        )
        print("no key: wrote retrieval-only latency to reports/perf_eval.json")
        return

    s.llm_fallbacks_enabled = False
    r = fakeredis.FakeRedis()
    rag = RagPipeline(make_retriever(PRODUCTION), SessionMemory(r), r, s.prompt_version, use_cache=True)
    golden = [q for q in load_golden() if q["answerable"]][: args.n]
    calib = json.loads((REPORTS / "semantic_cache_calibration.json").read_text())
    pairs = [(p["a"], p["b"]) for p in calib["paraphrases"]]

    def ask(question: str, role: str = "employee") -> dict:
        res = [
            d
            for n, d in rag.run(
                AskRequest(question=question, session_id=uuid.uuid4().hex, role=role, stream=False), use_memory=False
            )
            if n == "result"
        ][0]
        return {
            "question": question,
            "cache": res.cache,
            "found": res.found,
            "timings": res.timings_ms,
            "tokens": res.tokens.total_tokens,
            "cost": res.tokens.cost_usd,
        }

    replay = []
    for phase, items in (
        ("pass1", [(q.get("standalone") or q["question"], q["role"]) for q in golden]),
        ("pass2", [(q.get("standalone") or q["question"], q["role"]) for q in golden]),
        ("paraphrase_originals", [(a, "employee") for a, _ in pairs]),
        ("paraphrases", [(b, "employee") for _, b in pairs]),
    ):
        for question, role in items:
            row = ask(question, role)
            row["phase"] = phase
            replay.append(row)
            print(f"{phase:22s} cache={row['cache']:8s} {row['timings']['total']:7.0f}ms tokens={row['tokens']}")

    misses = [x for x in replay if x["cache"] == "none"]
    hits = [x for x in replay if x["cache"] != "none"]
    costs = [x["cost"] for x in misses if x["cost"] is not None]
    report = {
        "mode": "full pipeline",
        "config": PRODUCTION.params(),
        "generator": s.groq_chat_model,
        "requests": len(replay),
        "cache_hit_rate": round(len(hits) / len(replay), 4),
        "cache_hits_by_kind": {k: sum(x["cache"] == k for x in replay) for k in ("exact", "semantic")},
        "semantic_hits_on_paraphrases": sum(x["cache"] == "semantic" for x in replay if x["phase"] == "paraphrases"),
        "n_paraphrases": len(pairs),
        "uncached_stages_ms": stage_stats(misses),
        "cached_total_ms": {
            "p50": round(percentile(sorted(x["timings"]["total"] for x in hits), 50), 1) if hits else None,
            "p95": round(percentile(sorted(x["timings"]["total"] for x in hits), 95), 1) if hits else None,
        },
        "tokens_per_uncached_query": round(sum(x["tokens"] for x in misses) / max(1, len(misses)), 1),
        "est_cost_per_uncached_query_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "est_cost_per_query_with_cache_usd": round(sum(costs) / len(replay), 6) if costs else None,
        "replay": replay,
    }
    write_report("perf_eval.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("replay", "uncached_stages_ms")}, indent=2))


if __name__ == "__main__":
    main()
