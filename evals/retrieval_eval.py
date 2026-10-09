"""Retrieval eval + access-control leak test + CI gate. Needs NO API key.

    uv run python evals/retrieval_eval.py            # production config
    uv run python evals/retrieval_eval.py --gate     # also enforce evals/gate.json (CI)

Writes reports/retrieval_eval.json.
"""

import argparse
import json
import sys
from collections import defaultdict

from common import (
    EVAL_DIR,
    PRODUCTION,
    EvalConfig,
    eval_database_url,
    load_golden,
    make_retriever,
    retrieval_metrics,
    run_retrieval,
    write_report,
)

from app.ingest.models import ROLES
from app.retrieval.retriever import RetrievalConfig


def validate_golden(golden: list[dict], store) -> list[str]:
    """Every expected (doc_id, section) must exist in the corpus, or the metric is meaningless."""
    known = {(c.doc_id, c.section) for c in store.all_chunks()}
    return [f"{q['id']}: {tuple(s)}" for q in golden for s in q["expected_sources"] if tuple(s) not in known]


def leak_test(cfg: EvalConfig) -> dict:
    """Adversarial access-control test: for every chunk a role may NOT see, query as that role
    with the chunk's own text, its heading and its first sentence, through vector-only,
    hybrid, and hybrid+rerank retrieval. Any restricted chunk in any result is a leak."""
    base = make_retriever(cfg)
    store = base.store
    visible = {r: {c.chunk_id for c in store.all_chunks(r)} for r in ROLES}
    modes = {
        "vector": RetrievalConfig(top_k=20, use_hybrid=False, use_rerank=False),
        "hybrid": RetrievalConfig(top_k=20, use_hybrid=True, use_rerank=False),
        "hybrid_rerank": RetrievalConfig(top_k=20, use_hybrid=True, use_rerank=True),
    }
    queries = leaks = 0
    leak_examples = []
    for role in ROLES:
        restricted = [c for c in store.all_chunks() if c.chunk_id not in visible[role]]
        for chunk in restricted:
            body = chunk.text.split("\n", 1)[-1]
            probes = [chunk.text, chunk.metadata.get("heading_path", ""), body.split(".")[0]]
            for mode, rc in modes.items():
                base.config = rc
                for probe in probes:
                    queries += 1
                    for sc in base.retrieve(probe, role).chunks:
                        if sc.chunk.chunk_id not in visible[role]:
                            leaks += 1
                            leak_examples.append({"role": role, "mode": mode, "chunk": sc.chunk.chunk_id})
    base.config = RetrievalConfig()
    return {"queries": queries, "leaks": leaks, "examples": leak_examples[:10]}


def threshold_sweep(rows: list[dict]) -> list[dict]:
    """No-answer accuracy at different relevance thresholds (top rerank logit)."""
    out = []
    for t in [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2]:
        correct = sum(((r["top_score"] or -99) >= t) == r["q"]["answerable"] for r in rows)
        out.append({"threshold": t, "no_answer_accuracy": round(correct / len(rows), 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="exit 1 if metrics regress below evals/gate.json")
    args = ap.parse_args()

    cfg = PRODUCTION
    golden = load_golden()
    missing = validate_golden(golden, make_retriever(cfg).store)
    if missing:
        sys.exit(f"golden set references sections that don't exist: {missing}")

    rows = run_retrieval(cfg, golden)
    metrics = retrieval_metrics(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["q"]["type"]].append(r)
    per_type = {
        t: retrieval_metrics(rs)
        if any(x["q"]["answerable"] for x in rs)
        else {"no_answer_accuracy": round(sum(x["found"] == x["q"]["answerable"] for x in rs) / len(rs), 4)}
        for t, rs in by_type.items()
    }
    leak = leak_test(cfg)
    # Same adversarial test on the other stacks: the role filter must hold in every implementation.
    leak_by_backend = {cfg.store: {k: leak[k] for k in ("queries", "leaks")}}
    others = [EvalConfig(store="llamaindex")] + ([EvalConfig(store="pgvector")] if eval_database_url() else [])
    for other in others:
        r = leak_test(other)
        leak_by_backend[other.store] = {k: r[k] for k in ("queries", "leaks")}
    if not eval_database_url():
        leak_by_backend["pgvector"] = "not run (no EVAL_DATABASE_URL / DATABASE_URL)"
    misses = [
        {
            "id": r["q"]["id"],
            "question": r["q"]["question"],
            "role": r["q"]["role"],
            "found": r["found"],
            "top_score": round(r["top_score"], 2) if r["top_score"] is not None else None,
            "retrieved": [f"{c['doc_id']}|{c['section']}" for c in r["retrieved"][:3]],
        }
        for r in rows
        if r["found"] != r["q"]["answerable"]
        or (
            r["q"]["answerable"]
            and not any([c["doc_id"], c["section"]] in r["q"]["expected_sources"] for c in r["retrieved"])
        )
    ]
    stage_ms = defaultdict(list)
    for r in rows:
        for k, v in r["timings"].items():
            stage_ms[k].append(v)
    report = {
        "config": cfg.params(),
        "metrics": metrics,
        "per_type": per_type,
        "leak_test": leak,
        "leak_test_by_backend": leak_by_backend,
        "threshold_sweep": threshold_sweep(rows),
        "stage_latency_ms_median": {k: round(sorted(v)[len(v) // 2], 2) for k, v in stage_ms.items()},
        "misses": misses,
    }
    write_report("retrieval_eval.json", report)
    print(json.dumps({"metrics": metrics, "leak_test_by_backend": leak_by_backend}, indent=2))

    if args.gate:
        gate = json.loads((EVAL_DIR / "gate.json").read_text())
        failures = []
        if metrics["recall_at_k"] < gate["min_recall_at_5"]:
            failures.append(f"recall@5 {metrics['recall_at_k']} < {gate['min_recall_at_5']}")
        if metrics["mrr"] < gate["min_mrr"]:
            failures.append(f"MRR {metrics['mrr']} < {gate['min_mrr']}")
        backend_leaks = sum(v["leaks"] for v in leak_by_backend.values() if isinstance(v, dict))
        if backend_leaks > gate["max_leaks"] or metrics["leaks"] > gate["max_leaks"]:
            failures.append(f"access-control leaks: {backend_leaks + metrics['leaks']}")
        if failures:
            sys.exit("EVAL GATE FAILED: " + "; ".join(failures))
        print("eval gate passed")


if __name__ == "__main__":
    main()
