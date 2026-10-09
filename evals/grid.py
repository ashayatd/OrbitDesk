"""Retrieval experiment grid (NO LLM calls), logged to MLflow.

Factors: chunk size {100, 300} x top-k {3, 5} x rerank {on, off} x hybrid vs vector-only
x Chroma vs FAISS = 32 configs, each scored on the 40 golden questions.
Prompt v1 vs v2 is a generation factor, so it is compared in generation_eval.py instead
(Groq free-tier budget: LLM-judged metrics only for the baseline and the best configs).

    uv run python evals/grid.py   ->  reports/grid_results.json + MLflow experiment "orbitdesk-retrieval-grid"
"""

import itertools
import json

from common import BASELINE, EvalConfig, load_golden, retrieval_metrics, run_retrieval, write_report
from mlflow_utils import init_mlflow

GRID = {
    "chunk_size": [100, 300],
    "top_k": [3, 5],
    "rerank": [True, False],
    "hybrid": [True, False],
    "store": ["chroma", "faiss"],
}


def rank_key(r: dict):
    m = r["metrics"]
    return (-m["recall_at_k"], -m["mrr"], -m["no_answer_accuracy"], m["latency_p50_ms"])


def main() -> None:
    golden = load_golden()
    mlflow = init_mlflow("orbitdesk-retrieval-grid")
    results = []
    for values in itertools.product(*GRID.values()):
        cfg = EvalConfig(**dict(zip(GRID.keys(), values, strict=True)))
        rows = run_retrieval(cfg, golden)
        metrics = retrieval_metrics(rows)
        results.append({"name": cfg.name, "config": cfg.params(), "metrics": metrics})
        with mlflow.start_run(run_name=cfg.name):
            mlflow.log_params(cfg.params())
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
            mlflow.set_tag("baseline", str(cfg == BASELINE))
        print(
            f"{cfg.name:38s} recall@k={metrics['recall_at_k']:.3f} mrr={metrics['mrr']:.3f} "
            f"noans={metrics['no_answer_accuracy']:.3f} p50={metrics['latency_p50_ms']:.0f}ms leaks={metrics['leaks']}"
        )

    results.sort(key=rank_key)
    baseline = next(r for r in results if r["name"] == BASELINE.name)
    report = {
        "ranking_rule": "recall@k desc, MRR desc, no-answer accuracy desc, p50 latency asc",
        "baseline": baseline,
        "best": results[:2],
        "all": results,
    }
    write_report("grid_results.json", report)
    print("\nbest:", json.dumps([r["name"] for r in results[:2]]), "baseline:", BASELINE.name)


if __name__ == "__main__":
    main()
