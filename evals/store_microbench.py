"""Micro-benchmark of the raw store query (top-20, role filter, no rerank) for each backend.

    EVAL_DATABASE_URL=postgresql+psycopg://... uv run --group eval python evals/store_microbench.py
    -> reports/store_microbench.json

Isolates the vector-store call from the rest of retrieval (embedding, BM25, rerank), which the
grid's end-to-end p50 includes.
"""

import statistics
import time

from common import EvalConfig, build_store, eval_database_url, load_golden, make_retriever, write_report

from app.retrieval.embeddings import get_embedder

REPEATS = 5


def main() -> None:
    emb = get_embedder()
    queries = [q.get("standalone") or q["question"] for q in load_golden()]
    qvecs = [emb.embed_query(q) for q in queries]
    out = {"n_queries": len(queries), "repeats": REPEATS, "k": 20, "role": "employee", "results_ms": {}}
    backends = ["chroma", "faiss"] + (["pgvector"] if eval_database_url() else [])
    for name in backends:
        store = build_store(300, name)
        for v in qvecs[:3]:
            store.query(v, 20, "employee")  # warm-up
        times = []
        for _ in range(REPEATS):
            for v in qvecs:
                t0 = time.perf_counter()
                store.query(v, 20, "employee")
                times.append((time.perf_counter() - t0) * 1000)
        out["results_ms"][name] = {
            "median": round(statistics.median(times), 3),
            "p95": round(sorted(times)[int(0.95 * len(times))], 3),
        }
        if name == "pgvector":  # the in-database hybrid query (vector + full-text + RRF) for comparison
            times = []
            for _ in range(REPEATS):
                for q, v in zip(queries, qvecs, strict=True):
                    t0 = time.perf_counter()
                    store.hybrid_query(v, q, 20, "employee")
                    times.append((time.perf_counter() - t0) * 1000)
            out["results_ms"]["pgvector_hybrid_sql"] = {
                "median": round(statistics.median(times), 3),
                "p95": round(sorted(times)[int(0.95 * len(times))], 3),
            }
    li = make_retriever(EvalConfig(store="llamaindex", rerank=False, hybrid=False))
    vec = li.index.as_retriever(similarity_top_k=20)
    times = []
    for _ in range(REPEATS):
        for q in queries:
            t0 = time.perf_counter()
            vec.retrieve(q)  # includes LlamaIndex's own query embedding call
            times.append((time.perf_counter() - t0) * 1000)
    out["results_ms"]["llamaindex_vector_incl_embedding"] = {
        "median": round(statistics.median(times), 3),
        "p95": round(sorted(times)[int(0.95 * len(times))], 3),
    }
    if not eval_database_url():
        out["pgvector"] = "not run (no EVAL_DATABASE_URL / DATABASE_URL)"
    write_report("store_microbench.json", out)
    print(out["results_ms"])


if __name__ == "__main__":
    main()
