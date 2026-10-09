"""Benchmark embedding throughput: one-by-one vs batched, on the real corpus chunks.

uv run python scripts/bench_embeddings.py  ->  reports/embedding_benchmark.json
"""

import json
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingest.chunker import chunk_document  # noqa: E402
from app.ingest.parsers import parse_file  # noqa: E402
from app.ingest.pipeline import load_manifest  # noqa: E402
from app.retrieval.embeddings import get_embedder  # noqa: E402

REPEATS = 3
BATCH_SIZES = [8, 32, 64]


def corpus_texts() -> list[str]:
    s = get_settings()
    emb = get_embedder()
    texts = []
    for meta in load_manifest(s.docs_dir / "manifest.yaml"):
        chunks = chunk_document(
            parse_file(s.docs_dir / meta.file), meta, "bench", emb.count_tokens, s.chunk_size
        )
        texts += [c.text for c in chunks]
    return texts


def time_it(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    import torch

    emb = get_embedder()
    texts = corpus_texts()
    emb.embed_documents(texts[:8])  # warm-up (lazy init, kernel selection)

    runs: dict[str, list[float]] = {"one_by_one": []}
    for _ in range(REPEATS):
        runs["one_by_one"].append(time_it(lambda: [emb.embed_documents([t], batch_size=1) for t in texts]))
    for bs in BATCH_SIZES:
        runs[f"batch_{bs}"] = [
            time_it(lambda bs=bs: emb.embed_documents(texts, batch_size=bs)) for _ in range(REPEATS)
        ]

    results = {}
    for name, secs in runs.items():
        med = statistics.median(secs)
        results[name] = {"median_s": round(med, 3), "chunks_per_s": round(len(texts) / med, 1)}
    base = results["one_by_one"]["chunks_per_s"]
    for r in results.values():
        r["speedup_vs_one_by_one"] = round(r["chunks_per_s"] / base, 2)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": emb.model_name,
        "n_chunks": len(texts),
        "repeats": REPEATS,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "results": results,
    }
    out = get_settings().reports_dir / "embedding_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
