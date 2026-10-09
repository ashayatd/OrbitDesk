"""Shared eval helpers: golden set, isolated eval indexes, retrieval metrics.

Eval indexes are separate from the app's index (own Chroma dir / FAISS in memory,
own SQLite registry, fake Redis), so experiments never touch production data.
A source matches when (doc_id, section) are equal: that stays stable across chunk
sizes, unlike chunk ids.
"""

import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.ingest.models import ROLES  # noqa: E402
from app.ingest.pipeline import ingest_document, load_manifest  # noqa: E402
from app.retrieval.embeddings import get_embedder  # noqa: E402
from app.retrieval.reranker import get_reranker  # noqa: E402
from app.retrieval.retriever import HybridRetriever, RetrievalConfig  # noqa: E402
from app.retrieval.vectorstore.base import VectorStore  # noqa: E402

EVAL_DIR = ROOT / "evals"
REPORTS = ROOT / "reports"
INDEX_DIR = ROOT / "data" / "eval_indexes"


def load_golden(path: Path = EVAL_DIR / "golden.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class EvalConfig:
    chunk_size: int = 300
    top_k: int = 5
    rerank: bool = True
    hybrid: bool = True
    store: str = "chroma"  # chroma | faiss
    prompt_version: str = "v1"

    @property
    def name(self) -> str:
        return (
            f"cs{self.chunk_size}-k{self.top_k}-{'rr' if self.rerank else 'norr'}-"
            f"{'hyb' if self.hybrid else 'vec'}-{self.store}-{self.prompt_version}"
        )

    def params(self) -> dict:
        return asdict(self)


BASELINE = EvalConfig(chunk_size=300, top_k=5, rerank=False, hybrid=False, store="chroma")
PRODUCTION = EvalConfig(chunk_size=300, top_k=5, rerank=True, hybrid=True, store="chroma")


@lru_cache(maxsize=8)
def build_store(chunk_size: int, store: str) -> VectorStore:
    """Index the corpus once per (chunk_size, store) with an isolated registry + fake Redis."""
    import fakeredis

    from app.retrieval.vectorstore.chroma_store import ChromaStore
    from app.retrieval.vectorstore.faiss_store import FaissStore

    emb = get_embedder()
    if store == "faiss":
        vs: VectorStore = FaissStore(dim=emb.dim)  # in memory
    else:
        vs = ChromaStore(collection=f"eval_cs{chunk_size}", path=INDEX_DIR / "chroma")
    s = get_settings()
    db = INDEX_DIR / f"registry_{store}_{chunk_size}.db"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    r = fakeredis.FakeRedis()
    for meta in load_manifest(s.docs_dir / "manifest.yaml"):
        ingest_document(
            s.docs_dir / meta.file,
            meta,
            store=vs,
            embedder=emb,
            r=r,
            db_path=db,
            chunk_size=chunk_size,
            overlap=max(10, chunk_size * 15 // 100),
            # FAISS lives in memory, so it always needs indexing; Chroma persists between runs.
            force=(store == "faiss"),
        )
    return vs


def make_retriever(cfg: EvalConfig) -> HybridRetriever:
    s = get_settings()
    rc = RetrievalConfig(
        candidate_k=s.candidate_k,
        top_k=cfg.top_k,
        use_hybrid=cfg.hybrid,
        use_rerank=cfg.rerank,
        rerank_threshold=s.relevance_threshold,
    )
    return HybridRetriever(build_store(cfg.chunk_size, cfg.store), get_embedder(), get_reranker(), rc)


def search_text(q: dict) -> str:
    """Retrieval evals use the gold standalone rewrite for follow-ups (no LLM needed)."""
    return q.get("standalone") or q["question"]


def is_match(retrieved: dict, expected: list) -> bool:
    return [retrieved["doc_id"], retrieved["section"]] in expected


def retrieval_metrics(rows: list[dict]) -> dict:
    """rows: one per question with keys retrieved (list of {doc_id, section}), found, latency_ms."""
    answerable = [r for r in rows if r["q"]["answerable"]]
    recalls, rrs = [], []
    for r in answerable:
        exp = r["q"]["expected_sources"]
        hits = {tuple(x) for x in exp if any(is_match(c, [x]) for c in r["retrieved"])}
        recalls.append(len(hits) / len(exp))
        rank = next((i for i, c in enumerate(r["retrieved"], 1) if is_match(c, exp)), None)
        rrs.append(1 / rank if rank else 0.0)
    no_answer_correct = [r["found"] == r["q"]["answerable"] for r in rows]
    unanswerable = [r for r in rows if not r["q"]["answerable"]]
    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "recall_at_k": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(rrs), 4),
        "no_answer_accuracy": round(sum(no_answer_correct) / len(rows), 4),
        "refusal_rate_on_unanswerable": round(sum(not r["found"] for r in unanswerable) / max(1, len(unanswerable)), 4),
        "false_refusal_rate_on_answerable": round(sum(not r["found"] for r in answerable) / len(answerable), 4),
        "leaks": sum(r["leaks"] for r in rows),
        "latency_p50_ms": round(percentile(lat, 50), 1),
        "latency_p95_ms": round(percentile(lat, 95), 1),
        "n_questions": len(rows),
    }


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def run_retrieval(cfg: EvalConfig, golden: list[dict]) -> list[dict]:
    retriever = make_retriever(cfg)
    store = retriever.store
    visible = {role: {c.chunk_id for c in store.all_chunks(role)} for role in ROLES}
    rows = []
    for q in golden:
        t0 = time.perf_counter()
        res = retriever.retrieve(search_text(q), q["role"])
        ms = (time.perf_counter() - t0) * 1000
        retrieved = [
            {"chunk_id": sc.chunk.chunk_id, "doc_id": sc.chunk.doc_id, "section": sc.chunk.section, "score": sc.score}
            for sc in res.chunks
        ]
        leaks = sum(c["chunk_id"] not in visible[q["role"]] for c in retrieved)
        rows.append(
            {
                "q": q,
                "retrieved": retrieved,
                "found": res.found,
                "top_score": res.top_score,
                "latency_ms": ms,
                "leaks": leaks,
                "timings": res.timings_ms,
            }
        )
    return rows


def write_report(name: str, data) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / name
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
