"""Hybrid retriever: role pre-filter -> (vector + BM25) -> RRF -> cross-encoder rerank -> threshold.

Every knob the eval grid varies lives in RetrievalConfig, so the same class
serves production (/ask, /search) and the experiments in evals/grid.py.
"""

import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.retrieval.bm25 import BM25Registry
from app.retrieval.embeddings import Embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.vectorstore.base import ScoredChunk, VectorStore


@dataclass
class RetrievalConfig:
    candidate_k: int = 20
    top_k: int = 5
    use_hybrid: bool = True
    use_rerank: bool = True
    rerank_threshold: float = -4.0  # cross-encoder logit
    vector_threshold: float = 0.55  # cosine, used only when rerank is off

    @classmethod
    def from_settings(cls) -> "RetrievalConfig":
        s = get_settings()
        return cls(s.candidate_k, s.top_k, s.use_hybrid, s.use_rerank, s.relevance_threshold)


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    found: bool  # False -> answer "I couldn't find this in the policies"
    top_score: float | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: int = 0


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig.from_settings()
        self.bm25 = BM25Registry(store)

    def embed(self, query: str):
        return self.embedder.embed_query(query)

    def retrieve(self, query: str, role: str, index_version: int = 0, qvec=None) -> RetrievalResult:
        """qvec: pass the query embedding if the caller already computed it (semantic cache lookup)."""
        cfg = self.config
        t: dict[str, float] = {}

        if qvec is None:
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            t["embed_query"] = _ms(t0)

        t0 = time.perf_counter()
        vector_hits = self.store.query(qvec, cfg.candidate_k, role)
        t["vector_search"] = _ms(t0)

        if cfg.use_hybrid:
            t0 = time.perf_counter()
            bm25_hits = self.bm25.for_role(role, index_version).search(query, cfg.candidate_k)
            t["bm25_search"] = _ms(t0)
            t0 = time.perf_counter()
            candidates = reciprocal_rank_fusion([vector_hits, bm25_hits])[: cfg.candidate_k]
            t["fusion"] = _ms(t0)
        else:
            candidates = vector_hits

        if cfg.use_rerank and self.reranker is not None:
            t0 = time.perf_counter()
            top = self.reranker.rerank(query, candidates, cfg.top_k)
            t["rerank"] = _ms(t0)
            top_score = top[0].score if top else None
            found = top_score is not None and top_score >= cfg.rerank_threshold
        else:
            # Without a reranker the only calibrated-ish score is cosine similarity.
            top = candidates[: cfg.top_k]
            top_score = vector_hits[0].score if vector_hits else None
            found = top_score is not None and top_score >= cfg.vector_threshold

        return RetrievalResult(top, found, top_score, t, len(candidates))


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
