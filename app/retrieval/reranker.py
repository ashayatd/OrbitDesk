"""Cross-encoder reranker (local, CPU).

A bi-encoder (the embedding model) encodes query and chunk separately, which is
fast but coarse. A cross-encoder reads (query, chunk) together and outputs one
relevance logit, which is much more accurate but too slow to run over the whole
corpus. So: retrieve ~20 candidates cheaply, then rerank them precisely.
"""

from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.retrieval.vectorstore.base import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c.chunk.text) for c in candidates], batch_size=32)
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
        return [ScoredChunk(c.chunk, float(s), "rerank") for c, s in ranked[:top_k]]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(get_settings().reranker_model)
