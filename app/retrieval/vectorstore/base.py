"""VectorStore interface. Chroma is the main store; FAISS implements the same
interface so the eval grid can swap them with one setting."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.ingest.models import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str = ""  # "vector" | "bm25" | "rrf" | "rerank"


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    def delete(self, chunk_ids: list[str]) -> None: ...

    def ids_for_doc(self, doc_id: str) -> list[str]: ...

    def query(self, embedding: np.ndarray, k: int, role: str) -> list[ScoredChunk]:
        """Top-k by cosine similarity among chunks the role may see.
        The role filter is applied INSIDE the search, never after it."""
        ...

    def all_chunks(self, role: str | None = None) -> list[Chunk]:
        """Every chunk (optionally only those visible to `role`); used to build BM25."""
        ...

    def count(self) -> int: ...
