"""Vector store factory: VECTOR_BACKEND=auto|chroma|faiss|pgvector.

auto = pgvector when DATABASE_URL is set (one database for data + vectors + keyword search),
otherwise Chroma (the no-Postgres fallback).
"""

from functools import lru_cache

from app.config import get_settings
from app.retrieval.vectorstore.base import ScoredChunk, VectorStore

__all__ = ["ScoredChunk", "VectorStore", "get_store"]


def resolve_backend() -> str:
    s = get_settings()
    if s.vector_backend == "auto":
        return "pgvector" if s.database_url else "chroma"
    return s.vector_backend


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    backend = resolve_backend()
    if backend == "pgvector":
        from app.retrieval.embeddings import get_embedder
        from app.retrieval.vectorstore.pgvector_store import PgVectorStore

        return PgVectorStore(s.database_url, dim=get_embedder().dim)
    if backend == "faiss":
        from app.retrieval.embeddings import get_embedder
        from app.retrieval.vectorstore.faiss_store import FaissStore

        return FaissStore(dim=get_embedder().dim, path=s.faiss_dir)
    from app.retrieval.vectorstore.chroma_store import ChromaStore

    return ChromaStore(path=s.chroma_dir, url=s.chroma_url)
