"""Vector store factory: VECTOR_BACKEND=chroma|faiss."""

from functools import lru_cache

from app.config import get_settings
from app.retrieval.vectorstore.base import ScoredChunk, VectorStore

__all__ = ["ScoredChunk", "VectorStore", "get_store"]


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    if s.vector_backend == "faiss":
        from app.retrieval.embeddings import get_embedder
        from app.retrieval.vectorstore.faiss_store import FaissStore

        return FaissStore(dim=get_embedder().dim, path=s.faiss_dir)
    from app.retrieval.vectorstore.chroma_store import ChromaStore

    return ChromaStore(path=s.chroma_dir, url=s.chroma_url)
