"""Process-wide singletons (models are loaded once per process)."""

from functools import lru_cache

from app.agent.memory import SessionMemory
from app.infra.redis_client import get_redis
from app.rag.pipeline import RagPipeline
from app.retrieval.embeddings import get_embedder
from app.retrieval.reranker import get_reranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vectorstore import get_store


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(get_store(), get_embedder(), get_reranker())


@lru_cache
def get_memory() -> SessionMemory:
    return SessionMemory(get_redis())


@lru_cache
def get_rag_pipeline() -> RagPipeline:
    return RagPipeline(get_retriever(), get_memory(), get_redis())
