"""LlamaIndex retrieval path behind the same interface as HybridRetriever (retrieve() -> RetrievalResult).

Built for a fair comparison with the main path: SAME chunks (read from the main vector store),
SAME local embedding model (bge-small via a BaseEmbedding adapter), SAME role pre-filter
(LlamaIndex MetadataFilters on the role flag; BM25 built per role from allowed nodes only),
SAME cross-encoder and refusal threshold. What differs is the retrieval stack itself:
VectorStoreIndex (SimpleVectorStore), BM25Retriever, QueryFusionRetriever (reciprocal rank),
SentenceTransformerRerank.

Install: uv sync --group llamaindex   (it is not part of the serving image)
"""

import time

from llama_index.core import Settings as LISettings
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever
from pydantic import PrivateAttr

from app.ingest.models import ROLES, Chunk
from app.retrieval.embeddings import Embedder
from app.retrieval.retriever import RetrievalConfig, RetrievalResult
from app.retrieval.vectorstore.base import ScoredChunk, VectorStore


class LocalLIEmbedding(BaseEmbedding):
    """Adapter so LlamaIndex uses OUR embedder (incl. bge's query prefix) instead of downloading its own."""

    _emb: Embedder = PrivateAttr()

    def __init__(self, embedder: Embedder, **kw) -> None:
        super().__init__(model_name="bge-small-local", embed_batch_size=8, **kw)
        self._emb = embedder

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._emb.embed_query(query).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._emb.embed_documents([text])[0].tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts, batch_size=8).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def _node(c: Chunk) -> TextNode:
    meta = dict(c.metadata)
    for r in ROLES:  # LlamaIndex filter values can't be bool: store the role flags as 0/1
        meta[f"role_{r}"] = int(bool(meta.get(f"role_{r}")))
    return TextNode(
        id_=c.chunk_id,
        text=c.text,
        metadata=meta,
        # metadata is for filtering only: keep it out of the embedded / reranked text
        excluded_embed_metadata_keys=list(meta),
        excluded_llm_metadata_keys=list(meta),
    )


class LlamaIndexRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, reranker_model: str, config: RetrievalConfig) -> None:
        LISettings.llm = MockLLM()  # nothing in this path may call an LLM
        self.store = store  # same attribute as HybridRetriever (used by the output guard)
        self.embedder = embedder
        self.config = config
        self.nodes = [_node(c) for c in store.all_chunks()]
        self.index = VectorStoreIndex(self.nodes, embed_model=LocalLIEmbedding(embedder), show_progress=False)
        self.reranker = SentenceTransformerRerank(model=reranker_model, top_n=config.top_k)
        self._bm25 = {
            role: BM25Retriever.from_defaults(
                nodes=[n for n in self.nodes if n.metadata.get(f"role_{role}")],
                similarity_top_k=config.candidate_k,
            )
            for role in ROLES
        }

    def embed(self, query: str):
        return self.embedder.embed_query(query)

    def retrieve(self, query: str, role: str, index_version: int = 0, qvec=None, department=None) -> RetrievalResult:
        cfg = self.config
        t: dict[str, float] = {}
        filters = [ExactMatchFilter(key=f"role_{role}", value=1)]
        if department:
            filters.append(ExactMatchFilter(key="department", value=department))
        vec = self.index.as_retriever(similarity_top_k=cfg.candidate_k, filters=MetadataFilters(filters=filters))

        t0 = time.perf_counter()
        vector_hits = vec.retrieve(query)
        if cfg.use_hybrid:
            fusion = QueryFusionRetriever(
                [vec, self._bm25[role]],
                mode="reciprocal_rerank",
                num_queries=1,  # no LLM query expansion
                similarity_top_k=cfg.candidate_k,
                use_async=False,
                llm=MockLLM(),
            )
            candidates = fusion.retrieve(query)
        else:
            candidates = vector_hits
        t["llamaindex_retrieve"] = round((time.perf_counter() - t0) * 1000, 2)

        if cfg.use_rerank:
            t0 = time.perf_counter()
            self.reranker.top_n = cfg.top_k
            top = self.reranker.postprocess_nodes(candidates, query_bundle=QueryBundle(query))
            t["rerank"] = round((time.perf_counter() - t0) * 1000, 2)
            top_score = float(top[0].score) if top else None
            found = top_score is not None and top_score >= cfg.rerank_threshold
        else:
            top = candidates[: cfg.top_k]
            top_score = float(vector_hits[0].score) if vector_hits else None
            found = top_score is not None and top_score >= cfg.vector_threshold

        chunks = [
            ScoredChunk(
                Chunk(n.node.node_id, n.node.metadata["doc_id"], n.node.get_content(), n.node.metadata),
                float(n.score or 0),
                "llamaindex",
            )
            for n in top
        ]
        return RetrievalResult(chunks, found, top_score, t, len(candidates))
