"""Chroma implementation.

Two modes:
- CHROMA_URL set (compose / Space): HttpClient to a Chroma server, so the API and
  the worker see the same index without sharing an embedded database.
- otherwise: embedded PersistentClient on disk (single-process dev, tests, evals).
"""

from pathlib import Path
from urllib.parse import urlparse

import chromadb
import numpy as np

from app.ingest.models import Chunk
from app.retrieval.vectorstore.base import ScoredChunk


class ChromaStore:
    def __init__(self, collection: str = "orbitdesk", path: Path | None = None, url: str = "") -> None:
        if url:
            u = urlparse(url)
            self.client = chromadb.HttpClient(host=u.hostname, port=u.port or 8000)
        elif path is not None:
            self.client = chromadb.PersistentClient(path=str(path))
        else:
            self.client = chromadb.EphemeralClient()
        # cosine distance; we pass our own (normalised) embeddings, Chroma never embeds.
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self.col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[c.metadata for c in chunks],
        )

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.col.delete(ids=chunk_ids)

    def ids_for_doc(self, doc_id: str) -> list[str]:
        return self.col.get(where={"doc_id": doc_id}, include=[])["ids"]

    def query(self, embedding: np.ndarray, k: int, role: str) -> list[ScoredChunk]:
        res = self.col.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            where={f"role_{role}": True},  # pre-filter: HNSW search only over allowed chunks
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            out.append(ScoredChunk(Chunk(cid, meta["doc_id"], doc, dict(meta)), 1.0 - dist, "vector"))
        return out

    def all_chunks(self, role: str | None = None) -> list[Chunk]:
        where = {f"role_{role}": True} if role else None
        res = self.col.get(where=where, include=["documents", "metadatas"])
        return [
            Chunk(cid, meta["doc_id"], doc, dict(meta))
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    def count(self) -> int:
        return self.col.count()
