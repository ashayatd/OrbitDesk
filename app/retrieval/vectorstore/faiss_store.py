"""FAISS implementation of the same VectorStore interface.

FAISS stores only vectors + int64 ids, so we keep chunk text/metadata in a side
dict and persist both. Exact search (IndexFlatIP on normalised vectors = cosine).
For ~100 chunks exact search is faster than any ANN index and has perfect recall.

Role filtering uses a search-time IDSelector, so the role filter happens inside
the search (pre-filter), the same guarantee Chroma's `where` gives.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.ingest.models import Chunk
from app.retrieval.vectorstore.base import ScoredChunk


class FaissStore:
    def __init__(self, dim: int, path: Path | None = None) -> None:
        self.dim = dim
        self.path = path
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.chunks: dict[int, Chunk] = {}  # faiss int id -> chunk
        self.id_of: dict[str, int] = {}  # chunk_id -> faiss int id
        self._next = 0
        if path and (path / "index.faiss").exists():
            self._load()

    # --- persistence ---
    def _save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path / "index.faiss"))
        side = {
            "next": self._next,
            "chunks": {i: [c.chunk_id, c.doc_id, c.text, c.metadata] for i, c in self.chunks.items()},
        }
        (self.path / "chunks.json").write_text(json.dumps(side), encoding="utf-8")

    def _load(self) -> None:
        self.index = faiss.read_index(str(self.path / "index.faiss"))
        side = json.loads((self.path / "chunks.json").read_text(encoding="utf-8"))
        self._next = side["next"]
        self.chunks = {int(i): Chunk(*v) for i, v in side["chunks"].items()}
        self.id_of = {c.chunk_id: i for i, c in self.chunks.items()}

    # --- VectorStore ---
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.delete([c.chunk_id for c in chunks if c.chunk_id in self.id_of])
        ids = np.arange(self._next, self._next + len(chunks), dtype=np.int64)
        self._next += len(chunks)
        self.index.add_with_ids(np.ascontiguousarray(embeddings, dtype=np.float32), ids)
        for i, c in zip(ids.tolist(), chunks, strict=True):
            self.chunks[i] = c
            self.id_of[c.chunk_id] = i
        self._save()

    def delete(self, chunk_ids: list[str]) -> None:
        ids = [self.id_of.pop(cid) for cid in chunk_ids if cid in self.id_of]
        if ids:
            self.index.remove_ids(np.array(ids, dtype=np.int64))
            for i in ids:
                self.chunks.pop(i, None)
            self._save()

    def ids_for_doc(self, doc_id: str) -> list[str]:
        return [c.chunk_id for c in self.chunks.values() if c.doc_id == doc_id]

    def query(self, embedding: np.ndarray, k: int, role: str) -> list[ScoredChunk]:
        allowed = np.array([i for i, c in self.chunks.items() if c.metadata.get(f"role_{role}")], dtype=np.int64)
        if allowed.size == 0:
            return []
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))
        scores, ids = self.index.search(embedding.reshape(1, -1).astype(np.float32), k, params=params)
        return [
            ScoredChunk(self.chunks[i], float(s), "vector") for s, i in zip(scores[0], ids[0], strict=True) if i != -1
        ]

    def all_chunks(self, role: str | None = None) -> list[Chunk]:
        return [c for c in self.chunks.values() if role is None or c.metadata.get(f"role_{role}")]

    def count(self) -> int:
        return len(self.chunks)
