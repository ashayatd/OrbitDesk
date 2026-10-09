"""BM25 keyword search (rank-bm25), one index per role.

Each role gets its own index built only from the chunks it may see. That is a
true pre-filter, and it also keeps HR-only words out of the IDF statistics a
regular employee's query is scored against.
"""

import re
import threading

from rank_bm25 import BM25Okapi

from app.ingest.models import Chunk
from app.retrieval.vectorstore.base import ScoredChunk, VectorStore

_STOP = frozenset(
    """a an and are as at be by can do for from how i if in is it my of on or our so that the this
    to what when where which who why will with you your me we""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumerics; keeps codes like 'vpn-809' as 'vpn', '809' and '0x0000007e'."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i]), "bm25") for i in ranked if scores[i] > 0]


class BM25Registry:
    """Lazily builds and caches one BM25Index per role; rebuilt when the index version changes."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._cache: dict[tuple[str, int], BM25Index] = {}
        self._lock = threading.Lock()

    def for_role(self, role: str, index_version: int) -> BM25Index:
        key = (role, index_version)
        with self._lock:
            if key not in self._cache:
                self._cache = {k: v for k, v in self._cache.items() if k[1] == index_version}  # drop stale
                self._cache[key] = BM25Index(self.store.all_chunks(role))
            return self._cache[key]
