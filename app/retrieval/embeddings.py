"""Local sentence-transformers embeddings (no API key needed).

bge-small-en-v1.5: 384 dimensions, 512-token max input. BGE recommends a short
instruction prefix on *queries* (not documents) for retrieval.
"""

from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import get_settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int
    max_tokens: int

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def count_tokens(self, text: str) -> int: ...


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_embedding_dimension()
        self.max_tokens = self.model.max_seq_length
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """L2-normalised, so cosine similarity == dot product (needed for FAISS IndexFlatIP)."""
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._query_prefix + text], batch_size=1)[0]

    def count_tokens(self, text: str) -> int:
        """Token count including [CLS]/[SEP], which also count against the 512 limit."""
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])


@lru_cache
def get_embedder() -> LocalEmbedder:
    return LocalEmbedder(get_settings().embedding_model)
