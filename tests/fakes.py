"""Test doubles: a deterministic embedder (no model download, millisecond-fast)."""

import hashlib
import re

import numpy as np


class FakeEmbedder:
    """Hashing-trick bag of words: texts sharing words get similar vectors."""

    dim = 64
    max_tokens = 512
    model_name = "fake"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 2
