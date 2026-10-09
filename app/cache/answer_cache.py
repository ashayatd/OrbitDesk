"""Answer cache: exact match first, then semantic (embedding similarity).

Both caches are namespaced by (index version, role, prompt version):
- index version: re-ingesting any document bumps it, so stale answers are never served;
- role: an HR answer must never be served to an employee;
- prompt version: switching PROMPT_VERSION (e.g. a rollback) doesn't serve answers made by the other prompt.

Exact:    cache:exact:{ns}:{sha256(normalised question)} -> AskResponse JSON
Semantic: cache:sem:{ns} -> Redis list of {"v": base64 float32 vector, "k": exact key}
          Lookup is a brute-force cosine scan over at most MAX_ENTRIES vectors
          (500 x 384 floats = one ~0.2 ms numpy matmul). That's simpler than
          RediSearch/vector indexes and fast enough at this scale.
Keys are built from the PII-masked, standalone (rewritten) question.
Semantic hits also require the same numbers/codes in both questions: "VPN error 809"
vs "VPN error 691" are 0.83 similar but need different answers.
Threshold calibration: reports/semantic_cache_calibration.json.
"""

import base64
import hashlib
import json
import re

import numpy as np
import redis

TTL_S = 24 * 3600
MAX_ENTRIES = 500


def normalise(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", q.lower())).strip()


def key_numbers(q: str) -> list[str]:
    return sorted(set(re.findall(r"\d+", q.lower())))


class AnswerCache:
    def __init__(self, r: redis.Redis, semantic_threshold: float = 0.95) -> None:
        self.r = r
        self.threshold = semantic_threshold

    @staticmethod
    def namespace(index_version: int, role: str, prompt_version: str) -> str:
        return f"v{index_version}:{role}:{prompt_version}"

    def _exact_key(self, ns: str, question: str) -> str:
        return f"cache:exact:{ns}:{hashlib.sha256(normalise(question).encode()).hexdigest()}"

    def get_exact(self, ns: str, question: str) -> dict | None:
        raw = self.r.get(self._exact_key(ns, question))
        return json.loads(raw) if raw else None

    def get_semantic(self, ns: str, question: str, qvec: np.ndarray) -> tuple[dict, float] | None:
        entries = self.r.lrange(f"cache:sem:{ns}", 0, -1)
        if not entries:
            return None
        parsed = [json.loads(e) for e in entries]
        mat = np.stack([np.frombuffer(base64.b64decode(p["v"]), dtype=np.float32) for p in parsed])
        sims = mat @ qvec.astype(np.float32)  # vectors are L2-normalised: dot = cosine
        best = int(np.argmax(sims))
        if sims[best] < self.threshold or parsed[best].get("n", []) != key_numbers(question):
            return None
        raw = self.r.get(parsed[best]["k"])
        return (json.loads(raw), float(sims[best])) if raw else None

    def put(self, ns: str, question: str, qvec: np.ndarray | None, response: dict) -> None:
        key = self._exact_key(ns, question)
        self.r.set(key, json.dumps(response), ex=TTL_S)
        if qvec is not None:
            sem_key = f"cache:sem:{ns}"
            vec = base64.b64encode(qvec.astype(np.float32).tobytes()).decode()
            entry = json.dumps({"v": vec, "k": key, "n": key_numbers(question)})
            pipe = self.r.pipeline()
            pipe.lpush(sem_key, entry)
            pipe.ltrim(sem_key, 0, MAX_ENTRIES - 1)  # newest first; oldest fall off
            pipe.expire(sem_key, TTL_S)
            pipe.execute()

    def record(self, outcome: str) -> None:
        """outcome: exact | semantic | miss (feeds the cache hit rate in /admin/stats)."""
        self.r.hincrby("stats:cache", outcome, 1)

    def stats(self) -> dict[str, int]:
        raw = self.r.hgetall("stats:cache")
        return {(k.decode() if isinstance(k, bytes) else k): int(v) for k, v in raw.items()}
