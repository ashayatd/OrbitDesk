"""Output guardrails, applied to every answer before it leaves the server.

1. Restricted-content leak check. Retrieval is already role-filtered, so restricted
   text never reaches the prompt. This is defence in depth against anything that
   bypasses that (memory, a cached answer, a future bug). We build the vocabulary
   that appears ONLY in documents the role may not see (e.g. "114,000", "OrbitVault")
   and block an answer that contains several of those terms.
2. System-prompt leak check: the answer must not echo our instructions.
"""

import re
import threading

from app.retrieval.vectorstore.base import VectorStore

BLOCKED_REPLY = "I can't share that information."
_TOKEN = re.compile(r"[a-z0-9][a-z0-9,.-]*[a-z0-9]", re.I)
_PROMPT_MARKERS = ("you are orbitdesk", "text inside <context>", "---meta---", "follow_up_suggestions")
LEAK_THRESHOLD = 3  # distinct restricted-only terms (numbers count double)


def _terms(text: str) -> set[str]:
    return {t.lower().rstrip(".,") for t in _TOKEN.findall(text) if len(t) >= 4}


class RestrictedVocabulary:
    """Per (role, index_version): terms that appear only in chunks the role cannot see."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._cache: dict[tuple[str, int], set[str]] = {}
        self._lock = threading.Lock()

    def for_role(self, role: str, index_version: int) -> set[str]:
        key = (role, index_version)
        with self._lock:
            if key not in self._cache:
                visible = {c.chunk_id for c in self.store.all_chunks(role)}
                allowed_terms: set[str] = set()
                restricted_terms: set[str] = set()
                for c in self.store.all_chunks(None):
                    (allowed_terms if c.chunk_id in visible else restricted_terms).update(_terms(c.text))
                self._cache = {k: v for k, v in self._cache.items() if k[1] == index_version}
                self._cache[key] = restricted_terms - allowed_terms
            return self._cache[key]


def check_output(answer: str, restricted_terms: set[str]) -> str | None:
    """Return a block reason, or None if the answer is fine."""
    low = answer.lower()
    if any(m in low for m in _PROMPT_MARKERS):
        return "system_prompt_leak"
    hits = _terms(answer) & restricted_terms
    weight = sum(2 if any(ch.isdigit() for ch in h) else 1 for h in hits)
    if weight >= LEAK_THRESHOLD:
        return f"restricted_content({len(hits)} terms)"
    return None
