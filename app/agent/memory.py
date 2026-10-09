"""Redis-backed per-session chat memory with a token budget.

Layout:
  mem:{sid}:turns    Redis list of JSON messages {"role", "content"}, oldest first
  mem:{sid}:summary  rolling summary of everything evicted from the list

Policy after each turn:
  1. keep at most `recent_turns` user+assistant pairs verbatim;
  2. if summary + turns exceed `token_budget`, evict the oldest pair too;
  3. evicted messages are folded into the summary by the small LLM (or, with no
     LLM available, a truncated extractive summary), under a per-session Redis
     lock so two concurrent requests can't both rewrite the summary.

Tokens are approximated as chars/4: Groq hosts several model families with
different tokenizers, and the budget only needs to be conservative, not exact.
"""

import json

import redis
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.infra.locks import LockNotAcquiredError, distributed_lock
from app.log import get_logger

log = get_logger(__name__)
TTL_S = 24 * 3600
SUMMARY_MAX_CHARS = 1200


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class SessionMemory:
    def __init__(
        self, r: redis.Redis, token_budget: int | None = None, recent_turns: int | None = None
    ) -> None:
        s = get_settings()
        self.r = r
        self.token_budget = token_budget or s.memory_token_budget
        self.recent_turns = recent_turns or s.memory_recent_turns

    def _k(self, sid: str, part: str) -> str:
        return f"mem:{sid}:{part}"

    def history(self, sid: str) -> list[dict]:
        return [json.loads(m) for m in self.r.lrange(self._k(sid, "turns"), 0, -1)]

    def summary(self, sid: str) -> str:
        v = self.r.get(self._k(sid, "summary"))
        return v.decode() if isinstance(v, bytes) else (v or "")

    def context(self, sid: str) -> tuple[str, list[dict]]:
        """(summary, recent messages) to give the LLM. Always within the token budget."""
        summary, turns = self.summary(sid), self.history(sid)
        while (
            turns
            and approx_tokens(summary) + sum(approx_tokens(m["content"]) for m in turns) > self.token_budget
        ):
            turns = turns[2:]  # defensive: trim if a compaction hasn't run yet
        return summary, turns

    def append(self, sid: str, user: str, assistant: str, summarize: bool = True) -> None:
        key = self._k(sid, "turns")
        pipe = self.r.pipeline()
        pipe.rpush(key, json.dumps({"role": "user", "content": user}))
        pipe.rpush(key, json.dumps({"role": "assistant", "content": assistant}))
        pipe.expire(key, TTL_S)
        pipe.execute()
        if summarize:
            self.compact(sid)

    def _over_budget(self, sid: str) -> bool:
        turns = self.history(sid)
        too_many = len(turns) > 2 * self.recent_turns
        too_big = (
            approx_tokens(self.summary(sid)) + sum(approx_tokens(m["content"]) for m in turns)
            > self.token_budget
        )
        return len(turns) > 2 and (too_many or too_big)

    def compact(self, sid: str) -> None:
        """Evict oldest pairs into the summary until within limits."""
        if not self._over_budget(sid):
            return
        try:
            with distributed_lock(self.r, f"mem:{sid}", ttl_s=60, wait_s=0.5):
                evicted: list[dict] = []
                while self._over_budget(sid):
                    raw = self.r.lpop(self._k(sid, "turns"), 2) or []
                    evicted += [json.loads(m) for m in raw]
                if evicted:
                    new = summarize(self.summary(sid), evicted)
                    self.r.set(self._k(sid, "summary"), new, ex=TTL_S)
        except LockNotAcquiredError:
            log.info("memory_compact_skipped", session_id=sid)  # another request is compacting

    def clear(self, sid: str) -> None:
        self.r.delete(self._k(sid, "turns"), self._k(sid, "summary"))


def summarize(previous: str, messages: list[dict]) -> str:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    if get_settings().llm_available:
        try:
            from app.llm.client import get_llm

            resp = get_llm("small", max_tokens=200).invoke(
                [
                    SystemMessage(
                        "Update the running summary of a helpdesk conversation. Keep facts the user "
                        "shared (their situation, ticket ids, dates) and topics asked. Max 120 words. "
                        "Do not include personal data like emails or phone numbers."
                    ),
                    HumanMessage(f"Current summary:\n{previous or '(empty)'}\n\nNew messages:\n{convo}"),
                ]
            )
            return str(resp.content).strip()[:SUMMARY_MAX_CHARS]
        except Exception:
            log.exception("memory_summarize_failed")
    # Extractive fallback: keep the user's questions only.
    asked = "; ".join(m["content"][:120] for m in messages if m["role"] == "user")
    return (f"{previous} Earlier the user asked: {asked}".strip())[-SUMMARY_MAX_CHARS:]
