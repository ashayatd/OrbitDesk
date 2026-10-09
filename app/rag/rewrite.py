"""Rewrite follow-up / vague questions into standalone search queries.

A cheap heuristic decides whether to call the LLM at all, so a clear standalone
question costs zero extra tokens and zero extra latency.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_llm
from app.llm.usage import Usage
from app.rag.prompts import load_prompt

_FOLLOWUP = re.compile(
    r"\b(it|its|that|this|those|these|they|them|there|same|other one|what about|how about|and if|also)\b",
    re.I,
)


def needs_rewrite(question: str, history: list[dict]) -> bool:
    words = len(question.split())
    if history and (words <= 6 or _FOLLOWUP.search(question)):
        return True
    return words <= 2  # "vpn broken", "leave?"


def format_history(history: list[dict], max_turns: int = 4) -> str:
    lines = [f"{m['role']}: {m['content'][:400]}" for m in history[-2 * max_turns :]]
    return "\n".join(lines) or "(no previous messages)"


def rewrite_question(question: str, history: list[dict], usage: Usage) -> str:
    p = load_prompt("rewrite", "v1")
    msgs = [
        SystemMessage(p["system"]),
        HumanMessage(p["user_template"].format(history=format_history(history), question=question)),
    ]
    resp = get_llm("small", max_tokens=80).invoke(msgs)
    usage.add_message("rewrite", resp, prompt_chars=sum(len(m.content) for m in msgs))
    text = str(resp.content).strip().strip('"').splitlines()[0] if str(resp.content).strip() else ""
    return text or question
