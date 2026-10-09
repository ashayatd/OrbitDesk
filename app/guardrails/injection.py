"""Prompt-injection / unsafe-request detection.

Layer 1 - heuristics (regex, ~0 ms, no tokens): applied to user input AND to every
          retrieved chunk (a poisoned document is "indirect" prompt injection).
Layer 2 - classifier on user input only, chosen by SAFETY_CLASSIFIER:
          promptguard (default): Llama Prompt Guard 2 86M on Groq, a purpose-built
            injection/jailbreak classifier (free plan, 512-token input, ~$0.04/M tokens);
          llm: the small chat model with prompts/safety_v1.yaml (also flags "unsafe" asks);
          off: heuristics only.
          Verdicts are cached by message hash, so a repeated message costs nothing.

A strong heuristic hit blocks without calling the LLM. Retrieved chunks that trip
the heuristics are dropped from the context instead of failing the request.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.infra.redis_client import get_redis
from app.llm.client import get_llm, groq_model
from app.llm.usage import Usage
from app.log import get_logger
from app.rag.prompts import load_prompt

log = get_logger(__name__)

# (pattern, weight). weight >= 1.0 on its own is a block.
_PATTERNS: list[tuple[re.Pattern, float]] = [
    (
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}"
            r"\b(instructions?|rules?|prompts?|guidelines|context)\b",
            re.I | re.S,
        ),
        1.0,
    ),
    (
        re.compile(
            r"\b(reveal|show|print|repeat|output|leak)\b.{0,30}\b(system|hidden|initial|original)\s+"
            r"(prompt|instructions?|message)",
            re.I | re.S,
        ),
        1.0,
    ),
    (
        re.compile(
            r"\byou are (now|no longer)\b|\bfrom now on,? you\b|\bact as (an? )?(admin|hr|root|developer)", re.I
        ),
        1.0,
    ),
    (re.compile(r"\b(developer|god|dan|jailbreak)\s*mode\b|\bjailbreak\b", re.I), 1.0),
    (re.compile(r"</?\s*(context|system|chunk|assistant|instructions?)\s*>", re.I), 0.8),
    (
        re.compile(
            r"\b(pretend|imagine|roleplay)\b.{0,30}\b(you|i)\b.{0,20}\b(are|am)\b.{0,20}\b(hr|admin|manager)\b",
            re.I | re.S,
        ),
        0.8,
    ),
    (re.compile(r"\bmy role is (hr|it_admin|admin)\b|\bi am (an? )?(hr|it) admin", re.I), 0.6),
    (re.compile(r"\bnew instructions?\b|\bsystem\s*:\s", re.I), 0.6),
    (re.compile(r"\b(base64|rot13)\b.{0,20}\b(decode|instructions?)\b", re.I | re.S), 0.5),
]
# Retrieved text: an instruction aimed at the assistant inside a document is suspicious on its own.
_CHUNK_PATTERNS = _PATTERNS + [
    (re.compile(r"\b(assistant|ai|chatbot|model|llm)\b.{0,30}\b(must|should|always|never)\b", re.I | re.S), 0.6),
]


@dataclass
class SafetyVerdict:
    label: str  # safe | injection | unsafe
    score: float
    source: str  # heuristic | llm | cache | disabled
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.label != "safe"

    def to_dict(self) -> dict:
        return asdict(self)


def heuristic_score(text: str, patterns=_PATTERNS) -> tuple[float, list[str]]:
    hits = [p.pattern[:40] for p, _ in patterns if p.search(text)]
    score = sum(w for p, w in patterns if p.search(text))
    return min(score, 2.0), hits


def is_suspicious_chunk(text: str) -> bool:
    score, _ = heuristic_score(text, _CHUNK_PATTERNS)
    return score >= 0.6


def _llm_classify(message: str, usage: Usage | None) -> SafetyVerdict:
    p = load_prompt("safety", "v1")
    msgs = [SystemMessage(p["system"]), HumanMessage(p["user_template"].format(message=message))]
    resp = get_llm("small", max_tokens=60).invoke(msgs)
    if usage is not None:
        usage.add_message("safety", resp, prompt_chars=sum(len(m.content) for m in msgs))
    text = str(resp.content)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    label = data.get("label") if data.get("label") in ("safe", "injection", "unsafe") else "safe"
    return SafetyVerdict(label, 1.0 if label != "safe" else 0.0, "llm", str(data.get("reason", ""))[:100])


def parse_prompt_guard(text: str) -> float:
    """Prompt Guard's output format isn't documented on Groq: accept a probability ("0.998",
    "1.2e-05") or a label ("MALICIOUS"/"JAILBREAK"/"INJECTION" vs "BENIGN")."""
    try:
        return float(text.strip())
    except ValueError:
        t = text.upper()
        return 1.0 if any(w in t for w in ("MALICIOUS", "JAILBREAK", "INJECTION", "UNSAFE")) else 0.0


def _prompt_guard_classify(message: str, usage: Usage | None) -> SafetyVerdict:
    s = get_settings()
    resp = groq_model(s.groq_safety_model, max_tokens=20).invoke([HumanMessage(message[:2000])])
    if usage is not None:
        usage.add_message("safety", resp, prompt_chars=len(message))
    prob = parse_prompt_guard(str(resp.content))
    label = "injection" if prob >= 0.5 else "safe"
    return SafetyVerdict(label, round(prob, 4), "promptguard", f"raw={str(resp.content)[:40]}")


def check_input(message: str, usage: Usage | None = None, use_llm: bool | None = None) -> SafetyVerdict:
    score, hits = heuristic_score(message)
    if score >= 1.0:
        return SafetyVerdict("injection", score, "heuristic", ", ".join(hits)[:200])

    s = get_settings()
    mode = s.safety_classifier
    available = bool(s.groq_api_key) if mode == "promptguard" else s.llm_available
    use_llm = mode != "off" and available if use_llm is None else use_llm
    if not use_llm:
        return SafetyVerdict("safe", score, "heuristic" if score else "disabled")

    r = get_redis()
    key = f"safety:{mode}:" + hashlib.sha256(message.encode()).hexdigest()
    cached = r.get(key)
    if cached:
        return SafetyVerdict(**{**json.loads(cached), "source": "cache"})
    try:
        verdict = _prompt_guard_classify(message, usage) if mode == "promptguard" else _llm_classify(message, usage)
    except Exception as e:  # classifier outage must not take the helpdesk down: fail open on layer 2 only
        log.warning("safety_classifier_failed", error=type(e).__name__)
        return SafetyVerdict("safe", score, "heuristic", "llm classifier unavailable")
    r.set(key, json.dumps(verdict.to_dict()), ex=7 * 24 * 3600)
    return verdict
