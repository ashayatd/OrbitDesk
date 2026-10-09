"""Router metrics, dependency-free (shared by the Colab notebook and compare_routers.py)."""

import json
import re

INTENTS = ["policy_question", "action", "small_talk", "out_of_scope"]
URGENCIES = ["low", "medium", "high"]


def parse_strict(text: str) -> dict | None:
    """Strict: the WHOLE output must be one JSON object with valid labels (that is the contract)."""
    try:
        d = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict) or d.get("intent") not in INTENTS or d.get("urgency") not in URGENCIES:
        return None
    return {"intent": d["intent"], "urgency": d["urgency"]}


def parse_lenient(text: str) -> dict | None:
    """Lenient: first {...} anywhere in the output (what the production router accepts)."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    return parse_strict(m.group(0)) if m else None


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def score(gold_rows: list[dict], raw_outputs: list[str]) -> dict:
    """Unparseable outputs count as wrong (predicted label "INVALID")."""
    strict = [parse_strict(o) for o in raw_outputs]
    lenient = [parse_lenient(o) for o in raw_outputs]
    pi = [(p or {}).get("intent", "INVALID") for p in lenient]
    pu = [(p or {}).get("urgency", "INVALID") for p in lenient]
    gi = [g["intent"] for g in gold_rows]
    gu = [g["urgency"] for g in gold_rows]
    n = len(gold_rows)
    return {
        "n": n,
        "strict_json_rate": round(sum(p is not None for p in strict) / n, 4),
        "intent_accuracy": round(sum(a == b for a, b in zip(gi, pi, strict=True)) / n, 4),
        "intent_macro_f1": round(macro_f1(gi, pi, INTENTS), 4),
        "urgency_accuracy": round(sum(a == b for a, b in zip(gu, pu, strict=True)) / n, 4),
        "urgency_macro_f1": round(macro_f1(gu, pu, URGENCIES), 4),
        "joint_accuracy": round(sum(a == b and c == d for a, b, c, d in zip(gi, pi, gu, pu, strict=True)) / n, 4),
    }
