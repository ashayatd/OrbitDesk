"""Calibrate the semantic-cache threshold on paraphrase vs near-miss question pairs.

A false hit serves the wrong answer, so the threshold must sit above every
near-miss pair, even at the cost of missing some paraphrases.

    uv run python scripts/calibrate_semantic_cache.py -> reports/semantic_cache_calibration.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.retrieval.embeddings import get_embedder  # noqa: E402

PARAPHRASES = [
    ("How many days of annual leave do I get?", "How many annual leave days do I get?"),
    ("How do I fix VPN error 809?", "How can I fix the VPN-809 error?"),
    ("What is the hotel limit in London?", "What's the nightly hotel limit for London?"),
    ("how do i reset my password", "How do I reset my password?"),
    ("What is the internet allowance for working from home?", "How much internet allowance do WFH employees get?"),
    ("Who do I call for a P1 incident?", "Which number do I call for a P1 incident?"),
]
NEAR_MISSES = [
    ("How many days of annual leave do I get?", "How many sick days do I get?"),
    ("How do I fix VPN error 809?", "How do I fix VPN error 691?"),
    ("What is the hotel limit in London?", "What is the hotel limit in Berlin?"),
    ("How many days of annual leave do I get?", "How many days of annual leave can I carry over?"),
    ("What is the meal allowance for domestic travel?", "What is the meal allowance for international travel?"),
    ("How long is primary caregiver parental leave?", "How long is secondary caregiver parental leave?"),
    ("What is the response target for a P1 incident?", "What is the response target for a P2 incident?"),
]


def main() -> None:
    e = get_embedder()

    def sim(a: str, b: str) -> float:
        return round(float(e.embed_query(a) @ e.embed_query(b)), 4)

    para = [{"a": a, "b": b, "cosine": sim(a, b)} for a, b in PARAPHRASES]
    near = [{"a": a, "b": b, "cosine": sim(a, b)} for a, b in NEAR_MISSES]
    max_near = max(p["cosine"] for p in near)
    threshold = get_settings().semantic_cache_threshold
    report = {
        "model": e.model_name,
        "configured_threshold": threshold,
        "max_near_miss_cosine": max_near,
        "min_paraphrase_cosine": min(p["cosine"] for p in para),
        "false_hits_at_threshold": sum(p["cosine"] >= threshold for p in near),
        "paraphrase_hits_at_threshold": sum(p["cosine"] >= threshold for p in para),
        "n_paraphrases": len(para),
        "n_near_misses": len(near),
        "paraphrases": para,
        "near_misses": near,
    }
    out = get_settings().reports_dir / "semantic_cache_calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("paraphrases", "near_misses")}, indent=2))


if __name__ == "__main__":
    main()
