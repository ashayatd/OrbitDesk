"""Eval plumbing that must work before any key exists: limit detection, checkpoint resume, metrics."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import generation_eval as ge  # noqa: E402
from common import load_golden, percentile, retrieval_metrics  # noqa: E402


def test_daily_vs_minute_limit_detection():
    daily = Exception("429 Rate limit reached for model openai/gpt-oss-120b on tokens per day (TPD): Limit 200000")
    minute = Exception("429 Rate limit reached on tokens per minute (TPM): Limit 8000, please try again in 7s")
    assert ge.is_daily_limit(daily) and not ge.is_daily_limit(minute)
    assert ge.is_rate_limit(minute)


def test_daily_limit_escalates_without_retrying():
    calls = []

    def boom():
        calls.append(1)
        raise Exception("429 rate limit: requests per day (RPD) exceeded")

    with pytest.raises(ge.DailyLimitError):
        ge.with_rate_limit_handling(boom)
    assert len(calls) == 1


def test_checkpoint_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    ge.append_jsonl(p, {"key": "q01", "answer": "a"})
    ge.append_jsonl(p, {"key": "q02", "answer": "b"})
    assert set(ge.read_jsonl(p)) == {"q01", "q02"}  # a rerun skips both


def test_golden_set_shape():
    g = load_golden()
    assert len(g) == 40
    types = {q["type"] for q in g}
    assert {"single", "multi_doc", "follow_up", "unanswerable", "role_restricted"} <= types
    assert all(q["standalone"] for q in g if q["type"] == "follow_up")
    assert all(not q["expected_sources"] for q in g if not q["answerable"])


def test_metrics_math():
    q1 = {"answerable": True, "expected_sources": [["a", "1"], ["b", "2"]]}
    q2 = {"answerable": False, "expected_sources": []}
    rows = [
        {
            "q": q1,
            "retrieved": [{"doc_id": "x", "section": "9"}, {"doc_id": "a", "section": "1"}],
            "found": True,
            "latency_ms": 10,
            "leaks": 0,
        },
        {"q": q2, "retrieved": [], "found": True, "latency_ms": 20, "leaks": 0},
    ]
    m = retrieval_metrics(rows)
    assert m["recall_at_k"] == 0.5 and m["mrr"] == 0.5 and m["no_answer_accuracy"] == 0.5
    assert percentile([10, 20], 50) == 15
