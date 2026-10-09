"""Phase 8: request log, feedback, admin stats, tracing off without keys."""

from fastapi.testclient import TestClient

from app.db.database import db_session, feedback, request_log
from app.main import create_app
from app.observability.request_log import admin_stats, log_request
from app.observability.tracing import trace_callbacks, traced


def test_tracing_is_noop_without_keys():
    assert trace_callbacks("req-1") == []
    assert traced(None) == {"callbacks": []}


def test_feedback_and_admin_stats():
    with db_session() as conn:
        conn.execute(request_log.delete())
        conn.execute(feedback.delete())
    log_request("/ask", request_id="r1", question="dog in office?", answered=False, latency_ms=100, cache_hit="none")
    log_request("/ask", request_id="r2", question="Dog in office?", answered=False, latency_ms=300, cache_hit="exact")
    log_request("/ask", request_id="r3", question="annual leave", answered=True, latency_ms=200, cache_hit="semantic")
    log_request(
        "/agent", request_id="r4", question="raise ticket", answered=True, latency_ms=400, error="error:Timeout"
    )

    client = TestClient(create_app())
    assert (
        client.post("/feedback", json={"request_id": "r1", "rating": "down", "comment": "no answer"}).status_code == 201
    )
    assert client.post("/feedback", json={"request_id": "r3", "rating": "up"}).status_code == 201
    assert client.post("/feedback", json={"rating": "meh"}).status_code == 422

    s = client.get("/admin/stats").json()
    assert s["requests"] == 4
    assert s["failure_rate"] == 0.25
    assert s["unanswered_rate"] == 0.5
    assert s["avg_latency_ms"] == 250.0
    assert s["cache_hit_rate"] == round(2 / 3, 4)  # /agent doesn't use the answer cache
    assert s["top_unanswered"][0]["times"] == 2  # grouped case-insensitively
    assert s["feedback"] == {"down": 1, "up": 1}
    assert s["recent_thumbs_down"][0]["question"] == "dog in office?"
    assert admin_stats(1)["questions_per_day"][0]["requests"] == 4


def test_request_id_header_and_ui_admin_tab(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "mount_ui", True)
    client = TestClient(create_app())
    r = client.get("/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    assert client.get("/ui/").status_code == 200
