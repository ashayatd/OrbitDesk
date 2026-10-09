from fastapi import Depends
from fastapi.testclient import TestClient

from app.deps import require_llm
from app.main import create_app


def test_health_without_keys():
    client = TestClient(create_app())
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_available"] is False
    assert body["redis_ok"] is True


def test_generation_dependency_returns_503_without_key():
    app = create_app()

    @app.get("/needs-llm")
    def needs_llm(_=Depends(require_llm)):
        return {"ok": True}

    r = TestClient(app).get("/needs-llm")
    assert r.status_code == 503
    assert r.json()["error"] == "llm_unavailable"
    assert "GROQ_API_KEY" in r.json()["detail"]
