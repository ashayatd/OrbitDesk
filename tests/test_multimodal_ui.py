"""Phase 5: /voice and /image degrade cleanly; UI builds and mounts."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.multimodal.vision as vision
from app.config import get_settings
from app.errors import ImageInputUnavailableError
from app.llm.usage import Usage
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "evals" / "samples"


def test_image_without_any_key_is_503_image_unavailable():
    client = TestClient(create_app())
    with open(SAMPLES / "vpn_error.png", "rb") as f:
        r = client.post("/image", files={"image": ("vpn_error.png", f, "image/png")}, data={"session_id": "s"})
    assert r.status_code == 503
    assert r.json()["error"] == "image_input_unavailable"


def test_voice_without_key_is_503():
    client = TestClient(create_app())
    with open(SAMPLES / "annual_leave.wav", "rb") as f:
        r = client.post("/voice", files={"audio": ("q.wav", f, "audio/wav")}, data={"session_id": "s"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]


def test_vision_falls_back_to_openrouter_then_reports_unavailable(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "groq_api_key", "fake")
    calls = []

    class Broken:
        def __init__(self, name):
            self.name = name

        def invoke(self, msgs, config=None):
            calls.append(self.name)
            raise RuntimeError("model_decommissioned")

    monkeypatch.setattr(vision, "groq_model", lambda *a, **k: Broken("groq"))
    monkeypatch.setattr(vision, "openrouter_model", lambda *a, **k: Broken("openrouter"))

    # 1) only Groq configured: clear message that mentions the fallback setting
    with pytest.raises(ImageInputUnavailableError) as e:
        vision.extract_error(b"png", "image/png", Usage())
    assert calls == ["groq"] and "FALLBACK_PROVIDER_KEY" in e.value.detail

    # 2) fallback configured: it is tried after Groq
    calls.clear()
    monkeypatch.setattr(s, "fallback_provider_key", "fake2")
    with pytest.raises(ImageInputUnavailableError):
        vision.extract_error(b"png", "image/png", Usage())
    assert calls == ["groq", "openrouter"]


def test_vision_fallback_success(monkeypatch):
    from langchain_core.messages import AIMessage

    s = get_settings()
    monkeypatch.setattr(s, "groq_api_key", "fake")
    monkeypatch.setattr(s, "fallback_provider_key", "fake2")

    class Broken:
        def invoke(self, msgs, config=None):
            raise RuntimeError("404 model not found")

    class Works:
        def invoke(self, msgs, config=None):
            return AIMessage(
                content='{"error_code": "VPN-809", "error_message": "Gateway unreachable", '
                '"application": "OrbitConnect", "search_query": "How do I fix VPN-809?"}'
            )

    monkeypatch.setattr(vision, "groq_model", lambda *a, **k: Broken())
    monkeypatch.setattr(vision, "openrouter_model", lambda *a, **k: Works())
    res = vision.extract_error(b"png", "image/png", Usage())
    assert res.provider == "openrouter" and res.error_code == "VPN-809"
    assert res.attempts[0]["error"] == "RuntimeError"


def test_other_endpoints_unaffected_by_missing_vision():
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200


def test_ui_builds_and_mounts(monkeypatch):
    monkeypatch.setattr(get_settings(), "mount_ui", True)
    client = TestClient(create_app())
    assert client.get("/ui/").status_code == 200
