import uuid

from fastapi.testclient import TestClient

import app.ingest.jobs as jobs
from app.main import create_app
from app.retrieval.vectorstore.chroma_store import ChromaStore
from tests.fakes import FakeEmbedder


def test_upload_enqueue_and_status(monkeypatch, tmp_path):
    store = ChromaStore(collection=f"t{uuid.uuid4().hex[:8]}")
    monkeypatch.setattr(jobs, "get_store", lambda: store)
    monkeypatch.setattr(jobs, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(jobs.get_settings(), "data_dir", tmp_path)
    client = TestClient(create_app())

    r = client.post(
        "/ingest",
        files={"file": ("printing.md", b"# Printing\n\n## Setup\n\nUse printer PRN-2.", "text/markdown")},
        data={"title": "Printing", "department": "IT", "allowed_roles": "employee,it_admin"},
    )
    assert r.status_code == 202
    status = client.get(r.json()["status_url"]).json()
    assert status["status"] == "finished"
    assert status["result"]["status"] == "indexed"
    assert store.ids_for_doc("printing")


def test_rejects_bad_roles_and_types():
    client = TestClient(create_app())
    bad_role = client.post(
        "/ingest",
        files={"file": ("a.md", b"x")},
        data={"title": "a", "department": "x", "allowed_roles": "ceo"},
    )
    assert bad_role.status_code == 422
    bad_type = client.post("/ingest", files={"file": ("a.exe", b"x")}, data={"title": "a", "department": "x"})
    assert bad_type.status_code == 415
