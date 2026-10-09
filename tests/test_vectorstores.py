"""Both stores must honour the same contract, especially the role pre-filter."""

import uuid

import pytest

from app.ingest.models import Chunk, role_flags
from app.retrieval.vectorstore.chroma_store import ChromaStore
from app.retrieval.vectorstore.faiss_store import FaissStore
from tests.fakes import FakeEmbedder

EMB = FakeEmbedder()


def make_chunks() -> list[Chunk]:
    def c(cid, doc, text, roles):
        return Chunk(cid, doc, text, {"doc_id": doc, **role_flags(roles)})

    return [
        c("pub::1", "pub", "annual leave is 24 days", ("employee", "hr", "it_admin")),
        c("hr::1", "hr", "salary band E3 senior engineer 98000 114000", ("hr",)),
        c("it::1", "it", "break glass account in the data centre safe", ("it_admin",)),
    ]


@pytest.fixture(params=["chroma", "faiss"])
def store(request, tmp_path):
    if request.param == "chroma":
        return ChromaStore(collection=f"t{uuid.uuid4().hex[:8]}")
    return FaissStore(dim=EMB.dim, path=tmp_path / "faiss")


def test_role_prefilter_never_returns_forbidden_chunks(store):
    chunks = make_chunks()
    store.upsert(chunks, EMB.embed_documents([c.text for c in chunks]))
    # Query with the HR chunk's exact text: the best possible match for it.
    q = EMB.embed_query("salary band E3 senior engineer 98000 114000")
    assert "hr::1" not in [s.chunk.chunk_id for s in store.query(q, 10, "employee")]
    assert store.query(q, 1, "hr")[0].chunk.chunk_id == "hr::1"
    assert {c.chunk_id for c in store.all_chunks("employee")} == {"pub::1"}
    assert {c.chunk_id for c in store.all_chunks("it_admin")} == {"pub::1", "it::1"}


def test_delete_and_ids_for_doc(store):
    chunks = make_chunks()
    store.upsert(chunks, EMB.embed_documents([c.text for c in chunks]))
    assert store.ids_for_doc("hr") == ["hr::1"]
    store.delete(["hr::1"])
    assert store.ids_for_doc("hr") == [] and store.count() == 2


def test_faiss_persists(tmp_path):
    chunks = make_chunks()
    s1 = FaissStore(dim=EMB.dim, path=tmp_path)
    s1.upsert(chunks, EMB.embed_documents([c.text for c in chunks]))
    s2 = FaissStore(dim=EMB.dim, path=tmp_path)
    assert s2.count() == 3
    assert s2.query(EMB.embed_query("annual leave"), 1, "employee")[0].chunk.chunk_id == "pub::1"
