import threading
import uuid

import pytest

from app.infra.locks import LockNotAcquiredError, distributed_lock
from app.infra.redis_client import get_redis
from app.ingest.models import DocMeta
from app.ingest.pipeline import ingest_document
from app.retrieval.vectorstore.chroma_store import ChromaStore
from tests.fakes import FakeEmbedder

V1 = "# Leave\n\n## Annual\n\nYou get 24 days.\n\n## Sick\n\nYou get 10 days.\n"
V2 = "# Leave\n\n## Annual\n\nYou get 26 days.\n"


@pytest.fixture
def env(tmp_path):
    doc = tmp_path / "leave.md"
    doc.write_text(V1, encoding="utf-8")
    meta = DocMeta("leave", "leave.md", "Leave", "HR", ("employee",), "1", "2026-01-01")
    store = ChromaStore(collection=f"t{uuid.uuid4().hex[:8]}")
    kw = dict(store=store, embedder=FakeEmbedder(), r=get_redis(), db_path=tmp_path / "t.db")
    return doc, meta, store, kw


def test_index_then_skip_unchanged(env):
    doc, meta, store, kw = env
    first = ingest_document(doc, meta, **kw)
    assert first.status == "indexed" and first.chunks_written == 2
    assert ingest_document(doc, meta, **kw).status == "skipped_unchanged"
    assert store.count() == 2


def test_changed_doc_replaces_old_chunks(env):
    doc, meta, store, kw = env
    ingest_document(doc, meta, **kw)
    old_ids = set(store.ids_for_doc("leave"))
    doc.write_text(V2, encoding="utf-8")
    res = ingest_document(doc, meta, **kw)
    assert res.status == "indexed" and res.chunks_deleted == 2
    new_ids = set(store.ids_for_doc("leave"))
    assert new_ids.isdisjoint(old_ids) and len(new_ids) == 1
    assert "26 days" in store.all_chunks()[0].text


def test_role_change_triggers_reindex(env):
    doc, meta, store, kw = env
    ingest_document(doc, meta, **kw)
    hr_only = DocMeta(**{**meta.__dict__, "allowed_roles": ("hr",)})
    assert ingest_document(doc, hr_only, **kw).status == "indexed"
    assert store.all_chunks("employee") == []


def test_ingest_bumps_index_version(env):
    doc, meta, _, kw = env
    v = ingest_document(doc, meta, **kw).index_version
    doc.write_text(V2, encoding="utf-8")
    assert ingest_document(doc, meta, **kw).index_version == v + 1


def test_lock_held_elsewhere_blocks_ingest(env):
    doc, meta, _, kw = env
    with distributed_lock(get_redis(), "ingest:leave"), pytest.raises(LockNotAcquiredError):
        ingest_document(doc, meta, lock_wait_s=0.2, **kw)


def test_concurrent_ingest_indexes_once(env):
    """Two workers race on the same doc: the lock + double-check means exactly one indexes."""
    doc, meta, store, kw = env
    results = []
    barrier = threading.Barrier(2)

    def run():
        barrier.wait()
        results.append(ingest_document(doc, meta, **kw).status)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["indexed", "skipped_unchanged"]
    assert store.count() == 2
