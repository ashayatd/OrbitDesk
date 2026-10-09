"""Integration tests against a real Postgres + pgvector (TEST_DATABASE_URL). Skipped without it.

CI runs these with a pgvector/pgvector service container.
"""

import threading
import uuid

import pytest

from app.agent.actions import confirm_action, propose_ticket
from app.db.repo import count_tickets, get_leave_balance, get_ticket
from app.infra.redis_client import get_redis
from app.ingest.models import Chunk, role_flags
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.postgres
TICKET = {"title": "Laptop stolen", "description": "Stolen from car", "category": "Security", "priority": "P2"}
EMB = FakeEmbedder()


def test_seed_and_identity_sequence(pg):
    assert count_tickets() == 24
    assert get_leave_balance("E1001")["annual"]["remaining"] == 15
    action = propose_ticket("E1001", "s", TICKET)
    t = confirm_action(action.confirmation_id, "E1001")["ticket"]
    assert t["id"] == 1025 and t["ref"] == "TCK-1025"  # sequence moved past the seeded ids
    assert get_ticket("TCK-1025")["title"] == "Laptop stolen"


def test_five_simultaneous_confirms_create_one_ticket_on_postgres(pg):
    before = count_tickets()
    action = propose_ticket("E1001", "s", TICKET)
    results, barrier = [], threading.Barrier(5)

    def click():
        barrier.wait()
        results.append(confirm_action(action.confirmation_id, "E1001")["status"])

    threads = [threading.Thread(target=click) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count_tickets() == before + 1
    assert sorted(results) == ["created", "duplicate", "duplicate", "duplicate", "duplicate"]


def test_unique_idempotency_key_across_different_pending_actions(pg):
    """Two different proposals confirmed with the same client key: ON CONFLICT keeps one ticket."""
    before = count_tickets()
    a1, a2 = propose_ticket("E1001", "s", TICKET), propose_ticket("E1001", "s", TICKET)
    r1 = confirm_action(a1.confirmation_id, "E1001", idempotency_key="client-key-1")
    r2 = confirm_action(a2.confirmation_id, "E1001", idempotency_key="client-key-1")
    assert r1["status"] == "created" and r2["status"] == "duplicate"
    assert r1["ticket"]["id"] == r2["ticket"]["id"] and count_tickets() == before + 1


def test_ingestion_registry_and_admin_stats_on_postgres(pg, tmp_path):
    from app.ingest.models import DocMeta
    from app.ingest.pipeline import ingest_document
    from app.observability.request_log import admin_stats, log_request, save_feedback
    from app.retrieval.vectorstore.chroma_store import ChromaStore

    doc = tmp_path / "d.md"
    doc.write_text("# D\n\n## S\n\nhello world", encoding="utf-8")
    meta = DocMeta("d", "d.md", "D", "IT", ("employee",), "1", "2026-01-01")
    kw = dict(store=ChromaStore(collection=f"t{uuid.uuid4().hex[:8]}"), embedder=EMB, r=get_redis())
    assert ingest_document(doc, meta, **kw).status == "indexed"
    assert ingest_document(doc, meta, **kw).status == "skipped_unchanged"  # registry row read from Postgres

    log_request("/ask", request_id="a1", question="dog?", answered=False, latency_ms=100)
    log_request("/ask", request_id="a2", question="Dog?", answered=False, latency_ms=300, cache_hit="exact")
    save_feedback("a1", "s", "down", "no answer")
    s = admin_stats(1)
    assert s["requests"] == 2 and s["avg_latency_ms"] == 200.0 and s["top_unanswered"][0]["times"] == 2


# ---------------- pgvector store ----------------


@pytest.fixture
def pgstore(pg):
    from app.retrieval.vectorstore.pgvector_store import PgVectorStore

    store = PgVectorStore(pg, collection=f"t{uuid.uuid4().hex[:8]}", dim=EMB.dim)
    yield store
    store.drop()


def _chunks():
    def c(cid, doc, txt, roles, dept="IT"):
        return Chunk(cid, doc, txt, {"doc_id": doc, "section": "s", "department": dept, **role_flags(roles)})

    return [
        c("pub::1", "pub", "annual leave is 24 days", ("employee", "hr", "it_admin"), "HR"),
        c("vpn::1", "vpn", "VPN error 809 gateway unreachable use TCP 443 fallback", ("employee", "hr", "it_admin")),
        c("hr::1", "hr", "salary band E3 senior engineer 98000 114000", ("hr",), "HR"),
        c("it::1", "it", "break glass account in the data centre safe", ("it_admin",)),
    ]


def test_pgvector_role_filter_and_crud(pgstore):
    chunks = _chunks()
    pgstore.upsert(chunks, EMB.embed_documents([c.text for c in chunks]))
    q = EMB.embed_query("salary band E3 senior engineer 98000 114000")
    assert "hr::1" not in [s.chunk.chunk_id for s in pgstore.query(q, 10, "employee")]
    assert pgstore.query(q, 1, "hr")[0].chunk.chunk_id == "hr::1"
    assert {c.chunk_id for c in pgstore.all_chunks("it_admin")} == {"pub::1", "vpn::1", "it::1"}
    assert {c.chunk_id for c in pgstore.all_chunks("employee", "IT")} == {"vpn::1"}
    pgstore.delete(["hr::1"])
    assert pgstore.ids_for_doc("hr") == [] and pgstore.count() == 3


def test_pgvector_full_text_and_in_db_hybrid(pgstore):
    chunks = _chunks()
    pgstore.upsert(chunks, EMB.embed_documents([c.text for c in chunks]))
    fts = pgstore.fts_query("How do I fix VPN error 809?", 5, "employee")
    assert fts[0].chunk.chunk_id == "vpn::1"
    assert all(s.chunk.chunk_id != "hr::1" for s in pgstore.fts_query("salary band engineer", 5, "employee"))
    hyb = pgstore.hybrid_query(EMB.embed_query("VPN 809"), "VPN 809", 5, "employee")
    assert hyb[0].chunk.chunk_id == "vpn::1" and hyb[0].source == "rrf"
    assert all(s.chunk.chunk_id not in ("hr::1", "it::1") for s in hyb)
