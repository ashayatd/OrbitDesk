"""Pipeline tests with the real corpus + fake embedder + fake streaming LLM (no keys, fast)."""

import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import app.rag.pipeline as pipeline_mod
from app.agent.memory import SessionMemory
from app.config import get_settings
from app.infra.redis_client import get_redis
from app.ingest.pipeline import ingest_document, load_manifest
from app.main import create_app
from app.rag.pipeline import NOT_FOUND, RagPipeline
from app.retrieval.bm25 import tokenize
from app.retrieval.retriever import HybridRetriever, RetrievalConfig
from app.retrieval.vectorstore.base import ScoredChunk
from app.retrieval.vectorstore.chroma_store import ChromaStore
from app.schemas.ask import AskRequest
from tests.fakes import FakeEmbedder


class KeywordReranker:
    """Stand-in for the cross-encoder: score = word overlap, so tests need no model download."""

    def rerank(self, query, candidates, top_k):
        q = set(tokenize(query))
        scored = [
            ScoredChunk(c.chunk, float(len(q & set(tokenize(c.chunk.text))) - 2), "rerank")
            for c in candidates
        ]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]


@pytest.fixture(scope="module")
def corpus_store(tmp_path_factory):
    store = ChromaStore(collection=f"t{uuid.uuid4().hex[:8]}")
    s = get_settings()
    db = tmp_path_factory.mktemp("db") / "t.db"
    for meta in load_manifest(s.docs_dir / "manifest.yaml"):
        ingest_document(
            s.docs_dir / meta.file, meta, store=store, embedder=FakeEmbedder(), r=get_redis(), db_path=db
        )
    return store


@pytest.fixture
def rag(corpus_store):
    retriever = HybridRetriever(
        corpus_store, FakeEmbedder(), KeywordReranker(), RetrievalConfig(rerank_threshold=0)
    )
    return RagPipeline(retriever, SessionMemory(get_redis()), get_redis(), "v1")


def fake_llm(text: str, calls: list | None = None):
    def factory(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))

    return factory


def test_answer_streams_and_verifies_citations(rag, monkeypatch):
    text = (
        "You can carry over up to 5 days, used by 31 March [leave-policy, 2.3 Carry-over]. "
        "Bands are E3 98k [salary-bands, 2. Engineering Bands].\n---META---\n"
        '{"follow_up_suggestions": ["How many sick days do I get?"]}'
    )
    monkeypatch.setattr(pipeline_mod, "get_llm", fake_llm(text))
    events = list(
        rag.run(AskRequest(question="How many unused annual leave days can I carry over?", session_id="s1"))
    )
    names = [e[0] for e in events]
    assert names[0] == "meta" and names[-1] == "result" and names.count("token") > 3
    streamed = "".join(d for n, d in events if n == "token")
    assert "---META---" not in streamed
    res = events[-1][1]
    assert res.found
    assert [c.doc_id for c in res.citations] == ["leave-policy"]
    assert res.invalid_citations == ["[salary-bands, 2. Engineering Bands]"]
    assert "salary-bands" not in res.answer
    assert res.follow_up_suggestions == ["How many sick days do I get?"]
    assert 0 < res.confidence < 1
    assert {"embed_query", "vector_search", "bm25_search", "rerank", "llm_generate", "total"} <= set(
        res.timings_ms
    )
    assert res.tokens.total_tokens > 0


def test_not_found_skips_llm(rag, monkeypatch):
    calls: list = []
    monkeypatch.setattr(pipeline_mod, "get_llm", fake_llm("should not be used", calls))
    res = list(
        rag.run(
            AskRequest(question="Who won the football world cup final yesterday evening?", session_id="s2")
        )
    )[-1][1]
    assert res.answer == NOT_FOUND and not res.found and res.confidence == 0
    assert calls == []  # no generation call: zero tokens spent on unanswerable questions


def test_employee_context_never_contains_hr_chunks(rag, monkeypatch):
    seen_prompts: list[str] = []

    class Spy(GenericFakeChatModel):
        def _stream(self, messages, *a, **k):
            seen_prompts.append(messages[-1].content)
            yield from super()._stream(messages, *a, **k)

    monkeypatch.setattr(
        pipeline_mod, "get_llm", lambda *a, **k: Spy(messages=iter([AIMessage(content="ok")]))
    )
    for i, q in enumerate(
        [
            "What is the salary band midpoint for a senior engineer E3?",
            "salary bands engineering maximum principal",
        ]
    ):
        res = list(rag.run(AskRequest(question=q, session_id=f"s3-{i}", role="employee")))[-1][1]
        assert all(r["doc_id"] != "salary-bands" for r in res.retrieved)
    assert all("salary-bands" not in p for p in seen_prompts)


def test_memory_records_turn(rag, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "get_llm", fake_llm("Answer [leave-policy, 3. Sick Leave]."))
    list(rag.run(AskRequest(question="How many sick leave days do I get each year?", session_id="mem1")))
    hist = rag.memory.history("mem1")
    assert [m["role"] for m in hist] == ["user", "assistant"]


def test_ask_endpoint_503_and_search_without_key():
    client = TestClient(create_app())
    r = client.post("/ask", json={"question": "hi", "session_id": "x"})
    assert r.status_code == 503 and r.json()["error"] == "llm_unavailable"
