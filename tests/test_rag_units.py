from app.ingest.models import Chunk
from app.rag.citations import strip_invalid_citations, verify_citations
from app.rag.output import MetaSplitter
from app.rag.rewrite import needs_rewrite
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.vectorstore.base import ScoredChunk


def sc(cid: str, doc: str = "d", section: str = "s") -> ScoredChunk:
    return ScoredChunk(Chunk(cid, doc, "text", {"doc_id": doc, "section": section, "title": doc.title()}), 1.0)


def test_meta_splitter_handles_delimiter_split_across_tokens():
    tokens = [
        "You get ",
        "24 days [leave",
        "-policy, 2.1 Entitlement].",
        "\n---ME",
        "TA---\n",
        '{"follow_up',
        '_suggestions": ["a"]}',
    ]
    sp = MetaSplitter()
    streamed = "".join(sp.feed(t) for t in tokens)
    answer, meta = sp.finish()
    assert "---" not in streamed and "META" not in streamed
    assert answer == "You get 24 days [leave-policy, 2.1 Entitlement]."
    assert meta == {"follow_up_suggestions": ["a"]}


def test_meta_splitter_without_meta_block():
    sp = MetaSplitter()
    sp.feed("Just an answer with a dash - here")
    assert sp.finish() == ("Just an answer with a dash - here", {})


def test_citation_verification():
    chunks = [
        sc("a::1", "leave-policy", "2.3 Carry-over"),
        sc("b::1", "vpn-setup", "4.1 Error VPN-809: Gateway Unreachable"),
    ]
    answer = (
        "Carry over 5 days [leave-policy, 2.3 Carry-over]. Use TCP 443 [vpn-setup, 4.1]. "
        "Bands are secret [salary-bands, 2. Engineering Bands]. Also [leave-policy, 9. Made Up]."
    )
    cites = verify_citations(answer, chunks)
    assert [(c.doc_id, c.valid) for c in cites] == [
        ("leave-policy", True),
        ("vpn-setup", True),
        ("salary-bands", False),  # real doc, but not retrieved for this user
        ("leave-policy", False),
    ]
    cleaned = strip_invalid_citations(answer, cites)
    assert "salary-bands" not in cleaned and "Made Up" not in cleaned
    assert "[leave-policy, 2.3 Carry-over]" in cleaned


def test_rrf_rewards_agreement():
    a, b, c = sc("a"), sc("b"), sc("c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert fused[0].chunk.chunk_id == "b"  # ranks 2 and 1 beat ranks 1 and 3
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-9


def test_needs_rewrite_heuristic():
    hist = [{"role": "user", "content": "How do I set up the VPN?"}]
    assert needs_rewrite("what about on mac?", hist)
    assert needs_rewrite("vpn broken", [])
    assert not needs_rewrite("How many days of annual leave do I get per year?", [])
