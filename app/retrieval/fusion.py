"""Reciprocal Rank Fusion.

score(d) = sum over result lists of 1 / (k + rank_in_list(d)), rank starting at 1.

Why RRF: BM25 scores (unbounded) and cosine similarities (-1..1) are on different
scales, so adding them needs fragile normalisation. RRF uses only ranks.
k=60 is the value from the original paper (Cormack et al., 2009); it damps the
advantage of being #1 vs #2 so one retriever can't dominate.
"""

from app.retrieval.vectorstore.base import ScoredChunk


def reciprocal_rank_fusion(result_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for rank, sc in enumerate(results, start=1):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, sc)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [ScoredChunk(by_id[cid].chunk, scores[cid], "rrf") for cid in ranked]
