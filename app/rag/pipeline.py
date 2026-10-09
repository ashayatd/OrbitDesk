"""/ask orchestration: memory -> rewrite -> hybrid retrieval -> generate (streamed) -> verify citations.

`run()` is a generator of (event, data) tuples, so the same code serves the SSE
endpoint (stream each event), the JSON endpoint and evals (take the last one).
Every stage records its latency in `timings_ms`; every LLM call adds to `Usage`.
"""

import math
import time
import uuid
from collections.abc import Iterator
from typing import Any

from app.agent.memory import SessionMemory
from app.cache.index_version import get_index_version
from app.config import get_settings
from app.llm.client import get_llm
from app.llm.usage import Usage
from app.log import get_logger
from app.rag.citations import strip_invalid_citations, verify_citations
from app.rag.output import MetaSplitter
from app.rag.prompts import build_answer_messages
from app.rag.rewrite import needs_rewrite, rewrite_question
from app.retrieval.retriever import HybridRetriever, RetrievalResult
from app.schemas.ask import AskRequest, AskResponse, CitationOut, TokenUsage

log = get_logger(__name__)
NOT_FOUND = "I couldn't find this in the policies."
Event = tuple[str, Any]


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def compute_confidence(top_score: float | None, n_valid: int, n_total: int, found: bool) -> float:
    """Confidence from evidence, not the model's self-report:
    sigmoid(top rerank logit) x share of citations that verified (0.5 if the answer cited nothing)."""
    if not found or top_score is None:
        return 0.0
    retrieval = 1 / (1 + math.exp(-top_score))
    citation_factor = n_valid / n_total if n_total else 0.5
    return round(retrieval * citation_factor, 3)


class RagPipeline:
    def __init__(
        self, retriever: HybridRetriever, memory: SessionMemory, redis_client, prompt_version: str = ""
    ) -> None:
        self.retriever = retriever
        self.memory = memory
        self.r = redis_client
        self.prompt_version = prompt_version or get_settings().prompt_version

    def run(self, req: AskRequest, request_id: str | None = None, use_memory: bool = True) -> Iterator[Event]:
        request_id = request_id or uuid.uuid4().hex
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        usage = Usage()

        # 1. Memory + query rewriting
        summary, history = self.memory.context(req.session_id) if use_memory else ("", [])
        if summary:
            history = [{"role": "system", "content": f"Summary of earlier conversation: {summary}"}, *history]
        rewritten = None
        if needs_rewrite(req.question, history):
            t0 = time.perf_counter()
            rewritten = rewrite_question(req.question, history, usage)
            timings["rewrite"] = _ms(t0)
        query = rewritten or req.question
        yield "meta", {"request_id": request_id, "rewritten_question": rewritten}

        # 2. Retrieval (role pre-filter inside)
        retrieval = self.retriever.retrieve(query, req.role, get_index_version(self.r))
        timings.update(retrieval.timings_ms)

        # 3. Generation (skipped entirely when nothing relevant was found)
        if not retrieval.found:
            answer, meta, citations = NOT_FOUND, {}, []
            yield "token", answer
        else:
            answer, meta = yield from self._generate(query, retrieval, timings, usage)
            t0 = time.perf_counter()
            citations = verify_citations(answer, retrieval.chunks)
            answer = strip_invalid_citations(answer, citations)
            timings["citation_check"] = _ms(t0)

        found = retrieval.found and not answer.startswith(NOT_FOUND)
        valid = [c for c in citations if c.valid]
        timings["total"] = _ms(t_start)
        result = AskResponse(
            request_id=request_id,
            answer=answer,
            citations=[
                CitationOut(doc_id=c.doc_id, section=c.section, title=c.title, chunk_id=c.chunk_id) for c in valid
            ],
            invalid_citations=[f"[{c.doc_id}, {c.section}]" for c in citations if not c.valid],
            confidence=compute_confidence(retrieval.top_score, len(valid), len(citations), found),
            follow_up_suggestions=[str(s) for s in meta.get("follow_up_suggestions", [])][:3] if found else [],
            found=found,
            question=req.question,
            rewritten_question=rewritten,
            retrieved=[
                {
                    "chunk_id": sc.chunk.chunk_id,
                    "doc_id": sc.chunk.doc_id,
                    "section": sc.chunk.section,
                    "score": round(sc.score, 3),
                }
                for sc in retrieval.chunks
            ],
            timings_ms=timings,
            tokens=TokenUsage(**usage.to_dict()),
            prompt_version=self.prompt_version,
        )
        log.info(
            "ask",
            request_id=request_id,
            role=req.role,
            found=found,
            timings_ms=timings,
            tokens=usage.to_dict()["total_tokens"],
        )
        yield "result", result

        if use_memory:  # after the result, so summarisation never delays the user
            self.memory.append(req.session_id, req.question, answer)

    def _generate(self, query: str, retrieval: RetrievalResult, timings: dict, usage: Usage):
        """Stream the answer; yields ("token", text) events and returns (answer, meta)."""
        msgs = build_answer_messages(query, retrieval.chunks, self.prompt_version)
        llm = get_llm("main", max_tokens=600)
        splitter, full = MetaSplitter(), None
        t0 = time.perf_counter()
        for chunk in llm.stream(msgs):
            if full is None:
                timings["llm_first_token"] = _ms(t0)
                full = chunk
            else:
                full = full + chunk
            emit = splitter.feed(str(chunk.content))
            if emit:
                yield "token", emit
        timings["llm_generate"] = _ms(t0)
        if full is not None:
            usage.add_message("answer", full, prompt_chars=sum(len(str(m.content)) for m in msgs))
        return splitter.finish()
