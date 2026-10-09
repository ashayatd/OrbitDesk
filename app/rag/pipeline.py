"""/ask orchestration.

    guard input (length cap, PII mask, injection/unsafe check)
    -> memory + rewrite follow-ups
    -> cache (exact, then semantic)
    -> hybrid retrieval (role pre-filter) + drop suspicious chunks
    -> generate (streamed) with an incremental output guard
    -> verify citations -> cache store -> memory

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
from app.cache.answer_cache import AnswerCache
from app.cache.index_version import get_index_version
from app.config import get_settings
from app.errors import InputRejectedError
from app.guardrails.injection import check_input, is_suspicious_chunk
from app.guardrails.output import BLOCKED_REPLY, RestrictedVocabulary, check_output
from app.guardrails.pii import mask_pii
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
REFUSAL = "I can't help with that request. I can answer questions about Orbit HR policies and IT support."
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
        self,
        retriever: HybridRetriever,
        memory: SessionMemory,
        redis_client,
        prompt_version: str = "",
        use_cache: bool = True,
    ) -> None:
        s = get_settings()
        self.retriever = retriever
        self.memory = memory
        self.r = redis_client
        self.prompt_version = prompt_version or s.prompt_version
        self.cache = AnswerCache(redis_client, s.semantic_cache_threshold) if use_cache else None
        self.restricted = RestrictedVocabulary(retriever.store)
        self.max_chars = s.max_question_chars

    def run(self, req: AskRequest, request_id: str | None = None, use_memory: bool = True) -> Iterator[Event]:
        request_id = request_id or uuid.uuid4().hex
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        usage = Usage()
        guard: dict[str, Any] = {}

        # 0. Input guardrails. The masked question is the only version used from here on
        #    (LLM prompts, cache keys, logs, memory).
        if len(req.question) > self.max_chars:
            raise InputRejectedError(f"question longer than {self.max_chars} characters")
        t0 = time.perf_counter()
        masked = mask_pii(req.question)
        question = masked.text
        if masked.masked:
            guard["pii_masked"] = masked.counts
        verdict = check_input(question, usage)
        guard["input_check"] = verdict.to_dict()
        timings["guard_input"] = _ms(t0)
        yield "meta", {"request_id": request_id, "rewritten_question": None, "guardrails": guard}
        if verdict.blocked:
            yield from self._finish_simple(
                req, request_id, question, REFUSAL, timings, usage, guard, t_start, use_memory
            )
            return

        # 1. Memory + query rewriting
        summary, history = self.memory.context(req.session_id) if use_memory else ("", [])
        if summary:
            history = [{"role": "system", "content": f"Summary of earlier conversation: {summary}"}, *history]
        rewritten = None
        if needs_rewrite(question, history):
            t0 = time.perf_counter()
            rewritten = rewrite_question(question, history, usage)
            timings["rewrite"] = _ms(t0)
        query = rewritten or question

        # 2. Cache: exact, then semantic. Namespaced by index version + role + prompt version.
        index_version = get_index_version(self.r)
        qvec = None
        if self.cache:
            ns = AnswerCache.namespace(index_version, req.role, self.prompt_version, req.department)
            t0 = time.perf_counter()
            hit, kind = self.cache.get_exact(ns, query), "exact"
            if hit is None:
                t1 = time.perf_counter()
                qvec = self.retriever.embed(query)  # reused by retrieval below
                timings["embed_query"] = _ms(t1)
                sem = self.cache.get_semantic(ns, query, qvec)
                hit, kind = (sem[0], "semantic") if sem else (None, "miss")
                if sem:
                    guard["semantic_similarity"] = round(sem[1], 4)
            timings["cache_lookup"] = _ms(t0)
            self.cache.record(kind)
            if hit is not None:
                yield from self._serve_cached(
                    req, request_id, question, hit, kind, rewritten, timings, guard, t_start, use_memory
                )
                return

        # 3. Retrieval (role pre-filter inside), then drop chunks that look like injected instructions
        retrieval = self.retriever.retrieve(query, req.role, index_version, qvec=qvec, department=req.department)
        timings.update(retrieval.timings_ms)
        clean = [sc for sc in retrieval.chunks if not is_suspicious_chunk(sc.chunk.text)]
        if len(clean) < len(retrieval.chunks):
            guard["chunks_dropped_injection"] = [sc.chunk.chunk_id for sc in retrieval.chunks if sc not in clean]
            retrieval.chunks = clean
            retrieval.found = retrieval.found and bool(clean)

        # 4. Generation (skipped entirely when nothing relevant was found)
        restricted_terms = self.restricted.for_role(req.role, index_version)
        if not retrieval.found:
            answer, meta, citations = NOT_FOUND, {}, []
            yield "token", answer
        else:
            answer, meta, blocked = yield from self._generate(query, retrieval, timings, usage, restricted_terms)
            if blocked:
                guard["output_blocked"] = blocked
                answer, meta = BLOCKED_REPLY, {}
                yield "replace", {"text": answer}
            t0 = time.perf_counter()
            citations = verify_citations(answer, retrieval.chunks)
            answer = strip_invalid_citations(answer, citations)
            timings["citation_check"] = _ms(t0)

        found = retrieval.found and not answer.startswith(NOT_FOUND) and "output_blocked" not in guard
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
            question=question,
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
            guardrails=guard,
        )
        log.info(
            "ask",
            request_id=request_id,
            role=req.role,
            found=found,
            timings_ms=timings,
            tokens=usage.to_dict()["total_tokens"],
            cache="miss",
        )
        yield "result", result

        if self.cache and "output_blocked" not in guard:
            ns = AnswerCache.namespace(index_version, req.role, self.prompt_version, req.department)
            self.cache.put(ns, query, qvec if qvec is not None else self.retriever.embed(query), result.model_dump())
        if use_memory:  # after the result, so summarisation never delays the user
            self.memory.append(req.session_id, question, answer)

    # ---- helpers ----

    def _generate(self, query: str, retrieval: RetrievalResult, timings: dict, usage: Usage, restricted: set[str]):
        """Stream the answer; yields ("token", text). Returns (answer, meta, blocked_reason).

        The output guard runs on the accumulated text after every streamed piece; if it
        trips we stop consuming the model immediately (no further tokens reach the user)."""
        msgs = build_answer_messages(query, retrieval.chunks, self.prompt_version)
        llm = get_llm("main", max_tokens=600)
        splitter, full, so_far, blocked = MetaSplitter(), None, "", None
        t0 = time.perf_counter()
        for chunk in llm.stream(msgs):
            if full is None:
                timings["llm_first_token"] = _ms(t0)
                full = chunk
            else:
                full = full + chunk
            emit = splitter.feed(str(chunk.content))
            if emit:
                so_far += emit
                blocked = check_output(so_far, restricted)
                if blocked:
                    break
                yield "token", emit
        timings["llm_generate"] = _ms(t0)
        if full is not None:
            usage.add_message("answer", full, prompt_chars=sum(len(str(m.content)) for m in msgs))
        answer, meta = splitter.finish()
        return answer, meta, blocked or check_output(answer, restricted)

    def _finish_simple(self, req, request_id, question, answer, timings, usage, guard, t_start, use_memory):
        """Refusal path: no retrieval, no generation."""
        yield "token", answer
        timings["total"] = _ms(t_start)
        yield (
            "result",
            AskResponse(
                request_id=request_id,
                answer=answer,
                citations=[],
                confidence=0.0,
                found=False,
                question=question,
                timings_ms=timings,
                tokens=TokenUsage(**usage.to_dict()),
                prompt_version=self.prompt_version,
                guardrails=guard,
            ),
        )
        if use_memory:
            self.memory.append(req.session_id, question, answer)

    def _serve_cached(self, req, request_id, question, cached, kind, rewritten, timings, guard, t_start, use_memory):
        yield "token", cached["answer"]
        timings["total"] = _ms(t_start)
        result = AskResponse(
            **{
                **cached,
                "request_id": request_id,
                "question": question,
                "cache": kind,
                "rewritten_question": rewritten,
                "timings_ms": timings,
                "tokens": TokenUsage(),  # a cache hit costs no generation tokens
                "guardrails": {**cached.get("guardrails", {}), **guard},
            }
        )
        log.info("ask", request_id=request_id, role=req.role, found=result.found, timings_ms=timings, cache=kind)
        yield "result", result
        if use_memory:
            self.memory.append(req.session_id, result.question, result.answer)
