"""POST /ask (SSE or JSON) and POST /search (retrieval only, works without any API key)."""

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from app.cache.index_version import get_index_version
from app.deps import rate_limit, require_llm
from app.errors import LLMUnavailableError
from app.infra.redis_client import get_redis
from app.log import get_logger
from app.observability.request_log import logged
from app.schemas.ask import AskRequest, AskResponse, SearchRequest
from app.services import get_rag_pipeline, get_retriever

router = APIRouter(tags=["rag"])
log = get_logger(__name__)


def sse_events(events: Iterator[tuple[str, object]]) -> Iterator[dict]:
    """Map pipeline events to SSE frames. Errors after streaming started become an `error` event."""
    try:
        for name, data in events:
            if isinstance(data, AskResponse):
                payload = data.model_dump_json()
            elif isinstance(data, str):
                payload = json.dumps({"text": data})
            else:
                payload = json.dumps(data)
            yield {"event": name, "data": payload}
    except LLMUnavailableError as e:
        yield {"event": "error", "data": json.dumps({"error": e.error, "detail": e.detail})}
    except Exception as e:  # provider failure mid-stream after all fallbacks
        log.exception("ask_stream_failed")
        yield {
            "event": "error",
            "data": json.dumps({"error": "generation_failed", "detail": type(e).__name__}),
        }


def _prepend(first, rest: Iterator) -> Iterator:
    yield first
    yield from rest


def collect(events: Iterator[tuple[str, object]]) -> AskResponse:
    result = None
    for name, data in events:
        if name == "result":
            result = data
    assert isinstance(result, AskResponse)
    return result


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, _=Depends(require_llm)):
    rate_limit(req.session_id)
    events = logged(get_rag_pipeline().run(req), "/ask", req.session_id, req.role, req.question)
    first = await run_in_threadpool(next, events)  # run input guardrails now: a 400 must not be a 200 stream
    events = _prepend(first, events)
    if req.stream:
        return EventSourceResponse(iterate_in_threadpool(sse_events(events)))
    return await run_in_threadpool(collect, events)


@router.post("/search")
def search(req: SearchRequest) -> dict:
    retriever = get_retriever()
    res = retriever.retrieve(req.query, req.role, get_index_version(get_redis()))
    return {
        "found": res.found,
        "top_score": res.top_score,
        "timings_ms": res.timings_ms,
        "results": [
            {
                "chunk_id": sc.chunk.chunk_id,
                "doc_id": sc.chunk.doc_id,
                "section": sc.chunk.section,
                "heading_path": sc.chunk.metadata.get("heading_path"),
                "score": round(sc.score, 3),
                "text": sc.chunk.text,
            }
            for sc in res.chunks[: req.top_k]
        ],
    }
