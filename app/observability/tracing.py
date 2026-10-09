"""Langfuse tracing (optional: a no-op when LANGFUSE_PUBLIC_KEY / SECRET_KEY are empty).

Every LLM call in a request is attached to ONE Langfuse trace whose id is derived
from the request id (`create_trace_id(seed=request_id)`). Passing the trace id
explicitly is deliberate: /ask runs as a generator that hops between threadpool
threads, so OpenTelemetry context propagation can't be relied on to group calls.
The agent graph gets the same callback, so every node, LLM call and tool call
shows up as a nested step.
"""

from functools import lru_cache

from app.config import get_settings
from app.log import get_logger

log = get_logger(__name__)


@lru_cache
def _client():
    s = get_settings()
    if not s.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host)
    except Exception as e:  # tracing must never break the app
        log.warning("langfuse_init_failed", error=str(e))
        return None


def trace_callbacks(trace_seed: str | None) -> list:
    """LangChain callbacks for one request (empty list when tracing is off)."""
    client = _client()
    if client is None or not trace_seed:
        return []
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    trace_id = Langfuse.create_trace_id(seed=trace_seed)
    return [CallbackHandler(public_key=get_settings().langfuse_public_key, trace_context={"trace_id": trace_id})]


def traced(usage) -> dict:
    """RunnableConfig for an LLM call, grouped under the request's trace."""
    return {"callbacks": trace_callbacks(getattr(usage, "trace_id", None))}


def flush() -> None:
    client = _client()
    if client is not None:
        client.flush()
