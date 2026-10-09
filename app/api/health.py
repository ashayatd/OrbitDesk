from fastapi import APIRouter

from app.cache.index_version import get_index_version
from app.config import get_settings
from app.infra.redis_client import get_redis, redis_ok

router = APIRouter(tags=["health"])


def _store_chunks() -> int | None:
    try:
        from app.retrieval.vectorstore import get_store

        return get_store().count()
    except Exception:
        return None  # store unreachable (e.g. Chroma server still starting)


@router.get("/health")
def health() -> dict:
    s = get_settings()
    ok = redis_ok()
    return {
        "status": "ok",
        "llm_available": s.llm_available,
        "redis_ok": ok,
        "index_version": get_index_version(get_redis()) if ok else None,
        "vector_backend": s.vector_backend,
        "indexed_chunks": _store_chunks(),
        "prompt_version": s.prompt_version,
        "router_mode": s.router_mode,
    }
