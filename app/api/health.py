from fastapi import APIRouter

from app.config import get_settings
from app.infra.redis_client import redis_ok

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm_available": s.llm_available,
        "redis_ok": redis_ok(),
        "prompt_version": s.prompt_version,
        "router_mode": s.router_mode,
        "vector_backend": s.vector_backend,
    }
