"""FastAPI dependencies shared by routers."""

from app.config import Settings, get_settings
from app.errors import LLMUnavailableError
from app.infra.ratelimit import check_rate_limit
from app.infra.redis_client import get_redis


def require_llm() -> Settings:
    """Dependency for generation endpoints: raises 503 when no LLM key is configured."""
    settings = get_settings()
    if not settings.llm_available:
        raise LLMUnavailableError()
    return settings


def rate_limit(session_id: str) -> None:
    """Per-session limit for LLM endpoints (raises 429 with Retry-After)."""
    check_rate_limit(get_redis(), session_id, get_settings().rate_limit_per_minute)
