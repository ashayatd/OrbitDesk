"""FastAPI dependencies shared by routers."""

from app.config import Settings, get_settings
from app.errors import LLMUnavailableError


def require_llm() -> Settings:
    """Dependency for generation endpoints: raises 503 when no LLM key is configured."""
    settings = get_settings()
    if not settings.llm_available:
        raise LLMUnavailableError()
    return settings
