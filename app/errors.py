"""Domain errors mapped to HTTP responses in app.main."""


class LLMUnavailableError(Exception):
    """No LLM key configured, or every provider in the fallback chain failed."""

    def __init__(self, detail: str | None = None, error: str = "llm_unavailable") -> None:
        self.error = error
        self.detail = detail or (
            "GROQ_API_KEY is not set. Ingestion and retrieval (/search) still work; "
            "set the key in .env to enable generation."
        )
        super().__init__(self.detail)


class ImageInputUnavailableError(LLMUnavailableError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, error="image_input_unavailable")


class RateLimitedError(Exception):
    def __init__(self, retry_after_s: int) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(f"rate limited, retry after {retry_after_s}s")


class InputRejectedError(Exception):
    """Guardrail rejected the user's input (too long, injection, ...)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
