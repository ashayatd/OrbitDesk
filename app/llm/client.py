"""LLM access with timeouts, retries and a fallback chain.

    main tier : Groq gpt-oss-120b -> Groq gpt-oss-20b -> OpenRouter (if FALLBACK_PROVIDER_KEY)
    small tier: Groq gpt-oss-20b  -> Groq gpt-oss-120b -> OpenRouter

Each model is wrapped in `.with_retry()` (exponential backoff + jitter, transient
errors only), and the chain in `.with_fallbacks()` (any error: timeout, 5xx,
model decommissioned, daily quota hit). Tools are bound to each model *before*
wrapping, because a retry/fallback wrapper no longer exposes `.bind_tools()`.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from app.config import get_settings
from app.errors import LLMUnavailableError

Tier = Literal["main", "small"]
REASONING_HEADROOM = 1024


def is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss") or "qwen3" in model


def _transient_errors() -> tuple[type[BaseException], ...]:
    """Errors worth retrying on the same model. 4xx (bad request, auth) are not."""
    import groq
    import httpx
    import openai

    return (
        groq.RateLimitError,
        groq.APITimeoutError,
        groq.APIConnectionError,
        groq.InternalServerError,
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        httpx.TimeoutException,
        TimeoutError,
    )


def groq_model(model: str, temperature: float = 0.0, max_tokens: int | None = None) -> BaseChatModel:
    from langchain_groq import ChatGroq

    s = get_settings()
    extra = {}
    if is_reasoning_model(model):
        # Hidden reasoning tokens count against max_tokens: leave headroom or the visible
        # answer gets cut off (a router asked for 30 tokens would return nothing).
        extra["reasoning_effort"] = s.reasoning_effort
        max_tokens = (max_tokens or 0) + REASONING_HEADROOM if max_tokens else None
    return ChatGroq(
        model=model,
        api_key=s.groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=s.llm_timeout_s,
        max_retries=0,  # retries are handled by .with_retry so behaviour is uniform across providers
        **extra,
    )


def openrouter_model(model: str, temperature: float = 0.0, max_tokens: int | None = None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    s = get_settings()
    return ChatOpenAI(
        model=model,
        api_key=s.fallback_provider_key,
        base_url=s.fallback_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=s.llm_timeout_s,
        max_retries=0,
        stream_usage=True,
    )


def candidate_models(tier: Tier, temperature: float = 0.0, max_tokens: int | None = None) -> list[BaseChatModel]:
    s = get_settings()
    models: list[BaseChatModel] = []
    if s.groq_api_key:
        order = [s.groq_chat_model, s.groq_small_model]
        if tier == "small":
            order.reverse()
        models += [groq_model(m, temperature, max_tokens) for m in dict.fromkeys(order)]
    if s.fallback_provider_key:
        models.append(openrouter_model(s.fallback_chat_model, temperature, max_tokens))
    return models if s.llm_fallbacks_enabled else models[:1]


def get_llm(
    tier: Tier = "main",
    *,
    tools: list | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> Runnable:
    models = candidate_models(tier, temperature, max_tokens)
    if not models:
        raise LLMUnavailableError()
    s = get_settings()
    wrapped = [
        (m.bind_tools(tools) if tools else m).with_retry(
            retry_if_exception_type=_transient_errors(),
            wait_exponential_jitter=True,
            stop_after_attempt=1 + s.llm_max_retries,
        )
        for m in models
    ]
    return wrapped[0].with_fallbacks(wrapped[1:]) if len(wrapped) > 1 else wrapped[0]
