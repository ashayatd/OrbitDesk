"""List prices used for *estimated* cost per query (USD per 1M tokens).

Sources (checked 2026-10-09):
- gpt-oss: https://groq.com/blog/day-zero-support-for-openai-open-models
  (third-party trackers list slightly different numbers; we use Groq's own announcement)
- llama 3.x: https://eesel.ai/blog/groq-pricing (now "contact sales" tier on Groq)
- prompt guard: https://www.getmaxim.ai/bifrost/model-library/compare/groq/llama-prompt-guard-2-86m
Free-plan usage is not billed; these numbers estimate what the same traffic would cost on a paid plan.
Models without a known price return None, and the cost is reported as unknown, not as 0.
"""

PRICES_PER_M: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "meta-llama/llama-prompt-guard-2-86m": (0.04, 0.04),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}

STT_PRICE_PER_HOUR: dict[str, float] = {
    "whisper-large-v3-turbo": 0.04,
    "whisper-large-v3": 0.111,
}


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICES_PER_M.get(model) or PRICES_PER_M.get(model.split("/")[-1])
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
