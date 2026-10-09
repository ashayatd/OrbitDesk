"""List prices used for *estimated* cost per query (USD per 1M tokens).

Source: https://eesel.ai/blog/groq-pricing (summary of groq.com pricing), checked 2026-10-09.
Models without a known price return None, and the cost is reported as unknown, not as 0.
"""

PRICES_PER_M: dict[str, tuple[float, float]] = {
    # model: (input, output)
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
