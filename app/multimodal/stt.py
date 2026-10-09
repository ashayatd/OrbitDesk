"""Speech-to-text with Groq Whisper (whisper-large-v3-turbo by default)."""

import time
from dataclasses import dataclass

from app.config import get_settings
from app.errors import LLMUnavailableError
from app.llm.pricing import STT_PRICE_PER_HOUR

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Groq free-tier upload limit
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4", ".mpeg", ".mpga"}


@dataclass
class Transcript:
    text: str
    duration_s: float | None
    latency_ms: float
    model: str
    cost_usd: float | None


def transcribe(filename: str, data: bytes) -> Transcript:
    s = get_settings()
    if not s.groq_api_key:
        raise LLMUnavailableError("GROQ_API_KEY is not set, so speech-to-text is unavailable.")
    from groq import Groq

    client = Groq(api_key=s.groq_api_key, timeout=s.llm_timeout_s, max_retries=s.llm_max_retries)
    t0 = time.perf_counter()
    resp = client.audio.transcriptions.create(
        file=(filename, data),
        model=s.groq_stt_model,
        response_format="verbose_json",  # includes audio duration, used for the cost estimate
        language="en",
        temperature=0.0,
    )
    latency = round((time.perf_counter() - t0) * 1000, 1)
    duration = getattr(resp, "duration", None)
    price = STT_PRICE_PER_HOUR.get(s.groq_stt_model)
    # Groq bills a minimum of 10 seconds per request.
    cost = round(max(duration, 10) / 3600 * price, 6) if duration is not None and price else None
    return Transcript(resp.text.strip(), duration, latency, s.groq_stt_model, cost)
