"""Extract error text from a screenshot with a vision model.

Graceful degradation (the Groq vision model is a *preview* model and may be removed):
    1. Groq GROQ_VISION_MODEL
    2. OpenRouter FALLBACK_VISION_MODEL, if FALLBACK_PROVIDER_KEY is set
    3. otherwise ImageInputUnavailableError -> HTTP 503 "image_input_unavailable"
Only /image is affected; every other endpoint keeps working.
"""

import base64
import json
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.errors import ImageInputUnavailableError
from app.llm.client import groq_model, openrouter_model
from app.llm.usage import Usage
from app.log import get_logger

log = get_logger(__name__)
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # base64-encoded request limit on Groq
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

SYSTEM = (
    "You read screenshots of software errors for an IT helpdesk. Extract, exactly as shown: the error code, "
    "the error message, and the application name. Do not guess anything that is not visible. "
    'Reply with ONLY JSON: {"error_code": "...", "error_message": "...", "application": "...", '
    '"search_query": "<one short question an employee would ask to fix this>"}. Use "" for missing fields.'
)


@dataclass
class VisionResult:
    error_code: str
    error_message: str
    application: str
    search_query: str
    model: str
    provider: str
    latency_ms: float
    attempts: list[dict] = field(default_factory=list)


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def extract_error(image: bytes, mime: str, usage: Usage) -> VisionResult:
    s = get_settings()
    data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    msgs = [
        SystemMessage(SYSTEM),
        HumanMessage(
            content=[
                {"type": "text", "text": "Extract the error from this screenshot."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]
    providers = []
    if s.groq_api_key:
        providers.append(("groq", s.groq_vision_model, lambda: groq_model(s.groq_vision_model, max_tokens=300)))
    if s.fallback_provider_key:
        providers.append(
            ("openrouter", s.fallback_vision_model, lambda: openrouter_model(s.fallback_vision_model, max_tokens=300))
        )
    if not providers:
        raise ImageInputUnavailableError(
            "Image input unavailable: no vision provider is configured (set GROQ_API_KEY or FALLBACK_PROVIDER_KEY)."
        )

    attempts: list[dict] = []
    for provider, model, factory in providers:
        t0 = time.perf_counter()
        try:
            resp = factory().invoke(msgs)
        except Exception as e:  # decommissioned model (404), timeout, 5xx, quota...
            attempts.append({"provider": provider, "model": model, "error": type(e).__name__})
            log.warning("vision_provider_failed", provider=provider, model=model, error=type(e).__name__)
            continue
        latency = round((time.perf_counter() - t0) * 1000, 1)
        usage.add_message("vision", resp)
        data = _parse(str(resp.content))
        attempts.append({"provider": provider, "model": model, "ok": True})
        return VisionResult(
            error_code=str(data.get("error_code", "")),
            error_message=str(data.get("error_message", "")),
            application=str(data.get("application", "")),
            search_query=str(data.get("search_query", "")),
            model=model,
            provider=provider,
            latency_ms=latency,
            attempts=attempts,
        )

    tried = ", ".join(f"{a['provider']}:{a['model']} ({a['error']})" for a in attempts)
    hint = "" if s.fallback_provider_key else " Set FALLBACK_PROVIDER_KEY to enable the OpenRouter vision fallback."
    raise ImageInputUnavailableError(f"Image input unavailable: every vision model failed [{tried}].{hint}")
