"""POST /voice (audio -> Whisper -> /ask flow) and POST /image (screenshot -> vision -> IT runbooks)."""

import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.ask import collect
from app.deps import rate_limit
from app.llm.usage import Usage
from app.multimodal.stt import AUDIO_SUFFIXES, MAX_AUDIO_BYTES, transcribe
from app.multimodal.vision import IMAGE_TYPES, MAX_IMAGE_BYTES, extract_error
from app.schemas.ask import AskRequest, Role
from app.services import get_rag_pipeline

router = APIRouter(tags=["multimodal"])


@router.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    role: Role = Form("employee"),
) -> dict:
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(415, f"unsupported audio type {suffix!r}")
    data = await audio.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio larger than 25 MB")
    rate_limit(session_id)

    transcript = await run_in_threadpool(transcribe, audio.filename or f"audio{suffix}", data)
    if not transcript.text:
        raise HTTPException(422, "no speech detected in the audio")
    req = AskRequest(question=transcript.text, session_id=session_id, role=role, stream=False)
    answer = await run_in_threadpool(collect, get_rag_pipeline().run(req))
    answer.timings_ms = {"stt": transcript.latency_ms, **answer.timings_ms}
    return {
        "transcript": transcript.text,
        "stt": {
            "model": transcript.model,
            "audio_seconds": transcript.duration_s,
            "latency_ms": transcript.latency_ms,
            "cost_usd": transcript.cost_usd,
        },
        "answer": answer.model_dump(),
    }


@router.post("/image")
async def image(
    image: UploadFile = File(...),
    session_id: str = Form(...),
    role: Role = Form("employee"),
    note: str = Form("", max_length=500, description="optional text from the user, e.g. what they were doing"),
) -> dict:
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in IMAGE_TYPES:
        raise HTTPException(415, f"unsupported image type {suffix!r}; use png, jpg or webp")
    data = await image.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image larger than 4 MB")
    rate_limit(session_id)

    usage = Usage()
    t0 = time.perf_counter()
    extracted = await run_in_threadpool(extract_error, data, IMAGE_TYPES[suffix], usage)
    vision_ms = round((time.perf_counter() - t0) * 1000, 1)
    query = extracted.search_query or f"How do I fix {extracted.error_code} {extracted.error_message}".strip()
    if note:
        query = f"{query} ({note})"
    if not (extracted.error_code or extracted.error_message):
        raise HTTPException(422, "no error message could be read from the image")

    # Search IT runbooks only (department pre-filter), still role-filtered.
    req = AskRequest(question=query, session_id=session_id, role=role, stream=False, department="IT")
    answer = await run_in_threadpool(collect, get_rag_pipeline().run(req))
    answer.timings_ms = {"vision": vision_ms, **answer.timings_ms}
    return {
        "extracted": {
            "error_code": extracted.error_code,
            "error_message": extracted.error_message,
            "application": extracted.application,
            "search_query": query,
            "provider": extracted.provider,
            "model": extracted.model,
            "attempts": extracted.attempts,
            "tokens": usage.to_dict(),
        },
        "answer": answer.model_dump(),
    }
