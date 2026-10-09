"""FastAPI app factory. Gradio UI is mounted at /ui (same process, same port)."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.api import ask, health, ingest
from app.config import get_settings
from app.errors import InputRejectedError, LLMUnavailableError, RateLimitedError
from app.log import configure_logging, get_logger

log = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup")
    from app.db.seed import seed

    seed()
    if get_settings().ingest_on_startup:
        # Background thread: /health is up immediately; unchanged docs are skipped by hash.
        threading.Thread(target=_startup_ingest, daemon=True, name="startup-ingest").start()
    yield
    log.info("shutdown")


def _startup_ingest() -> None:
    from app.ingest.jobs import ingest_manifest

    try:
        results = ingest_manifest()
        log.info(
            "startup_ingest_done", indexed=sum(r["status"] == "indexed" for r in results), total=len(results)
        )
    except Exception:
        log.exception("startup_ingest_failed")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="OrbitDesk", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(ask.router)

    @app.exception_handler(LLMUnavailableError)
    async def _llm_unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": exc.error, "detail": exc.detail})

    @app.exception_handler(RateLimitedError)
    async def _rate_limited(_: Request, exc: RateLimitedError) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "detail": str(exc)},
            headers={"Retry-After": str(exc.retry_after_s)},
        )

    @app.exception_handler(InputRejectedError)
    async def _rejected(_: Request, exc: InputRejectedError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "input_rejected", "detail": exc.reason})

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    return app


app = create_app()
