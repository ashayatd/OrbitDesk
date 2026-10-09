"""FastAPI app factory. Gradio UI is mounted at /ui (same process, same port)."""

import os

# Internal tool: no usage analytics leave the process (set before gradio / huggingface_hub import).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import threading
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.api import agent, ask, feedback, health, ingest, multimodal
from app.config import get_settings
from app.errors import InputRejectedError, LLMUnavailableError, RateLimitedError
from app.log import configure_logging, get_logger

log = get_logger()


async def access_log(request: Request, call_next):
    """One structured JSON line per HTTP request, with a request id bound for every log line inside it."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("http_request_failed", method=request.method)
        raise
    finally:
        structlog.contextvars.clear_contextvars()
    if not request.url.path.startswith(("/ui/", "/gradio")):  # Gradio polls a lot; keep logs readable
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            request_id=request_id,
        )
    response.headers["x-request-id"] = request_id
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup")
    from app.db.seed import seed

    seed()
    if get_settings().ingest_on_startup:
        # Background thread: /health is up immediately; unchanged docs are skipped by hash.
        threading.Thread(target=_startup_ingest, daemon=True, name="startup-ingest").start()
    async with app.state.mcp.session_manager.run():  # MCP streamable HTTP at /mcp/
        yield
    log.info("shutdown")


def _startup_ingest(attempts: int = 30, wait_s: float = 2.0) -> None:
    """Retry for ~60 s: in docker the Chroma server / Redis may start a few seconds after the API."""
    from app.ingest.jobs import ingest_manifest

    for attempt in range(1, attempts + 1):
        try:
            results = ingest_manifest()
            log.info("startup_ingest_done", indexed=sum(r["status"] == "indexed" for r in results), total=len(results))
            return
        except Exception as e:
            if attempt == attempts:
                log.exception("startup_ingest_failed")
                return
            log.info("startup_ingest_retry", attempt=attempt, error=type(e).__name__)
            time.sleep(wait_s)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="OrbitDesk", version="0.1.0", lifespan=lifespan)
    app.middleware("http")(access_log)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(ask.router)
    app.include_router(agent.router)
    app.include_router(multimodal.router)
    app.include_router(feedback.router)

    # MCP server (streamable HTTP, bearer-token auth) inside the same service.
    from app.mcp_server.server import build_server

    mcp = build_server(http=True, base_url=get_settings().mcp_url.rstrip("/").removesuffix("/mcp"))
    app.state.mcp = mcp
    app.mount("/mcp", mcp.streamable_http_app())

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
        return RedirectResponse("/ui" if get_settings().mount_ui else "/docs")

    if get_settings().mount_ui:
        import gradio as gr

        from app.ui.gradio_app import build_ui

        app = gr.mount_gradio_app(app, build_ui(), path="/ui")
    return app


app = create_app()
