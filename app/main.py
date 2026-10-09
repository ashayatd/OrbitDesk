"""FastAPI app factory. Gradio UI is mounted at /ui (same process, same port)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.api import health
from app.errors import InputRejectedError, LLMUnavailableError, RateLimitedError
from app.log import configure_logging, get_logger

log = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup")
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="OrbitDesk", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)

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
        return RedirectResponse("/ui")

    return app


app = create_app()
