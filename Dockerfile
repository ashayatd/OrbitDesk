# OrbitDesk demo image: API + Gradio UI + RQ worker + Redis + Chroma server in ONE container,
# for Hugging Face Spaces (Docker SDK, port 7860). docker-compose.yml reuses this image for
# separate api / worker services. Python 3.11 to match local development.
FROM python:3.11-slim

# redis-server: queue, cache, locks. supervisor: runs the 4 processes. curl: healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends redis-server supervisor curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

# Spaces run the container as uid 1000. /var/orbitdesk is the shared data volume in docker-compose
# (named volumes copy this directory's ownership on first mount, so uid 1000 can write to it).
RUN useradd -m -u 1000 user && mkdir -p /var/orbitdesk && chown user:user /var/orbitdesk
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /home/user/app

# 1) Dependencies first (cached layer). Runtime deps only: no dev/eval/finetune groups.
#    torch comes from the CPU-only index configured in pyproject.toml ([tool.uv.sources]).
COPY --chown=user pyproject.toml uv.lock ./
# --no-cache: without it uv leaves ~1.6 GB of downloaded wheels in this layer
RUN uv sync --frozen --no-default-groups --no-install-project --no-cache

# 2) Bake the local models into the image so the Space starts without downloading them.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 3) App code + data, then ingest the docs at BUILD time (embedded Chroma on disk; the Chroma
#    server started by supervisord at runtime serves the same directory). Startup ingestion
#    runs again at boot but skips every unchanged doc by content hash.
COPY --chown=user . .
RUN redis-server --daemonize yes --save "" --appendonly no \
 && QUEUE_MODE=inline python scripts/ingest_all.py \
 && redis-cli shutdown nosave

ENV REDIS_URL=redis://127.0.0.1:6379/0 \
    CHROMA_URL=http://127.0.0.1:8001 \
    QUEUE_MODE=rq
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://127.0.0.1:7860/health || exit 1
CMD ["supervisord", "-c", "/home/user/app/docker/supervisord.conf"]
