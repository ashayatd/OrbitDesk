"""Test setup: no API keys, in-process fake Redis, temp data dir."""

import os

import pytest

# Must be set before app modules read settings.
os.environ.update(
    {
        "ORBITDESK_ENV_FILE": "__no_env_file__",
        "GROQ_API_KEY": "",
        "FALLBACK_PROVIDER_KEY": "",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "",
        "REDIS_URL": "fakeredis://",
        "QUEUE_MODE": "inline",
        "INGEST_ON_STARTUP": "false",
    }
)


@pytest.fixture(autouse=True)
def _flush_redis():
    from app.infra.redis_client import get_redis

    get_redis().flushall()
    yield
