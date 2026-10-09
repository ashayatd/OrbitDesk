"""Test setup: no API keys, in-process fake Redis, temp data dir."""

import os
import tempfile

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
        "MOUNT_UI": "false",  # test_ui.py builds the UI explicitly
        "DATA_DIR": tempfile.mkdtemp(prefix="orbitdesk-test-"),  # never touch the real data/ dir
    }
)


@pytest.fixture(autouse=True)
def _flush_redis():
    from app.infra.redis_client import get_redis

    get_redis().flushall()
    yield


@pytest.fixture(scope="session", autouse=True)
def _seed_db():
    from app.db.seed import seed

    seed()


@pytest.fixture
def pg(monkeypatch):
    """Point the app at a clean, seeded Postgres (TEST_DATABASE_URL) for one test."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from sqlalchemy import text

    from app.config import get_settings
    from app.db.database import get_engine, metadata, reset_initialised
    from app.db.seed import seed

    monkeypatch.setattr(get_settings(), "database_url", url)
    engine = get_engine(url)
    with engine.begin() as conn:
        for view in ("v_tickets", "v_departments"):
            conn.execute(text(f"DROP VIEW IF EXISTS {view}"))
    metadata.drop_all(engine)
    reset_initialised()
    seed()
    yield url
    reset_initialised()
