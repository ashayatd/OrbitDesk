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


@pytest.fixture(scope="session")
def live_server():
    """The real app (incl. the MCP mount at /mcp/) on a free port in a background thread."""
    import socket
    import threading
    import time

    import uvicorn

    from app.config import get_settings
    from app.main import create_app

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    get_settings().mcp_url = f"http://127.0.0.1:{port}/mcp/"
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session", autouse=True)
def _mcp_mode(request):
    """TOOLS_VIA_MCP=true pytest: every agent test goes through the MCP server."""
    from app.config import get_settings

    if os.environ.get("TOOLS_VIA_MCP", "").lower() == "true":
        get_settings().tools_via_mcp = True
        request.getfixturevalue("live_server")
    yield
