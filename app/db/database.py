"""SQLite: mock enterprise systems (employees, leave, tickets) + app tables.

Plain sqlite3, no ORM: the schema is small and the SQL is easier to explain.
"""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id TEXT PRIMARY KEY,            -- e.g. E1001
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('employee', 'hr', 'it_admin')),
    department TEXT NOT NULL,
    manager_id TEXT
);
CREATE TABLE IF NOT EXISTS leave_balances (
    employee_id TEXT NOT NULL REFERENCES employees(id),
    leave_type TEXT NOT NULL,       -- annual | sick | carry_over
    entitled REAL NOT NULL,
    used REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (employee_id, leave_type)
);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,            -- e.g. TCK-1001
    employee_id TEXT NOT NULL REFERENCES employees(id),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    idempotency_key TEXT UNIQUE     -- second line of defence against duplicate creates
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    version TEXT,
    chunk_count INTEGER NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS request_log (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    endpoint TEXT NOT NULL,
    session_id TEXT,
    role TEXT,
    question TEXT,                  -- PII-masked
    answered INTEGER,               -- 0 = "couldn't find" / refusal
    cache_hit TEXT,                 -- none | exact | semantic
    latency_ms REAL,
    timings TEXT,                   -- JSON per-stage ms
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    session_id TEXT,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT,
    ts TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)  # waits up to 10s on a locked DB instead of failing
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Connection that commits on success and rolls back on error."""
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_initialised: set[Path] = set()
_init_lock = threading.Lock()


def init_db(path: Path | None = None) -> None:
    """Create tables once per process. WAL mode (readers don't block the writer, so the
    api and worker can share the file) is persisted in the DB file, so it's set here only."""
    path = path or get_settings().db_path
    with _init_lock:
        if path in _initialised:
            return
        with db_session(path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        _initialised.add(path)
