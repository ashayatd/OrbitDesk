"""Database layer: SQLAlchemy Core, one schema for Postgres (main) and SQLite (fallback + unit tests).

DATABASE_URL set   -> Postgres (e.g. postgresql+psycopg://user:pw@host:5432/orbitdesk, or Neon)
DATABASE_URL empty -> SQLite file under DATA_DIR (logged at startup as a fallback)

Core (tables + SQL expressions) rather than the ORM: the queries stay visible and easy to
explain, and dialect-specific features (ON CONFLICT, FOR UPDATE, SET ROLE) are explicit.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
    text,
)
from sqlalchemy.engine import Connection

from app.config import get_settings
from app.log import get_logger

log = get_logger(__name__)
metadata = MetaData()

ROLE_CHECK = "role IN ('employee', 'hr', 'it_admin')"

employees = Table(
    "employees",
    metadata,
    Column("id", String(16), primary_key=True),  # E1001
    Column("name", String(120), nullable=False),
    Column("email", String(200), nullable=False, unique=True),
    Column("role", String(16), nullable=False),
    Column("department", String(64), nullable=False),
    Column("manager_id", String(16), ForeignKey("employees.id")),
    CheckConstraint(ROLE_CHECK, name="ck_employees_role"),
)
Index("ix_employees_department", employees.c.department)

leave_balances = Table(
    "leave_balances",
    metadata,
    Column("employee_id", String(16), ForeignKey("employees.id", ondelete="CASCADE"), primary_key=True),
    Column("leave_type", String(32), primary_key=True),  # annual | sick | carry_over
    Column("entitled", Float, nullable=False),
    Column("used", Float, nullable=False, server_default="0"),
    CheckConstraint("entitled >= 0 AND used >= 0", name="ck_leave_nonnegative"),
)

tickets = Table(
    "tickets",
    metadata,
    # Integer id shown as "TCK-<id>". Identity on Postgres; INTEGER PRIMARY KEY (rowid) on SQLite.
    Column("id", Integer, Identity(start=1001), primary_key=True),
    Column("employee_id", String(16), ForeignKey("employees.id"), nullable=False),
    Column("title", String(200), nullable=False),
    Column("description", Text, nullable=False),
    Column("category", String(40), nullable=False),
    Column("priority", String(4), nullable=False),
    Column("status", String(16), nullable=False, server_default="open"),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("resolved_at", DateTime),
    # Idempotency: one ticket per key, enforced by the database (INSERT ... ON CONFLICT DO NOTHING).
    Column("idempotency_key", String(128), unique=True),
    CheckConstraint("priority IN ('P1', 'P2', 'P3', 'P4')", name="ck_tickets_priority"),
    CheckConstraint("status IN ('open', 'in_progress', 'resolved', 'closed')", name="ck_tickets_status"),
)
Index("ix_tickets_employee", tickets.c.employee_id)
Index("ix_tickets_status_priority", tickets.c.status, tickets.c.priority)
Index("ix_tickets_created_at", tickets.c.created_at)

pending_actions = Table(
    "pending_actions",
    metadata,
    Column("confirmation_id", String(64), primary_key=True),
    Column("kind", String(32), nullable=False),  # create_ticket
    Column("employee_id", String(16), ForeignKey("employees.id"), nullable=False),
    Column("session_id", String(100), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("summary", String(300), nullable=False),
    Column("status", String(16), nullable=False, server_default="pending"),
    Column("ticket_id", Integer, ForeignKey("tickets.id")),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("expires_at", DateTime, nullable=False),
    CheckConstraint("status IN ('pending', 'executed', 'cancelled')", name="ck_pending_status"),
)
Index("ix_pending_employee_status", pending_actions.c.employee_id, pending_actions.c.status)

documents = Table(
    "documents",
    metadata,
    Column("doc_id", String(100), primary_key=True),
    Column("filename", String(200), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("version", String(32)),
    Column("chunk_count", Integer, nullable=False),
    Column("indexed_at", DateTime, nullable=False, server_default=func.now()),
)

request_log = Table(
    "request_log",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("ts", DateTime, nullable=False, server_default=func.now()),
    Column("endpoint", String(32), nullable=False),
    Column("session_id", String(100)),
    Column("role", String(16)),
    Column("question", Text),  # PII-masked
    Column("answered", Boolean),
    Column("cache_hit", String(16)),
    Column("latency_ms", Float),
    Column("timings", JSON),
    Column("tokens_in", Integer),
    Column("tokens_out", Integer),
    Column("cost_usd", Float),
    Column("error", String(200)),
)
Index("ix_request_log_ts", request_log.c.ts)
Index("ix_request_log_endpoint", request_log.c.endpoint)

feedback = Table(
    "feedback",
    metadata,
    Column("id", Integer, Identity(), primary_key=True),
    Column("request_id", String(64)),
    Column("session_id", String(100)),
    Column("rating", String(8), nullable=False),
    Column("comment", Text),
    Column("ts", DateTime, nullable=False, server_default=func.now()),
    CheckConstraint("rating IN ('up', 'down')", name="ck_feedback_rating"),
)
Index("ix_feedback_request", feedback.c.request_id)

# Read-only views for text-to-SQL (ticket_analytics). No names, emails or free text: no PII.
ANALYTICS_VIEWS = {
    "v_tickets": """
        SELECT t.id AS ticket_id, t.category, t.priority, t.status, t.created_at, t.resolved_at,
               e.department, e.role AS requester_role
        FROM tickets t JOIN employees e ON e.id = t.employee_id""",
    "v_departments": """
        SELECT department, COUNT(*) AS headcount FROM employees GROUP BY department""",
}
ANALYTICS_ROLE = "orbitdesk_analytics"


def database_url(path: Path | None = None) -> str:
    """Explicit SQLite path (tests) > DATABASE_URL > default SQLite file."""
    if path is not None:
        return f"sqlite:///{Path(path).as_posix()}"
    s = get_settings()
    return s.database_url or f"sqlite:///{s.db_path.as_posix()}"


def get_engine(url: str | None = None) -> Engine:
    """Engine for `url`, or for the current DATABASE_URL / SQLite default. The cache is keyed by the
    RESOLVED url: caching get_engine() under None would pin whichever database was asked for first."""
    return _engine(url or database_url())


@lru_cache(maxsize=16)
def _engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"timeout": 15, "check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):
            dbapi_conn.isolation_level = None  # let SQLAlchemy emit BEGIN (below)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")  # per connection; WAL mode is set once in init_db
            cur.close()

        @event.listens_for(engine, "begin")
        def _sqlite_begin(conn):
            # IMMEDIATE takes the write lock up front: SQLite's equivalent of SELECT ... FOR UPDATE
            conn.exec_driver_sql("BEGIN IMMEDIATE")

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def is_postgres(engine: Engine | None = None) -> bool:
    return (engine or get_engine()).dialect.name == "postgresql"


def _engine_for(path: Path | None) -> Engine:
    return get_engine(database_url(path))


@contextmanager
def db_session(path: Path | None = None) -> Iterator[Connection]:
    """One transaction: commits on success, rolls back on error."""
    with _engine_for(path).begin() as conn:
        yield conn


_initialised: set[str] = set()
_init_lock = threading.Lock()


def init_db(path: Path | None = None) -> None:
    """Create tables, indexes, analytics views (and on Postgres the read-only role). Idempotent,
    and serialised per process: several threads may hit a fresh database at the same moment."""
    url = database_url(path)
    if url in _initialised:
        return
    with _init_lock:
        if url in _initialised:
            return
        engine = get_engine(url)
        if not is_postgres(engine):
            # WAL lets the api and worker read while one writes. It's stored in the DB file, so set it once
            # here: running it on every new connection races when several connect to a fresh file at once.
            with engine.connect() as conn:
                conn.connection.driver_connection.execute("PRAGMA journal_mode=WAL")
        metadata.create_all(engine)
        with engine.begin() as conn:
            for name, sql in ANALYTICS_VIEWS.items():
                if is_postgres(engine):
                    conn.execute(text(f"CREATE OR REPLACE VIEW {name} AS {sql}"))
                else:
                    conn.execute(text(f"CREATE VIEW IF NOT EXISTS {name} AS {sql}"))
        if is_postgres(engine):
            _create_analytics_role(engine)
        _initialised.add(url)
    if not get_settings().database_url and path is None:
        log.info("database_fallback_sqlite", url=url, hint="set DATABASE_URL to use Postgres")


def _create_analytics_role(engine: Engine) -> None:
    """A NOLOGIN role that can SELECT the two views and nothing else. The app switches to it
    with SET LOCAL ROLE inside a READ ONLY transaction to run LLM-written SQL."""
    try:
        with engine.begin() as conn:
            # DO blocks can't take bind parameters; ANALYTICS_ROLE is a constant, not user input.
            conn.execute(
                text(
                    f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{ANALYTICS_ROLE}') "
                    f"THEN CREATE ROLE {ANALYTICS_ROLE} NOLOGIN; END IF; END $$"
                )
            )
            conn.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {ANALYTICS_ROLE}"))
            conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {ANALYTICS_ROLE}"))
            conn.execute(text(f"GRANT SELECT ON {', '.join(ANALYTICS_VIEWS)} TO {ANALYTICS_ROLE}"))
            conn.execute(text(f"GRANT {ANALYTICS_ROLE} TO CURRENT_USER"))
    except Exception as e:  # e.g. a hosted DB that forbids CREATE ROLE: validation + READ ONLY still apply
        log.warning("analytics_role_unavailable", error=str(e)[:200])


def reset_initialised() -> None:
    """Tests: forget which URLs were initialised (after dropping tables)."""
    _initialised.clear()
