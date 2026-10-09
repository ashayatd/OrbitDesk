"""Queries against the mock enterprise systems (SQLite)."""

import sqlite3
from pathlib import Path

from app.db.database import db_session, init_db


def get_employee(employee_id: str, db_path: Path | None = None) -> dict | None:
    init_db(db_path)
    with db_session(db_path) as conn:
        row = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    return dict(row) if row else None


def get_leave_balance(employee_id: str, db_path: Path | None = None) -> dict:
    """Remaining days per leave type for one employee (one query, not one per type)."""
    init_db(db_path)
    with db_session(db_path) as conn:
        rows = conn.execute(
            "SELECT leave_type, entitled, used, entitled - used AS remaining FROM leave_balances"
            " WHERE employee_id = ? ORDER BY leave_type",
            (employee_id,),
        ).fetchall()
    return {r["leave_type"]: {"entitled": r["entitled"], "used": r["used"], "remaining": r["remaining"]} for r in rows}


def _next_ticket_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT MAX(CAST(SUBSTR(id, 5) AS INTEGER)) AS n FROM tickets").fetchone()
    return f"TCK-{(row['n'] or 1000) + 1}"


def create_ticket(
    employee_id: str,
    title: str,
    description: str,
    category: str,
    priority: str,
    idempotency_key: str,
    db_path: Path | None = None,
) -> tuple[dict, bool]:
    """Insert a ticket. Returns (ticket, created). If the idempotency key was already used,
    returns the existing ticket with created=False: the DB UNIQUE constraint is the last line
    of defence even if the Redis idempotency check were bypassed."""
    init_db(db_path)
    with db_session(db_path) as conn:
        existing = conn.execute("SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
        if existing:
            return dict(existing), False
        conn.execute("BEGIN IMMEDIATE")  # serialise id allocation across processes
        tid = _next_ticket_id(conn)
        try:
            conn.execute(
                "INSERT INTO tickets (id, employee_id, title, description, category, priority, idempotency_key)"
                " VALUES (?,?,?,?,?,?,?)",
                (tid, employee_id, title, description, category, priority, idempotency_key),
            )
        except sqlite3.IntegrityError:  # lost a race on the same key
            conn.rollback()
            row = conn.execute("SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            return dict(row), False
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (tid,)).fetchone()
    return dict(row), True


def get_ticket(ticket_id: str, db_path: Path | None = None) -> dict | None:
    init_db(db_path)
    with db_session(db_path) as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id.upper(),)).fetchone()
    return dict(row) if row else None


def count_tickets(db_path: Path | None = None) -> int:
    init_db(db_path)
    with db_session(db_path) as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()["n"]
