"""Queries against the mock enterprise systems (Postgres or SQLite via SQLAlchemy Core)."""

import re
from pathlib import Path

from sqlalchemy import func, select

from app.db.database import db_session, employees, init_db, leave_balances, tickets
from app.db.dialect import insert_on_conflict_do_nothing

_TICKET_REF = re.compile(r"(?i)^TCK-(\d{4,6})$")


def ticket_ref(ticket_id: int) -> str:
    return f"TCK-{ticket_id}"


def parse_ticket_ref(ref: str) -> int | None:
    m = _TICKET_REF.match(ref.strip())
    return int(m.group(1)) if m else None


def ticket_dict(row) -> dict:
    d = dict(row._mapping)
    d["ref"] = ticket_ref(d["id"])
    for k in ("created_at", "resolved_at"):
        if d.get(k) is not None:
            d[k] = d[k].isoformat(sep=" ", timespec="seconds")
    return d


def get_employee(employee_id: str, db_path: Path | None = None) -> dict | None:
    init_db(db_path)
    with db_session(db_path) as conn:
        row = conn.execute(select(employees).where(employees.c.id == employee_id)).first()
    return dict(row._mapping) if row else None


def get_leave_balance(employee_id: str, db_path: Path | None = None) -> dict:
    """Remaining days per leave type: one query for all types, not one per type."""
    init_db(db_path)
    lb = leave_balances.c
    with db_session(db_path) as conn:
        rows = conn.execute(
            select(lb.leave_type, lb.entitled, lb.used, (lb.entitled - lb.used).label("remaining"))
            .where(lb.employee_id == employee_id)
            .order_by(lb.leave_type)
        ).all()
    return {r.leave_type: {"entitled": r.entitled, "used": r.used, "remaining": r.remaining} for r in rows}


def insert_ticket_once(conn, employee_id: str, payload: dict, idempotency_key: str) -> tuple[dict, bool]:
    """INSERT ... ON CONFLICT (idempotency_key) DO NOTHING RETURNING id, inside the caller's transaction.
    Returns (ticket, created). Repeating the key returns the first ticket with created=False."""
    stmt = (
        insert_on_conflict_do_nothing(conn, tickets, "idempotency_key")
        .values(employee_id=employee_id, idempotency_key=idempotency_key, **payload)
        .returning(tickets.c.id)
    )
    new_id = conn.execute(stmt).scalar_one_or_none()
    created = new_id is not None
    where = tickets.c.id == new_id if created else tickets.c.idempotency_key == idempotency_key
    return ticket_dict(conn.execute(select(tickets).where(where)).first()), created


def get_ticket(ref_or_id: str | int, db_path: Path | None = None) -> dict | None:
    tid = ref_or_id if isinstance(ref_or_id, int) else parse_ticket_ref(ref_or_id)
    if tid is None:
        return None
    init_db(db_path)
    with db_session(db_path) as conn:
        row = conn.execute(select(tickets).where(tickets.c.id == tid)).first()
    return ticket_dict(row) if row else None


def count_tickets(db_path: Path | None = None) -> int:
    init_db(db_path)
    with db_session(db_path) as conn:
        return conn.execute(select(func.count()).select_from(tickets)).scalar_one()
