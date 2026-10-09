"""Seed fictional employees, leave balances and tickets. Idempotent (ON CONFLICT DO NOTHING).

Tickets have fixed dates and statuses so the text-to-SQL eval (evals/sql_eval.py) has
deterministic expected answers.
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select, text

from app.db.database import database_url, db_session, employees, init_db, is_postgres, leave_balances, tickets
from app.db.dialect import insert_ignore

EMPLOYEES = [
    # id, name, email, role, department, manager_id  (managers first: manager_id is a foreign key)
    ("E1005", "Aisha Bello", "aisha.bello@orbit.example", "employee", "Engineering", None),
    ("E1006", "Lucas Martin", "lucas.martin@orbit.example", "employee", "Sales", None),
    ("E1001", "Priya Raman", "priya.raman@orbit.example", "employee", "Engineering", "E1005"),
    ("E1002", "Daniel Okafor", "daniel.okafor@orbit.example", "employee", "Sales", "E1006"),
    ("E1003", "Mei Lin", "mei.lin@orbit.example", "hr", "HR", None),
    ("E1004", "Tomas Novak", "tomas.novak@orbit.example", "it_admin", "IT", None),
    ("E1007", "Sofia Rossi", "sofia.rossi@orbit.example", "employee", "Finance", None),
    ("E1008", "Kenji Sato", "kenji.sato@orbit.example", "employee", "Engineering", "E1005"),
]

# employee_id -> {leave_type: (entitled, used)}
LEAVE = {
    "E1001": {"annual": (24, 9), "sick": (10, 2), "carry_over": (5, 1)},
    "E1002": {"annual": (24, 20.5), "sick": (10, 0), "carry_over": (3, 3)},
    "E1003": {"annual": (24, 4), "sick": (10, 1), "carry_over": (0, 0)},
    "E1004": {"annual": (24, 12), "sick": (10, 5), "carry_over": (5, 0)},
    "E1005": {"annual": (24, 15), "sick": (10, 3), "carry_over": (2, 2)},
    "E1006": {"annual": (24, 6), "sick": (10, 0), "carry_over": (5, 5)},
    "E1007": {"annual": (24, 10), "sick": (10, 4), "carry_over": (1, 0)},
    "E1008": {"annual": (24, 2), "sick": (10, 0), "carry_over": (0, 0)},
}

# id, employee, title, category, priority, status, created (YYYY-MM-DD), resolved (or None)
TICKETS = [
    (1001, "E1001", "VPN-809 at hotel", "VPN", "P3", "resolved", "2026-09-01", "2026-09-02"),
    (1002, "E1002", "Laptop blue screen", "Hardware", "P3", "open", "2026-09-03", None),
    (1003, "E1008", "MFA push not arriving", "MFA re-enrolment", "P3", "resolved", "2026-09-04", "2026-09-04"),
    (1004, "E1002", "Phishing email clicked", "Security", "P1", "resolved", "2026-09-05", "2026-09-05"),
    (1005, "E1007", "Need Tableau licence", "Software Request", "P4", "closed", "2026-09-07", "2026-09-12"),
    (1006, "E1001", "GitLab group access", "Access Request", "P4", "resolved", "2026-09-08", "2026-09-10"),
    (1007, "E1005", "VPN drops every hour", "VPN", "P3", "in_progress", "2026-09-10", None),
    (1008, "E1006", "Laptop stolen from car", "Security", "P2", "resolved", "2026-09-11", "2026-09-11"),
    (1009, "E1008", "Docker Desktop install", "Software Request", "P4", "resolved", "2026-09-12", "2026-09-15"),
    (1010, "E1007", "Password expired abroad", "Password", "P3", "resolved", "2026-09-14", "2026-09-14"),
    (1011, "E1002", "VPN-691 auth failed", "VPN", "P3", "resolved", "2026-09-15", "2026-09-16"),
    (1012, "E1001", "Monitor flickering", "Hardware", "P4", "open", "2026-09-16", None),
    (1013, "E1003", "OrbitHR access for new HRBP", "Access Request", "P3", "resolved", "2026-09-17", "2026-09-18"),
    (1014, "E1005", "Build server outage", "Other", "P1", "resolved", "2026-09-18", "2026-09-18"),
    (1015, "E1006", "CRM export blocked", "Access Request", "P3", "in_progress", "2026-09-21", None),
    (1016, "E1008", "Disk full warning", "Hardware", "P4", "resolved", "2026-09-22", "2026-09-22"),
    (1017, "E1007", "VPN on hotel wifi", "VPN", "P3", "open", "2026-09-23", None),
    (1018, "E1004", "Admin account review", "Access Request", "P4", "closed", "2026-09-24", "2026-09-25"),
    (1019, "E1001", "Suspicious login alert", "Security", "P2", "resolved", "2026-09-25", "2026-09-25"),
    (1020, "E1002", "New phone MFA", "MFA re-enrolment", "P3", "open", "2026-09-28", None),
    (1021, "E1005", "Python 3.12 request", "Software Request", "P4", "open", "2026-09-29", None),
    (1022, "E1006", "Laptop will not turn on", "Hardware", "P2", "in_progress", "2026-09-30", None),
    (1023, "E1008", "Wi-Fi keeps disconnecting", "Hardware", "P3", "open", "2026-10-01", None),
    (1024, "E1003", "Payroll tool access", "Access Request", "P3", "open", "2026-10-02", None),
]


def _dt(day: str | None) -> datetime | None:
    return datetime.fromisoformat(day + "T09:00:00") if day else None


def seed(path: Path | None = None) -> None:
    init_db(path)
    with db_session(path) as conn:
        cols = ("id", "name", "email", "role", "department", "manager_id")
        conn.execute(insert_ignore(conn, employees), [dict(zip(cols, e, strict=True)) for e in EMPLOYEES])
        conn.execute(
            insert_ignore(conn, leave_balances),
            [
                {"employee_id": emp, "leave_type": lt, "entitled": ent, "used": used}
                for emp, balances in LEAVE.items()
                for lt, (ent, used) in balances.items()
            ],
        )
        conn.execute(
            insert_ignore(conn, tickets),
            [
                {
                    "id": tid,
                    "employee_id": emp,
                    "title": title,
                    "description": f"{title} (seed data)",
                    "category": cat,
                    "priority": pri,
                    "status": status,
                    "created_at": _dt(created),
                    "resolved_at": _dt(resolved),
                    "idempotency_key": f"seed-{tid}",
                }
                for tid, emp, title, cat, pri, status, created, resolved in TICKETS
            ],
        )
        if is_postgres(conn.engine):
            # explicit ids were inserted, so move the identity sequence past them
            max_id = conn.execute(select(func.max(tickets.c.id))).scalar_one()
            conn.execute(text("SELECT setval(pg_get_serial_sequence('tickets', 'id'), :m)"), {"m": max_id})


if __name__ == "__main__":
    seed()
    print("seeded", database_url())
