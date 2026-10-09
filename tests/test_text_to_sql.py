"""Text-to-SQL safety: the validator (no DB, no LLM) and the database-level read-only defences."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
from sqlalchemy import text

from app.agent.core_tools import Identity, analytics
from app.analytics import text_to_sql as t2s
from app.analytics.text_to_sql import UnsafeSQLError, run_readonly, validate_sql
from app.db.seed import EMPLOYEES, TICKETS


def _status_counts() -> dict:
    from sqlalchemy import func, select

    from app.db.database import db_session, tickets

    with db_session() as conn:
        return dict(conn.execute(select(tickets.c.status, func.count()).group_by(tickets.c.status)).all())


ATTACKS = [
    json.loads(line)
    for line in (Path(__file__).resolve().parent.parent / "evals" / "suites" / "sql_attacks.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]


@pytest.mark.parametrize("dialect", ["postgres", "sqlite"])
@pytest.mark.parametrize("case", ATTACKS, ids=[c["id"] for c in ATTACKS])
def test_validator_blocks_every_attack(case, dialect):
    with pytest.raises(UnsafeSQLError):
        validate_sql(case["sql"], dialect=dialect)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT category, COUNT(*) AS n FROM v_tickets GROUP BY category",
        "SELECT department, headcount FROM v_departments ORDER BY headcount DESC",
        "SELECT t.department, COUNT(*) * 1.0 / d.headcount AS per_head FROM v_tickets t "
        "JOIN v_departments d ON d.department = t.department GROUP BY t.department, d.headcount",
    ],
)
def test_validator_allows_reads_of_views_and_adds_limit(sql):
    out = validate_sql(sql, dialect="postgres", row_limit=100)
    assert "LIMIT 100" in out


def test_validator_caps_a_large_limit():
    assert "LIMIT 100" in validate_sql("SELECT * FROM v_tickets LIMIT 100000", dialect="postgres", row_limit=100)
    assert "LIMIT 5" in validate_sql("SELECT * FROM v_tickets LIMIT 5", dialect="postgres", row_limit=100)


def test_extract_sql_from_model_output():
    assert t2s.extract_sql("Here you go:\n```sql\nSELECT 1 FROM v_tickets;\n```") == "SELECT 1 FROM v_tickets"


def test_analytics_tool_refuses_non_admins():
    out = json.loads(analytics(Identity("E1001", "employee"), "how many tickets?"))
    assert "only available to IT admins" in out["error"]


def test_analytics_end_to_end_with_stubbed_llm_sqlite(monkeypatch):
    monkeypatch.setattr(
        t2s, "generate_sql", lambda q, usage: "SELECT status, COUNT(*) AS n FROM v_tickets GROUP BY status"
    )
    out = json.loads(analytics(Identity("E1004", "it_admin"), "tickets by status"))
    assert "LIMIT" in out["sql"]
    assert dict(out["rows"]) == _status_counts()  # ground truth straight from the base table


def test_analytics_rejects_unsafe_llm_output(monkeypatch):
    monkeypatch.setattr(t2s, "generate_sql", lambda q, usage: "DELETE FROM tickets")
    out = json.loads(analytics(Identity("E1004", "it_admin"), "clean up tickets"))
    assert "rejected by the validator" in out["error"]


def test_sqlite_layer_is_read_only_even_without_the_validator():
    from sqlalchemy.exc import OperationalError as SAOperationalError

    from app.db.repo import count_tickets

    before = count_tickets()
    with pytest.raises((sqlite3.OperationalError, SAOperationalError), match="readonly"):
        run_readonly("DELETE FROM tickets")
    assert count_tickets() == before


# ---------------- Postgres: database-level defences (validator bypassed on purpose) ----------------


@pytest.mark.postgres
def test_pg_readonly_role_blocks_writes_and_base_tables(pg):
    from app.db.database import get_engine

    for sql in [
        "DELETE FROM tickets",
        "SELECT * FROM employees",
        "SELECT * FROM leave_balances",
        "UPDATE tickets SET status = 'closed'",
        "DROP TABLE tickets",
    ]:
        with pytest.raises(Exception) as e:
            run_readonly(sql)
        assert any(w in str(e.value).lower() for w in ("permission denied", "read-only", "must be owner"))
    with get_engine(pg).connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tickets")).scalar() == 24
        assert conn.execute(text("SELECT COUNT(*) FROM employees")).scalar() == 8


@pytest.mark.postgres
def test_pg_statement_timeout(pg):
    with pytest.raises(Exception) as e:
        run_readonly("SELECT pg_sleep(2) FROM v_departments", timeout_ms=200)
    assert "statement timeout" in str(e.value).lower() or "canceling statement" in str(e.value).lower()


@pytest.mark.postgres
def test_pg_analytics_end_to_end_with_stubbed_llm(pg, monkeypatch):
    monkeypatch.setattr(
        t2s,
        "generate_sql",
        lambda q, usage: (
            "SELECT department, COUNT(*) AS open_tickets FROM v_tickets "
            "WHERE status IN ('open', 'in_progress') GROUP BY department ORDER BY open_tickets DESC, department"
        ),
    )
    out = json.loads(analytics(Identity("E1004", "it_admin"), "open tickets by department"))
    dept = {e[0]: e[4] for e in EMPLOYEES}
    expected = Counter(dept[t[1]] for t in TICKETS if t[5] in ("open", "in_progress"))
    top = sorted(expected.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    assert out["rows"][0] == [top[0], top[1]]
