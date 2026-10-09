"""Text-to-SQL for the ticket_analytics tool (it_admin only), defended in depth.

1. The LLM sees only two views (v_tickets, v_departments): no names, emails or free text.
2. validate_sql() parses the SQL with sqlglot and rejects anything that is not exactly one
   SELECT (UNION of SELECTs allowed) over the allowlisted views: other tables, WITH, SELECT INTO,
   comments, semicolons, unknown/dangerous functions (pg_sleep, set_config, dblink, ...).
   It adds or caps LIMIT.
3. run_readonly() executes in a READ ONLY transaction, as the NOLOGIN role orbitdesk_analytics
   (SELECT on the two views only), with statement_timeout. On SQLite: PRAGMA query_only + a
   progress-handler timeout.
Even if layer 2 had a bug, layers 3 would refuse writes and reads of the base tables.
"""

import re
import time
from dataclasses import asdict, dataclass

import sqlglot
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import text
from sqlglot import exp

from app.config import get_settings
from app.db.database import ANALYTICS_ROLE, ANALYTICS_VIEWS, get_engine, init_db, is_postgres
from app.llm.client import get_llm
from app.llm.usage import Usage
from app.observability.tracing import traced
from app.rag.prompts import load_prompt

ALLOWED_VIEWS = frozenset(ANALYTICS_VIEWS)
# Functions sqlglot doesn't model become exp.Anonymous; only these are allowed by name.
SAFE_ANONYMOUS_FUNCS = frozenset({"date_part", "strftime", "julianday", "to_char", "age", "date"})
DANGEROUS_FUNC_PREFIXES = ("pg_", "lo_", "dblink", "set_config", "current_setting", "query_to_xml", "txid")
ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


class UnsafeSQLError(ValueError):
    pass


@dataclass
class AnalyticsResult:
    question: str
    sql: str
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def dialect_name() -> str:
    return "postgres" if is_postgres() else "sqlite"


def validate_sql(sql: str, dialect: str | None = None, row_limit: int | None = None) -> str:
    """Return a safe, LIMITed version of `sql` or raise UnsafeSQLError with the reason."""
    dialect = dialect or dialect_name()
    row_limit = row_limit or get_settings().analytics_row_limit
    raw = sql.strip()
    if not raw:
        raise UnsafeSQLError("empty query")
    if ";" in raw:
        raise UnsafeSQLError("semicolons are not allowed (single statement only)")
    if "--" in raw or "/*" in raw or "*/" in raw:
        raise UnsafeSQLError("comments are not allowed")
    try:
        statements = sqlglot.parse(raw, read=dialect)
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"could not parse SQL: {str(e)[:120]}") from e
    if len(statements) != 1 or statements[0] is None:
        raise UnsafeSQLError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, ALLOWED_ROOTS):
        raise UnsafeSQLError(f"only SELECT is allowed, got {type(tree).__name__.upper()}")
    for node in tree.walk():
        if isinstance(node, exp.With | exp.CTE):
            raise UnsafeSQLError("WITH clauses are not allowed")
        if isinstance(node, exp.Into):
            raise UnsafeSQLError("SELECT INTO is not allowed")
        if isinstance(node, exp.Insert | exp.Update | exp.Delete | exp.Drop | exp.Create | exp.Alter | exp.Command):
            raise UnsafeSQLError(f"{type(node).__name__.upper()} is not allowed")
        if isinstance(node, exp.Table):
            schema = (node.db or "").lower()
            if node.name.lower() not in ALLOWED_VIEWS or schema not in ("", "public", "main"):
                raise UnsafeSQLError(f"table/view '{node.sql(dialect=dialect)}' is not allowed")
        if isinstance(node, exp.Anonymous):
            name = node.name.lower()
            if name.startswith(DANGEROUS_FUNC_PREFIXES) or name not in SAFE_ANONYMOUS_FUNCS:
                raise UnsafeSQLError(f"function '{name}' is not allowed")
    if not any(isinstance(n, exp.Table) for n in tree.walk()):
        raise UnsafeSQLError("query must read from v_tickets or v_departments")

    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        lit = limit.expression if hasattr(limit, "expression") else None
        current = int(lit.this) if isinstance(lit, exp.Literal) and lit.is_int else None
    if current is None or current > row_limit:
        tree = tree.limit(row_limit)
    return tree.sql(dialect=dialect)


def _analytics_role_available(conn) -> bool:
    return bool(conn.execute(text("SELECT pg_has_role(current_user, :r, 'MEMBER')"), {"r": ANALYTICS_ROLE}).scalar())


def run_readonly(
    sql: str, timeout_ms: int | None = None, row_limit: int | None = None
) -> tuple[list[str], list[list], bool]:
    """Execute already-validated SQL with database-level read-only protections."""
    s = get_settings()
    timeout_ms = timeout_ms or s.analytics_statement_timeout_ms
    row_limit = row_limit or s.analytics_row_limit
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        if is_postgres(engine):
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                if _analytics_role_available(conn):
                    conn.execute(text(f"SET LOCAL ROLE {ANALYTICS_ROLE}"))
                conn.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
                result = conn.execute(text(sql))
                cols, rows = list(result.keys()), [list(r) for r in result.fetchmany(row_limit + 1)]
                conn.rollback()
        else:
            # Raw sqlite3 connection (autocommit, no implicit BEGIN IMMEDIATE from our engine hook).
            raw = conn.connection.driver_connection
            deadline = time.monotonic() + timeout_ms / 1000
            raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            raw.execute("PRAGMA query_only = ON")  # any write now fails with "readonly database"
            try:
                cur = raw.execute(sql)
                cols, rows = [d[0] for d in cur.description], [list(r) for r in cur.fetchmany(row_limit + 1)]
            finally:
                raw.execute("PRAGMA query_only = OFF")
                raw.set_progress_handler(None, 0)
    truncated = len(rows) > row_limit
    return cols, [_jsonable(r) for r in rows[:row_limit]], truncated


def _jsonable(row: list) -> list:
    return [
        v.isoformat(sep=" ") if hasattr(v, "isoformat") else (float(v) if hasattr(v, "is_finite") else v) for v in row
    ]


def generate_sql(question: str, usage: Usage) -> str:
    p = load_prompt("text_to_sql", "v1")
    dialect = dialect_name()
    date_hint = (
        "Dates: use date_trunc('week', created_at), created_at::date, and intervals like INTERVAL '7 days'."
        if dialect == "postgres"
        else "Dates: use date(created_at), strftime('%Y-%m', created_at) and julianday() for differences."
    )
    msgs = [
        SystemMessage(
            p["system"].format(dialect="PostgreSQL" if dialect == "postgres" else "SQLite", date_hint=date_hint)
        )
    ]
    for ex in p["few_shot"]:
        msgs += [
            HumanMessage(p["user_template"].format(question=ex["question"])),
            AIMessage(f"```sql\n{ex['sql']}\n```"),
        ]
    msgs.append(HumanMessage(p["user_template"].format(question=question)))
    resp = get_llm("main", max_tokens=400).invoke(msgs, config=traced(usage))
    usage.add_message("text_to_sql", resp, prompt_chars=sum(len(str(m.content)) for m in msgs))
    return extract_sql(str(resp.content))


def extract_sql(text_: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text_, re.S | re.I)
    sql = (m.group(1) if m else text_).strip()
    return sql.rstrip(";").strip()  # a single trailing semicolon is the only one we forgive


def ticket_analytics(question: str, role: str, usage: Usage | None = None) -> AnalyticsResult:
    """Full path: role check -> LLM writes SQL -> validate -> run read-only. Raises PermissionError /
    UnsafeSQLError; callers turn those into a refusal message."""
    if role != "it_admin":
        raise PermissionError("ticket_analytics is only available to IT admins")
    usage = usage or Usage()
    t0 = time.perf_counter()
    safe = validate_sql(generate_sql(question, usage))
    cols, rows, truncated = run_readonly(safe)
    return AnalyticsResult(
        question, safe, cols, rows, len(rows), truncated, round((time.perf_counter() - t0) * 1000, 1)
    )
