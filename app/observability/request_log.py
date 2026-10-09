"""Request log + feedback (SQLite) and the aggregates shown in the admin view.

Only PII-masked questions are stored (the pipeline masks before anything else sees them).
"""

import json
import uuid

from app.db.database import db_session, init_db
from app.log import get_logger

log = get_logger(__name__)


def log_request(
    endpoint: str,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    role: str | None = None,
    question: str | None = None,
    answered: bool | None = None,
    cache_hit: str = "none",
    latency_ms: float | None = None,
    timings: dict | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float | None = None,
    error: str | None = None,
) -> None:
    """Never raises: losing a log row must not fail a user request."""
    try:
        init_db()
        with db_session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO request_log (id, endpoint, session_id, role, question, answered, cache_hit,"
                " latency_ms, timings, tokens_in, tokens_out, cost_usd, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id or uuid.uuid4().hex,
                    endpoint,
                    session_id,
                    role,
                    (question or "")[:500],
                    None if answered is None else int(answered),
                    cache_hit,
                    latency_ms,
                    json.dumps(timings or {}),
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    error,
                ),
            )
    except Exception:
        log.exception("request_log_failed")


def log_ask_result(endpoint: str, res, session_id: str, role: str) -> None:
    log_request(
        endpoint,
        request_id=res.request_id,
        session_id=session_id,
        role=role,
        question=res.question,
        answered=res.found,
        cache_hit=res.cache,
        latency_ms=res.timings_ms.get("total"),
        timings=res.timings_ms,
        tokens_in=res.tokens.input_tokens,
        tokens_out=res.tokens.output_tokens,
        cost_usd=res.tokens.cost_usd,
    )


def logged(events, endpoint: str, session_id: str, role: str, question: str = ""):
    """Pass pipeline events through unchanged; write one request_log row for the result or the error."""
    try:
        for name, data in events:
            if name == "result":
                log_ask_result(endpoint, data, session_id, role)
            yield name, data
    except Exception as e:
        log_request(endpoint, session_id=session_id, role=role, question=question[:200], error=type(e).__name__)
        raise


def save_feedback(request_id: str | None, session_id: str | None, rating: str, comment: str | None) -> int:
    init_db()
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO feedback (request_id, session_id, rating, comment) VALUES (?,?,?,?)",
            (request_id, session_id, rating, (comment or "")[:2000]),
        )
        return int(cur.lastrowid)


def admin_stats(days: int = 14) -> dict:
    """Aggregations are done in SQL (GROUP BY), one query per panel, not row-by-row in Python."""
    init_db()
    since = f"-{int(days)} days"
    with db_session() as conn:
        per_day = conn.execute(
            "SELECT date(ts) AS day, COUNT(*) AS requests,"
            " SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,"
            " SUM(CASE WHEN answered = 0 THEN 1 ELSE 0 END) AS unanswered,"
            " ROUND(AVG(latency_ms), 1) AS avg_latency_ms"
            " FROM request_log WHERE ts >= datetime('now', ?) GROUP BY day ORDER BY day",
            (since,),
        ).fetchall()
        totals = conn.execute(
            "SELECT COUNT(*) AS n,"
            " SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,"
            " SUM(CASE WHEN answered = 0 THEN 1 ELSE 0 END) AS unanswered,"
            " SUM(CASE WHEN cache_hit IN ('exact','semantic') THEN 1 ELSE 0 END) AS cache_hits,"
            " SUM(CASE WHEN endpoint NOT IN ('/agent') THEN 1 ELSE 0 END) AS cacheable,"  # RAG paths use the cache
            " ROUND(AVG(latency_ms), 1) AS avg_latency_ms, ROUND(SUM(COALESCE(cost_usd,0)), 6) AS cost_usd"
            " FROM request_log WHERE ts >= datetime('now', ?)",
            (since,),
        ).fetchone()
        top_unanswered = conn.execute(
            "SELECT question, COUNT(*) AS times FROM request_log"
            " WHERE answered = 0 AND error IS NULL AND ts >= datetime('now', ?)"
            " GROUP BY lower(trim(question)) ORDER BY times DESC LIMIT 10",
            (since,),
        ).fetchall()
        by_endpoint = conn.execute(
            "SELECT endpoint, COUNT(*) AS n, ROUND(AVG(latency_ms),1) AS avg_latency_ms FROM request_log"
            " WHERE ts >= datetime('now', ?) GROUP BY endpoint ORDER BY n DESC",
            (since,),
        ).fetchall()
        fb = conn.execute(
            "SELECT rating, COUNT(*) AS n FROM feedback WHERE ts >= datetime('now', ?) GROUP BY rating", (since,)
        ).fetchall()
        recent_down = conn.execute(
            "SELECT f.ts, f.comment, r.question FROM feedback f LEFT JOIN request_log r ON r.id = f.request_id"
            " WHERE f.rating = 'down' ORDER BY f.ts DESC LIMIT 10"
        ).fetchall()
    n = totals["n"] or 0
    return {
        "window_days": days,
        "requests": n,
        "failure_rate": round((totals["errors"] or 0) / n, 4) if n else None,
        "unanswered_rate": round((totals["unanswered"] or 0) / n, 4) if n else None,
        "avg_latency_ms": totals["avg_latency_ms"],
        "cache_hit_rate": round((totals["cache_hits"] or 0) / totals["cacheable"], 4) if totals["cacheable"] else None,
        "est_cost_usd": totals["cost_usd"],
        "questions_per_day": [dict(r) for r in per_day],
        "by_endpoint": [dict(r) for r in by_endpoint],
        "top_unanswered": [dict(r) for r in top_unanswered],
        "feedback": {r["rating"]: r["n"] for r in fb},
        "recent_thumbs_down": [dict(r) for r in recent_down],
    }
