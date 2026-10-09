"""Request log + feedback (SQLite) and the aggregates shown in the admin view.

Only PII-masked questions are stored (the pipeline masks before anything else sees them).
"""

import uuid
from datetime import datetime, timedelta

from sqlalchemy import case, func, insert, select

from app.db.database import db_session, feedback, init_db, request_log
from app.db.dialect import upsert
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
        values = {
            "endpoint": endpoint,
            "session_id": session_id,
            "role": role,
            "question": (question or "")[:500],
            "answered": answered,
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "timings": timings or {},
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "error": error,
        }
        with db_session() as conn:
            conn.execute(upsert(conn, request_log, "id", {"id": request_id or uuid.uuid4().hex, **values}, values))
    except Exception:
        log.exception("request_log_failed")


def _refused(res) -> bool:
    """A guardrail refusal (blocked input or output) is not a knowledge gap: keep it out of 'unanswered'."""
    g = res.guardrails or {}
    return g.get("input_check", {}).get("label", "safe") != "safe" or "output_blocked" in g


def log_ask_result(endpoint: str, res, session_id: str, role: str) -> None:
    log_request(
        endpoint,
        request_id=res.request_id,
        session_id=session_id,
        role=role,
        question=res.question,
        answered=None if _refused(res) else res.found,
        cache_hit=res.cache,
        latency_ms=res.timings_ms.get("total"),
        timings=res.timings_ms,
        tokens_in=res.tokens.input_tokens,
        tokens_out=res.tokens.output_tokens,
        cost_usd=res.tokens.cost_usd,
    )


def log_agent_result(endpoint: str, out: dict, session_id: str, role: str) -> None:
    stopped = out.get("stopped") or ""
    refused = out.get("guardrails", {}).get("input_check", {}).get("label", "safe") != "safe"
    log_request(
        endpoint,
        request_id=out.get("request_id"),
        session_id=session_id,
        role=role,
        question=out.get("question"),  # PII-masked by the runner
        answered=None if refused else not stopped,
        latency_ms=out["timings_ms"]["total"],
        tokens_in=out["tokens"]["input_tokens"],
        tokens_out=out["tokens"]["output_tokens"],
        cost_usd=out["tokens"]["cost_usd"],
        error=stopped if stopped.startswith("error") else None,
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
        return conn.execute(
            insert(feedback)
            .values(request_id=request_id, session_id=session_id, rating=rating, comment=(comment or "")[:2000])
            .returning(feedback.c.id)
        ).scalar_one()


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def admin_stats(days: int = 14) -> dict:
    """Aggregations are done in the database (GROUP BY), one query per panel, not row by row in Python."""
    init_db()
    since = datetime.now() - timedelta(days=int(days))
    rl = request_log.c
    errors = func.sum(case((rl.error.is_not(None), 1), else_=0))
    unanswered = func.sum(case((rl.answered.is_(False), 1), else_=0))
    in_window = rl.ts >= since
    day = func.date(rl.ts)
    with db_session() as conn:
        per_day = conn.execute(
            select(
                day.label("day"),
                func.count().label("requests"),
                errors.label("errors"),
                unanswered.label("unanswered"),
                func.avg(rl.latency_ms).label("avg_latency_ms"),
            )
            .where(in_window)
            .group_by(day)
            .order_by(day)
        ).all()
        totals = conn.execute(
            select(
                func.count().label("n"),
                errors.label("errors"),
                unanswered.label("unanswered"),
                func.sum(case((rl.cache_hit.in_(("exact", "semantic")), 1), else_=0)).label("cache_hits"),
                func.sum(case((rl.endpoint != "/agent", 1), else_=0)).label("cacheable"),  # RAG paths use the cache
                func.avg(rl.latency_ms).label("avg_latency_ms"),
                func.sum(func.coalesce(rl.cost_usd, 0)).label("cost_usd"),
            ).where(in_window)
        ).one()
        q_norm = func.lower(func.trim(rl.question))
        top_unanswered = conn.execute(
            select(func.min(rl.question).label("question"), func.count().label("times"))
            .where(rl.answered.is_(False), rl.error.is_(None), in_window)
            .group_by(q_norm)
            .order_by(func.count().desc())
            .limit(10)
        ).all()
        by_endpoint = conn.execute(
            select(rl.endpoint, func.count().label("n"), func.avg(rl.latency_ms).label("avg_latency_ms"))
            .where(in_window)
            .group_by(rl.endpoint)
            .order_by(func.count().desc())
        ).all()
        fb = conn.execute(
            select(feedback.c.rating, func.count().label("n")).where(feedback.c.ts >= since).group_by(feedback.c.rating)
        ).all()
        recent_down = conn.execute(
            select(feedback.c.ts, feedback.c.comment, rl.question)
            .select_from(feedback.outerjoin(request_log, rl.id == feedback.c.request_id))
            .where(feedback.c.rating == "down")
            .order_by(feedback.c.ts.desc())
            .limit(10)
        ).all()
    n = totals.n or 0
    return {
        "window_days": days,
        "requests": n,
        "failure_rate": round((totals.errors or 0) / n, 4) if n else None,
        "unanswered_rate": round((totals.unanswered or 0) / n, 4) if n else None,
        "avg_latency_ms": _r(totals.avg_latency_ms),
        "cache_hit_rate": round((totals.cache_hits or 0) / totals.cacheable, 4) if totals.cacheable else None,
        "est_cost_usd": _r(totals.cost_usd, 6),
        "questions_per_day": [
            {
                "day": str(r.day),
                "requests": r.requests,
                "errors": r.errors,
                "unanswered": r.unanswered,
                "avg_latency_ms": _r(r.avg_latency_ms),
            }
            for r in per_day
        ],
        "by_endpoint": [
            {"endpoint": r.endpoint, "n": r.n, "avg_latency_ms": _r(r.avg_latency_ms)} for r in by_endpoint
        ],
        "top_unanswered": [{"question": r.question, "times": r.times} for r in top_unanswered],
        "feedback": {r.rating: r.n for r in fb},
        "recent_thumbs_down": [{"ts": str(r.ts), "comment": r.comment, "question": r.question} for r in recent_down],
    }
