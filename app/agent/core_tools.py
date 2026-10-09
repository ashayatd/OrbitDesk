"""Tool logic with an EXPLICIT, trusted identity. Shared by:
- the in-process LangChain tools (app/agent/tools.py), and
- the MCP server (app/mcp_server/server.py),
so both paths behave the same. Identity is a parameter here, never something the model supplies.
"""

import json
from dataclasses import dataclass

from app.analytics.text_to_sql import UnsafeSQLError, ticket_analytics
from app.db.repo import get_leave_balance, get_ticket, parse_ticket_ref
from app.errors import LLMUnavailableError
from app.llm.usage import Usage
from app.schemas.ask import AskRequest


@dataclass(frozen=True)
class Identity:
    employee_id: str
    role: str
    session_id: str = "mcp"


def search_policies(rag_pipeline, who: Identity, question: str, trace_id: str | None = None) -> dict:
    """Phase 2 RAG pipeline, role-filtered by `who.role`. Returns answer + verified citations + usage."""
    events = rag_pipeline.run(
        AskRequest(question=question, session_id=who.session_id, role=who.role, stream=False),
        trace_id,
        use_memory=False,
    )
    res = [d for n, d in events if n == "result"][0]
    return {
        "answer": res.answer,
        "citations": [c.model_dump() for c in res.citations],
        "found": res.found,
        "usage_calls": res.tokens.calls,
    }


def ticket_status(who: Identity, ticket_id: str) -> str:
    if parse_ticket_ref(ticket_id) is None:
        return "Invalid ticket id format; expected something like TCK-1002."
    t = get_ticket(ticket_id)
    # it_admin may see any ticket; everyone else only their own (same message: don't leak existence)
    if not t or (who.role != "it_admin" and t["employee_id"] != who.employee_id):
        return f"No ticket {ticket_id.strip().upper()} found for you."
    return f"{t['ref']}: '{t['title']}' status={t['status']} priority={t['priority']} created={t['created_at']}"


def leave_balance(who: Identity) -> str:
    bal = get_leave_balance(who.employee_id)
    if not bal:
        return "No leave records found for this employee."
    return "; ".join(f"{t}: {v['remaining']:g} of {v['entitled']:g} days remaining" for t, v in bal.items())


def analytics(who: Identity, question: str, usage: Usage | None = None) -> str:
    """Returns JSON text with the SQL that ran and the rows (or a refusal/error message)."""
    try:
        res = ticket_analytics(question, who.role, usage)
    except PermissionError as e:
        return json.dumps({"error": str(e)})
    except LLMUnavailableError as e:
        return json.dumps({"error": e.detail})
    except UnsafeSQLError as e:
        return json.dumps({"error": f"generated SQL was rejected by the validator: {e}"})
    except Exception as e:  # e.g. statement_timeout, SQL error on a valid-but-wrong query
        return json.dumps({"error": f"query failed: {type(e).__name__}: {str(e)[:150]}"})
    return json.dumps(
        {"sql": res.sql, "columns": res.columns, "rows": res.rows, "truncated": res.truncated}, default=str
    )
