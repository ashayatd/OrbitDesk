"""Agent tools, built per request as closures over a trusted AgentContext.

Security property: identity (employee_id, role) is captured from the request
context when the tools are built. It is NOT a tool argument, so the model can't
read someone else's leave balance even if a prompt injection tells it to.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agent.actions import TICKET_CATEGORIES, PendingAction, propose_ticket
from app.db.repo import get_leave_balance, get_ticket
from app.infra.redis_client import get_redis
from app.llm.usage import Usage
from app.schemas.ask import AskRequest


@dataclass
class AgentContext:
    employee_id: str
    role: str
    session_id: str
    usage: Usage = field(default_factory=Usage)
    pending: list[PendingAction] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)


class NoArgs(BaseModel):
    pass


class SearchArgs(BaseModel):
    question: str = Field(description="Standalone question about Orbit HR/IT policies or runbooks")


class TicketArgs(BaseModel):
    title: str = Field(max_length=120, description="Short summary of the issue")
    description: str = Field(max_length=2000, description="Details: what happened, error codes, impact")
    category: Literal[TICKET_CATEGORIES] = Field(description="Ticket category")  # type: ignore[valid-type]
    priority: Literal["P1", "P2", "P3", "P4"] = Field(
        description="P1 company-wide outage/security breach, P2 team blocked, P3 one user blocked, P4 question"
    )


class TicketStatusArgs(BaseModel):
    ticket_id: str = Field(description="Ticket id like TCK-1002")


def build_tools(ctx: AgentContext, rag_pipeline) -> list[StructuredTool]:
    def search_policies(question: str) -> str:
        """Search Orbit Systems HR policies and IT runbooks; returns a cited answer."""
        events = rag_pipeline.run(  # request id = the agent's trace id, so one trace per agent turn
            AskRequest(question=question, session_id=ctx.session_id, role=ctx.role, stream=False),
            ctx.usage.trace_id,
            use_memory=False,
        )
        res = [d for n, d in events if n == "result"][0]
        ctx.usage.merge(usage_from_response(res))
        ctx.citations += [c.model_dump() for c in res.citations]
        return res.answer

    def get_my_leave_balance() -> str:
        """Get the current user's remaining leave days (annual, sick, carry-over)."""
        bal = get_leave_balance(ctx.employee_id)  # identity from context, never from the model
        if not bal:
            return "No leave records found for this employee."
        return "; ".join(f"{t}: {v['remaining']:g} of {v['entitled']:g} days remaining" for t, v in bal.items())

    def create_ticket(title: str, description: str, category: str, priority: str) -> str:
        """Propose an IT ticket. It is NOT created until the user confirms it in the UI."""
        action = propose_ticket(
            get_redis(),
            ctx.employee_id,
            ctx.session_id,
            {"title": title, "description": description, "category": category, "priority": priority},
        )
        ctx.pending.append(action)
        return (
            f"PENDING_CONFIRMATION: {action.summary}. Tell the user to press Confirm to create it. "
            "Do not say it has been created."
        )

    def get_ticket_status(ticket_id: str) -> str:
        """Get the status of one of the current user's IT tickets."""
        if not re.fullmatch(r"(?i)TCK-\d{4,6}", ticket_id.strip()):
            return "Invalid ticket id format; expected something like TCK-1002."
        t = get_ticket(ticket_id.strip())
        # it_admin may see any ticket; everyone else only their own (same message: don't leak existence)
        if not t or (ctx.role != "it_admin" and t["employee_id"] != ctx.employee_id):
            return f"No ticket {ticket_id.upper()} found for you."
        return f"{t['id']}: '{t['title']}' status={t['status']} priority={t['priority']} created={t['created_at']}"

    def logged(fn):
        def wrapper(**kwargs):
            out = fn(**kwargs)
            ctx.tool_log.append({"tool": fn.__name__, "args": kwargs, "result": out[:300]})
            return out

        return wrapper

    return [
        StructuredTool.from_function(
            logged(search_policies),
            name="search_policies",
            description=search_policies.__doc__,
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            logged(get_my_leave_balance),
            name="get_leave_balance",
            description=get_my_leave_balance.__doc__,
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            logged(create_ticket),
            name="create_ticket",
            description=create_ticket.__doc__,
            args_schema=TicketArgs,
        ),
        StructuredTool.from_function(
            logged(get_ticket_status),
            name="get_ticket_status",
            description=get_ticket_status.__doc__,
            args_schema=TicketStatusArgs,
        ),
    ]


def usage_from_response(res) -> Usage:
    u = Usage()
    for c in res.tokens.calls:
        u.add(f"search_policies/{c['stage']}", c["model"], c["in"], c["out"])
    return u
