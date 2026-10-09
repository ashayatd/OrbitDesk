"""Agent tools, built per request as closures over a trusted AgentContext.

Security property: identity (employee_id, role) is captured from the request context when the
tools are built. It is NOT a tool argument, so the model can't read someone else's leave balance
even if a prompt injection tells it to.

TOOLS_VIA_MCP=true: search_policies, get_ticket_status and ticket_analytics are loaded from the
MCP server instead (app/agent/mcp_tools.py); the identity then travels as a signed bearer token.
get_leave_balance and create_ticket always stay in-process (create_ticket is part of the
confirm-before-write flow).
"""

import json
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.agent import core_tools
from app.agent.actions import TICKET_CATEGORIES, PendingAction, propose_ticket
from app.config import get_settings
from app.llm.usage import Usage


@dataclass
class AgentContext:
    employee_id: str
    role: str
    session_id: str
    usage: Usage = field(default_factory=Usage)
    pending: list[PendingAction] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)

    @property
    def identity(self) -> core_tools.Identity:
        return core_tools.Identity(self.employee_id, self.role, self.session_id)


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


class AnalyticsArgs(BaseModel):
    question: str = Field(description="Question about ticket statistics, e.g. 'open P1 tickets by department'")


SEARCH_DOC = "Search Orbit Systems HR policies and IT runbooks; returns a cited answer."
STATUS_DOC = "Get the status of one of the current user's IT tickets."
ANALYTICS_DOC = (
    "IT admins only: answer statistics questions about helpdesk tickets (counts, trends, breakdowns by "
    "category/priority/status/department). Returns the SQL that ran and the result rows."
)


def build_tools(ctx: AgentContext, rag_pipeline) -> list[BaseTool]:
    def search_policies(question: str) -> str:
        out = core_tools.search_policies(rag_pipeline, ctx.identity, question, ctx.usage.trace_id)
        for c in out["usage_calls"]:
            ctx.usage.add(f"search_policies/{c['stage']}", c["model"], c["in"], c["out"])
        ctx.citations += out["citations"]
        return out["answer"]

    def get_leave_balance() -> str:
        return core_tools.leave_balance(ctx.identity)  # identity from context, never from the model

    def create_ticket(title: str, description: str, category: str, priority: str) -> str:
        action = propose_ticket(
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
        return core_tools.ticket_status(ctx.identity, ticket_id)

    def ticket_analytics(question: str) -> str:
        return core_tools.analytics(ctx.identity, question, ctx.usage)

    def logged(name: str, fn):
        def wrapper(**kwargs):
            out = fn(**kwargs)
            ctx.tool_log.append({"tool": name, "args": kwargs, "result": str(out)[:300]})
            return out

        return wrapper

    def tool(name, fn, doc, schema):
        return StructuredTool.from_function(logged(name, fn), name=name, description=doc, args_schema=schema)

    local = [
        tool("get_leave_balance", get_leave_balance, "Get the current user's remaining leave days.", NoArgs),
        tool("create_ticket", create_ticket, "Propose an IT ticket. Not created until the user confirms.", TicketArgs),
    ]
    if get_settings().tools_via_mcp:
        from app.agent.mcp_tools import load_mcp_tools

        remote = load_mcp_tools(ctx, logged)
    else:
        remote = [
            tool("search_policies", search_policies, SEARCH_DOC, SearchArgs),
            tool("get_ticket_status", get_ticket_status, STATUS_DOC, TicketStatusArgs),
        ]
        if ctx.role == "it_admin":  # other roles never even see the tool
            remote.append(tool("ticket_analytics", ticket_analytics, ANALYTICS_DOC, AnalyticsArgs))
    return remote + local


def usage_from_response(res) -> Usage:
    u = Usage()
    for c in res.tokens.calls:
        u.add(f"search_policies/{c['stage']}", c["model"], c["in"], c["out"])
    return u


def citations_from_mcp(text: str) -> tuple[str, list[dict]]:
    """MCP search_policies returns JSON {answer, citations}; the model sees only the answer."""
    try:
        data = json.loads(text)
        return data.get("answer", text), data.get("citations", [])
    except (json.JSONDecodeError, AttributeError):
        return text, []
