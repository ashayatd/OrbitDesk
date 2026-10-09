"""TOOLS_VIA_MCP=true: load search_policies / get_ticket_status / ticket_analytics from the MCP server.

The agent mints a short-lived bearer token for the CURRENT user (employee + role from the
request context), and langchain-mcp-adapters turns the server's tools into LangChain tools.
The model still never supplies an identity: it travels in the Authorization header.

The adapter's tools are async-only, while this graph runs synchronously in a worker thread, so
calls are executed on one background event loop and wrapped as ordinary sync tools.
"""

import asyncio
import threading
from concurrent.futures import Future

from langchain_core.tools import BaseTool, StructuredTool

from app.agent.tools import (
    ANALYTICS_DOC,
    SEARCH_DOC,
    STATUS_DOC,
    AnalyticsArgs,
    SearchArgs,
    TicketStatusArgs,
    citations_from_mcp,
)
from app.config import get_settings
from app.mcp_server.auth import mint_token

_loop: asyncio.AbstractEventLoop | None = None
_lock = threading.Lock()


def _run(coro, timeout: float = 120):
    """Run a coroutine on the shared background loop and wait for it."""
    global _loop
    with _lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True, name="mcp-client-loop").start()
    fut: Future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result(timeout=timeout)


def _text(result) -> str:
    """Adapter results may be a string or a list of content blocks; the model needs text."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    return str(result)


def fetch_remote_tools(employee_id: str, role: str) -> dict[str, BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    token = mint_token(employee_id, role, ttl_s=300)
    client = MultiServerMCPClient(
        {
            "orbitdesk": {
                "transport": "streamable_http",
                "url": get_settings().mcp_url,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    )
    return {t.name: t for t in _run(client.get_tools())}


def load_mcp_tools(ctx, logged) -> list[BaseTool]:
    remote = fetch_remote_tools(ctx.employee_id, ctx.role)

    def call(name: str, **kwargs) -> str:
        return _text(_run(remote[name].ainvoke(kwargs)))

    def search_policies(question: str) -> str:
        answer, citations = citations_from_mcp(call("search_policies", question=question))
        ctx.citations += citations  # same response shape as in-process mode
        return answer

    def get_ticket_status(ticket_id: str) -> str:
        return call("get_ticket_status", ticket_id=ticket_id)

    def ticket_analytics(question: str) -> str:
        return call("ticket_analytics", question=question)

    def tool(name, fn, doc, schema):
        return StructuredTool.from_function(logged(name, fn), name=name, description=doc, args_schema=schema)

    tools = [
        tool("search_policies", search_policies, SEARCH_DOC, SearchArgs),
        tool("get_ticket_status", get_ticket_status, STATUS_DOC, TicketStatusArgs),
    ]
    if ctx.role == "it_admin":  # same visibility as in-process mode; the server enforces it too
        tools.append(tool("ticket_analytics", ticket_analytics, ANALYTICS_DOC, AnalyticsArgs))
    return tools
