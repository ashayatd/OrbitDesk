"""OrbitDesk MCP server (official Python MCP SDK, FastMCP).

Tools: search_policies, get_ticket_status, ticket_analytics. They call the same core functions
as the in-process agent tools (app/agent/core_tools.py), so behaviour is identical.

Who is calling (employee + role) is NEVER a tool argument:
- streamable HTTP: from the verified bearer token (app/mcp_server/auth.py), via the SDK's auth middleware;
- stdio (local, e.g. Claude Desktop): from server config, MCP_EMPLOYEE_ID + MCP_ROLE env vars,
  checked against the employee record at startup.
"""

import json
import os

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.agent import core_tools
from app.agent.core_tools import Identity
from app.db.repo import get_employee
from app.mcp_server.auth import HmacTokenVerifier

INSTRUCTIONS = (
    "OrbitDesk tools for the fictional company Orbit Systems: search HR policies and IT runbooks "
    "(answers come with verified citations), check a ticket's status, and (IT admins only) ask "
    "statistics questions about tickets. Results are filtered by the caller's role."
)


def stdio_identity_from_env() -> Identity:
    employee_id, role = os.environ.get("MCP_EMPLOYEE_ID", ""), os.environ.get("MCP_ROLE", "")
    emp = get_employee(employee_id) if employee_id else None
    if emp is None or emp["role"] != role:
        raise SystemExit("Set MCP_EMPLOYEE_ID and MCP_ROLE to a real employee and their role (e.g. E1004 / it_admin).")
    return Identity(employee_id, role, session_id=f"mcp-stdio-{employee_id}")


def build_server(
    *, http: bool, stdio_identity: Identity | None = None, base_url: str = "http://127.0.0.1:8000"
) -> FastMCP:
    kwargs: dict = {}
    if http:
        kwargs.update(
            token_verifier=HmacTokenVerifier(),
            auth=AuthSettings(issuer_url=base_url, resource_server_url=f"{base_url}/mcp"),
            streamable_http_path="/",  # mounted at /mcp by app.main
            stateless_http=True,  # no server-side session state: any API replica can serve any call
            json_response=True,
            # Host header checks would reject the Space's public hostname; every request is
            # still authenticated with a signed bearer token.
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
    mcp = FastMCP("orbitdesk", instructions=INSTRUCTIONS, **kwargs)

    def caller() -> Identity:
        tok = get_access_token()
        if tok is not None:
            role = tok.claims["role"] if tok.claims else tok.scopes[0].removeprefix("role:")
            return Identity(tok.client_id, role, session_id=f"mcp-{tok.client_id}")
        if stdio_identity is not None:
            return stdio_identity
        raise PermissionError("no authenticated caller")

    @mcp.tool()
    def search_policies(question: str) -> str:
        """Search Orbit Systems HR policies and IT runbooks. Returns JSON {answer, citations}; results are
        filtered by the caller's role, so restricted documents are never searched."""
        from app.services import get_rag_pipeline  # lazy: loads the embedding + rerank models

        out = core_tools.search_policies(get_rag_pipeline(), caller(), question)
        return json.dumps({"answer": out["answer"], "citations": out["citations"], "found": out["found"]})

    @mcp.tool()
    def get_ticket_status(ticket_id: str) -> str:
        """Status of an IT ticket like TCK-1002. Employees see only their own tickets; IT admins see all."""
        return core_tools.ticket_status(caller(), ticket_id)

    @mcp.tool()
    def ticket_analytics(question: str) -> str:
        """IT admins only: statistics about helpdesk tickets (counts, trends, breakdowns). Returns JSON with
        the SQL that ran (read-only, validated) and the rows."""
        return core_tools.analytics(caller(), question)

    return mcp
