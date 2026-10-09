"""Run the OrbitDesk MCP server or mint a token.

    python -m app.mcp_server stdio                       # local clients; identity from MCP_EMPLOYEE_ID/MCP_ROLE
    python -m app.mcp_server http --port 8100            # standalone streamable HTTP (bearer tokens)
    python -m app.mcp_server token E1004 [--ttl 3600]    # print a bearer token for an MCP client

(The main API also serves streamable HTTP at /mcp/, so a separate process is optional.)
"""

import argparse
import sys

from app.db.repo import get_employee
from app.db.seed import seed
from app.log import configure_logging
from app.mcp_server.auth import mint_token
from app.mcp_server.server import build_server, stdio_identity_from_env


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m app.mcp_server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stdio")
    h = sub.add_parser("http")
    h.add_argument("--host", default="127.0.0.1")
    h.add_argument("--port", type=int, default=8100)
    t = sub.add_parser("token")
    t.add_argument("employee_id")
    t.add_argument("--ttl", type=int, default=3600)
    args = ap.parse_args()

    if args.cmd == "stdio":
        configure_logging("WARNING")  # stdout is the MCP channel: keep logs off it
        seed()
        build_server(http=False, stdio_identity=stdio_identity_from_env()).run("stdio")
    elif args.cmd == "http":
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        configure_logging()
        seed()
        mcp = build_server(http=True, base_url=f"http://{args.host}:{args.port}")
        app = Starlette(
            routes=[Mount("/mcp", app=mcp.streamable_http_app())], lifespan=lambda _: mcp.session_manager.run()
        )
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        seed()
        emp = get_employee(args.employee_id)
        if emp is None:
            sys.exit(f"unknown employee {args.employee_id}")
        print(mint_token(emp["id"], emp["role"], args.ttl))  # role comes from the employee record


if __name__ == "__main__":
    main()
