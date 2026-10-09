"""MCP server: auth, identity from token/config only, tools over streamable HTTP and stdio."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

import app.analytics.text_to_sql as t2s
from app.mcp_server.auth import mint_token, verify

ROOT = Path(__file__).resolve().parent.parent


def test_token_roundtrip_tamper_and_expiry():
    tok = mint_token("E1004", "it_admin")
    assert verify(tok)["sub"] == "E1004" and verify(tok)["role"] == "it_admin"
    payload, sig = tok.split(".")
    assert verify(payload + "." + sig[:-2] + "AA") is None  # bad signature
    assert verify(mint_token("E1001", "employee", ttl_s=-1)) is None  # expired
    with pytest.raises(ValueError):
        mint_token("E1001", "ceo")


async def _call(url: str, token: str | None, tool: str | None = None, args: dict | None = None):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with streamablehttp_client(url, headers=headers) as (read, write, _), ClientSession(read, write) as s:
        await s.initialize()
        if tool is None:
            return await s.list_tools()
        res = await s.call_tool(tool, args or {})
        return "".join(c.text for c in res.content if hasattr(c, "text"))


def test_http_requires_a_valid_token(live_server):
    url = f"{live_server}/mcp/"
    assert httpx.post(url, json={}).status_code == 401
    assert httpx.post(url, json={}, headers={"Authorization": "Bearer forged.token"}).status_code == 401


def test_http_tools_take_no_identity_arguments(live_server):
    tools = asyncio.run(_call(f"{live_server}/mcp/", mint_token("E1001", "employee"))).tools
    assert {t.name for t in tools} == {"search_policies", "get_ticket_status", "ticket_analytics"}
    for t in tools:
        assert not {"employee_id", "role", "user"} & set(t.inputSchema.get("properties", {}))


def test_http_identity_comes_from_the_token(live_server):
    url = f"{live_server}/mcp/"
    # TCK-1002 belongs to E1002
    as_other = asyncio.run(_call(url, mint_token("E1001", "employee"), "get_ticket_status", {"ticket_id": "TCK-1002"}))
    as_owner = asyncio.run(_call(url, mint_token("E1002", "employee"), "get_ticket_status", {"ticket_id": "TCK-1002"}))
    assert "No ticket" in as_other and "status=open" in as_owner


def test_http_analytics_admin_only(live_server, monkeypatch):
    url = f"{live_server}/mcp/"
    monkeypatch.setattr(t2s, "generate_sql", lambda q, usage: "SELECT COUNT(*) AS n FROM v_tickets")
    denied = json.loads(
        asyncio.run(_call(url, mint_token("E1001", "employee"), "ticket_analytics", {"question": "count"}))
    )
    assert "only available to IT admins" in denied["error"]
    ok = json.loads(asyncio.run(_call(url, mint_token("E1004", "it_admin"), "ticket_analytics", {"question": "count"})))
    assert ok["sql"].startswith("SELECT COUNT(*)") and ok["rows"][0][0] >= 24


def test_stdio_server_uses_configured_identity(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {
        **os.environ,
        "MCP_EMPLOYEE_ID": "E1004",
        "MCP_ROLE": "it_admin",
        "DATA_DIR": str(tmp_path),
        "ORBITDESK_ENV_FILE": "__none__",
        "REDIS_URL": "fakeredis://",
        "PYTHONPATH": str(ROOT),
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "app.mcp_server", "stdio"], env=env, cwd=str(ROOT)
    )

    async def go():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            res = await s.call_tool("get_ticket_status", {"ticket_id": "TCK-1002"})
            return names, "".join(c.text for c in res.content)

    t0 = time.time()
    names, text = asyncio.run(go())
    assert names == {"search_policies", "get_ticket_status", "ticket_analytics"}
    assert "status=open" in text  # it_admin from server config may see any ticket
    assert time.time() - t0 < 120


def test_stdio_refuses_to_start_without_a_valid_identity(tmp_path):
    import subprocess

    env = {
        **os.environ,
        "MCP_EMPLOYEE_ID": "E1001",
        "MCP_ROLE": "it_admin",
        "DATA_DIR": str(tmp_path),
        "ORBITDESK_ENV_FILE": "__none__",
        "REDIS_URL": "fakeredis://",
        "PYTHONPATH": str(ROOT),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "app.mcp_server", "stdio"],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        input="",
    )
    assert proc.returncode != 0 and "MCP_ROLE" in proc.stderr  # E1001 is not an it_admin
