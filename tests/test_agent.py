"""Agent, actions, router and memory tests: fake LLM, fake router, no keys."""

import threading

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import app.agent.graph as graph_mod
from app.agent.actions import ConfirmError, confirm_action, propose_ticket
from app.agent.graph import STEP_LIMIT_REPLY, AgentRunner
from app.agent.memory import SessionMemory
from app.agent.router import RouteDecision, parse_route
from app.agent.tools import AgentContext, build_tools
from app.db.repo import count_tickets
from app.infra.redis_client import get_redis
from app.main import create_app

TICKET = {
    "title": "Laptop stolen",
    "description": "Stolen from car",
    "category": "Security",
    "priority": "P2",
}


class ScriptedLLM(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools and replays scripted AIMessages."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages: AIMessage):
    return lambda *a, **k: ScriptedLLM(messages=iter(messages))


class FixedRouter:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    def route(self, message, usage=None):
        return RouteDecision(self.intent, "low", "test")


def tool_call(name: str, args: dict, i: int = 1) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def runner(intent: str) -> AgentRunner:
    return AgentRunner(FixedRouter(intent), rag=None, memory=SessionMemory(get_redis()))


# ---------- actions / idempotency ----------


def test_double_confirm_creates_one_ticket():
    r = get_redis()
    before = count_tickets()
    action = propose_ticket(r, "E1001", "s", TICKET)
    first = confirm_action(r, action.confirmation_id, "E1001")
    second = confirm_action(r, action.confirmation_id, "E1001")  # double click / retry
    assert first["status"] == "created" and second["status"] == "duplicate"
    assert first["ticket"]["id"] == second["ticket"]["id"]
    assert count_tickets() == before + 1


def test_concurrent_confirms_create_one_ticket():
    r = get_redis()
    before = count_tickets()
    action = propose_ticket(r, "E1001", "s", TICKET)
    results, barrier = [], threading.Barrier(5)

    def click():
        barrier.wait()
        try:
            results.append(confirm_action(r, action.confirmation_id, "E1001")["status"])
        except ConfirmError as e:
            results.append(e.status)

    threads = [threading.Thread(target=click) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count_tickets() == before + 1
    assert results.count("created") == 1  # others: "duplicate" or 409 in-progress


def test_confirm_rejects_other_employee_and_unknown_id():
    r = get_redis()
    action = propose_ticket(r, "E1001", "s", TICKET)
    with pytest.raises(ConfirmError) as e:
        confirm_action(r, action.confirmation_id, "E1002")
    assert e.value.status == 403
    with pytest.raises(ConfirmError) as e:
        confirm_action(r, "does-not-exist", "E1001")
    assert e.value.status == 404


# ---------- tools: identity comes from context ----------


def test_tools_use_session_identity_not_model_args():
    ctx = AgentContext(employee_id="E1001", role="employee", session_id="s")
    tools = {t.name: t for t in build_tools(ctx, rag_pipeline=None)}
    assert tools["get_leave_balance"].args == {}  # the model cannot even pass an employee id
    assert "annual: 15 of 24" in tools["get_leave_balance"].invoke({})
    # TCK-1002 belongs to E1002: hidden from E1001, visible to it_admin
    assert "No ticket" in tools["get_ticket_status"].invoke({"ticket_id": "TCK-1002"})
    admin = {t.name: t for t in build_tools(AgentContext("E1004", "it_admin", "s"), None)}
    assert "status=open" in admin["get_ticket_status"].invoke({"ticket_id": "TCK-1002"})


# ---------- graph ----------


def test_agent_proposes_ticket_without_creating_it(monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "get_llm",
        scripted(tool_call("create_ticket", TICKET), AIMessage(content="Please confirm the ticket below.")),
    )
    before = count_tickets()
    out = runner("action").run("my laptop was stolen, open a ticket", "g1", "employee", "E1001")
    assert out["pending_action"]["kind"] == "create_ticket"
    assert out["steps"] == 2 and [t["tool"] for t in out["tool_calls"]] == ["create_ticket"]
    assert count_tickets() == before  # nothing written until /agent/confirm
    assert confirm_action(get_redis(), out["pending_action"]["confirmation_id"], "E1001")["status"] == "created"


def test_agent_step_limit(monkeypatch):
    loop = [tool_call("get_leave_balance", {}, i) for i in range(20)]
    monkeypatch.setattr(graph_mod, "get_llm", scripted(*loop))
    out = runner("action").run("balance?", "g2", "employee", "E1001")
    assert out["stopped"] == "step_limit" and out["reply"] == STEP_LIMIT_REPLY


def test_small_talk_and_out_of_scope_use_no_llm(monkeypatch):
    monkeypatch.setattr(graph_mod, "get_llm", scripted())  # would raise StopIteration if called
    assert "OrbitDesk" in runner("small_talk").run("hello", "g3", "employee", "E1001")["reply"]
    assert "only help" in runner("out_of_scope").run("tell me a joke", "g4", "employee", "E1001")["reply"]


def test_agent_rejects_role_mismatch(monkeypatch):
    monkeypatch.setattr(graph_mod, "get_llm", scripted())
    with pytest.raises(PermissionError):
        runner("action").run("show salary bands", "g5", "hr", "E1001")  # E1001 is not HR


# ---------- router parsing ----------


def test_parse_route():
    assert parse_route('{"intent": "action", "urgency": "high"}') == ("action", "high")
    assert parse_route('Sure! ```json\n{"intent":"small_talk","urgency":"low"}```') == ("small_talk", "low")
    assert parse_route('{"intent": "delete_db", "urgency": "low"}') is None
    assert parse_route("no json") is None


# ---------- memory ----------


def test_memory_compacts_into_summary_within_budget():
    mem = SessionMemory(get_redis(), token_budget=200, recent_turns=2)
    for i in range(6):
        mem.append("m1", f"question number {i} about VPN error {800 + i}", "answer " * 30)
    summary, turns = mem.context("m1")
    assert len(turns) <= 4
    assert "question number 0" in summary  # oldest turns survive as summary (extractive fallback, no LLM)
    total = len(summary) // 4 + sum(len(m["content"]) // 4 for m in turns)
    assert total <= 200


# ---------- API ----------


def test_agent_endpoints_without_key():
    client = TestClient(create_app())
    r = client.post("/agent", json={"message": "hi", "session_id": "x", "employee_id": "E1001"})
    assert r.status_code == 503
    action = propose_ticket(get_redis(), "E1001", "x", TICKET)
    ok = client.post("/agent/confirm", json={"confirmation_id": action.confirmation_id, "employee_id": "E1001"})
    assert ok.status_code == 200 and ok.json()["status"] == "created"
    again = client.post("/agent/confirm", json={"confirmation_id": action.confirmation_id, "employee_id": "E1001"})
    assert again.json()["status"] == "duplicate"
