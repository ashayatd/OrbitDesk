"""LangGraph agent for POST /agent.

    START -> route --policy_question--> policy (Phase 2 RAG pipeline, no tools) -> END
                   --action-----------> agent <-> tools (loop, max steps + deadline) -> END
                   --small_talk-------> small_talk (canned, 0 tokens) -> END
                   --out_of_scope-----> refuse (canned, 0 tokens) -> END

Policy questions skip the tool loop: one retrieval + one generation is cheaper and
more predictable than letting the agent decide to search. The agent path still has
search_policies for mixed messages ("my laptop was stolen, what do I do? open a ticket").
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.memory import SessionMemory
from app.agent.router import Router
from app.agent.tools import AgentContext, build_tools, usage_from_response
from app.cache.index_version import get_index_version
from app.config import get_settings
from app.db.repo import get_employee
from app.errors import InputRejectedError
from app.guardrails.injection import check_input
from app.guardrails.output import BLOCKED_REPLY, check_output
from app.guardrails.pii import mask_pii
from app.llm.client import get_llm
from app.llm.usage import Usage
from app.log import get_logger
from app.observability.tracing import trace_callbacks, traced
from app.rag.pipeline import RagPipeline
from app.rag.prompts import load_prompt
from app.schemas.ask import AskRequest

log = get_logger(__name__)

SMALL_TALK_REPLY = (
    "Hi! I'm OrbitDesk. I can answer questions about Orbit HR policies and IT runbooks, "
    "check your leave balance, and raise or check IT tickets. What do you need?"
)
OUT_OF_SCOPE_REPLY = "Sorry, I can only help with Orbit Systems HR policies, IT support, leave balances and IT tickets."
STEP_LIMIT_REPLY = "I couldn't finish this within the allowed number of steps. Please rephrase or split the request."
TIMEOUT_REPLY = "This is taking too long, so I stopped. Please try again in a moment."
ERROR_REPLY = "Sorry, something went wrong on my side. Please try again."
REFUSAL_REPLY = "I can't help with that request. I can answer questions about Orbit HR policies and IT support."


class AgentState(TypedDict, total=False):
    message: str
    messages: Annotated[list[BaseMessage], add_messages]
    route: dict
    reply: str
    steps: int
    deadline: float
    stopped: str  # "" | step_limit | timeout


class AgentRunner:
    def __init__(self, router: Router, rag: RagPipeline, memory: SessionMemory) -> None:
        self.router = router
        self.rag = rag
        self.memory = memory
        self.settings = get_settings()

    # ---- graph construction (per request: tools close over the request's identity) ----
    def _build(self, ctx: AgentContext):
        tools = build_tools(ctx, self.rag)
        llm = get_llm("main", tools=tools, max_tokens=500)
        max_steps = self.settings.agent_max_steps

        def route(state: AgentState) -> AgentState:
            return {"route": self.router.route(state["message"], ctx.usage).to_dict()}

        def policy(state: AgentState) -> AgentState:
            req = AskRequest(question=state["message"], session_id=ctx.session_id, role=ctx.role, stream=False)
            # same request id -> the RAG calls land in the agent's Langfuse trace
            res = [d for n, d in self.rag.run(req, ctx.usage.trace_id, use_memory=False) if n == "result"][0]
            ctx.usage.merge(usage_from_response(res))
            ctx.citations += [c.model_dump() for c in res.citations]
            return {"reply": res.answer}

        def agent(state: AgentState) -> AgentState:
            steps = state.get("steps", 0)
            if steps >= max_steps:
                return {"stopped": "step_limit", "reply": STEP_LIMIT_REPLY}
            if time.monotonic() > state["deadline"]:
                return {"stopped": "timeout", "reply": TIMEOUT_REPLY}
            resp = llm.invoke(state["messages"], config=traced(ctx.usage))
            ctx.usage.add_message(f"agent_step_{steps + 1}", resp)
            return {"messages": [resp], "steps": steps + 1}

        def after_agent(state: AgentState) -> str:
            if state.get("stopped"):
                return END
            last = state["messages"][-1]
            return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finish"

        def finish(state: AgentState) -> AgentState:
            return {"reply": str(state["messages"][-1].content).strip()}

        g = StateGraph(AgentState)
        g.add_node("route", route)
        g.add_node("policy", policy)
        g.add_node("small_talk", lambda s: {"reply": SMALL_TALK_REPLY})
        g.add_node("refuse", lambda s: {"reply": OUT_OF_SCOPE_REPLY})
        g.add_node("agent", agent)
        g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        g.add_node("finish", finish)
        g.add_edge(START, "route")
        g.add_conditional_edges(
            "route",
            lambda s: {"policy_question": "policy", "action": "agent", "small_talk": "small_talk"}.get(
                s["route"]["intent"], "refuse"
            ),
        )
        g.add_conditional_edges("agent", after_agent, ["tools", "finish", END])
        g.add_edge("tools", "agent")
        for n in ("policy", "small_talk", "refuse", "finish"):
            g.add_edge(n, END)
        return g.compile()

    def _initial_messages(self, ctx: AgentContext, employee: dict, message: str) -> list[BaseMessage]:
        p = load_prompt("agent", "v1")
        msgs: list[BaseMessage] = [
            SystemMessage(
                p["system"].format(
                    name=employee["name"],
                    employee_id=ctx.employee_id,
                    role=ctx.role,
                    department=employee["department"],
                )
            )
        ]
        summary, history = self.memory.context(ctx.session_id)
        if summary:
            msgs.append(SystemMessage(f"Summary of the earlier conversation: {summary}"))
        for m in history:
            msgs.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
        msgs.append(HumanMessage(message))
        return msgs

    # ---- public API ----
    def run(self, message: str, session_id: str, role: str, employee_id: str) -> dict:
        t0 = time.perf_counter()
        employee = get_employee(employee_id)
        if employee is None:
            raise PermissionError(f"unknown employee {employee_id}")
        if employee["role"] != role:
            raise PermissionError("role does not match the employee record")
        request_id = uuid.uuid4().hex
        ctx = AgentContext(employee_id=employee_id, role=role, session_id=session_id, usage=Usage(trace_id=request_id))

        # Input guardrails (same as /ask): length cap, PII masking, injection/unsafe check.
        if len(message) > self.settings.max_question_chars:
            raise InputRejectedError(f"message longer than {self.settings.max_question_chars} characters")
        masked = mask_pii(message)
        message = masked.text
        verdict = check_input(message, ctx.usage)
        guard: dict = {"input_check": verdict.to_dict(), **({"pii_masked": masked.counts} if masked.masked else {})}
        if verdict.blocked:
            self.memory.append(session_id, message, REFUSAL_REPLY)
            return self._response(REFUSAL_REPLY, {}, ctx, {}, t0, guard, message)

        timeout = self.settings.agent_timeout_s
        state: AgentState = {
            "message": message,
            "messages": self._initial_messages(ctx, employee, message),
            "steps": 0,
            "deadline": time.monotonic() + timeout,
        }
        graph = self._build(ctx)
        # Hard wall-clock cap on top of the in-graph deadline check (which can't interrupt
        # a single slow LLM call). recursion_limit is a second guard against loops.
        pool = ThreadPoolExecutor(max_workers=1)
        config = {"recursion_limit": 4 * self.settings.agent_max_steps + 6, "callbacks": trace_callbacks(request_id)}
        future = pool.submit(graph.invoke, state, config)
        try:
            final = future.result(timeout=timeout + 5)
        except FutureTimeout:
            final = {"reply": TIMEOUT_REPLY, "stopped": "timeout", "route": {}}
        except Exception as e:
            log.exception("agent_failed")
            final = {
                "reply": ERROR_REPLY,
                "stopped": f"error:{type(e).__name__}",
                "route": state.get("route", {}),
            }
        finally:
            pool.shutdown(wait=False)

        reply = final.get("reply", ERROR_REPLY)
        # Output guardrail: never return restricted-document content to this role.
        restricted = self.rag.restricted.for_role(role, get_index_version(self.rag.r)) if self.rag else set()
        blocked = check_output(reply, restricted)
        if blocked:
            guard["output_blocked"] = blocked
            reply = BLOCKED_REPLY
        self.memory.append(session_id, message, reply)
        return self._response(reply, final.get("route", {}), ctx, final, t0, guard, message)

    @staticmethod
    def _response(
        reply: str, route: dict, ctx: AgentContext, final: dict, t0: float, guard: dict, question: str
    ) -> dict:
        return {
            "request_id": ctx.usage.trace_id,
            "question": question,  # PII-masked
            "reply": reply,
            "route": route,
            "pending_action": ctx.pending[-1].to_dict() if ctx.pending else None,
            "citations": ctx.citations,
            "tool_calls": ctx.tool_log,
            "steps": final.get("steps", 0),
            "stopped": final.get("stopped") or None,
            "timings_ms": {"total": round((time.perf_counter() - t0) * 1000, 1)},
            "tokens": ctx.usage.to_dict(),
            "guardrails": guard,
        }
