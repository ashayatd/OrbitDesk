"""POST /agent and POST /agent/confirm."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.agent.actions import ConfirmError, confirm_action
from app.deps import rate_limit, require_llm
from app.observability.request_log import log_request
from app.schemas.agent import AgentRequest, AgentResponse, ConfirmRequest
from app.services import get_agent_runner

router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse)
async def agent(req: AgentRequest, _=Depends(require_llm)) -> AgentResponse:
    rate_limit(req.session_id)
    try:
        out = await run_in_threadpool(get_agent_runner().run, req.message, req.session_id, req.role, req.employee_id)
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except Exception as e:
        log_request("/agent", session_id=req.session_id, role=req.role, error=type(e).__name__)
        raise
    stopped = out.get("stopped") or ""
    log_request(
        "/agent",
        request_id=out["request_id"],
        session_id=req.session_id,
        role=req.role,
        question=out["question"],  # PII-masked by the runner
        answered=not stopped,
        latency_ms=out["timings_ms"]["total"],
        tokens_in=out["tokens"]["input_tokens"],
        tokens_out=out["tokens"]["output_tokens"],
        cost_usd=out["tokens"]["cost_usd"],
        error=stopped if stopped.startswith("error") else None,
    )
    pending = out.pop("pending_action")
    return AgentResponse(
        **out,
        pending_action=pending and {k: pending[k] for k in ("confirmation_id", "kind", "summary", "payload")},
    )


@router.post("/agent/confirm")
def confirm(req: ConfirmRequest) -> dict:
    """Executes a pending action. Needs no LLM: it's a plain, idempotent DB write."""
    try:
        return confirm_action(req.confirmation_id, req.employee_id, req.idempotency_key)
    except ConfirmError as e:
        raise HTTPException(e.status, e.detail) from e
