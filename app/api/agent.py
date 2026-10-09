"""POST /agent and POST /agent/confirm."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.agent.actions import ConfirmError, confirm_action
from app.deps import rate_limit, require_llm
from app.infra.redis_client import get_redis
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
    pending = out.pop("pending_action")
    return AgentResponse(
        **out,
        pending_action=pending and {k: pending[k] for k in ("confirmation_id", "kind", "summary", "payload")},
    )


@router.post("/agent/confirm")
def confirm(req: ConfirmRequest) -> dict:
    """Executes a pending action. Needs no LLM: it's a plain, idempotent DB write."""
    try:
        return confirm_action(get_redis(), req.confirmation_id, req.employee_id, req.idempotency_key)
    except ConfirmError as e:
        raise HTTPException(e.status, e.detail) from e
