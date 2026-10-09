"""Pydantic models for /agent and /agent/confirm."""

from pydantic import BaseModel, Field

from app.schemas.ask import Role, TokenUsage


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=100)
    role: Role = "employee"
    employee_id: str = Field(pattern=r"^E\d{4}$")


class PendingActionOut(BaseModel):
    confirmation_id: str
    kind: str
    summary: str
    payload: dict


class AgentResponse(BaseModel):
    reply: str
    route: dict
    pending_action: PendingActionOut | None = None
    citations: list[dict] = []
    tool_calls: list[dict] = []
    steps: int = 0
    stopped: str | None = None
    timings_ms: dict[str, float] = {}
    tokens: TokenUsage = TokenUsage()
    guardrails: dict = {}


class ConfirmRequest(BaseModel):
    confirmation_id: str = Field(min_length=8, max_length=64)
    employee_id: str = Field(pattern=r"^E\d{4}$")
    idempotency_key: str | None = Field(None, max_length=128, description="defaults to confirmation_id")
