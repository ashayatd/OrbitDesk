"""POST /feedback and GET /admin/stats."""

from typing import Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.observability.request_log import admin_stats, save_feedback

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    request_id: str | None = Field(None, max_length=64)
    session_id: str | None = Field(None, max_length=100)
    rating: Literal["up", "down"]
    comment: str | None = Field(None, max_length=2000)


@router.post("/feedback", status_code=201)
def feedback(req: FeedbackRequest) -> dict:
    return {"id": save_feedback(req.request_id, req.session_id, req.rating, req.comment)}


@router.get("/admin/stats")
def stats(days: int = Query(14, ge=1, le=365)) -> dict:
    """No auth in this demo (documented in DECISIONS.md); in production this sits behind SSO + an admin role."""
    return admin_stats(days)
