"""Pydantic models for /ask and /search."""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["employee", "hr", "it_admin"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)  # hard cap; guardrails apply a stricter one
    session_id: str = Field(min_length=1, max_length=100)
    role: Role = "employee"
    stream: bool = True
    department: str | None = Field(None, description="optional filter, e.g. IT (used by /image)")


class CitationOut(BaseModel):
    doc_id: str
    section: str
    title: str | None = None
    chunk_id: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = 0.0
    estimated: bool = False
    calls: list[dict] = []


class AskResponse(BaseModel):
    """Structured output of the RAG pipeline (the final SSE `result` event)."""

    request_id: str
    answer: str
    citations: list[CitationOut]
    invalid_citations: list[str] = []  # removed from the answer; kept here for debugging/evals
    confidence: float = Field(ge=0, le=1)
    follow_up_suggestions: list[str] = []
    found: bool  # False -> "I couldn't find this in the policies"
    question: str
    rewritten_question: str | None = None
    retrieved: list[dict] = []  # chunk_id, doc_id, section, score (for evals / UI "sources")
    timings_ms: dict[str, float] = {}
    tokens: TokenUsage = TokenUsage()
    cache: Literal["none", "exact", "semantic"] = "none"
    prompt_version: str = "v1"
    guardrails: dict = {}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    role: Role = "employee"
    top_k: int = Field(5, ge=1, le=20)
