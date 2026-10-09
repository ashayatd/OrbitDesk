"""POST /ingest: upload a document + its access metadata, enqueue indexing."""

import re
from datetime import date
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.infra.queue import get_queue, job_status
from app.ingest.jobs import ingest_upload_job
from app.ingest.models import ROLES

router = APIRouter(tags=["ingest"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")


@router.post("/ingest", status_code=202)
async def ingest(
    file: UploadFile = File(...),
    title: str = Form(...),
    department: str = Form(...),
    allowed_roles: str = Form("employee,hr,it_admin", description="comma-separated"),
    version: str = Form("1.0"),
    updated_at: str = Form(default_factory=lambda: date.today().isoformat()),
    doc_id: str | None = Form(None),
    force: bool = Form(False),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"unsupported file type {suffix!r}; allowed: {sorted(ALLOWED_SUFFIXES)}")
    roles = [r.strip() for r in allowed_roles.split(",") if r.strip()]
    if not roles or set(roles) - set(ROLES):
        raise HTTPException(422, f"allowed_roles must be a subset of {ROLES}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file larger than 5 MB")

    doc_id = doc_id or _slug(file.filename or "upload")
    meta = {
        "doc_id": doc_id,
        "title": title,
        "department": department,
        "allowed_roles": roles,
        "version": version,
        "updated_at": updated_at,
    }
    job = get_queue().enqueue(
        ingest_upload_job, f"{doc_id}{suffix}", data, meta, force, job_timeout=600, result_ttl=86400
    )
    return {"job_id": job.id, "doc_id": doc_id, "status_url": f"/ingest/{job.id}"}


@router.get("/ingest/{job_id}")
def ingest_status(job_id: str) -> dict:
    status = job_status(job_id)
    if status is None:
        raise HTTPException(404, "unknown job id")
    return status
