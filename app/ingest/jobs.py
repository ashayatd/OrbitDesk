"""Job entrypoints executed by the RQ worker (or in-process when QUEUE_MODE=inline).

The upload's bytes travel inside the job payload, not as a path on a shared
disk, so a worker on another machine/container needs nothing but Redis + the
vector store. Uploads are capped at 5 MB (see app/api/ingest.py).
"""

from pathlib import Path

from app.config import get_settings
from app.infra.redis_client import get_redis
from app.ingest.models import DocMeta
from app.ingest.pipeline import IngestResult, ingest_document, load_manifest
from app.retrieval.embeddings import get_embedder
from app.retrieval.vectorstore import get_store


def _ingest(path: Path, meta: DocMeta, force: bool = False) -> IngestResult:
    s = get_settings()
    return ingest_document(
        path,
        meta,
        store=get_store(),
        embedder=get_embedder(),
        r=get_redis(),
        chunk_size=s.chunk_size,
        overlap=s.chunk_overlap,
        batch_size=s.embed_batch_size,
        force=force,
    )


def ingest_upload_job(filename: str, data: bytes, meta: dict, force: bool = False) -> dict:
    """RQ job: persist the uploaded bytes locally, then run the pipeline."""
    s = get_settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    path = s.uploads_dir / filename
    path.write_bytes(data)
    doc = DocMeta(**{**meta, "file": filename, "allowed_roles": tuple(meta["allowed_roles"])})
    return _ingest(path, doc, force).to_dict()


def ingest_manifest(force: bool = False) -> list[dict]:
    """Ingest every document in data/docs/manifest.yaml (startup + scripts/ingest_all.py)."""
    s = get_settings()
    results = []
    for meta in load_manifest(s.docs_dir / "manifest.yaml"):
        results.append(_ingest(s.docs_dir / meta.file, meta, force).to_dict())
    return results
