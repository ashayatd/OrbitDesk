"""Ingestion pipeline: hash check -> lock -> parse -> chunk -> embed (batched) -> swap chunks.

Re-ingesting a changed document replaces its chunks with an *upsert-then-delete*
swap: new chunks (new ids, because the id contains the content hash) are written
first, then the old ids are deleted. Readers briefly see old+new rather than
nothing, which is the safer failure mode for a helpdesk.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import redis
import yaml

from app.cache.index_version import bump_index_version
from app.db.database import db_session, init_db
from app.infra.locks import distributed_lock
from app.ingest.chunker import chunk_document
from app.ingest.models import DocMeta
from app.ingest.parsers import parse_file
from app.log import get_logger
from app.retrieval.embeddings import Embedder
from app.retrieval.vectorstore.base import VectorStore

log = get_logger(__name__)


@dataclass
class IngestResult:
    doc_id: str
    status: str  # indexed | skipped_unchanged
    chunks_written: int = 0
    chunks_deleted: int = 0
    content_hash: str = ""
    index_version: int | None = None
    timings_ms: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def content_hash(path: Path, meta: DocMeta) -> str:
    """Hash of file bytes + access metadata: a role change must also trigger re-indexing."""
    h = hashlib.sha256(path.read_bytes())
    h.update(json.dumps([sorted(meta.allowed_roles), meta.version, meta.title, meta.department]).encode())
    return h.hexdigest()


def load_manifest(path: Path) -> list[DocMeta]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [DocMeta(**{**d, "allowed_roles": tuple(d["allowed_roles"])}) for d in data["documents"]]


def _registered_hash(doc_id: str, db_path: Path | None) -> str | None:
    with db_session(db_path) as conn:
        row = conn.execute("SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return row["content_hash"] if row else None


def ingest_document(
    path: Path,
    meta: DocMeta,
    *,
    store: VectorStore,
    embedder: Embedder,
    r: redis.Redis,
    db_path: Path | None = None,
    chunk_size: int = 300,
    overlap: int = 45,
    batch_size: int = 8,
    force: bool = False,
    lock_wait_s: float = 30,
) -> IngestResult:
    init_db(db_path)
    h = content_hash(path, meta)

    def unchanged() -> bool:
        return not force and _registered_hash(meta.doc_id, db_path) == h and bool(store.ids_for_doc(meta.doc_id))

    if unchanged():  # cheap check before taking the lock
        return IngestResult(meta.doc_id, "skipped_unchanged", content_hash=h)

    with distributed_lock(r, f"ingest:{meta.doc_id}", wait_s=lock_wait_s):
        if unchanged():  # double-check: another worker may have finished while we waited
            return IngestResult(meta.doc_id, "skipped_unchanged", content_hash=h)

        t: dict[str, float] = {}
        t0 = time.perf_counter()
        sections = parse_file(path)
        t["parse"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        chunks = chunk_document(sections, meta, h, embedder.count_tokens, chunk_size, overlap)
        t["chunk"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        vectors = embedder.embed_documents([c.text for c in chunks], batch_size=batch_size)
        t["embed"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        old_ids = set(store.ids_for_doc(meta.doc_id))
        store.upsert(chunks, vectors)
        stale = sorted(old_ids - {c.chunk_id for c in chunks})
        store.delete(stale)
        t["upsert"] = time.perf_counter() - t0

        with db_session(db_path) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, filename, content_hash, version, chunk_count)"
                " VALUES (?,?,?,?,?) ON CONFLICT(doc_id) DO UPDATE SET filename=excluded.filename,"
                " content_hash=excluded.content_hash, version=excluded.version,"
                " chunk_count=excluded.chunk_count, indexed_at=datetime('now')",
                (meta.doc_id, path.name, h, meta.version, len(chunks)),
            )
        version = bump_index_version(r)

    timings = {k: round(v * 1000, 1) for k, v in t.items()}
    log.info("ingested", doc_id=meta.doc_id, chunks=len(chunks), deleted=len(stale), timings_ms=timings)
    return IngestResult(meta.doc_id, "indexed", len(chunks), len(stale), h, version, timings)
