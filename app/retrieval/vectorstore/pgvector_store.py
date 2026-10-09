"""pgvector implementation of VectorStore: vectors, metadata, role filter and keyword search in Postgres.

One table per collection:
  chunk_id PK | doc_id | text | metadata JSONB | department | role_employee/hr/it_admin BOOL
  | embedding vector(dim)  -- HNSW index, cosine distance (vector_cosine_ops)
  | tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED  -- GIN index

- Role filter = a SQL WHERE clause on the boolean column, so restricted rows are never candidates.
  With HNSW + WHERE, pgvector >= 0.8's iterative scan keeps scanning the graph until it finds k
  rows that pass the filter (otherwise a selective filter could return fewer than k).
- hybrid_query(): vector ranking + Postgres full-text ranking + Reciprocal Rank Fusion in ONE
  SQL statement, so hybrid search runs inside the database with no Python-side BM25 index.
"""

import json
import re

import numpy as np
from sqlalchemy import text

from app.db.database import get_engine
from app.ingest.models import Chunk
from app.retrieval.vectorstore.base import ScoredChunk

_COLS = "chunk_id, doc_id, text, metadata"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"what", "how", "the", "and", "for", "can", "does", "with", "you", "your", "are", "is", "do", "my", "a", "an"}


def _vec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in np.asarray(v, dtype=np.float32)) + "]"


def ts_or_query(query: str) -> str:
    """Natural-language question -> OR tsquery ('vpn | 809 | hotel'). plainto_tsquery would AND
    every word, which almost never matches a whole question."""
    toks = [t for t in _TOKEN.findall(query.lower()) if t not in _STOP and len(t) > 1]
    return " | ".join(dict.fromkeys(toks))


class PgVectorStore:
    def __init__(self, url: str, collection: str = "orbitdesk", dim: int = 384, ef_search: int = 100) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", collection):
            raise ValueError("collection must be [a-z0-9_]+ (it becomes part of a table name)")
        self.engine = get_engine(url)
        self.table = f"chunks_{collection}"
        self.dim = dim
        self.ef_search = ef_search
        self._create()

    def _create(self) -> None:
        t = self.table
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""CREATE TABLE IF NOT EXISTS {t} (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    department TEXT,
                    role_employee BOOLEAN NOT NULL,
                    role_hr BOOLEAN NOT NULL,
                    role_it_admin BOOLEAN NOT NULL,
                    embedding vector({self.dim}) NOT NULL,
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED)"""
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {t}_hnsw ON {t} USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                )
            )
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {t}_tsv ON {t} USING gin (tsv)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {t}_doc ON {t} (doc_id)"))

    @staticmethod
    def _filter(role: str | None, department: str | None) -> tuple[str, dict]:
        clauses, params = ["TRUE"], {}
        if role:
            if role not in ("employee", "hr", "it_admin"):
                raise ValueError(f"unknown role {role!r}")
            clauses.append(f"role_{role}")  # whitelisted column name, not user text
        if department:
            clauses.append("department = :department")
            params["department"] = department
        return " AND ".join(clauses), params

    def _session_settings(self, conn) -> None:
        conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}"))
        conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))

    @staticmethod
    def _chunk(row) -> Chunk:
        meta = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata)
        return Chunk(row.chunk_id, row.doc_id, row.text, meta)

    # --- VectorStore ---
    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "metadata": json.dumps(c.metadata),
                "department": c.metadata.get("department"),
                "re": bool(c.metadata.get("role_employee")),
                "rh": bool(c.metadata.get("role_hr")),
                "ra": bool(c.metadata.get("role_it_admin")),
                "emb": _vec(e),
            }
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""INSERT INTO {self.table} (chunk_id, doc_id, text, metadata, department,
                        role_employee, role_hr, role_it_admin, embedding)
                    VALUES (:chunk_id, :doc_id, :text, CAST(:metadata AS jsonb), :department, :re, :rh, :ra,
                            CAST(:emb AS vector))
                    ON CONFLICT (chunk_id) DO UPDATE SET doc_id = EXCLUDED.doc_id, text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata, department = EXCLUDED.department,
                        role_employee = EXCLUDED.role_employee, role_hr = EXCLUDED.role_hr,
                        role_it_admin = EXCLUDED.role_it_admin, embedding = EXCLUDED.embedding"""
                ),
                rows,
            )

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            with self.engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {self.table} WHERE chunk_id = ANY(:ids)"), {"ids": list(chunk_ids)})

    def ids_for_doc(self, doc_id: str) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT chunk_id FROM {self.table} WHERE doc_id = :d ORDER BY chunk_id"), {"d": doc_id}
            )
            return [r.chunk_id for r in rows]

    def query(self, embedding: np.ndarray, k: int, role: str, department: str | None = None) -> list[ScoredChunk]:
        where, params = self._filter(role, department)
        with self.engine.begin() as conn:
            self._session_settings(conn)
            rows = conn.execute(
                text(
                    f"SELECT {_COLS}, 1 - (embedding <=> CAST(:q AS vector)) AS score FROM {self.table} "
                    f"WHERE {where} ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"
                ),
                {"q": _vec(embedding), "k": k, **params},
            ).all()
        out = [ScoredChunk(self._chunk(r), float(r.score), "vector") for r in rows]
        return sorted(out, key=lambda s: s.score, reverse=True)  # relaxed_order may be slightly out of order

    def fts_query(self, query: str, k: int, role: str, department: str | None = None) -> list[ScoredChunk]:
        tsq = ts_or_query(query)
        if not tsq:
            return []
        where, params = self._filter(role, department)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {_COLS}, ts_rank_cd(tsv, q) AS score FROM {self.table}, to_tsquery('english', :tq) q "
                    f"WHERE {where} AND tsv @@ q ORDER BY score DESC LIMIT :k"
                ),
                {"tq": tsq, "k": k, **params},
            ).all()
        return [ScoredChunk(self._chunk(r), float(r.score), "fts") for r in rows]

    def hybrid_query(
        self, embedding: np.ndarray, query: str, k: int, role: str, department: str | None = None, rrf_k: int = 60
    ) -> list[ScoredChunk]:
        """Vector top-k and full-text top-k fused with RRF in one statement (same formula as fusion.py)."""
        tsq = ts_or_query(query) or "__nothing__"
        where, params = self._filter(role, department)
        sql = f"""
            WITH vec AS (
                SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> CAST(:q AS vector)) AS r
                FROM {self.table} WHERE {where}
                ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k),
            fts AS (
                SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(tsv, tq) DESC) AS r
                FROM {self.table}, to_tsquery('english', :tq) tq
                WHERE {where} AND tsv @@ tq
                ORDER BY ts_rank_cd(tsv, tq) DESC LIMIT :k)
            SELECT c.chunk_id, c.doc_id, c.text, c.metadata,
                   COALESCE(1.0 / (:rrf + vec.r), 0) + COALESCE(1.0 / (:rrf + fts.r), 0) AS score
            FROM vec FULL OUTER JOIN fts USING (chunk_id)
            JOIN {self.table} c USING (chunk_id)
            ORDER BY score DESC LIMIT :k"""
        with self.engine.begin() as conn:
            self._session_settings(conn)
            rows = conn.execute(text(sql), {"q": _vec(embedding), "tq": tsq, "k": k, "rrf": rrf_k, **params}).all()
        return [ScoredChunk(self._chunk(r), float(r.score), "rrf") for r in rows]

    def all_chunks(self, role: str | None = None, department: str | None = None) -> list[Chunk]:
        where, params = self._filter(role, department)
        with self.engine.connect() as conn:
            rows = conn.execute(text(f"SELECT {_COLS} FROM {self.table} WHERE {where} ORDER BY chunk_id"), params).all()
        return [self._chunk(r) for r in rows]

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {self.table}")).scalar_one())

    def drop(self) -> None:
        """Tests/evals only."""
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {self.table}"))
