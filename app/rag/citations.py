"""Parse [doc_id, section] citations from an answer and verify each against the retrieved chunks.

A citation is valid only if it points at a chunk that was actually in the context.
That catches both hallucinated sources and real documents the model "remembers"
but that were not retrieved (e.g. an HR-only doc for an employee).
"""

import re
from dataclasses import dataclass

from app.retrieval.vectorstore.base import ScoredChunk

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9-]*)\s*,\s*([^\]]+?)\]")


@dataclass
class Citation:
    doc_id: str
    section: str
    valid: bool
    chunk_id: str | None = None
    title: str | None = None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _section_matches(cited: str, actual: str) -> bool:
    """Exact match after normalisation, or the model cited only the number ("2.2")
    or only the name ("Hotels")."""
    c, a = _norm(cited), _norm(actual)
    if not c:
        return False
    if c == a:
        return True
    number = re.match(r"^(\d+(?: \d+)*)", a)
    return (number is not None and c == number.group(1)) or a.endswith(c)


def verify_citations(answer: str, chunks: list[ScoredChunk]) -> list[Citation]:
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for doc_id, section in CITATION_RE.findall(answer):
        key = (doc_id, section.strip())
        if key in seen:
            continue
        seen.add(key)
        match = next(
            (
                sc.chunk
                for sc in chunks
                if sc.chunk.doc_id == doc_id and _section_matches(section, sc.chunk.section)
            ),
            None,
        )
        out.append(
            Citation(
                doc_id=doc_id,
                section=section.strip(),
                valid=match is not None,
                chunk_id=match.chunk_id if match else None,
                title=match.metadata.get("title") if match else None,
            )
        )
    return out


def strip_invalid_citations(answer: str, citations: list[Citation]) -> str:
    """Remove citation markers that failed verification, so the user never sees a fake source."""
    for c in citations:
        if not c.valid:
            answer = re.sub(
                r"\s*\[" + re.escape(c.doc_id) + r"\s*,\s*" + re.escape(c.section) + r"\]", "", answer
            )
    return answer
