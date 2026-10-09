"""Recursive, token-aware chunking with overlap.

Each section (text under one heading) is chunked on its own, so chunks never span
two headings and every chunk carries its heading path. Sizes are measured with
the *embedding model's* tokenizer, because that model truncates at 512 tokens:
a chunk that is too long silently loses its tail at embedding time.
"""

from collections.abc import Callable

from app.ingest.models import Chunk, DocMeta, Section, role_flags

SEPARATORS = ["\n\n", "\n", ". ", " "]

TokenCounter = Callable[[str], int]


def split_recursive(
    text: str, max_tokens: int, count: TokenCounter, seps: list[str] = SEPARATORS
) -> list[str]:
    """Split text into pieces that each fit max_tokens, using the coarsest separator that works."""
    if count(text) <= max_tokens:
        return [text]
    if not seps:  # no separator left: hard cut by words (rare; one giant "word")
        words = text.split()
        half = max(1, len(words) // 2)
        return split_recursive(" ".join(words[:half]), max_tokens, count, []) + split_recursive(
            " ".join(words[half:]), max_tokens, count, []
        )
    sep, rest = seps[0], seps[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return split_recursive(text, max_tokens, count, rest)
    pieces: list[str] = []
    for i, p in enumerate(parts):
        p = p + (sep if sep == ". " and i < len(parts) - 1 else "")  # keep sentence periods
        pieces += split_recursive(p, max_tokens, count, rest)
    return pieces


def merge_with_overlap(pieces: list[str], max_tokens: int, overlap: int, count: TokenCounter) -> list[str]:
    """Greedily pack pieces into chunks of <= max_tokens; each new chunk starts with the
    trailing pieces of the previous one (up to `overlap` tokens) for context continuity."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    for piece in pieces:
        candidate = cur + [piece]
        if cur and count(" ".join(candidate)) > max_tokens:
            chunks.append(cur)
            tail: list[str] = []
            for prev in reversed(cur):  # take trailing pieces for overlap
                if count(" ".join([prev] + tail)) > overlap:
                    break
                tail.insert(0, prev)
            cur = tail + [piece]
            if count(" ".join(cur)) > max_tokens:  # overlap would overflow: drop it
                cur = [piece]
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return [" ".join(c).strip() for c in chunks]


def chunk_document(
    sections: list[Section],
    meta: DocMeta,
    content_hash: str,
    count: TokenCounter,
    chunk_size: int = 300,
    overlap: int = 45,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        heading = " > ".join(section.heading_path) or meta.title
        header = f"[{heading}]\n"
        budget = chunk_size - count(header)  # heading line is part of every chunk
        pieces = split_recursive(section.text, budget, count)
        for body in merge_with_overlap(pieces, budget, overlap, count):
            text = header + body
            n = len(chunks)
            chunks.append(
                Chunk(
                    # hash prefix in the id: new chunks never collide with the old version's ids
                    chunk_id=f"{meta.doc_id}::{content_hash[:8]}::{n}",
                    doc_id=meta.doc_id,
                    text=text,
                    metadata={
                        "doc_id": meta.doc_id,
                        "title": meta.title,
                        "heading_path": heading,
                        "section": section.heading_path[-1] if section.heading_path else meta.title,
                        "department": meta.department,
                        "version": meta.version,
                        "updated_at": meta.updated_at,
                        "content_hash": content_hash,
                        "chunk_index": n,
                        "token_count": count(text),
                        **role_flags(meta.allowed_roles),
                    },
                )
            )
    return chunks
