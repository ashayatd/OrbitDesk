from pathlib import Path

from app.ingest.chunker import chunk_document
from app.ingest.models import DocMeta
from app.ingest.parsers import parse_file, parse_markdown

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"
META = DocMeta("d", "d.md", "Doc", "IT", ("employee",), "1", "2026-01-01")


def words(text: str) -> int:
    return len(text.split()) + 2


def test_markdown_heading_path():
    md = "# Title\n\n## A\n\ntext a\n\n### A1\n\ntext a1\n\n## B\n\ntext b\n"
    sections = parse_markdown(md)
    assert [s.heading_path for s in sections] == [["Title", "A"], ["Title", "A", "A1"], ["Title", "B"]]


def test_pdf_heading_path_rebuilt():
    sections = parse_file(DOCS / "travel-reimbursement.pdf")
    paths = [" > ".join(s.heading_path) for s in sections]
    assert "Travel and Expense Reimbursement Policy > 2. Booking Travel > 2.2 Hotels" in paths
    hotels = next(s for s in sections if s.heading_path[-1] == "2.2 Hotels")
    assert "USD 180" in hotels.text and "USD 260" in hotels.text


def test_chunks_fit_budget_and_overlap():
    long_text = " ".join(f"Sentence number {i} talks about leave rules." for i in range(200))
    sections = parse_markdown(f"# T\n\n## Long\n\n{long_text}")
    chunks = chunk_document(sections, META, "abcdef1234", words, chunk_size=60, overlap=15)
    assert len(chunks) > 5
    assert all(words(c.text) <= 60 for c in chunks)
    assert all(c.text.startswith("[T > Long]") for c in chunks)
    # overlap: the first sentence of chunk 2 also appears in chunk 1
    first_body, second_body = chunks[0].text.split("\n", 1)[1], chunks[1].text.split("\n", 1)[1]
    assert second_body[:30] in first_body


def test_chunk_metadata_roles_and_ids():
    chunks = chunk_document(parse_markdown("# T\n\n## S\n\nhello"), META, "abcdef1234", words)
    m = chunks[0].metadata
    assert chunks[0].chunk_id == "d::abcdef12::0"
    assert m["role_employee"] is True and m["role_hr"] is False and m["role_it_admin"] is False
    assert m["section"] == "S" and m["heading_path"] == "T > S"
