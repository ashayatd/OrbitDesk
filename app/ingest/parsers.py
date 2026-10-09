"""Turn a source file into a list of Sections, each tagged with its heading path."""

import re
from pathlib import Path

from pypdf import PdfReader

from app.ingest.models import Section

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# PDF headings come out as plain lines like "2.1 Hotels" or "3. Meals and Per Diem".
_PDF_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^.:]{1,60})$")
_DISCLAIMER = "Fictional document created for the OrbitDesk demo"


def parse_file(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return parse_markdown(path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return parse_pdf(path)
    raise ValueError(f"unsupported file type: {suffix}")


def parse_markdown(text: str) -> list[Section]:
    """Split on headings; heading_path tracks the stack of headings above each block."""
    stack: list[tuple[int, str]] = []  # (level, title)
    sections: list[Section] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body and _DISCLAIMER not in body:
            sections.append(Section([t for _, t in stack], body))
        buf.clear()

    for line in text.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def parse_pdf(path: Path) -> list[Section]:
    """Extract text page by page, then rebuild markdown-like structure from numbered headings.

    The first non-empty line is the document title. "2." is level 2, "2.1" is level 3.
    """
    lines = [ln.strip() for p in PdfReader(path).pages for ln in (p.extract_text() or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []
    md = [f"# {lines[0]}"]
    for ln in lines[1:]:
        m = _PDF_HEADING.match(ln)
        if m:
            depth = m.group(1).count(".") + 2
            md.append(f"{'#' * depth} {ln}")
        else:
            md.append(ln)
    return parse_markdown(_join_wrapped_lines(md))


def _join_wrapped_lines(lines: list[str]) -> str:
    """PDF extraction breaks paragraphs at the page width; rejoin lines into paragraphs."""
    out: list[str] = []
    for ln in lines:
        if ln.startswith("#"):
            out += ["", ln, ""]
        elif out and out[-1] and not out[-1].startswith("#") and not out[-1].endswith((".", ":")):
            out[-1] += " " + ln  # continuation of the previous wrapped line
        else:
            out.append(ln)
    return "\n".join(out)
