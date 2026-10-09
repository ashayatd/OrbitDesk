"""Render the two PDF policies from markdown sources in scripts/pdf_sources/.

Run once (the PDFs are committed): `uv run python scripts/generate_pdfs.py`.
Headings are rendered as numbered bold lines ("2.1 Hotels") so the PDF parser
can rebuild the heading path from plain extracted text, just like a real PDF.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "scripts" / "pdf_sources"
OUT = ROOT / "data" / "docs"


def md_to_flowables(md: str) -> list:
    styles = getSampleStyleSheet()
    flow = []
    for line in md.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            flow += [Paragraph(line[2:], styles["Title"]), Spacer(1, 8)]
        elif line.startswith("### "):
            flow.append(Paragraph(line[4:], styles["Heading3"]))
        elif line.startswith("## "):
            flow.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("> "):
            flow.append(Paragraph(f"<i>{line[2:]}</i>", styles["Italic"]))
        else:
            flow += [Paragraph(line, styles["BodyText"]), Spacer(1, 4)]
    return flow


def main() -> None:
    for src in sorted(SOURCES.glob("*.md")):
        out = OUT / f"{src.stem}.pdf"
        doc = SimpleDocTemplate(str(out), pagesize=A4, title=src.stem, author="Orbit Systems (fictional)")
        doc.build(md_to_flowables(src.read_text(encoding="utf-8")))
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
