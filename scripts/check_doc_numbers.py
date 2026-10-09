"""Check that every measured number quoted in the docs exists in reports/*.json.

    uv run python scripts/check_doc_numbers.py      # exit 1 if a doc quotes a number no report contains

Scans decimal numbers (0.962, 152.8, 1.59x, 0.95) in README.md and docs/*.md. Integers are skipped
(too many are counts, ports or ids); configuration constants that are set, not measured, are allowlisted.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
SKIP_DOCS = {"PLAN.md"}  # the plan predates the measurements

# Set by configuration or by the task definition, not measured.
CONFIG_CONSTANTS = {
    "0.95": "semantic cache threshold / CI recall gate",
    "0.9": "CI MRR gate",
    "0.5": "prompt-guard decision threshold / no-citation confidence factor",
    "0.2": "max ID-based context precision with k=5 and one reference",
    "0.60": "travel policy mileage rate (document content)",
    "0.04": "price per audio hour / prompt guard price",
    "0.15": "price",
    "0.75": "price",
    "0.10": "price",
    "0.50": "price",
    "2.5": "one question as a share of 40 (arithmetic)",
    "3.11": "Python version",
    "1.5.9": "chromadb version",
    "1.14.2": "trl version",
    "1.15": "trl version",
    "0.4.3": "ragas version",
    "1.30": "mcp SDK version",
    "5.1": "Windows PowerShell version",
    "0.8": "pgvector version (iterative scan)",
    "0.4": "langchain-community version",
    "3.0": "version",
    "2.0": "doc version",
    "0.5B": "model size",
    "1.5B": "model size",
}


def report_numbers() -> set[str]:
    nums: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, bool):
            return
        elif isinstance(x, int | float):
            for nd in range(0, 5):
                nums.add(f"{abs(x):.{nd}f}")
            nums.add(str(abs(x)))
        elif isinstance(x, str):
            nums.update(re.findall(r"\d+\.\d+", x))

    for p in (ROOT / "reports").glob("*.json"):
        walk(json.loads(p.read_text(encoding="utf-8")))
    return nums


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # docs contain emoji
    known = report_numbers()
    problems = []
    for doc in DOCS:
        if doc.name in SKIP_DOCS:
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"(?<![\w.-])-?(\d+\.\d+)(?![\d.]|-\w)", line):
                n = m.group(1)
                if n in known or n in CONFIG_CONSTANTS or n.rstrip("0").rstrip(".") in known:
                    continue
                problems.append(f"{doc.relative_to(ROOT)}:{lineno}: {n}  | {line.strip()[:110]}")
    if problems:
        print("Numbers not found in reports/*.json (or the config allowlist):")
        print("\n".join(problems))
        sys.exit(1)
    print(f"ok: every decimal number in {len(DOCS) - len(SKIP_DOCS)} docs is backed by reports/ or config")


if __name__ == "__main__":
    main()
