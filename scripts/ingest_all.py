"""Seed the SQLite DB and ingest every document in data/docs/manifest.yaml.

uv run python scripts/ingest_all.py           # skips unchanged docs (content hash)
uv run python scripts/ingest_all.py --force   # re-index everything
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.seed import seed  # noqa: E402
from app.ingest.jobs import ingest_manifest  # noqa: E402
from app.log import configure_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    configure_logging("WARNING")
    seed()
    results = ingest_manifest(force=args.force)
    print(
        json.dumps(
            [{k: r[k] for k in ("doc_id", "status", "chunks_written", "chunks_deleted")} for r in results],
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
