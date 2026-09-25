"""Build the seed index from data/seed/manifest.json + data/seed/pdfs/.

    python -m scripts.build_index            # full rebuild into data/seed/index
    python -m scripts.build_index --only wipro-q1-fy2027-earnings

Runs fully offline (local ONNX embeddings), so rebuilding costs no API quota.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_DIR  # noqa: E402
from app.ingest.pipeline import ingest_pdf  # noqa: E402
from app.store import Index  # noqa: E402

SEED = BACKEND_DIR / "data" / "seed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="doc ids to include (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = json.loads((SEED / "manifest.json").read_text(encoding="utf-8"))["documents"]
    out = SEED / "index"
    if out.exists():
        shutil.rmtree(out)
    index = Index(out)
    for doc in manifest:
        if args.only and doc["id"] not in args.only:
            continue
        ingest_pdf(index, str(SEED / "pdfs" / f"{doc['id']}.pdf"), doc, save=False)
    index.save()
    (out / "VERSION").write_text(time.strftime("%Y%m%d%H%M%S"))
    print(f"seed index: {len(index.documents)} docs, {len(index.chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
