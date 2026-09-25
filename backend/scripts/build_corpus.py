"""Download the seed filings from SEC EDGAR and print them to PDF.

    python -m scripts.build_corpus --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe"

EDGAR serves filings as (inline-XBRL) HTML. The app is PDF-first, since that's
what people actually upload and what page-level citations need, so each filing
is cleaned up and printed to PDF with headless Chrome. The PDFs are committed
under data/seed/pdfs, so this only needs re-running to refresh the corpus.

SEC asks automated clients to send a descriptive User-Agent; set SEC_USER_AGENT
to "<your app> <contact email>".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SEED = Path(__file__).resolve().parent.parent / "data" / "seed"


def fetch(url: str) -> bytes:
    ua = os.environ.get("SEC_USER_AGENT", "FinSight-RAG-Demo research@example.com")
    for attempt in range(6):
        try:
            time.sleep(0.5)  # stay well under EDGAR's 10 requests/second limit
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"giving up on {url}")


def clean_html(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    # C1 control characters are cp1252 punctuation that was mis-decoded upstream.
    text = re.sub(r"[\x80-\x9f]", lambda m: bytes([ord(m.group())]).decode("cp1252", errors="replace"), text)
    text = re.sub(r"^.*?(<html)", r"\1", text, count=1, flags=re.S | re.I)  # drop the SGML <DOCUMENT> wrapper
    text = re.sub(r"</TEXT>\s*</DOCUMENT>\s*$", "", text, flags=re.I)
    text = re.sub(r"<ix:header>.*?</ix:header>", "", text, flags=re.S | re.I)  # hidden XBRL facts
    text = re.sub(r"<img[^>]*>", "", text, flags=re.I)  # relative image links can't resolve offline
    text = re.sub(
        r"(<head[^>]*>)", r"\1<meta charset='utf-8'><style>@page{size:Letter;margin:0.6in}</style>", text, count=1, flags=re.I
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", required=True, help="path to a Chrome/Chromium/Edge binary")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    manifest = json.loads((SEED / "manifest.json").read_text(encoding="utf-8"))["documents"]
    out_dir = SEED / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for doc in manifest:
            if args.only and doc["id"] not in args.only:
                continue
            html = Path(tmp) / f"{doc['id']}.html"
            # Non-ASCII as numeric entities: immune to however Chrome guesses the charset.
            html.write_text(clean_html(fetch(doc["source"])), encoding="ascii", errors="xmlcharrefreplace")
            pdf = out_dir / f"{doc['id']}.pdf"
            subprocess.run(
                [args.chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                 f"--print-to-pdf={pdf}", html.resolve().as_uri()],
                check=True, capture_output=True,
            )
            print(f"{doc['id']}: {pdf.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
